from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ExpandedExperimentJob, SearchExecutionResult
from .utils import now_utc_iso


class RunStateStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.state_path = self.run_root / "state.json"
        self.events_path = self.run_root / "events.jsonl"
        self.logs_dir = self.run_root / "logs"
        self.summary_json_path = self.run_root / "summary.json"
        self.summary_md_path = self.run_root / "summary.md"

    def initialize_new(
        self,
        *,
        run_id: str,
        manifest_path: Path,
        jobs: list[ExpandedExperimentJob],
    ) -> dict[str, Any]:
        if self.state_path.exists():
            raise FileExistsError(f"run state already exists: {self.state_path}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = now_utc_iso()
        state = {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "status": "running",
            "created_at": ts,
            "updated_at": ts,
            "jobs": [self._job_payload(job) for job in jobs],
        }
        self.save(state)
        self.append_event(
            state,
            event_type="run_initialized",
            experiment_id=None,
            payload={
                "run_id": run_id,
                "job_count": len(jobs),
            },
        )
        return state

    def load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"run state does not exist: {self.state_path}")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("run state file is malformed")
        return payload

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_utc_iso()
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def append_event(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        experiment_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        event = {
            "ts": now_utc_iso(),
            "event_type": event_type,
            "experiment_id": experiment_id,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
        state["updated_at"] = now_utc_iso()

    def reconcile_jobs(self, state: dict[str, Any]) -> None:
        changed = False
        for job in state.get("jobs", []):
            if job.get("status") == "running":
                job["status"] = "planned"
                changed = True
            result_dir = Path(str(job["result_dir"]))
            search_trace = result_dir / "search_trace.json"
            final_report_json = result_dir / "final_report.json"
            final_report_md = result_dir / "final_report.md"
            if search_trace.is_file() and final_report_json.is_file() and job.get("status") != "succeeded":
                job["status"] = "succeeded"
                job["last_error"] = None
                job["artifacts"] = {
                    "search_trace": str(search_trace),
                    "final_report_json": str(final_report_json),
                    "final_report_md": str(final_report_md) if final_report_md.is_file() else None,
                }
                changed = True
        if changed:
            self.save(state)
            self.append_event(
                state,
                event_type="state_reconciled",
                experiment_id=None,
                payload={"reason": "resume_reconciliation"},
            )

    def find_job(self, state: dict[str, Any], experiment_id: str) -> dict[str, Any]:
        for job in state.get("jobs", []):
            if job.get("experiment_id") == experiment_id:
                return job
        raise KeyError(f"job not found in state: {experiment_id}")

    def increment_attempt(self, state: dict[str, Any], experiment_id: str, *, kind: str) -> None:
        if kind not in {"startup", "search"}:
            raise ValueError(f"unsupported attempt kind: {kind!r}")
        job = self.find_job(state, experiment_id)
        attempts = job.setdefault("attempts", {"startup": 0, "search": 0})
        attempts[kind] = int(attempts.get(kind, 0)) + 1
        self.save(state)

    def set_job_status(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        status: str,
        last_error: str | None = None,
    ) -> None:
        job = self.find_job(state, experiment_id)
        job["status"] = status
        if last_error is not None:
            job["last_error"] = last_error
        self.save(state)

    def mark_job_succeeded(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        result: SearchExecutionResult,
    ) -> None:
        job = self.find_job(state, experiment_id)
        job["status"] = "succeeded"
        job["last_error"] = None
        job["artifacts"] = {
            "search_trace": None if result.search_trace_path is None else str(result.search_trace_path),
            "final_report_json": (
                None if result.final_report_json_path is None else str(result.final_report_json_path)
            ),
            "final_report_md": None if result.final_report_md_path is None else str(result.final_report_md_path),
            "stdout_log": str(result.stdout_log),
            "stderr_log": str(result.stderr_log),
        }
        self.save(state)

    def mark_job_failed(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        error: str,
    ) -> None:
        job = self.find_job(state, experiment_id)
        job["status"] = "failed"
        job["last_error"] = error
        self.save(state)

    def summarize(self, state: dict[str, Any]) -> dict[str, Any]:
        jobs = state.get("jobs", [])
        counts = {"planned": 0, "running": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        job_summaries: list[dict[str, Any]] = []
        for job in jobs:
            status = str(job.get("status", "planned"))
            if status not in counts:
                status = "planned"
            counts[status] += 1
            job_summaries.append(
                {
                    "experiment_id": job.get("experiment_id"),
                    "status": status,
                    "result_dir": job.get("result_dir"),
                    "attempts": job.get("attempts", {}),
                    "last_error": job.get("last_error"),
                }
            )
        return {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "updated_at": state.get("updated_at"),
            "counts": counts,
            "jobs": job_summaries,
        }

    def write_summary_files(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = self.summarize(state)
        self.summary_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        lines = [
            f"# Orchestrator Summary: {summary['run_id']}",
            "",
            f"- Status: {summary['status']}",
            f"- Updated At: {summary['updated_at']}",
            f"- Planned: {summary['counts']['planned']}",
            f"- Running: {summary['counts']['running']}",
            f"- Succeeded: {summary['counts']['succeeded']}",
            f"- Failed: {summary['counts']['failed']}",
            f"- Skipped: {summary['counts']['skipped']}",
            "",
            "## Jobs",
            "",
            "| Experiment ID | Status | Startup Attempts | Search Attempts | Result Dir |",
            "| --- | --- | --- | --- | --- |",
        ]
        for job in summary["jobs"]:
            attempts = job.get("attempts", {})
            lines.append(
                "| "
                f"{job.get('experiment_id')} | "
                f"{job.get('status')} | "
                f"{attempts.get('startup', 0)} | "
                f"{attempts.get('search', 0)} | "
                f"{job.get('result_dir')} |"
            )
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary

    @staticmethod
    def _job_payload(job: ExpandedExperimentJob) -> dict[str, Any]:
        return {
            "experiment_id": job.experiment_id,
            "source_index": job.source_index,
            "model": job.model,
            "workload": str(job.workload),
            "result_dir": str(job.result_dir),
            "server_signature_key": job.server_signature_key,
            "status": "planned",
            "attempts": {"startup": 0, "search": 0},
            "last_error": None,
            "artifacts": {
                "search_trace": None,
                "final_report_json": None,
                "final_report_md": None,
                "stdout_log": None,
                "stderr_log": None,
            },
        }
