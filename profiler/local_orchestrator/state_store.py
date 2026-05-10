from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from .models import ExpandedExperimentJob, SearchExecutionResult
from .utils import now_utc_iso


def build_job_state_payload(job: ExpandedExperimentJob) -> dict[str, Any]:
    return {
        "experiment_id": job.experiment_id,
        "source_index": job.source_index,
        "model": job.model,
        "workload": str(job.workload),
        "endpoint": job.endpoint,
        "hardware": job.hardware.name,
        "gpu_count": job.launch.gpu_count,
        "tensor_parallel_size": job.launch.tensor_parallel_size,
        "max_model_len": job.launch.max_model_len,
        "search": {
            "search_mode": job.search.search_mode,
            "trial_min_duration_s": job.search.trial_min_duration_s,
            "trial_max_duration_s": job.search.trial_max_duration_s,
            "final_confirmation_duration_s": job.search.final_confirmation_duration_s,
            "rate_precision": job.search.rate_precision,
            "initial_request_rate": job.search.initial_request_rate,
            "max_request_rate": job.search.max_request_rate,
            "open_loop_bracket_growth_factor": job.search.open_loop_bracket_growth_factor,
            "client_limited_retry_attempts": job.search.client_limited_retry_attempts,
            "client_limited_retry_cooldown_s": job.search.client_limited_retry_cooldown_s,
            "ttft_slo_ms": job.search.ttft_slo_ms,
            "tpot_slo_ms": job.search.tpot_slo_ms,
            "ttft_slo_field": job.search.ttft_slo_field,
            "tpot_slo_field": job.search.tpot_slo_field,
            "ttft_slo_mode": job.search.ttft_slo_mode,
            "longbench_ttft_static_preset": job.search.longbench_ttft_static_preset,
        },
        "probe": None if job.probe is None else job.probe.to_payload(),
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
        mst_output_root: Path | None = None,
    ) -> dict[str, Any]:
        if self.state_path.exists():
            raise FileExistsError(f"run state already exists: {self.state_path}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = now_utc_iso()
        state = {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "mst_output_root": None if mst_output_root is None else str(mst_output_root),
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
            original_status = str(job.get("status", "planned"))
            if original_status == "running":
                job["status"] = "planned"
                changed = True
            result_dir = Path(str(job["result_dir"]))
            search_trace = result_dir / "search_trace.json"
            final_report_json = result_dir / "final_report.json"
            final_report_md = result_dir / "final_report.md"
            if search_trace.is_file() and final_report_json.is_file() and original_status == "running":
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

    def refresh_job_plan(self, state: dict[str, Any], *, job: ExpandedExperimentJob) -> None:
        job_state = self.find_job(state, job.experiment_id)
        fresh = self._job_payload(job)
        for key in (
            "source_index",
            "model",
            "workload",
            "endpoint",
            "hardware",
            "gpu_count",
            "tensor_parallel_size",
            "max_model_len",
            "search",
            "probe",
            "result_dir",
            "server_signature_key",
        ):
            job_state[key] = fresh[key]
        self.save(state)

    def reset_job_for_rerun(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
    ) -> None:
        job = self.find_job(state, experiment_id)
        job["status"] = "planned"
        job["last_error"] = None
        job["attempts"] = {"startup": 0, "search": 0}
        job["artifacts"] = {
            "search_trace": None,
            "final_report_json": None,
            "final_report_md": None,
            "stdout_log": None,
            "stderr_log": None,
        }
        self.save(state)

    def summarize(self, state: dict[str, Any]) -> dict[str, Any]:
        jobs = state.get("jobs", [])
        counts = {"planned": 0, "running": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        termination_reason_counts: dict[str, int] = {}
        bottleneck_class_counts: dict[str, int] = {}
        slo_policy_counts: dict[str, int] = {}
        max_no_drift_rates: list[float] = []
        max_slo_rates: list[float] = []
        job_summaries: list[dict[str, Any]] = []
        for job in jobs:
            status = str(job.get("status", "planned"))
            if status not in counts:
                status = "planned"
            counts[status] += 1

            trace_payload = self._extract_search_trace_payload(job)
            search_result = self._extract_search_result_from_trace(trace_payload)
            termination_reason = None if search_result is None else search_result.get("termination_reason")
            bottleneck_class = None if search_result is None else search_result.get("bottleneck_class")
            slo_policy = self._extract_slo_policy(job, trace_payload)
            slo_policy_label = self._format_slo_policy(slo_policy)

            if isinstance(termination_reason, str) and termination_reason:
                termination_reason_counts[termination_reason] = termination_reason_counts.get(termination_reason, 0) + 1
            if isinstance(bottleneck_class, str) and bottleneck_class:
                bottleneck_class_counts[bottleneck_class] = bottleneck_class_counts.get(bottleneck_class, 0) + 1
            slo_policy_counts[slo_policy_label] = slo_policy_counts.get(slo_policy_label, 0) + 1

            max_no_drift = self._as_finite_float(
                None if search_result is None else search_result.get("max_no_drift_request_rate")
            )
            if max_no_drift is not None:
                max_no_drift_rates.append(max_no_drift)

            max_slo = self._as_finite_float(
                None if search_result is None else search_result.get("max_slo_satisfying_request_rate")
            )
            if max_slo is not None:
                max_slo_rates.append(max_slo)

            artifacts = job.get("artifacts")
            if not isinstance(artifacts, dict):
                artifacts = {}

            job_summaries.append(
                {
                    "experiment_id": job.get("experiment_id"),
                    "status": status,
                    "result_dir": job.get("result_dir"),
                    "attempts": job.get("attempts", {}),
                    "last_error": job.get("last_error"),
                    "termination_reason": termination_reason,
                    "bottleneck_class": bottleneck_class,
                    "slo_policy": slo_policy,
                    "slo_policy_label": slo_policy_label,
                    "max_no_drift_request_rate": max_no_drift,
                    "max_slo_satisfying_request_rate": max_slo,
                    "artifacts": {
                        "search_trace": artifacts.get("search_trace"),
                        "final_report_json": artifacts.get("final_report_json"),
                        "final_report_md": artifacts.get("final_report_md"),
                        "stdout_log": artifacts.get("stdout_log"),
                        "stderr_log": artifacts.get("stderr_log"),
                    },
                }
            )

        failed_jobs = [
            {
                "experiment_id": str(job.get("experiment_id")),
                "error": job.get("last_error"),
            }
            for job in job_summaries
            if job.get("status") == "failed"
        ]

        return {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "updated_at": state.get("updated_at"),
            "counts": counts,
            "aggregate": {
                "termination_reason_counts": dict(sorted(termination_reason_counts.items())),
                "bottleneck_class_counts": dict(sorted(bottleneck_class_counts.items())),
                "slo_policy_counts": dict(sorted(slo_policy_counts.items())),
                "max_no_drift_request_rate": self._rate_stats(max_no_drift_rates),
                "max_slo_satisfying_request_rate": self._rate_stats(max_slo_rates),
                "failed_jobs": failed_jobs,
            },
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
            f"- Termination Reasons: {json.dumps(summary['aggregate']['termination_reason_counts'], sort_keys=True)}",
            f"- Bottleneck Classes: {json.dumps(summary['aggregate']['bottleneck_class_counts'], sort_keys=True)}",
            "",
            "## SLO Policies",
            "",
            "| Policy | Jobs |",
            "| --- | ---: |",
        ]
        for policy, count in summary["aggregate"]["slo_policy_counts"].items():
            lines.append(f"| {policy} | {count} |")
        lines.extend(
            [
                "",
                "## Jobs",
                "",
                (
                    "| Experiment ID | Status | Startup Attempts | Search Attempts | "
                    "Termination | Bottleneck | TTFT SLO | TPOT SLO | Max No-Drift RPS | "
                    "Max SLO RPS | Result Dir |"
                ),
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for job in summary["jobs"]:
            attempts = job.get("attempts", {})
            slo_policy = job.get("slo_policy")
            if not isinstance(slo_policy, dict):
                slo_policy = {}
            lines.append(
                "| "
                f"{job.get('experiment_id')} | "
                f"{job.get('status')} | "
                f"{attempts.get('startup', 0)} | "
                f"{attempts.get('search', 0)} | "
                f"{job.get('termination_reason') or '-'} | "
                f"{job.get('bottleneck_class') or '-'} | "
                f"{self._format_single_slo(slo_policy, metric='ttft')} | "
                f"{self._format_single_slo(slo_policy, metric='tpot')} | "
                f"{self._format_rate(job.get('max_no_drift_request_rate'))} | "
                f"{self._format_rate(job.get('max_slo_satisfying_request_rate'))} | "
                f"{job.get('result_dir')} |"
            )
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary

    @staticmethod
    def _extract_search_result(job: dict[str, Any]) -> dict[str, Any] | None:
        return RunStateStore._extract_search_result_from_trace(
            RunStateStore._extract_search_trace_payload(job)
        )

    @staticmethod
    def _extract_search_trace_payload(job: dict[str, Any]) -> dict[str, Any] | None:
        artifacts = job.get("artifacts")
        artifact_path: Path | None = None
        if isinstance(artifacts, dict):
            raw_search_trace = artifacts.get("search_trace")
            if isinstance(raw_search_trace, str) and raw_search_trace:
                artifact_path = Path(raw_search_trace)
        if artifact_path is None:
            artifact_path = Path(str(job.get("result_dir"))) / "search_trace.json"
        if not artifact_path.is_file():
            return None
        return RunStateStore._read_json_mapping(artifact_path)

    @staticmethod
    def _extract_search_result_from_trace(trace_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if trace_payload is None:
            return None
        result = trace_payload.get("result")
        if not isinstance(result, dict):
            return None
        return result

    @staticmethod
    def _extract_slo_policy(
        job: dict[str, Any],
        trace_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace_policy = RunStateStore._extract_trace_slo_policy(trace_payload)
        search = job.get("search")
        if isinstance(search, dict) and any(
            key in search
            for key in (
                "ttft_slo_ms",
                "tpot_slo_ms",
                "ttft_slo_field",
                "tpot_slo_field",
                "ttft_slo_mode",
                "longbench_ttft_static_preset",
            )
        ):
            return {
                "ttft_slo_ms": search.get("ttft_slo_ms"),
                "ttft_slo_field": search.get("ttft_slo_field", "ttft_p90_ms"),
                "ttft_slo_mode": search.get(
                    "ttft_slo_mode",
                    trace_policy.get("ttft_slo_mode", "static") if trace_policy else "static",
                ),
                "longbench_ttft_static_preset": search.get(
                    "longbench_ttft_static_preset",
                    trace_policy.get("longbench_ttft_static_preset") if trace_policy else None,
                ),
                "tpot_slo_ms": search.get("tpot_slo_ms"),
                "tpot_slo_field": search.get("tpot_slo_field", "tpot_p90_ms"),
            }
        if trace_policy is not None:
            return trace_policy
        return {
            "ttft_slo_ms": None,
            "ttft_slo_field": "ttft_p90_ms",
            "ttft_slo_mode": "static",
            "longbench_ttft_static_preset": None,
            "tpot_slo_ms": None,
            "tpot_slo_field": "tpot_p90_ms",
        }

    @staticmethod
    def _extract_trace_slo_policy(trace_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if trace_payload is not None:
            config = trace_payload.get("config")
            if isinstance(config, dict):
                metadata = config.get("metadata")
                if isinstance(metadata, dict):
                    policy = metadata.get("stability_policy")
                    if isinstance(policy, dict):
                        return {
                            "ttft_slo_ms": policy.get("ttft_slo_ms"),
                            "ttft_slo_field": policy.get("ttft_slo_field", "ttft_p90_ms"),
                            "ttft_slo_mode": policy.get("ttft_slo_mode", "static"),
                            "longbench_ttft_static_preset": policy.get(
                                "longbench_ttft_static_preset"
                            ),
                            "tpot_slo_ms": policy.get("tpot_slo_ms"),
                            "tpot_slo_field": policy.get("tpot_slo_field", "tpot_p90_ms"),
                        }
        return None

    @staticmethod
    def _format_slo_policy(policy: dict[str, Any]) -> str:
        return (
            f"{RunStateStore._format_single_slo(policy, metric='ttft')}; "
            f"{RunStateStore._format_single_slo(policy, metric='tpot')}"
        )

    @staticmethod
    def _format_single_slo(policy: dict[str, Any], *, metric: str) -> str:
        if metric not in {"ttft", "tpot"}:
            raise ValueError(f"unsupported SLO metric: {metric!r}")
        value = RunStateStore._as_finite_float(policy.get(f"{metric}_slo_ms"))
        field = policy.get(f"{metric}_slo_field") or f"{metric}_p90_ms"
        if metric == "ttft":
            mode = policy.get("ttft_slo_mode", "static")
            preset = policy.get("longbench_ttft_static_preset")
            if mode == "length_scaled":
                return f"{field} length_scaled"
            if preset is not None:
                return f"{field} longbench_static:{preset}"
        if value is None:
            return f"{field} disabled"
        return f"{field}<={value:.6g}ms"

    @staticmethod
    def _read_json_mapping(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _as_finite_float(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _rate_stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    @staticmethod
    def _format_rate(value: object) -> str:
        numeric = RunStateStore._as_finite_float(value)
        if numeric is None:
            return "-"
        return f"{numeric:.6g}"

    @staticmethod
    def _job_payload(job: ExpandedExperimentJob) -> dict[str, Any]:
        return build_job_state_payload(job)
