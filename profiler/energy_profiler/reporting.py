from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from local_orchestrator.utils import now_utc_iso

from .models import EnergyJobStatus, EnergyPlan


_COUNTS_TEMPLATE = {
    "planned": 0,
    "running": 0,
    "succeeded": 0,
    "failed": 0,
    "skipped": 0,
}


class EnergyRunStateStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.state_path = run_root / "state.json"
        self.summary_json_path = run_root / "summary.json"
        self.summary_md_path = run_root / "summary.md"
        self.summary_csv_path = run_root / "summary_compact.csv"
        self.plan_copy_path = run_root / "plan.yaml"
        self.logs_dir = run_root / "logs"
        self.jobs_dir = run_root / "jobs"

    def initialize_new(self, *, plan_path: Path, plan: EnergyPlan) -> dict[str, Any]:
        if self.state_path.exists():
            raise FileExistsError(f"energy run state already exists: {self.state_path}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.plan_copy_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
        ts = now_utc_iso()
        state = {
            "plan_id": plan.plan.plan_id,
            "plan_path": str(plan_path),
            "status": "running",
            "created_at": ts,
            "updated_at": ts,
            "jobs": [self._job_payload(plan_job) for plan_job in plan.jobs],
        }
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"energy run state does not exist: {self.state_path}")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("energy run state is malformed")
        return payload

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_utc_iso()
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def reconcile_jobs(self, state: dict[str, Any]) -> None:
        changed = False
        for job in state.get("jobs", []):
            if job.get("status") == "running":
                job["status"] = "planned"
                changed = True
            result_dir = Path(str(job.get("result_dir")))
            summary_path = result_dir / "summary.json"
            energy_summary_path = result_dir / "energy_summary.json"
            gpu_power_path = result_dir / "gpu_power.json"
            if summary_path.is_file() and energy_summary_path.is_file() and job.get("status") != "succeeded":
                job["status"] = "succeeded"
                job["last_error"] = None
                artifacts = job.setdefault("artifacts", {})
                artifacts.update(
                    {
                        "summary_json": str(summary_path),
                        "request_records_jsonl": str(result_dir / "request_records.jsonl"),
                        "server_metrics_jsonl": str(result_dir / "server_metrics.jsonl"),
                        "windows_csv": str(result_dir / "windows.csv"),
                        "gpu_power_json": str(gpu_power_path) if gpu_power_path.is_file() else None,
                        "energy_summary_json": str(energy_summary_path),
                        "repeats": _repeat_artifacts(result_dir),
                    }
                )
                changed = True
        if changed:
            self.save(state)

    def find_job(self, state: dict[str, Any], job_id: str) -> dict[str, Any]:
        for job in state.get("jobs", []):
            if job.get("job_id") == job_id:
                return job
        raise KeyError(f"energy job not found in state: {job_id}")

    def increment_attempt(self, state: dict[str, Any], *, job_id: str) -> None:
        job = self.find_job(state, job_id)
        attempts = int(job.get("attempts", 0))
        job["attempts"] = attempts + 1
        self.save(state)

    def set_job_status(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        status: EnergyJobStatus,
        last_error: str | None = None,
    ) -> None:
        job = self.find_job(state, job_id)
        job["status"] = status
        if last_error is not None:
            job["last_error"] = last_error
        self.save(state)

    def mark_job_succeeded(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        gpu_ids: tuple[int, ...],
        base_url: str,
        artifacts: Mapping[str, str | None],
    ) -> None:
        job = self.find_job(state, job_id)
        job["status"] = "succeeded"
        job["last_error"] = None
        job["gpu_ids"] = list(gpu_ids)
        job["base_url"] = base_url
        job["artifacts"] = dict(artifacts)
        self.save(state)

    def mark_job_failed(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        error: str,
    ) -> None:
        job = self.find_job(state, job_id)
        job["status"] = "failed"
        job["last_error"] = error
        self.save(state)

    def summarize(self, state: dict[str, Any]) -> dict[str, Any]:
        counts = dict(_COUNTS_TEMPLATE)
        total_energy_joules = 0.0
        total_incremental_energy_joules = 0.0
        succeeded_with_energy = 0
        jobs_summary: list[dict[str, Any]] = []
        failed_jobs: list[dict[str, Any]] = []
        plan_jobs = self._load_plan_jobs_by_id()

        for job in state.get("jobs", []):
            status = str(job.get("status", "planned"))
            if status not in counts:
                status = "planned"
            counts[status] += 1
            plan_job = plan_jobs.get(str(job.get("job_id")))
            energy_summary = self._load_energy_summary(job)
            if energy_summary is not None:
                total_energy = _optional_finite_float(energy_summary.get("energy_joules"))
                total_incremental = _optional_finite_float(energy_summary.get("incremental_energy_joules"))
                if total_energy is not None:
                    total_energy_joules += total_energy
                    succeeded_with_energy += 1
                if total_incremental is not None:
                    total_incremental_energy_joules += total_incremental
            else:
                total_energy = None
                total_incremental = None

            gpu_count = _optional_positive_int(job.get("gpu_count"))
            tensor_parallel_size = _optional_positive_int(job.get("tensor_parallel_size"))
            if plan_job is not None:
                launch = plan_job.get("launch")
                if isinstance(launch, Mapping):
                    gpu_count = gpu_count or _optional_positive_int(launch.get("gpu_count"))
                    tensor_parallel_size = tensor_parallel_size or _optional_positive_int(
                        launch.get("tensor_parallel_size")
                    )

            job_summary = {
                "job_id": job.get("job_id"),
                "status": status,
                "source_experiment_id": job.get("source_experiment_id"),
                "model": job.get("model"),
                "workload": job.get("workload"),
                "workload_name": _workload_name(job.get("workload")),
                "gpu_count": gpu_count,
                "tensor_parallel_size": tensor_parallel_size,
                "request_rate": job.get("request_rate"),
                "mst_rate": job.get("mst_rate"),
                "gpu_ids": job.get("gpu_ids"),
                "result_dir": job.get("result_dir"),
                "attempts": job.get("attempts", 0),
                "last_error": job.get("last_error"),
                "energy_joules": total_energy,
                "incremental_energy_joules": total_incremental,
                "energy_kwh": None if energy_summary is None else _optional_finite_float(energy_summary.get("energy_kwh")),
                "avg_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("avg_power_w")),
                "idle_avg_power_w": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("idle_avg_power_w")),
                "incremental_avg_power_w": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("incremental_avg_power_w")),
                "min_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("min_power_w")),
                "p50_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("p50_power_w")),
                "p90_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("p90_power_w")),
                "p95_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("p95_power_w")),
                "p99_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("p99_power_w")),
                "max_power_w": None if energy_summary is None else _optional_finite_float(energy_summary.get("max_power_w")),
                "energy_per_successful_request_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("energy_per_successful_request_j")),
                "incremental_energy_per_successful_request_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("incremental_energy_per_successful_request_j")),
                "energy_per_total_request_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("energy_per_total_request_j")),
                "incremental_energy_per_total_request_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("incremental_energy_per_total_request_j")),
                "energy_per_total_token_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("energy_per_total_token_j")),
                "incremental_energy_per_total_token_j": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("incremental_energy_per_total_token_j")),
                "successful_requests": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("successful_requests"), allow_zero=True),
                "started_requests": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("started_requests"), allow_zero=True),
                "total_input_tokens": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("total_input_tokens"), allow_zero=True),
                "total_output_tokens": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("total_output_tokens"), allow_zero=True),
                "total_tokens": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("total_tokens"), allow_zero=True),
                "monitor_duration_s": None
                if energy_summary is None
                else _optional_finite_float(energy_summary.get("monitor_duration_s")),
                "repeat_count": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("repeat_count"), allow_zero=True),
                "successful_repeat_count": None
                if energy_summary is None
                else _optional_positive_int(energy_summary.get("successful_repeat_count"), allow_zero=True),
                "artifacts": dict(job.get("artifacts", {})),
            }
            jobs_summary.append(job_summary)
            if status == "failed":
                failed_jobs.append(
                    {
                        "job_id": str(job.get("job_id")),
                        "error": job.get("last_error"),
                    }
                )

        return {
            "plan_id": state.get("plan_id"),
            "status": state.get("status"),
            "updated_at": state.get("updated_at"),
            "counts": counts,
            "aggregate": {
                "total_energy_joules": total_energy_joules if succeeded_with_energy else None,
                "total_incremental_energy_joules": total_incremental_energy_joules if succeeded_with_energy else None,
                "failed_jobs": failed_jobs,
            },
            "jobs": jobs_summary,
        }

    def write_summary_files(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = self.summarize(state)
        self.summary_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_compact_csv(summary)

        lines = [
            f"# Energy Profiling Summary: {summary['plan_id']}",
            "",
            f"- Status: {summary['status']}",
            f"- Updated At: {summary['updated_at']}",
            f"- Planned: {summary['counts']['planned']}",
            f"- Running: {summary['counts']['running']}",
            f"- Succeeded: {summary['counts']['succeeded']}",
            f"- Failed: {summary['counts']['failed']}",
            f"- Skipped: {summary['counts']['skipped']}",
            f"- Total Energy (J): {_format_float(summary['aggregate']['total_energy_joules'])}",
            f"- Total Incremental Energy (J): {_format_float(summary['aggregate']['total_incremental_energy_joules'])}",
            "",
            "## Jobs",
            "",
            "| Job ID | Status | Request Rate | MST Rate | Energy (J) | Incremental Energy (J) | Avg Power (W) | Energy / Req (J) | Incremental Energy / Req (J) | Energy / Token (J) | Result Dir |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for job in summary["jobs"]:
            lines.append(
                "| "
                f"{job['job_id']} | "
                f"{job['status']} | "
                f"{_format_float(job.get('request_rate'))} | "
                f"{_format_float(job.get('mst_rate'))} | "
                f"{_format_float(job.get('energy_joules'))} | "
                f"{_format_float(job.get('incremental_energy_joules'))} | "
                f"{_format_float(job.get('avg_power_w'))} | "
                f"{_format_float(job.get('energy_per_successful_request_j'))} | "
                f"{_format_float(job.get('incremental_energy_per_successful_request_j'))} | "
                f"{_format_float(job.get('energy_per_total_token_j'))} | "
                f"{job.get('result_dir')} |"
            )
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary

    def _job_payload(self, plan_job) -> dict[str, Any]:
        result_dir = self.jobs_dir / plan_job.id
        return {
            "job_id": plan_job.id,
            "status": "planned",
            "source_experiment_id": plan_job.source_experiment_id,
            "model": plan_job.model,
            "workload": str(plan_job.workload),
            "endpoint": plan_job.endpoint,
            "gpu_count": plan_job.launch.gpu_count,
            "tensor_parallel_size": plan_job.launch.tensor_parallel_size,
            "request_rate": plan_job.request_rate,
            "mst_rate": plan_job.mst_rate,
            "result_dir": str(result_dir),
            "server_signature_key": plan_job.server_signature_key,
            "attempts": 0,
            "last_error": None,
            "gpu_ids": None,
            "base_url": None,
            "artifacts": {
                "summary_json": None,
                "request_records_jsonl": None,
                "server_metrics_jsonl": None,
                "windows_csv": None,
                "gpu_power_json": None,
                "energy_summary_json": None,
                "repeats": [],
                "profile_stdout_log": None,
                "profile_stderr_log": None,
                "vllm_stdout_log": None,
                "vllm_stderr_log": None,
            },
        }

    @staticmethod
    def _load_energy_summary(job: Mapping[str, Any]) -> Mapping[str, Any] | None:
        artifacts = job.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return None
        path = artifacts.get("energy_summary_json")
        if not isinstance(path, str) or not path:
            return None
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, Mapping):
            return None
        return payload

    def _load_plan_jobs_by_id(self) -> dict[str, Mapping[str, Any]]:
        if not self.plan_copy_path.is_file():
            return {}
        try:
            payload = yaml.safe_load(self.plan_copy_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        raw_jobs = payload.get("jobs")
        if not isinstance(raw_jobs, list):
            return {}
        jobs: dict[str, Mapping[str, Any]] = {}
        for raw_job in raw_jobs:
            if not isinstance(raw_job, Mapping):
                continue
            raw_id = raw_job.get("id")
            if isinstance(raw_id, str) and raw_id:
                jobs[raw_id] = raw_job
        return jobs

    def _write_compact_csv(self, summary: Mapping[str, Any]) -> None:
        fieldnames = [
            "job_id",
            "status",
            "source_experiment_id",
            "model",
            "workload",
            "gpu_count",
            "tensor_parallel_size",
            "request_rate",
            "mst_rate",
            "energy_joules",
            "incremental_energy_joules",
            "energy_kwh",
            "avg_power_w",
            "idle_avg_power_w",
            "incremental_avg_power_w",
            "min_power_w",
            "p50_power_w",
            "p90_power_w",
            "p95_power_w",
            "p99_power_w",
            "max_power_w",
            "energy_per_successful_request_j",
            "incremental_energy_per_successful_request_j",
            "energy_per_total_request_j",
            "incremental_energy_per_total_request_j",
            "energy_per_total_token_j",
            "incremental_energy_per_total_token_j",
            "successful_requests",
            "started_requests",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "monitor_duration_s",
            "repeat_count",
            "successful_repeat_count",
        ]
        raw_jobs = summary.get("jobs")
        jobs = raw_jobs if isinstance(raw_jobs, list) else []
        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for job in jobs:
                if not isinstance(job, Mapping):
                    continue
                row = dict(job)
                row["workload"] = job.get("workload_name") or _workload_name(job.get("workload"))
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _optional_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not isfinite(numeric):
        return None
    return numeric


def _optional_positive_int(value: Any, *, allow_zero: bool = False) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or (value == 0 and not allow_zero):
        return None
    return value


def _workload_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).stem


def _format_float(value: Any) -> str:
    numeric = _optional_finite_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _repeat_artifacts(result_dir: Path) -> list[dict[str, Any]]:
    repeats = []
    for repeat_dir in sorted(result_dir.glob("repeat_[0-9][0-9][0-9]")):
        if not repeat_dir.is_dir():
            continue
        repeats.append(
            {
                "repeat_index": _repeat_index_from_dir(repeat_dir),
                "result_dir": str(repeat_dir),
                "summary_json": str(repeat_dir / "summary.json") if (repeat_dir / "summary.json").is_file() else None,
                "request_records_jsonl": str(repeat_dir / "request_records.jsonl")
                if (repeat_dir / "request_records.jsonl").is_file()
                else None,
                "server_metrics_jsonl": str(repeat_dir / "server_metrics.jsonl")
                if (repeat_dir / "server_metrics.jsonl").is_file()
                else None,
                "windows_csv": str(repeat_dir / "windows.csv") if (repeat_dir / "windows.csv").is_file() else None,
                "gpu_power_json": str(repeat_dir / "gpu_power.json") if (repeat_dir / "gpu_power.json").is_file() else None,
                "energy_summary_json": str(repeat_dir / "energy_summary.json")
                if (repeat_dir / "energy_summary.json").is_file()
                else None,
            }
        )
    return repeats


def _repeat_index_from_dir(path: Path) -> int | None:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
