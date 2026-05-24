from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from local_orchestrator.models import (
    ExpandedExperimentJob,
    HardwareConfig,
    RunConfig,
    SearchConfig,
)
from local_orchestrator.utils import now_utc_iso

from .executor import EnergyExecutor
from .models import EnergyPlan, EnergyPlanJob
from .reporting import EnergyRunStateStore


def default_local_energy_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"local-energy-{ts}"


def local_energy_run_root(plan: EnergyPlan, run_id: str) -> Path:
    return (plan.plan.output_root / plan.plan.plan_id / run_id).resolve()


def run_config_from_energy_plan(plan: EnergyPlan, *, run_id: str | None = None) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        output_root=plan.plan.output_root,
        allowed_gpu_ids=plan.execution.allowed_gpu_ids,
        max_active_gpus=plan.execution.max_active_gpus,
        keep_one_gpu_spare=False,
        default_endpoint="/v1/chat/completions",
        base_port_start=plan.execution.base_port_start,
        base_port_end=plan.execution.base_port_end,
        metrics_port_offset=plan.execution.metrics_port_offset,
        python_executable=plan.plan.python_executable,
    )


def expand_energy_plan_for_local_scheduler(
    plan: EnergyPlan,
    *,
    run_root: Path,
) -> list[ExpandedExperimentJob]:
    jobs: list[ExpandedExperimentJob] = []
    hardware = HardwareConfig(name="energy-local")
    for source_index, plan_job in enumerate(plan.jobs):
        jobs.append(
            ExpandedExperimentJob(
                experiment_id=plan_job.id,
                source_index=source_index,
                model=plan_job.model,
                workload=plan_job.workload,
                endpoint=plan_job.endpoint,
                launch=plan_job.launch.to_launch_config(),
                search=_search_config_from_energy_job(plan=plan, job=plan_job),
                hardware=hardware,
                probe=None,
                result_dir=run_root / "jobs" / plan_job.id,
                model_slug=plan_job.id,
                dataset_slug=Path(plan_job.workload).stem,
                server_config_slug=plan_job.server_config_slug,
                server_signature_key=plan_job.server_signature_key,
                server_metadata_file=None,
            )
        )
    return jobs


@dataclass(frozen=True, slots=True)
class EnergyExecutionResult:
    success: bool
    return_code: int
    commands: tuple[tuple[str, ...], ...]
    stdout_log: Path
    stderr_log: Path
    gpu_ids: tuple[int, ...]
    base_url: str
    artifacts: Mapping[str, Any]
    error: str | None = None


class EnergyProfilingAdapter:
    def __init__(
        self,
        *,
        plan: EnergyPlan,
        executor_factory: Callable[[], EnergyExecutor] | None = None,
    ) -> None:
        self._plan = plan
        self._jobs_by_id = {job.id: job for job in plan.jobs}
        self._executor_factory = executor_factory or EnergyExecutor

    def invoke(self, *, job: ExpandedExperimentJob, server, logs_dir) -> EnergyExecutionResult:
        del logs_dir
        plan_job = self._jobs_by_id[job.experiment_id]
        executor = self._executor_factory()
        metrics_url = f"{str(server.base_url).rstrip('/')}/metrics"
        try:
            result = executor.run_live_trial(
                trial_id=plan_job.id,
                output_dir=job.result_dir,
                workload=plan_job.workload,
                model=plan_job.model,
                base_url=str(server.base_url),
                endpoint=plan_job.endpoint,
                metrics_url=metrics_url,
                gpu_ids=tuple(server.gpu_ids),
                duration_s=self._plan.defaults.duration_s,
                request_rate=plan_job.request_rate,
                request_timeout_s=self._plan.defaults.request_timeout_s,
                metrics_interval_s=self._plan.defaults.metrics_interval_s,
                window_s=self._plan.defaults.window_s,
                idle_monitor_duration_s=self._plan.defaults.idle_monitor_duration_s,
                traffic_warmup_s=self._plan.defaults.traffic_warmup_s,
                repeats=self._plan.defaults.repeats,
                repeat_cooldown_s=self._plan.defaults.repeat_cooldown_s,
                warmup_each_repeat=self._plan.defaults.warmup_each_repeat,
                gpu_monitor_interval_s=self._plan.defaults.gpu_monitor_interval_s,
                gpu_monitor_truncate_s=self._plan.defaults.gpu_monitor_truncate_s,
                monitor_clock=self._plan.defaults.monitor_clock,
                safety_max_outstanding=self._plan.defaults.safety_max_outstanding,
                force=True,
            )
        except Exception as exc:
            stdout_log, stderr_log = _default_profile_logs(job.result_dir, plan_job.id)
            return EnergyExecutionResult(
                success=False,
                return_code=1,
                commands=(),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                gpu_ids=tuple(server.gpu_ids),
                base_url=str(server.base_url),
                artifacts={},
                error=str(exc),
            )

        artifacts = _energy_artifacts_from_result(job.result_dir, result)
        stdout_log = Path(str(artifacts.get("profile_stdout_log") or ""))
        stderr_log = Path(str(artifacts.get("profile_stderr_log") or ""))
        return EnergyExecutionResult(
            success=True,
            return_code=0,
            commands=(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            gpu_ids=tuple(server.gpu_ids),
            base_url=str(server.base_url),
            artifacts=artifacts,
        )


class SchedulerEnergyStateStore:
    def __init__(self, delegate: EnergyRunStateStore, *, plan: EnergyPlan) -> None:
        self._delegate = delegate
        self._plan = plan
        self._jobs_by_id = {job.id: job for job in plan.jobs}
        self.events_path = delegate.run_root / "events.jsonl"

    @property
    def run_root(self) -> Path:
        return self._delegate.run_root

    @property
    def logs_dir(self) -> Path:
        return self._delegate.logs_dir

    def initialize_new(self, *, plan_path: Path) -> dict[str, Any]:
        return self._delegate.initialize_new(plan_path=plan_path, plan=self._plan)

    def load(self) -> dict[str, Any]:
        return self._delegate.load()

    def save(self, state: dict[str, Any]) -> None:
        self._delegate.save(state)

    def write_summary_files(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._delegate.write_summary_files(state)

    def summarize(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._delegate.summarize(state)

    def reconcile_jobs(self, state: dict[str, Any]) -> None:
        self._delegate.reconcile_jobs(state)

    def find_job(self, state: dict[str, Any], experiment_id: str) -> dict[str, Any]:
        return self._delegate.find_job(state, experiment_id)

    def increment_attempt(self, state: dict[str, Any], experiment_id: str, *, kind: str) -> None:
        del kind
        self._delegate.increment_attempt(state, job_id=experiment_id)

    def set_job_status(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        status: str,
        last_error: str | None = None,
    ) -> None:
        self._delegate.set_job_status(
            state,
            job_id=experiment_id,
            status=status,  # type: ignore[arg-type]
            last_error=last_error,
        )

    def mark_job_succeeded(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        result: EnergyExecutionResult,
    ) -> None:
        self._delegate.mark_job_succeeded(
            state,
            job_id=experiment_id,
            gpu_ids=result.gpu_ids,
            base_url=result.base_url,
            artifacts=result.artifacts,
        )

    def mark_job_failed(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        error: str,
    ) -> None:
        self._delegate.mark_job_failed(state, job_id=experiment_id, error=error)

    def refresh_job_plan(self, state: dict[str, Any], *, job: ExpandedExperimentJob) -> None:
        job_state = self.find_job(state, job.experiment_id)
        plan_job = self._jobs_by_id[job.experiment_id]
        job_state.update(
            {
                "source_experiment_id": plan_job.source_experiment_id,
                "model": plan_job.model,
                "workload": str(plan_job.workload),
                "endpoint": plan_job.endpoint,
                "gpu_count": plan_job.launch.gpu_count,
                "tensor_parallel_size": plan_job.launch.tensor_parallel_size,
                "request_rate": plan_job.request_rate,
                "mst_rate": plan_job.mst_rate,
                "result_dir": str(job.result_dir),
                "server_signature_key": plan_job.server_signature_key,
            }
        )
        self._delegate.save(state)

    def reset_job_for_rerun(self, state: dict[str, Any], *, experiment_id: str) -> None:
        job_state = self.find_job(state, experiment_id)
        result_dir = Path(str(job_state.get("result_dir")))
        if result_dir.exists():
            shutil.rmtree(result_dir)
        job_state["status"] = "planned"
        job_state["attempts"] = 0
        job_state["last_error"] = None
        job_state["gpu_ids"] = None
        job_state["base_url"] = None
        job_state["artifacts"] = {
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
        }
        self._delegate.save(state)

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
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
        state["updated_at"] = now_utc_iso()


def _search_config_from_energy_job(*, plan: EnergyPlan, job: EnergyPlanJob) -> SearchConfig:
    return SearchConfig(
        search_mode="open-loop",
        trial_min_duration_s=plan.defaults.duration_s,
        trial_max_duration_s=plan.defaults.duration_s,
        final_confirmation_duration_s=plan.defaults.duration_s,
        initial_request_rate=job.request_rate,
        max_request_rate=job.request_rate,
        metrics_interval_s=plan.defaults.metrics_interval_s,
        window_s=plan.defaults.window_s,
        max_num_seqs=None if job.launch.max_num_seqs is None else int(job.launch.max_num_seqs),
        max_num_batched_tokens=(
            None if job.launch.max_num_batched_tokens is None else int(job.launch.max_num_batched_tokens)
        ),
    )


def _energy_artifacts_from_result(result_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    repeats = result.get("repeats")
    logs_dir = result_dir / "logs"
    trial_id = str(result.get("trial_id") or result_dir.name)
    stdout_log, stderr_log = _default_profile_logs(result_dir, trial_id)
    return {
        "summary_json": result.get("summary_json"),
        "request_records_jsonl": result.get("request_records_jsonl"),
        "server_metrics_jsonl": result.get("server_metrics_jsonl"),
        "windows_csv": result.get("windows_csv"),
        "gpu_power_json": result.get("gpu_power_json"),
        "energy_summary_json": result.get("energy_summary_json"),
        "repeats": repeats if isinstance(repeats, list) else [],
        "profile_stdout_log": str(stdout_log) if stdout_log.is_file() else None,
        "profile_stderr_log": str(stderr_log) if stderr_log.is_file() else None,
        "vllm_stdout_log": None,
        "vllm_stderr_log": None,
        "live_trial_json": str(result_dir / "live_trial.json") if (result_dir / "live_trial.json").is_file() else None,
        "logs_dir": str(logs_dir) if logs_dir.is_dir() else None,
    }


def _default_profile_logs(result_dir: Path, trial_id: str) -> tuple[Path, Path]:
    return (
        result_dir / "logs" / f"{trial_id}.profile.stdout.log",
        result_dir / "logs" / f"{trial_id}.profile.stderr.log",
    )
