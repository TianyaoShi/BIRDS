from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

from local_orchestrator.lifecycle import VLLMLifecycleManager
from local_orchestrator.resources import GPULeaseManager, PortAllocator
from local_orchestrator.utils import runtime_server_signature

from .models import EnergyPlan, EnergyPlanExecution, EnergyPlanJob
from .planning import load_energy_plan
from .reporting import EnergyRunStateStore


class MonitorProtocol(Protocol):
    gpu_id: int

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_snapshot(self, wait: bool = False, timeout: float | None = None) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class EnergyExecutorConfig:
    allowed_gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    max_active_gpus: int | None = None
    base_port_start: int = 8000
    base_port_end: int = 8099
    metrics_port_offset: int = 1000

    def __post_init__(self) -> None:
        if not self.allowed_gpu_ids:
            raise ValueError("allowed_gpu_ids must be non-empty")
        if self.max_active_gpus is not None and self.max_active_gpus <= 0:
            raise ValueError("max_active_gpus must be positive when provided")
        if self.base_port_start <= 0 or self.base_port_end <= 0:
            raise ValueError("base ports must be positive")
        if self.base_port_end < self.base_port_start:
            raise ValueError("base_port_end must be >= base_port_start")
        if self.metrics_port_offset <= 0:
            raise ValueError("metrics_port_offset must be positive")

    @property
    def effective_max_active_gpus(self) -> int:
        return self.max_active_gpus or len(self.allowed_gpu_ids)

    @classmethod
    def from_plan_execution(cls, execution: EnergyPlanExecution) -> "EnergyExecutorConfig":
        return cls(
            allowed_gpu_ids=execution.allowed_gpu_ids,
            max_active_gpus=execution.max_active_gpus,
            base_port_start=execution.base_port_start,
            base_port_end=execution.base_port_end,
            metrics_port_offset=execution.metrics_port_offset,
        )


@dataclass(frozen=True, slots=True)
class PowerSnapshot:
    gpu_id: int
    avg_power_mw: float
    power_stats: dict[str, Any]
    power_trace_mw: tuple[int, ...]
    clock_trace_mhz: tuple[tuple[int, int, int], ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "avg_power_mw": self.avg_power_mw,
            "power_stats": dict(self.power_stats),
            "power_trace_mw": list(self.power_trace_mw),
            "clock_trace_mhz": None if self.clock_trace_mhz is None else [list(item) for item in self.clock_trace_mhz],
        }


@dataclass(frozen=True, slots=True)
class MonitorRunResult:
    duration_s: float
    per_gpu: tuple[PowerSnapshot, ...]
    aggregate_trace_mw: tuple[float, ...]

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(snapshot.gpu_id for snapshot in self.per_gpu)

    @property
    def avg_power_mw(self) -> float:
        return sum(snapshot.avg_power_mw for snapshot in self.per_gpu)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "gpu_ids": list(self.gpu_ids),
            "aggregate_trace_mw": list(self.aggregate_trace_mw),
            "per_gpu": [snapshot.to_dict() for snapshot in self.per_gpu],
        }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]
MonitorFactory = Callable[..., MonitorProtocol]


class EnergyExecutor:
    def __init__(
        self,
        *,
        config: EnergyExecutorConfig | None = None,
        lifecycle_factory: Callable[[], VLLMLifecycleManager] | None = None,
        run_command: RunCommand | None = None,
        monitor_factory: MonitorFactory | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config_override = config is not None
        self._config = config or EnergyExecutorConfig()
        self._lifecycle = (lifecycle_factory or VLLMLifecycleManager)()
        self._run_command = run_command or _default_run_command
        self._monitor_factory = monitor_factory or _default_monitor_factory
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn

        self._gpu_manager: GPULeaseManager
        self._port_allocator: PortAllocator
        self._apply_config(self._config)
        self._active_lease = None
        self._active_ports = None
        self._active_server = None
        self._active_server_signature_key: str | None = None
        self._idle_snapshot: MonitorRunResult | None = None

    def run_plan(self, plan_path: str | Path) -> dict[str, Any]:
        plan_file = Path(plan_path)
        plan = load_energy_plan(plan_file)
        self._configure_for_plan(plan)
        run_root = plan.plan.output_root / plan.plan.plan_id
        state_store = EnergyRunStateStore(run_root)
        state = state_store.initialize_new(plan_path=plan_file, plan=plan)
        return self._execute(plan=plan, state_store=state_store, state=state, force=False)

    def resume_run(self, run_root: str | Path, *, force: bool = False) -> dict[str, Any]:
        resolved_run_root = Path(run_root)
        state_store = EnergyRunStateStore(resolved_run_root)
        state = state_store.load()
        state_store.reconcile_jobs(state)
        plan = load_energy_plan(state_store.plan_copy_path)
        self._configure_for_plan(plan)
        return self._execute(plan=plan, state_store=state_store, state=state, force=force)

    def status(self, run_root: str | Path) -> dict[str, Any]:
        state_store = EnergyRunStateStore(Path(run_root))
        return state_store.summarize(state_store.load())

    def _configure_for_plan(self, plan: EnergyPlan) -> None:
        if self._config_override:
            return
        self._apply_config(EnergyExecutorConfig.from_plan_execution(plan.execution))

    def _apply_config(self, config: EnergyExecutorConfig) -> None:
        self._config = config
        self._gpu_manager = GPULeaseManager(
            allowed_gpu_ids=config.allowed_gpu_ids,
            max_active_gpus=config.effective_max_active_gpus,
        )
        self._port_allocator = PortAllocator(
            base_port_start=config.base_port_start,
            base_port_end=config.base_port_end,
            metrics_port_offset=config.metrics_port_offset,
        )

    def run_live_trial(
        self,
        *,
        trial_id: str,
        output_dir: str | Path,
        workload: str | Path,
        model: str,
        base_url: str,
        endpoint: str,
        metrics_url: str | None,
        gpu_ids: Sequence[int],
        duration_s: float,
        request_rate: float,
        request_timeout_s: float,
        metrics_interval_s: float,
        window_s: float,
        idle_monitor_duration_s: float,
        gpu_monitor_interval_s: float,
        gpu_monitor_truncate_s: float,
        monitor_clock: bool,
        safety_max_outstanding: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        resolved_output_dir = Path(output_dir)
        if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
            if not force:
                raise RuntimeError(f"refusing to overwrite existing artifacts in {resolved_output_dir}")
            shutil.rmtree(resolved_output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = resolved_output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        resolved_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        if not resolved_gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        resolved_metrics_url = metrics_url or f"{base_url.rstrip('/')}/metrics"

        idle_snapshot = self._collect_idle_baseline(
            gpu_ids=resolved_gpu_ids,
            duration_s=idle_monitor_duration_s,
            interval_s=gpu_monitor_interval_s,
            truncate_s=0.0,
            monitor_clock=monitor_clock,
        )
        traffic_snapshot = self._run_live_monitored_trial(
            trial_id=trial_id,
            output_dir=resolved_output_dir,
            workload=Path(workload),
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            metrics_url=resolved_metrics_url,
            gpu_ids=resolved_gpu_ids,
            duration_s=duration_s,
            request_rate=request_rate,
            request_timeout_s=request_timeout_s,
            metrics_interval_s=metrics_interval_s,
            window_s=window_s,
            gpu_monitor_interval_s=gpu_monitor_interval_s,
            gpu_monitor_truncate_s=gpu_monitor_truncate_s,
            monitor_clock=monitor_clock,
            safety_max_outstanding=safety_max_outstanding,
            logs_dir=logs_dir,
        )
        trial_summary_payload = _load_json_mapping(resolved_output_dir / "summary.json")
        energy_summary = compute_energy_summary(
            trial_summary_payload=trial_summary_payload,
            idle_snapshot=idle_snapshot,
            traffic_snapshot=traffic_snapshot,
        )
        gpu_power_payload = {
            "idle": idle_snapshot.to_dict(),
            "traffic": traffic_snapshot.to_dict(),
        }
        _write_json(resolved_output_dir / "gpu_power.json", gpu_power_payload)
        _write_json(resolved_output_dir / "energy_summary.json", energy_summary)
        _write_json(
            resolved_output_dir / "live_trial.json",
            {
                "trial_id": trial_id,
                "workload": str(workload),
                "model": model,
                "base_url": base_url,
                "endpoint": endpoint,
                "metrics_url": resolved_metrics_url,
                "gpu_ids": list(resolved_gpu_ids),
                "duration_s": duration_s,
                "request_rate": request_rate,
            },
        )
        return {
            "trial_id": trial_id,
            "output_dir": str(resolved_output_dir),
            "summary_json": str(resolved_output_dir / "summary.json"),
            "request_records_jsonl": str(resolved_output_dir / "request_records.jsonl"),
            "server_metrics_jsonl": str(resolved_output_dir / "server_metrics.jsonl"),
            "windows_csv": str(resolved_output_dir / "windows.csv"),
            "gpu_power_json": str(resolved_output_dir / "gpu_power.json"),
            "energy_summary_json": str(resolved_output_dir / "energy_summary.json"),
            "energy_summary": energy_summary,
        }

    def _execute(
        self,
        *,
        plan: EnergyPlan,
        state_store: EnergyRunStateStore,
        state: dict[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        jobs_by_id = {job.id: job for job in plan.jobs}
        pending = []
        for job_state in state.get("jobs", []):
            status = str(job_state.get("status", "planned"))
            if force:
                pending.append(job_state)
                continue
            if status in {"succeeded", "skipped"}:
                continue
            pending.append(job_state)

        try:
            for job_state in pending:
                job = jobs_by_id[str(job_state["job_id"])]
                if force:
                    self._reset_job_for_rerun(job_state)
                self._run_single_job(
                    plan=plan,
                    plan_job=job,
                    job_state=job_state,
                    state=state,
                    state_store=state_store,
                )
            summary = state_store.summarize(state)
            state["status"] = "failed" if summary["counts"]["failed"] else "completed"
            state_store.save(state)
            return state_store.write_summary_files(state)
        finally:
            self._release_active_server()
            self._lifecycle.shutdown()

    def _run_single_job(
        self,
        *,
        plan: EnergyPlan,
        plan_job: EnergyPlanJob,
        job_state: dict[str, Any],
        state: dict[str, Any],
        state_store: EnergyRunStateStore,
    ) -> None:
        result_dir = Path(str(job_state["result_dir"]))
        if result_dir.exists() and any(result_dir.iterdir()):
            raise_if_conflicting = {
                result_dir / "summary.json",
                result_dir / "energy_summary.json",
                result_dir / "gpu_power.json",
            }
            conflicts = [path for path in raise_if_conflicting if path.exists()]
            if conflicts:
                state_store.mark_job_failed(
                    state,
                    job_id=plan_job.id,
                    error=f"refusing to overwrite existing artifacts in {result_dir}",
                )
                return
        result_dir.mkdir(parents=True, exist_ok=True)

        if plan_job.launch.gpu_count > self._config.effective_max_active_gpus:
            state_store.mark_job_failed(
                state,
                job_id=plan_job.id,
                error=(
                    f"job requires gpu_count={plan_job.launch.gpu_count}, "
                    f"but executor max_active_gpus={self._config.effective_max_active_gpus}"
                ),
            )
            return

        state_store.increment_attempt(state, job_id=plan_job.id)
        state_store.set_job_status(state, job_id=plan_job.id, status="running")

        try:
            server = self._ensure_server(plan, plan_job, state_store)
            idle_snapshot = self._idle_snapshot
            if idle_snapshot is None:
                raise RuntimeError("idle baseline snapshot is unavailable after server startup")
            traffic_snapshot = self._run_monitored_trial(
                plan=plan,
                plan_job=plan_job,
                server=server,
                result_dir=result_dir,
                logs_dir=state_store.logs_dir,
            )
            if plan.defaults.cooldown_s > 0.0:
                self._sleep_fn(plan.defaults.cooldown_s)
            trial_summary_payload = _load_json_mapping(result_dir / "summary.json")
            energy_summary = compute_energy_summary(
                trial_summary_payload=trial_summary_payload,
                idle_snapshot=idle_snapshot,
                traffic_snapshot=traffic_snapshot,
            )
            gpu_power_payload = {
                "idle": idle_snapshot.to_dict(),
                "traffic": traffic_snapshot.to_dict(),
            }
            _write_json(result_dir / "gpu_power.json", gpu_power_payload)
            _write_json(result_dir / "energy_summary.json", energy_summary)

            vllm_stdout_log = state_store.logs_dir / f"{plan_job.id}.vllm.stdout.log"
            vllm_stderr_log = state_store.logs_dir / f"{plan_job.id}.vllm.stderr.log"
            _copy_file_if_exists(Path(str(server.stdout_log)), vllm_stdout_log)
            _copy_file_if_exists(Path(str(server.stderr_log)), vllm_stderr_log)

            state_store.mark_job_succeeded(
                state,
                job_id=plan_job.id,
                gpu_ids=tuple(server.gpu_ids),
                base_url=str(server.base_url),
                artifacts={
                    "summary_json": str(result_dir / "summary.json"),
                    "request_records_jsonl": str(result_dir / "request_records.jsonl"),
                    "server_metrics_jsonl": str(result_dir / "server_metrics.jsonl"),
                    "windows_csv": str(result_dir / "windows.csv"),
                    "gpu_power_json": str(result_dir / "gpu_power.json"),
                    "energy_summary_json": str(result_dir / "energy_summary.json"),
                    "profile_stdout_log": str(state_store.logs_dir / f"{plan_job.id}.profile.stdout.log"),
                    "profile_stderr_log": str(state_store.logs_dir / f"{plan_job.id}.profile.stderr.log"),
                    "vllm_stdout_log": str(vllm_stdout_log),
                    "vllm_stderr_log": str(vllm_stderr_log),
                },
            )
        except Exception as exc:
            state_store.mark_job_failed(state, job_id=plan_job.id, error=str(exc))

    def _run_monitored_trial(
        self,
        *,
        plan: EnergyPlan,
        plan_job: EnergyPlanJob,
        server,
        result_dir: Path,
        logs_dir: Path,
    ) -> MonitorRunResult:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"{plan_job.id}.profile.stdout.log"
        stderr_log = logs_dir / f"{plan_job.id}.profile.stderr.log"
        command = build_run_trial_command(
            plan=plan,
            job=plan_job,
            base_url=str(server.base_url),
            metrics_url=f"{server.base_url}/metrics",
            output_dir=result_dir,
        )

        env = os.environ.copy()
        profiler_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = profiler_root if not existing_pythonpath else f"{profiler_root}{os.pathsep}{existing_pythonpath}"

        monitors = self._build_monitors(
            gpu_ids=tuple(server.gpu_ids),
            interval_s=plan.defaults.gpu_monitor_interval_s,
            truncate_s=plan.defaults.gpu_monitor_truncate_s,
            monitor_clock=plan.defaults.monitor_clock,
        )
        start_ts = self._time_fn()
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle:
            for monitor in monitors:
                monitor.start()
            try:
                result = self._run_command(
                    command,
                    env=env,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            finally:
                end_ts = self._time_fn()
                for monitor in monitors:
                    monitor.stop()
        if result.returncode != 0:
            raise RuntimeError(f"run-trial failed with exit code {result.returncode}")
        return _collect_monitor_result(monitors=monitors, duration_s=end_ts - start_ts)

    def _run_live_monitored_trial(
        self,
        *,
        trial_id: str,
        output_dir: Path,
        workload: Path,
        model: str,
        base_url: str,
        endpoint: str,
        metrics_url: str,
        gpu_ids: tuple[int, ...],
        duration_s: float,
        request_rate: float,
        request_timeout_s: float,
        metrics_interval_s: float,
        window_s: float,
        gpu_monitor_interval_s: float,
        gpu_monitor_truncate_s: float,
        monitor_clock: bool,
        safety_max_outstanding: int | None,
        logs_dir: Path,
    ) -> MonitorRunResult:
        stdout_log = logs_dir / f"{trial_id}.profile.stdout.log"
        stderr_log = logs_dir / f"{trial_id}.profile.stderr.log"
        command = build_live_run_trial_command(
            trial_id=trial_id,
            output_dir=output_dir,
            workload=workload,
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            metrics_url=metrics_url,
            duration_s=duration_s,
            request_rate=request_rate,
            request_timeout_s=request_timeout_s,
            metrics_interval_s=metrics_interval_s,
            window_s=window_s,
            safety_max_outstanding=safety_max_outstanding,
        )

        env = os.environ.copy()
        profiler_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = profiler_root if not existing_pythonpath else f"{profiler_root}{os.pathsep}{existing_pythonpath}"

        monitors = self._build_monitors(
            gpu_ids=gpu_ids,
            interval_s=gpu_monitor_interval_s,
            truncate_s=gpu_monitor_truncate_s,
            monitor_clock=monitor_clock,
        )
        start_ts = self._time_fn()
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
            for monitor in monitors:
                monitor.start()
            try:
                result = self._run_command(
                    command,
                    env=env,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            finally:
                end_ts = self._time_fn()
                for monitor in monitors:
                    monitor.stop()
        if result.returncode != 0:
            raise RuntimeError(f"run-trial failed with exit code {result.returncode}")
        return _collect_monitor_result(monitors=monitors, duration_s=end_ts - start_ts)

    def _ensure_server(
        self,
        plan: EnergyPlan,
        plan_job: EnergyPlanJob,
        state_store: EnergyRunStateStore,
    ):
        if (
            self._active_server is not None
            and self._active_server_signature_key == plan_job.server_signature_key
            and self._lifecycle.is_ready(self._active_server)
        ):
            return self._active_server

        self._release_active_server()
        self._active_lease = self._gpu_manager.acquire(plan_job.launch.gpu_count)
        self._active_ports = self._port_allocator.reserve()
        runtime_signature = runtime_server_signature(
            server_signature_key=plan_job.server_signature_key,
            gpu_ids=self._active_lease.gpu_ids,
            base_port=self._active_ports.base_port,
            metrics_port=self._active_ports.metrics_port,
        )
        lifecycle_job = SimpleNamespace(
            model=plan_job.model,
            endpoint=plan_job.endpoint,
            launch=plan_job.launch.to_launch_config(),
            server_signature_key=plan_job.server_signature_key,
        )
        server = self._lifecycle.ensure_server(
            job=lifecycle_job,
            gpu_ids=self._active_lease.gpu_ids,
            ports=self._active_ports,
            runtime_signature=runtime_signature,
            logs_dir=state_store.logs_dir,
            force_restart=False,
        )
        self._active_server = server
        self._active_server_signature_key = plan_job.server_signature_key
        self._idle_snapshot = self._collect_idle_baseline(
            gpu_ids=self._active_lease.gpu_ids,
            duration_s=plan.defaults.idle_monitor_duration_s,
            interval_s=plan.defaults.gpu_monitor_interval_s,
            truncate_s=0.0,
            monitor_clock=plan.defaults.monitor_clock,
        )
        return server

    def _collect_idle_baseline(
        self,
        *,
        gpu_ids: tuple[int, ...],
        duration_s: float,
        interval_s: float,
        truncate_s: float,
        monitor_clock: bool,
    ) -> MonitorRunResult:
        if duration_s <= 0.0:
            return MonitorRunResult(duration_s=0.0, per_gpu=(), aggregate_trace_mw=())
        monitors = self._build_monitors(
            gpu_ids=gpu_ids,
            interval_s=interval_s,
            truncate_s=truncate_s,
            monitor_clock=monitor_clock,
        )
        start_ts = self._time_fn()
        for monitor in monitors:
            monitor.start()
        try:
            self._sleep_fn(duration_s)
        finally:
            end_ts = self._time_fn()
            for monitor in monitors:
                monitor.stop()
        return _collect_monitor_result(monitors=monitors, duration_s=end_ts - start_ts)

    def _build_monitors(
        self,
        *,
        gpu_ids: tuple[int, ...],
        interval_s: float,
        truncate_s: float,
        monitor_clock: bool,
    ) -> list[MonitorProtocol]:
        return [
            self._monitor_factory(
                gpu_id=gpu_id,
                interval=interval_s,
                truncate=truncate_s,
                monitor_clock=monitor_clock,
            )
            for gpu_id in gpu_ids
        ]

    def _release_active_server(self) -> None:
        if self._active_server is not None:
            self._lifecycle.stop_active_server(reason="energy_executor_release")
            self._active_server = None
        if self._active_ports is not None:
            self._port_allocator.release(self._active_ports)
            self._active_ports = None
        if self._active_lease is not None:
            self._gpu_manager.release(self._active_lease)
            self._active_lease = None
        self._active_server_signature_key = None
        self._idle_snapshot = None

    @staticmethod
    def _reset_job_for_rerun(job_state: dict[str, Any]) -> None:
        result_dir = Path(str(job_state["result_dir"]))
        if result_dir.exists():
            shutil.rmtree(result_dir)
        job_state["status"] = "planned"
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
            "profile_stdout_log": None,
            "profile_stderr_log": None,
            "vllm_stdout_log": None,
            "vllm_stderr_log": None,
        }


def build_run_trial_command(
    *,
    plan: EnergyPlan,
    job: EnergyPlanJob,
    base_url: str,
    metrics_url: str,
    output_dir: Path,
) -> tuple[str, ...]:
    python_executable = plan.plan.python_executable or sys.executable
    command: list[str] = [
        python_executable,
        "-m",
        "llm_mst_finder.cli",
        "run-trial",
        "--trial-id",
        job.id,
        "--output-dir",
        str(output_dir),
        "--workload",
        str(job.workload),
        "--mode",
        "open-loop",
        "--duration-s",
        str(plan.defaults.duration_s),
        "--base-url",
        base_url,
        "--endpoint",
        job.endpoint,
        "--model",
        job.model,
        "--request-rate",
        str(job.request_rate),
        "--request-timeout-s",
        str(plan.defaults.request_timeout_s),
        "--metrics-url",
        metrics_url,
        "--metrics-interval-s",
        str(plan.defaults.metrics_interval_s),
        "--window-s",
        str(plan.defaults.window_s),
    ]
    if plan.defaults.safety_max_outstanding is not None:
        command.extend(["--safety-max-outstanding", str(plan.defaults.safety_max_outstanding)])
    if job.launch.max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(job.launch.max_num_seqs)])
    if job.launch.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(job.launch.max_num_batched_tokens)])
    return tuple(command)


def build_live_run_trial_command(
    *,
    trial_id: str,
    output_dir: Path,
    workload: Path,
    model: str,
    base_url: str,
    endpoint: str,
    metrics_url: str,
    duration_s: float,
    request_rate: float,
    request_timeout_s: float,
    metrics_interval_s: float,
    window_s: float,
    safety_max_outstanding: int | None = None,
) -> tuple[str, ...]:
    command: list[str] = [
        sys.executable,
        "-m",
        "llm_mst_finder.cli",
        "run-trial",
        "--trial-id",
        trial_id,
        "--output-dir",
        str(output_dir),
        "--workload",
        str(workload),
        "--mode",
        "open-loop",
        "--duration-s",
        str(duration_s),
        "--base-url",
        base_url,
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--request-rate",
        str(request_rate),
        "--request-timeout-s",
        str(request_timeout_s),
        "--metrics-url",
        metrics_url,
        "--metrics-interval-s",
        str(metrics_interval_s),
        "--window-s",
        str(window_s),
    ]
    if safety_max_outstanding is not None:
        command.extend(["--safety-max-outstanding", str(safety_max_outstanding)])
    return tuple(command)


def compute_energy_summary(
    *,
    trial_summary_payload: Mapping[str, Any],
    idle_snapshot: MonitorRunResult,
    traffic_snapshot: MonitorRunResult,
) -> dict[str, Any]:
    summary_payload = _require_mapping(trial_summary_payload.get("summary"), "summary.json.summary")
    benchmark_metrics = _require_mapping(summary_payload.get("benchmark_metrics"), "summary.json.summary.benchmark_metrics")

    successful_requests = _optional_non_negative_int(summary_payload.get("successful_requests"))
    started_requests = _optional_non_negative_int(summary_payload.get("started_requests"))
    total_input_tokens = _optional_non_negative_int(benchmark_metrics.get("total_input_tokens"))
    total_output_tokens = _optional_non_negative_int(benchmark_metrics.get("total_output_tokens"))
    total_tokens = None
    if total_input_tokens is not None and total_output_tokens is not None:
        total_tokens = total_input_tokens + total_output_tokens

    avg_power_w = traffic_snapshot.avg_power_mw / 1000.0
    idle_avg_power_w = idle_snapshot.avg_power_mw / 1000.0 if idle_snapshot.per_gpu else 0.0
    incremental_avg_power_w = avg_power_w - idle_avg_power_w
    energy_joules = avg_power_w * traffic_snapshot.duration_s
    incremental_energy_joules = incremental_avg_power_w * traffic_snapshot.duration_s

    aggregate_stats = _trace_stats_w(traffic_snapshot.aggregate_trace_mw)
    per_gpu = []
    idle_by_gpu = {snapshot.gpu_id: snapshot for snapshot in idle_snapshot.per_gpu}
    for snapshot in traffic_snapshot.per_gpu:
        idle_gpu = idle_by_gpu.get(snapshot.gpu_id)
        gpu_avg_power_w = snapshot.avg_power_mw / 1000.0
        idle_gpu_avg_power_w = 0.0 if idle_gpu is None else idle_gpu.avg_power_mw / 1000.0
        per_gpu.append(
            {
                "gpu_id": snapshot.gpu_id,
                "avg_power_w": gpu_avg_power_w,
                "idle_avg_power_w": idle_gpu_avg_power_w,
                "incremental_avg_power_w": gpu_avg_power_w - idle_gpu_avg_power_w,
                "energy_joules": gpu_avg_power_w * traffic_snapshot.duration_s,
                "incremental_energy_joules": (gpu_avg_power_w - idle_gpu_avg_power_w) * traffic_snapshot.duration_s,
                "power_stats": _convert_power_stats_w(snapshot.power_stats),
            }
        )

    return {
        "energy_joules": energy_joules,
        "incremental_energy_joules": incremental_energy_joules,
        "energy_kwh": energy_joules / 3_600_000.0,
        "avg_power_w": avg_power_w,
        "idle_avg_power_w": idle_avg_power_w,
        "incremental_avg_power_w": incremental_avg_power_w,
        "min_power_w": aggregate_stats["min_power_w"],
        "p50_power_w": aggregate_stats["p50_power_w"],
        "p90_power_w": aggregate_stats["p90_power_w"],
        "p95_power_w": aggregate_stats["p95_power_w"],
        "p99_power_w": aggregate_stats["p99_power_w"],
        "max_power_w": aggregate_stats["max_power_w"],
        "energy_per_successful_request_j": _safe_divide(energy_joules, successful_requests),
        "incremental_energy_per_successful_request_j": _safe_divide(incremental_energy_joules, successful_requests),
        "energy_per_total_request_j": _safe_divide(energy_joules, started_requests),
        "incremental_energy_per_total_request_j": _safe_divide(incremental_energy_joules, started_requests),
        "energy_per_total_token_j": _safe_divide(energy_joules, total_tokens),
        "incremental_energy_per_total_token_j": _safe_divide(incremental_energy_joules, total_tokens),
        "monitor_duration_s": traffic_snapshot.duration_s,
        "idle_monitor_duration_s": idle_snapshot.duration_s,
        "gpu_ids": list(traffic_snapshot.gpu_ids),
        "successful_requests": successful_requests,
        "started_requests": started_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "per_gpu": per_gpu,
    }


def _collect_monitor_result(
    *,
    monitors: Sequence[MonitorProtocol],
    duration_s: float,
) -> MonitorRunResult:
    per_gpu: list[PowerSnapshot] = []
    traces: list[list[int]] = []
    for monitor in monitors:
        snapshot = monitor.get_snapshot(wait=True, timeout=1.0)
        power_trace = list(getattr(snapshot, "power_trace_mw", []))
        traces.append(power_trace)
        raw_clocks = getattr(snapshot, "clock_trace_mhz", None)
        clock_trace = None
        if raw_clocks is not None:
            clock_trace = tuple(tuple(int(item) for item in entry) for entry in raw_clocks)
        per_gpu.append(
            PowerSnapshot(
                gpu_id=int(getattr(monitor, "gpu_id")),
                avg_power_mw=float(getattr(snapshot, "avg_power_mw", 0.0)),
                power_stats=dict(getattr(snapshot, "power_stats", {})),
                power_trace_mw=tuple(int(item) for item in power_trace),
                clock_trace_mhz=clock_trace,
            )
        )
    aggregate_trace = _aggregate_trace_mw(traces)
    return MonitorRunResult(
        duration_s=max(0.0, duration_s),
        per_gpu=tuple(per_gpu),
        aggregate_trace_mw=tuple(aggregate_trace),
    )


def _aggregate_trace_mw(traces: Sequence[Sequence[int]]) -> list[float]:
    if not traces or any(not trace for trace in traces):
        return []
    min_len = min(len(trace) for trace in traces)
    aggregate: list[float] = []
    for index in range(min_len):
        aggregate.append(float(sum(trace[index] for trace in traces)))
    return aggregate


def _trace_stats_w(trace_mw: Sequence[float]) -> dict[str, float | None]:
    if not trace_mw:
        return {
            "min_power_w": None,
            "p50_power_w": None,
            "p90_power_w": None,
            "p95_power_w": None,
            "p99_power_w": None,
            "max_power_w": None,
        }
    ordered = sorted(float(value) / 1000.0 for value in trace_mw)
    return {
        "min_power_w": ordered[0],
        "p50_power_w": _percentile(ordered, 50.0),
        "p90_power_w": _percentile(ordered, 90.0),
        "p95_power_w": _percentile(ordered, 95.0),
        "p99_power_w": _percentile(ordered, 99.0),
        "max_power_w": ordered[-1],
    }


def _convert_power_stats_w(power_stats: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in power_stats.items():
        if key.endswith("power") or key.startswith("power_") or key in {"min_power", "max_power", "median_power"}:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                converted[f"{key}_w"] = float(value) / 1000.0
                continue
        converted[key] = value
    return converted


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    low_index = int(position)
    high_index = min(low_index + 1, len(values) - 1)
    fraction = position - low_index
    return values[low_index] + (values[high_index] - values[low_index]) * fraction


def _safe_divide(numerator: float, denominator: int | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must decode to a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_file_if_exists(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copyfile(source, destination)


def _default_monitor_factory(*, gpu_id: int, interval: float, truncate: float, monitor_clock: bool) -> MonitorProtocol:
    from gpu_monitor import GPUMonitor

    return GPUMonitor(
        gpu_id=gpu_id,
        interval=interval,
        truncate=truncate,
        monitor_clock=monitor_clock,
    )


def _default_run_command(
    command: tuple[str, ...],
    *,
    env: dict[str, str],
    cwd: str,
    stdout,
    stderr,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        check=False,
    )
