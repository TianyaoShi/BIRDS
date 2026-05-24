from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from energy_profiler.local_scheduler import (
    EnergyProfilingAdapter,
    SchedulerEnergyStateStore,
    expand_energy_plan_for_local_scheduler,
    run_config_from_energy_plan,
)
from energy_profiler.models import (
    EnergyLaunchConfig,
    EnergyPlan,
    EnergyPlanDefaults,
    EnergyPlanExecution,
    EnergyPlanHeader,
    EnergyPlanJob,
)
from energy_profiler.planning import write_energy_plan
from energy_profiler.reporting import EnergyRunStateStore
from local_orchestrator.resources import GPULeaseManager, PortAllocator
from local_orchestrator.scheduler import OrchestratorScheduler


class _FakeLifecycle:
    def __init__(self) -> None:
        self.active_server = None

    def ensure_server(
        self,
        *,
        job,
        gpu_ids,
        ports,
        runtime_signature,
        logs_dir,
        force_restart: bool = False,
    ):
        del job, runtime_signature, force_restart
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"vllm_gpu{'-'.join(str(gpu_id) for gpu_id in gpu_ids)}.stdout.log"
        stderr_log = logs_dir / f"vllm_gpu{'-'.join(str(gpu_id) for gpu_id in gpu_ids)}.stderr.log"
        stdout_log.write_text("server stdout\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        self.active_server = SimpleNamespace(
            gpu_ids=tuple(gpu_ids),
            base_url=f"http://127.0.0.1:{ports.base_port}",
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        return self.active_server

    def is_ready(self, server, *, timeout_s: float = 2.0) -> bool:
        del timeout_s
        return self.active_server is server

    def stop_active_server(self, *, reason: str) -> None:
        del reason
        self.active_server = None

    def shutdown(self) -> None:
        self.active_server = None


class _RecordingExecutor:
    calls: list[dict[str, object]] = []

    def run_live_trial(self, **kwargs):
        self.calls.append(dict(kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        energy_summary_path = output_dir / "energy_summary.json"
        gpu_power_path = output_dir / "gpu_power.json"
        summary_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "successful_requests": 1,
                        "started_requests": 1,
                        "benchmark_metrics": {
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                        },
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        energy_summary_path.write_text(
            json.dumps(
                {
                    "gpu_ids": list(kwargs["gpu_ids"]),
                    "energy_joules": 1.0,
                    "incremental_energy_joules": 0.5,
                    "avg_power_w": 1.0,
                    "successful_requests": 1,
                    "started_requests": 1,
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "total_tokens": 15,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        gpu_power_path.write_text("{}\n", encoding="utf-8")
        return {
            "trial_id": kwargs["trial_id"],
            "summary_json": str(summary_path),
            "request_records_jsonl": None,
            "server_metrics_jsonl": None,
            "windows_csv": None,
            "gpu_power_json": str(gpu_power_path),
            "energy_summary_json": str(energy_summary_path),
            "repeats": [],
        }


def _energy_job(tmp_path: Path, *, job_id: str, gpu_count: int) -> EnergyPlanJob:
    return EnergyPlanJob(
        id=job_id,
        source_experiment_id=f"source-{job_id}",
        source_result_dir=tmp_path / "mst" / job_id,
        model=f"model/{job_id}",
        workload=tmp_path / "workload.yaml",
        endpoint="/v1/chat/completions",
        request_rate=1.0,
        mst_rate=1.0,
        mst_rate_source="max_slo_satisfying_request_rate",
        launch=EnergyLaunchConfig(
            gpu_count=gpu_count,
            tensor_parallel_size=gpu_count,
            max_model_len=8192,
        ),
        server_signature_key=f"sig-{job_id}",
        server_config_slug=f"server-{job_id}",
    )


def test_local_energy_scheduler_uses_physical_gpu_ids_for_monitoring(tmp_path: Path) -> None:
    _RecordingExecutor.calls = []
    workload = tmp_path / "workload.yaml"
    workload.write_text("name: workload\n", encoding="utf-8")
    plan = EnergyPlan(
        plan=EnergyPlanHeader(
            plan_id="energy-plan",
            source_orchestrator_run_root=tmp_path / "orchestrator",
            output_root=tmp_path / "energy",
            python_executable="python",
        ),
        defaults=EnergyPlanDefaults(
            duration_s=1.0,
            warmup_s=0.0,
            traffic_warmup_s=0.0,
            cooldown_s=0.0,
            repeats=1,
        ),
        execution=EnergyPlanExecution(
            allowed_gpu_ids=(2, 4, 7),
            max_active_gpus=3,
            base_port_start=9100,
            base_port_end=9199,
        ),
        jobs=(
            _energy_job(tmp_path, job_id="tp1-a", gpu_count=1),
            _energy_job(tmp_path, job_id="tp2", gpu_count=2),
            _energy_job(tmp_path, job_id="tp1-b", gpu_count=1),
        ),
    )
    plan_path = write_energy_plan(plan, tmp_path / "energy_plan.yaml")
    run_root = tmp_path / "energy" / "energy-plan" / "local-energy-test"
    jobs = expand_energy_plan_for_local_scheduler(plan, run_root=run_root)
    state_store = SchedulerEnergyStateStore(EnergyRunStateStore(run_root), plan=plan)
    state = state_store.initialize_new(plan_path=plan_path)
    run_config = run_config_from_energy_plan(plan, run_id="local-energy-test")
    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(
            allowed_gpu_ids=run_config.allowed_gpu_ids,
            max_active_gpus=run_config.max_active_gpus,
        ),
        port_allocator=PortAllocator(
            base_port_start=run_config.base_port_start,
            base_port_end=run_config.base_port_end,
            metrics_port_offset=run_config.metrics_port_offset,
        ),
        lifecycle=_FakeLifecycle(),
        adapter=EnergyProfilingAdapter(
            plan=plan,
            executor_factory=_RecordingExecutor,
        ),
        state_store=state_store,
        lifecycle_factory=_FakeLifecycle,
    )

    summary = scheduler.run(jobs=jobs, state=state, resume=False)

    assert summary["counts"]["succeeded"] == 3
    assert any(call["gpu_ids"] == (2, 4) for call in _RecordingExecutor.calls)
    assert all(
        set(call["gpu_ids"]).issubset({2, 4, 7})
        for call in _RecordingExecutor.calls
    )
    state_payload = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    tp2_state = next(job for job in state_payload["jobs"] if job["job_id"] == "tp2")
    assert tp2_state["gpu_ids"] == [2, 4]
