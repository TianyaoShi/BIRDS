from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from energy_profiler.executor import (
    EnergyExecutor,
    EnergyExecutorConfig,
    MonitorRunResult,
    PowerSnapshot,
    build_run_trial_command,
    compute_energy_summary,
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


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration_s: float) -> None:
        self.now += duration_s

    def advance(self, duration_s: float) -> None:
        self.now += duration_s


class _FakeLifecycle:
    def __init__(self) -> None:
        self.active_server = None
        self.ensure_calls = 0
        self.stop_calls = 0

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
        stdout_log = logs_dir / "server.stdout.log"
        stderr_log = logs_dir / "server.stderr.log"
        stdout_log.write_text("server stdout\n", encoding="utf-8")
        stderr_log.write_text("server stderr\n", encoding="utf-8")
        self.ensure_calls += 1
        self.active_server = SimpleNamespace(
            gpu_ids=gpu_ids,
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
        self.stop_calls += 1
        self.active_server = None

    def shutdown(self) -> None:
        self.active_server = None


class _FakeMonitor:
    def __init__(self, *, gpu_id: int, interval: float, truncate: float, monitor_clock: bool) -> None:
        del interval, monitor_clock
        self.gpu_id = gpu_id
        self._is_idle = truncate == 0.0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_snapshot(self, wait: bool = False, timeout: float | None = None):
        del wait, timeout
        avg_power = 200_000 if self._is_idle else 300_000
        return SimpleNamespace(
            avg_power_mw=avg_power,
            power_stats={
                "min_power": avg_power - 10_000,
                "median_power": avg_power,
                "power_95p": avg_power + 10_000,
                "max_power": avg_power + 20_000,
            },
            power_trace_mw=[avg_power - 10_000, avg_power, avg_power + 10_000],
            clock_trace_mhz=None,
        )


def _make_plan(tmp_path: Path) -> EnergyPlan:
    launch = EnergyLaunchConfig(
        executable="vllm",
        tensor_parallel_size=1,
        gpu_count=1,
        dtype="float16",
        max_model_len=32768,
        max_num_seqs=64,
    )
    return EnergyPlan(
        plan=EnergyPlanHeader(
            plan_id="energy-plan-a",
            source_orchestrator_run_root=tmp_path / "results" / "orchestrator" / "run-a",
            output_root=tmp_path / "results" / "energy",
            python_executable="python",
            mode="sweep",
        ),
        defaults=EnergyPlanDefaults(duration_s=15.0, warmup_s=5.0, cooldown_s=0.0),
        execution=EnergyPlanExecution(allowed_gpu_ids=(2,), max_active_gpus=1),
        jobs=(
            EnergyPlanJob(
                id="job-a-r1",
                source_experiment_id="exp-a",
                source_result_dir=tmp_path / "results" / "mst" / "job-a",
                model="Qwen/Qwen3-8B",
                workload=tmp_path / "sharegpt.yaml",
                endpoint="/v1/chat/completions",
                request_rate=1.0,
                mst_rate=1.84,
                mst_rate_source="max_slo_satisfying_request_rate",
                launch=launch,
                server_signature_key="sig-a",
                server_config_slug="server-a",
                metadata={"rounding_step": 0.1},
            ),
            EnergyPlanJob(
                id="job-a-r2",
                source_experiment_id="exp-a",
                source_result_dir=tmp_path / "results" / "mst" / "job-a",
                model="Qwen/Qwen3-8B",
                workload=tmp_path / "sharegpt.yaml",
                endpoint="/v1/chat/completions",
                request_rate=2.0,
                mst_rate=1.84,
                mst_rate_source="max_slo_satisfying_request_rate",
                launch=launch,
                server_signature_key="sig-a",
                server_config_slug="server-a",
                metadata={"rounding_step": 0.1},
            ),
        ),
    )


def test_build_run_trial_command_uses_plain_mst_finder_cli(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    command = build_run_trial_command(
        plan=plan,
        job=plan.jobs[0],
        base_url="http://127.0.0.1:8000",
        metrics_url="http://127.0.0.1:8000/metrics",
        output_dir=tmp_path / "job-a-r1",
    )

    assert command[:4] == ("python", "-m", "llm_mst_finder.cli", "run-trial")
    assert "--request-rate" in command
    assert "--metrics-url" in command
    assert "--gpu-id" not in command
    assert "--gpu-monitor-interval-s" not in command
    assert "--energy-summary-path" not in command


def test_compute_energy_summary_handles_zero_successful_requests() -> None:
    idle = MonitorRunResult(
        duration_s=5.0,
        per_gpu=(
            PowerSnapshot(
                gpu_id=0,
                avg_power_mw=200_000,
                power_stats={"min_power": 190_000, "median_power": 200_000, "power_95p": 210_000, "max_power": 220_000},
                power_trace_mw=(190_000, 200_000, 210_000),
                clock_trace_mhz=None,
            ),
        ),
        aggregate_trace_mw=(190_000, 200_000, 210_000),
    )
    traffic = MonitorRunResult(
        duration_s=10.0,
        per_gpu=(
            PowerSnapshot(
                gpu_id=0,
                avg_power_mw=300_000,
                power_stats={"min_power": 290_000, "median_power": 300_000, "power_95p": 310_000, "max_power": 320_000},
                power_trace_mw=(290_000, 300_000, 310_000),
                clock_trace_mhz=None,
            ),
        ),
        aggregate_trace_mw=(290_000, 300_000, 310_000),
    )
    summary = compute_energy_summary(
        trial_summary_payload={
            "summary": {
                "successful_requests": 0,
                "started_requests": 5,
                "benchmark_metrics": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                },
            }
        },
        idle_snapshot=idle,
        traffic_snapshot=traffic,
    )

    assert summary["energy_joules"] == pytest.approx(3000.0)
    assert summary["incremental_energy_joules"] == pytest.approx(1000.0)
    assert summary["energy_per_successful_request_j"] is None
    assert summary["energy_per_total_request_j"] == pytest.approx(600.0)
    assert summary["energy_per_total_token_j"] == pytest.approx(20.0)
    assert "energy_per_output_token_j" not in summary


def test_executor_run_plan_reuses_server_and_writes_energy_artifacts(tmp_path: Path) -> None:
    clock = _FakeClock()
    lifecycle = _FakeLifecycle()
    recorded_commands: list[tuple[str, ...]] = []
    plan = _make_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "experiments" / "energy" / "energy-plan-a.yaml")

    def run_command(command, *, env, cwd, stdout, stderr):
        del env, cwd
        recorded_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "summary": {
                "successful_requests": 4,
                "started_requests": 5,
                "benchmark_metrics": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                },
            }
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "request_records.jsonl").write_text("", encoding="utf-8")
        (output_dir / "server_metrics.jsonl").write_text("", encoding="utf-8")
        (output_dir / "windows.csv").write_text("trial_id,window_idx\n", encoding="utf-8")
        stdout.write("{}\n")
        stderr.write("")
        duration_s = float(command[command.index("--duration-s") + 1])
        clock.advance(duration_s)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    executor = EnergyExecutor(
        config=EnergyExecutorConfig(allowed_gpu_ids=(0,), max_active_gpus=1),
        lifecycle_factory=lambda: lifecycle,
        run_command=run_command,
        monitor_factory=_FakeMonitor,
        sleep_fn=clock.sleep,
        time_fn=clock,
    )
    summary = executor.run_plan(plan_path)

    assert summary["counts"]["succeeded"] == 2
    assert lifecycle.ensure_calls == 1
    assert len(recorded_commands) == 4
    assert recorded_commands[0][recorded_commands[0].index("--trial-id") + 1] == "job-a-r1-warmup"
    assert recorded_commands[0][recorded_commands[0].index("--duration-s") + 1] == "30.0"
    assert recorded_commands[1][recorded_commands[1].index("--trial-id") + 1] == "job-a-r1"
    assert recorded_commands[1][recorded_commands[1].index("--duration-s") + 1] == "15.0"
    run_root = plan.plan.output_root / plan.plan.plan_id
    assert (run_root / "jobs" / "job-a-r1" / "warmup" / "summary.json").is_file()
    assert (run_root / "jobs" / "job-a-r1" / "gpu_power.json").is_file()
    assert (run_root / "jobs" / "job-a-r1" / "energy_summary.json").is_file()
    energy_payload = json.loads((run_root / "jobs" / "job-a-r1" / "energy_summary.json").read_text(encoding="utf-8"))
    assert energy_payload["gpu_ids"] == [0]
    assert energy_payload["energy_per_total_token_j"] is not None
    assert (run_root / "logs" / "job-a-r1.profile.stdout.log").is_file()
    assert (run_root / "logs" / "job-a-r1.vllm.stdout.log").is_file()


def test_executor_run_plan_repeats_trial_and_aggregates_energy(tmp_path: Path) -> None:
    clock = _FakeClock()
    lifecycle = _FakeLifecycle()
    recorded_commands: list[tuple[str, ...]] = []
    base_plan = _make_plan(tmp_path)
    plan = replace(
        base_plan,
        defaults=replace(
            base_plan.defaults,
            traffic_warmup_s=3.0,
            repeats=2,
            repeat_cooldown_s=2.0,
            cooldown_s=0.0,
        ),
        jobs=(base_plan.jobs[0],),
    )
    plan_path = write_energy_plan(plan, tmp_path / "experiments" / "energy" / "energy-plan-repeats.yaml")

    def run_command(command, *, env, cwd, stdout, stderr):
        del env, cwd
        recorded_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "summary": {
                "successful_requests": 2,
                "started_requests": 2,
                "benchmark_metrics": {
                    "total_input_tokens": 40,
                    "total_output_tokens": 20,
                },
            }
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "request_records.jsonl").write_text("", encoding="utf-8")
        (output_dir / "server_metrics.jsonl").write_text("", encoding="utf-8")
        (output_dir / "windows.csv").write_text("trial_id,window_idx\n", encoding="utf-8")
        stdout.write("{}\n")
        stderr.write("")
        clock.advance(float(command[command.index("--duration-s") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    executor = EnergyExecutor(
        config=EnergyExecutorConfig(allowed_gpu_ids=(0,), max_active_gpus=1),
        lifecycle_factory=lambda: lifecycle,
        run_command=run_command,
        monitor_factory=_FakeMonitor,
        sleep_fn=clock.sleep,
        time_fn=clock,
    )
    summary = executor.run_plan(plan_path)

    assert summary["counts"]["succeeded"] == 1
    assert [command[command.index("--trial-id") + 1] for command in recorded_commands] == [
        "job-a-r1-repeat-001-warmup",
        "job-a-r1-repeat-001",
        "job-a-r1-repeat-002",
    ]
    run_root = plan.plan.output_root / plan.plan.plan_id
    result_dir = run_root / "jobs" / "job-a-r1"
    assert (result_dir / "repeat_001" / "warmup" / "summary.json").is_file()
    assert (result_dir / "repeat_001" / "energy_summary.json").is_file()
    assert (result_dir / "repeat_002" / "energy_summary.json").is_file()
    aggregate = json.loads((result_dir / "energy_summary.json").read_text(encoding="utf-8"))
    assert aggregate["repeat_count"] == 2
    assert aggregate["successful_repeat_count"] == 2
    assert aggregate["energy_joules"] == pytest.approx(4500.0)
    assert aggregate["repeat_statistics"]["energy_joules"]["stdev"] == pytest.approx(0.0)
    state_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert state_summary["jobs"][0]["artifacts"]["repeats"][0]["repeat_index"] == 1


def test_executor_warmup_failure_fails_before_measured_trial(tmp_path: Path) -> None:
    clock = _FakeClock()
    lifecycle = _FakeLifecycle()
    recorded_commands: list[tuple[str, ...]] = []
    base_plan = _make_plan(tmp_path)
    plan = replace(base_plan, jobs=(base_plan.jobs[0],), defaults=replace(base_plan.defaults, cooldown_s=0.0))
    plan_path = write_energy_plan(plan, tmp_path / "experiments" / "energy" / "energy-plan-warmup-fail.yaml")

    def run_command(command, *, env, cwd, stdout, stderr):
        del env, cwd, stdout
        recorded_commands.append(command)
        stderr.write("warmup failed\n")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="warmup failed\n")

    executor = EnergyExecutor(
        config=EnergyExecutorConfig(allowed_gpu_ids=(0,), max_active_gpus=1),
        lifecycle_factory=lambda: lifecycle,
        run_command=run_command,
        monitor_factory=_FakeMonitor,
        sleep_fn=clock.sleep,
        time_fn=clock,
    )
    summary = executor.run_plan(plan_path)

    assert summary["counts"]["failed"] == 1
    assert len(recorded_commands) == 1
    assert recorded_commands[0][recorded_commands[0].index("--trial-id") + 1] == "job-a-r1-warmup"
    run_root = plan.plan.output_root / plan.plan.plan_id
    assert not (run_root / "jobs" / "job-a-r1" / "energy_summary.json").exists()
    assert "warmup run-trial failed" in summary["jobs"][0]["last_error"]


def test_executor_run_plan_uses_plan_execution_when_config_is_not_overridden(tmp_path: Path) -> None:
    clock = _FakeClock()
    lifecycle = _FakeLifecycle()
    recorded_commands: list[tuple[str, ...]] = []
    plan = _make_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "experiments" / "energy" / "energy-plan-a.yaml")

    def run_command(command, *, env, cwd, stdout, stderr):
        del env, cwd
        recorded_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "summary": {
                "successful_requests": 1,
                "started_requests": 1,
                "benchmark_metrics": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                },
            }
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "request_records.jsonl").write_text("", encoding="utf-8")
        (output_dir / "server_metrics.jsonl").write_text("", encoding="utf-8")
        (output_dir / "windows.csv").write_text("trial_id,window_idx\n", encoding="utf-8")
        stdout.write("{}\n")
        stderr.write("")
        clock.advance(float(command[command.index("--duration-s") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    executor = EnergyExecutor(
        lifecycle_factory=lambda: lifecycle,
        run_command=run_command,
        monitor_factory=_FakeMonitor,
        sleep_fn=clock.sleep,
        time_fn=clock,
    )
    summary = executor.run_plan(plan_path)

    assert summary["counts"]["succeeded"] == 2
    assert lifecycle.active_server is None
    run_root = plan.plan.output_root / plan.plan.plan_id
    energy_payload = json.loads((run_root / "jobs" / "job-a-r1" / "energy_summary.json").read_text(encoding="utf-8"))
    assert energy_payload["gpu_ids"] == [2]
