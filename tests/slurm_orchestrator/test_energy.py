from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from energy_profiler.models import (
    EnergyLaunchConfig,
    EnergyPlan,
    EnergyPlanDefaults,
    EnergyPlanExecution,
    EnergyPlanHeader,
    EnergyPlanJob,
    EnergyPlanSlurm,
)
from energy_profiler.planning import write_energy_plan
from local_orchestrator.models import SlurmConfig
from slurm_orchestrator.cli import main as slurm_main
from slurm_orchestrator.energy import (
    collect_energy_run,
    finalize_energy_task,
    materialize_energy_run_plan,
    render_energy_task_shell,
)


def _make_energy_plan(tmp_path: Path) -> EnergyPlan:
    launch_a = EnergyLaunchConfig(
        executable="vllm",
        tensor_parallel_size=1,
        gpu_count=1,
        dtype="float16",
        max_model_len=4096,
        env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
    )
    launch_b = EnergyLaunchConfig(
        executable="vllm",
        tensor_parallel_size=2,
        gpu_count=2,
        max_model_len=8192,
    )
    return EnergyPlan(
        plan=EnergyPlanHeader(
            plan_id="energy-plan-a",
            source_orchestrator_run_root=tmp_path / "orchestrator" / "run-a",
            output_root=tmp_path / "energy",
            python_executable="/plan/python",
            mode="mst-rounded",
        ),
        defaults=EnergyPlanDefaults(
            duration_s=12.0,
            warmup_s=3.0,
            cooldown_s=0.0,
            metrics_interval_s=0.5,
            window_s=4.0,
            gpu_monitor_interval_s=0.25,
            gpu_monitor_truncate_s=1.0,
            safety_max_outstanding=8,
        ),
        execution=EnergyPlanExecution(
            allowed_gpu_ids=(0, 1),
            max_active_gpus=2,
            base_port_start=8100,
            base_port_end=8199,
            metrics_port_offset=2000,
        ),
        jobs=(
            EnergyPlanJob(
                id="job-a",
                source_experiment_id="exp-a",
                source_result_dir=tmp_path / "mst" / "exp-a",
                model="model-a",
                workload=tmp_path / "sharegpt.yaml",
                endpoint="/v1/chat/completions",
                request_rate=1.5,
                mst_rate=1.6,
                mst_rate_source="max_slo_satisfying_request_rate",
                launch=launch_a,
                server_signature_key="sig-a",
                server_config_slug="tp1",
            ),
            EnergyPlanJob(
                id="job-b",
                source_experiment_id="exp-b",
                source_result_dir=tmp_path / "mst" / "exp-b",
                model="model-b",
                workload=tmp_path / "sharegpt.yaml",
                endpoint="/v1/chat/completions",
                request_rate=2.5,
                mst_rate=2.6,
                mst_rate_source="max_slo_satisfying_request_rate",
                launch=launch_b,
                server_signature_key="sig-b",
                server_config_slug="tp2",
            ),
        ),
    )


def test_materialize_energy_run_plan_groups_jobs_and_renders_energy_script(tmp_path: Path) -> None:
    plan = _make_energy_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")
    slurm = SlurmConfig(
        partition="ai",
        account="research",
        qos="normal",
        time="00:45:00",
        modules=("cuda/12.4",),
        setup_commands=("source /venv/bin/activate",),
        python_executable="/slurm/python",
        array_concurrency_limit=3,
        base_port=8500,
    )

    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
        slurm=slurm,
    )

    assert run_plan["run_root"] == str((tmp_path / "energy" / "energy-plan-a" / "energy-slurm-run").resolve())
    assert run_plan["job_count"] == 2
    assert [group["group_key"] for group in run_plan["groups"]] == ["gpu1", "gpu2"]
    assert run_plan["jobs"][0]["base_port"] == 8500
    assert run_plan["jobs"][1]["base_port"] == 8501

    gpu1_group = json.loads(Path(run_plan["groups"][0]["plan_path"]).read_text(encoding="utf-8"))
    assert gpu1_group["python_executable"] == "/slurm/python"
    assert gpu1_group["jobs"][0]["metrics_port"] == 10500

    script = Path(run_plan["groups"][0]["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=14" in script
    assert "#SBATCH --array=0-0%3" in script
    assert "#SBATCH -t 00:45:00" in script
    assert "module load cuda/12.4" in script
    assert "source /venv/bin/activate" in script
    assert 'export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"' in script
    assert 'local raw="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"' in script
    assert ': >"$VLLM_STDOUT"' in script
    assert ': >"$VLLM_STDERR"' in script
    assert ': >"$PROFILE_STDOUT"' in script
    assert ': >"$PROFILE_STDERR"' in script
    assert "emit-energy-task-shell" in script
    assert "terminate_vllm()" in script
    assert 'kill -KILL -- -"$VLLM_PID"' in script
    assert "mark-energy-task-running" in script
    assert "finalize-energy-task" in script
    assert "energy_profiler.cli" in render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "--idle-monitor-duration-s 3.0" in render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "--traffic-warmup-s 30.0" in render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "--repeats 1" in render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "export PYTORCH_ALLOC_CONF=expandable_segments:True" in render_energy_task_shell(
        run_plan["groups"][0]["plan_path"],
        0,
    )
    assert (Path(run_plan["run_root"]) / "state.json").is_file()
    assert (Path(run_plan["run_root"]) / "plan.yaml").is_file()


def test_energy_sbatch_honors_slurm_cpu_overrides(tmp_path: Path) -> None:
    plan = _make_energy_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")
    slurm = SlurmConfig(base_port=8500, cpus_per_gpu=32)

    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
        slurm=slurm,
    )

    gpu2_script = Path(run_plan["groups"][1]["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in gpu2_script
    assert "#SBATCH --cpus-per-task=64" in gpu2_script


def test_sweep_energy_plan_bundles_same_server_rates_into_one_slurm_task(tmp_path: Path) -> None:
    base_plan = _make_energy_plan(tmp_path)
    first_job = base_plan.jobs[0]
    second_rate_same_server = replace(
        first_job,
        id="job-a-r2",
        request_rate=2.0,
    )
    plan = replace(
        base_plan,
        plan=replace(base_plan.plan, mode="sweep"),
        jobs=(first_job, second_rate_same_server, base_plan.jobs[1]),
    )
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")

    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
        slurm=SlurmConfig(base_port=8500, array_concurrency_limit=3),
    )

    assert run_plan["job_count"] == 3
    assert [group["group_key"] for group in run_plan["groups"]] == ["gpu1", "gpu2"]
    assert run_plan["groups"][0]["task_count"] == 1
    assert run_plan["groups"][0]["array_spec"] == "0-0%3"

    gpu1_group = json.loads(Path(run_plan["groups"][0]["plan_path"]).read_text(encoding="utf-8"))
    assert len(gpu1_group["jobs"]) == 1
    assert len(gpu1_group["jobs"][0]["jobs"]) == 2
    assert gpu1_group["jobs"][0]["jobs"][0]["job_id"] == "job-a"
    assert gpu1_group["jobs"][0]["jobs"][1]["job_id"] == "job-a-r2"
    assert gpu1_group["jobs"][0]["jobs"][0]["base_port"] == 8500
    assert gpu1_group["jobs"][0]["jobs"][1]["base_port"] == 8500

    task_shell = render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "ENERGY_TASK_JOB_COUNT=2" in task_shell
    assert "job-a-r2" in task_shell


def test_materialize_energy_run_plan_uses_plan_slurm_config(tmp_path: Path) -> None:
    plan = replace(
        _make_energy_plan(tmp_path),
        slurm=EnergyPlanSlurm(
            partition="plan-ai",
            python_executable="/plan/slurm-python",
            array_concurrency_limit=1,
            base_port=8900,
        ),
    )
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")

    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
    )

    assert run_plan["jobs"][0]["base_port"] == 8900
    assert run_plan["groups"][0]["array_spec"] == "0-0%1"
    script = Path(run_plan["groups"][0]["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH -p plan-ai" in script
    task_shell = render_energy_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "declare -a ENERGY_TRIAL_CMD=(/plan/slurm-python" in task_shell


def test_finalize_and_collect_energy_run_write_profiler_compatible_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_energy_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")
    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
        slurm=SlurmConfig(base_port=8600),
    )
    group_plan_path = Path(run_plan["groups"][0]["plan_path"])
    group_payload = json.loads(group_plan_path.read_text(encoding="utf-8"))
    result_dir = Path(group_payload["jobs"][0]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "successful_requests": 1,
                    "started_requests": 1,
                    "benchmark_metrics": {"total_input_tokens": 10, "total_output_tokens": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "energy_summary.json").write_text(
        json.dumps({"energy_joules": 100.0, "incremental_energy_joules": 25.0, "avg_power_w": 20.0}),
        encoding="utf-8",
    )
    (result_dir / "gpu_power.json").write_text("{}", encoding="utf-8")
    (result_dir / "request_records.jsonl").write_text("", encoding="utf-8")
    (result_dir / "server_metrics.jsonl").write_text("", encoding="utf-8")
    (result_dir / "windows.csv").write_text("trial_id,window_idx\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_GPUS", "2")

    succeeded = finalize_energy_task(group_plan_path, 0, exit_code=0, trial_started=True)
    failed = finalize_energy_task(Path(run_plan["groups"][1]["plan_path"]), 0, exit_code=1, trial_started=False)
    collected = collect_energy_run(run_plan["run_root"])

    assert succeeded["status"] == "succeeded"
    assert succeeded["gpu_ids"] == [2]
    assert failed["status"] == "failed"
    assert collected["summary"]["counts"]["succeeded"] == 1
    assert collected["summary"]["counts"]["failed"] == 1
    assert collected["summary"]["status"] == "failed"
    assert collected["summary"]["aggregate"]["total_energy_joules"] == 100.0
    assert (Path(run_plan["run_root"]) / "summary.json").is_file()
    assert (Path(run_plan["run_root"]) / "summary.md").is_file()


def test_energy_collect_cli_syncs_energy_summaries_without_raw_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _make_energy_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")
    run_plan = materialize_energy_run_plan(
        plan=plan,
        plan_path=plan_path,
        run_id="energy-slurm-run",
        slurm=SlurmConfig(base_port=8600),
    )
    group_plan_path = Path(run_plan["groups"][0]["plan_path"])
    group_payload = json.loads(group_plan_path.read_text(encoding="utf-8"))
    result_dir = Path(group_payload["jobs"][0]["result_dir"])
    repeat_dir = result_dir / "repeat_001"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "successful_requests": 1,
                    "started_requests": 1,
                    "benchmark_metrics": {"total_input_tokens": 10, "total_output_tokens": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "energy_summary.json").write_text(
        json.dumps({"energy_joules": 100.0, "incremental_energy_joules": 25.0, "avg_power_w": 20.0}),
        encoding="utf-8",
    )
    (result_dir / "gpu_power.json").write_text("{}", encoding="utf-8")
    (result_dir / "request_records.jsonl").write_text("", encoding="utf-8")
    (result_dir / "server_metrics.jsonl").write_text("", encoding="utf-8")
    (result_dir / "windows.csv").write_text("trial_id,window_idx\n", encoding="utf-8")
    (repeat_dir / "summary.json").write_text("{}", encoding="utf-8")
    (repeat_dir / "energy_summary.json").write_text("{}", encoding="utf-8")
    (repeat_dir / "gpu_power.json").write_text("{}", encoding="utf-8")

    finalize_energy_task(group_plan_path, 0, exit_code=0, trial_started=True)
    synced_files = []

    def fake_run(command, **kwargs):
        del kwargs
        files_from = command[command.index("--files-from") + 1]
        synced_files.extend(Path(files_from).read_text(encoding="utf-8").splitlines())
        return subprocess.CompletedProcess(command, 0, stdout="Number of files: 7\n", stderr="")

    monkeypatch.setattr("slurm_orchestrator.cli.subprocess.run", fake_run)

    rc = slurm_main(
        [
            "energy-collect",
            "--run-root",
            run_plan["run_root"],
            "--sync-results-root",
            str(tmp_path),
            "--sync-results-to",
            str(tmp_path / "shared-results"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    result_rel = result_dir.relative_to(tmp_path).as_posix()
    repeat_rel = repeat_dir.relative_to(tmp_path).as_posix()
    run_rel = Path(run_plan["run_root"]).relative_to(tmp_path).as_posix()
    assert rc == 0
    assert payload["result_sync"]["status"] == "succeeded"
    assert payload["result_sync"]["scope"] == "run"
    assert f"{run_rel}/summary.json" in synced_files
    assert f"{run_rel}/summary.md" in synced_files
    assert f"{run_rel}/summary_compact.csv" in synced_files
    assert f"{result_rel}/summary.json" in synced_files
    assert f"{result_rel}/energy_summary.json" in synced_files
    assert f"{result_rel}/gpu_power.json" in synced_files
    assert f"{repeat_rel}/summary.json" in synced_files
    assert f"{repeat_rel}/energy_summary.json" in synced_files
    assert f"{repeat_rel}/gpu_power.json" in synced_files
    assert not any(path.endswith("request_records.jsonl") for path in synced_files)
    assert not any(path.endswith("server_metrics.jsonl") for path in synced_files)
    assert not any(path.endswith("windows.csv") for path in synced_files)


def test_energy_submit_cli_materializes_from_source_slurm_config_and_submits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "orchestrator" / "run-a"
    source_root.mkdir(parents=True)
    (tmp_path / "sharegpt.yaml").write_text("name: stub\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path / "orchestrator")},
                "slurm": {
                    "partition": "ai",
                    "python_executable": "/source/python",
                    "array_concurrency_limit": 2,
                    "base_port": 8700,
                },
                "experiments": [
                    {"id": "exp-a", "model": "model-a", "workload": str(tmp_path / "sharegpt.yaml")}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (source_root / "state.json").write_text(
        json.dumps({"manifest_path": str(manifest_path)}),
        encoding="utf-8",
    )
    plan = _make_energy_plan(tmp_path)
    plan_path = write_energy_plan(plan, tmp_path / "plans" / "energy-plan-a.yaml")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"]
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    monkeypatch.setattr("slurm_orchestrator.energy.subprocess.run", fake_run)

    rc = slurm_main(["energy-submit", "--plan", str(plan_path), "--run-id", "cli-run"])
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert len(calls) == 2
    assert calls[0][:2] == ["sbatch", "--parsable"]
    assert output["run_id"] == "cli-run"
    assert all(group["return_code"] == 0 for group in output["groups"])
    run_root = tmp_path / "energy" / "energy-plan-a" / "cli-run"
    script = (run_root / "scripts" / "gpu1.sbatch.sh").read_text(encoding="utf-8")
    assert "#SBATCH -p ai" in script
    assert "#SBATCH --array=0-0%2" in script
    task_shell = render_energy_task_shell(run_root / "groups" / "gpu1.json", 0)
    assert task_shell.startswith("ENERGY_JOB_ID=job-a")
    assert "declare -a ENERGY_TRIAL_CMD=(/source/python" in task_shell
