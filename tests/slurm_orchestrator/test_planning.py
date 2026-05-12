from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from local_orchestrator.manifest import load_manifest
from slurm_orchestrator.planning import (
    load_run_plan,
    materialize_run_plan,
    refresh_run_plan_for_resume,
    render_task_shell,
    submit_run_plan_tasks,
)
from slurm_orchestrator.state import collect_run, finalize_task


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_workload(tmp_path: Path, name: str) -> Path:
    workload_path = tmp_path / name
    workload_path.write_text("name: stub\n", encoding="utf-8")
    return workload_path


def test_materialize_run_plan_groups_jobs_and_writes_absolute_payloads(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "slurm": {"base_port": 8400},
            "experiments": [
                {
                    "id": "exp-a",
                    "model": "model-a",
                    "workload": str(workload),
                },
                {
                    "id": "exp-b",
                    "model": "model-b",
                    "workload": str(workload),
                    "launch": {"gpu_count": 2, "tensor_parallel_size": 2},
                },
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-a")

    assert plan["job_count"] == 2
    assert [group["group_key"] for group in plan["groups"]] == ["gpu1", "gpu2"]
    assert plan["jobs"][0]["base_port"] == 8400
    assert plan["jobs"][1]["base_port"] == 8401

    gpu2_plan = json.loads(Path(plan["groups"][1]["plan_path"]).read_text(encoding="utf-8"))
    assert gpu2_plan["gpu_count"] == 2
    assert gpu2_plan["jobs"][0]["job"]["launch"]["gpu_count"] == 2
    assert Path(gpu2_plan["jobs"][0]["job"]["result_dir"]).is_absolute()

    loaded = load_run_plan(tmp_path / "runs" / "run-a")
    assert loaded["run_id"] == "run-a"
    assert loaded["jobs"][0]["experiment_id"] == "exp-a"


def test_rendered_sbatch_script_includes_array_limit_wait_search_report_and_cleanup(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs"), "python_executable": "/venv/bin/python"},
            "slurm": {
                "partition": "ai",
                "account": "research",
                "qos": "preemptible",
                "time": "04:00:00",
                "array_concurrency_limit": 3,
                "modules": ["cuda/12.4"],
                "setup_commands": ["source /venv/bin/activate"],
            },
            "experiments": [
                {
                    "id": "exp-a",
                    "model": "model-a",
                    "workload": str(workload),
                    "launch": {"env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}},
                },
                {"id": "exp-b", "model": "model-b", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-b")
    script = Path(plan["groups"][0]["script_path"]).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=14" in script
    assert "#SBATCH --array=0-1%3" in script
    assert "#SBATCH --output=" in script
    assert "set -euo pipefail" in script
    assert 'export PYTHONPATH="$PROFILER_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert "module load cuda/12.4" in script
    assert "source /venv/bin/activate" in script
    assert 'export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"' in script
    assert ': >"$VLLM_STDOUT"' in script
    assert ': >"$VLLM_STDERR"' in script
    assert ': >"$MST_STDOUT"' in script
    assert ': >"$MST_STDERR"' in script
    assert 'setsid "${VLLM_CMD[@]}"' in script
    assert "log_gpu_diagnostics()" in script
    assert 'echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"' in script
    assert 'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"' in script
    assert "nvidia-smi pmon -c 1" in script
    assert 'log_gpu_diagnostics "before_vllm_start"' in script
    assert 'log_gpu_diagnostics "after_vllm_start"' in script
    assert "wait-ready" in script
    assert '"${SEARCH_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"' in script
    assert '"${REPORT_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"' in script
    assert "trap cleanup EXIT" in script
    task_shell = render_task_shell(plan["groups"][0]["plan_path"], 0)
    assert "export PYTORCH_ALLOC_CONF=expandable_segments:True" in task_shell


def test_rendered_sbatch_script_scales_cpus_with_gpu_count(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "experiments": [
                {
                    "id": "exp-b",
                    "model": "model-b",
                    "workload": str(workload),
                    "launch": {"gpu_count": 4, "tensor_parallel_size": 4},
                },
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-d")
    script = Path(plan["groups"][0]["script_path"]).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=56" in script


def test_rendered_sbatch_script_accepts_cpus_per_gpu_override(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "slurm": {"cpus_per_gpu": 32},
            "experiments": [
                {
                    "id": "exp-b",
                    "model": "model-b",
                    "workload": str(workload),
                    "launch": {"gpu_count": 4, "tensor_parallel_size": 4},
                },
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-cpu-per-gpu")
    script = Path(plan["groups"][0]["script_path"]).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=128" in script


def test_rendered_sbatch_script_accepts_fixed_cpus_per_task_override(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "slurm": {"cpus_per_gpu": 32, "cpus_per_task": 16},
            "experiments": [
                {
                    "id": "exp-b",
                    "model": "model-b",
                    "workload": str(workload),
                    "launch": {"gpu_count": 4, "tensor_parallel_size": 4},
                },
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-cpu-per-task")
    script = Path(plan["groups"][0]["script_path"]).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=16" in script


def test_collect_run_aggregates_succeeded_and_failed_jobs(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "slurm": {"array_concurrency_limit": 2},
            "search": {"search_mode": "open-loop"},
            "experiments": [
                {"id": "exp-a", "model": "model-a", "workload": str(workload)},
                {"id": "exp-b", "model": "model-b", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-c")
    group_plan_path = Path(plan["groups"][0]["plan_path"])
    group_payload = json.loads(group_plan_path.read_text(encoding="utf-8"))

    success_result_dir = Path(group_payload["jobs"][0]["job"]["result_dir"])
    success_result_dir.mkdir(parents=True, exist_ok=True)
    (success_result_dir / "search_trace.json").write_text(
        json.dumps({"result": {"termination_reason": "stable", "max_no_drift_request_rate": 3.0}}),
        encoding="utf-8",
    )
    (success_result_dir / "final_report.json").write_text("{}\n", encoding="utf-8")

    finalize_task(group_plan_path, 0, exit_code=0, search_started=True)
    finalize_task(group_plan_path, 1, exit_code=2, search_started=False)

    collected = collect_run(tmp_path / "runs" / "run-c")

    assert collected["summary"]["counts"]["succeeded"] == 1
    assert collected["summary"]["counts"]["failed"] == 1
    assert collected["summary"]["status"] == "failed"
    assert (tmp_path / "runs" / "run-c" / "state.json").is_file()
    assert (tmp_path / "runs" / "run-c" / "summary.json").is_file()
    assert (tmp_path / "runs" / "run-c" / "summary.md").is_file()


def test_collect_run_marks_stale_nonterminal_slurm_tasks_failed(tmp_path: Path, monkeypatch) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "experiments": [
                {"id": "exp-running", "model": "model-a", "workload": str(workload)},
                {"id": "exp-planned", "model": "model-b", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-stale")
    run_root = tmp_path / "runs" / "run-stale"
    (run_root / "resume-submission.json").write_text(
        json.dumps(
            {
                "groups": [{"group_key": "gpu1", "job_id": "12345"}],
                "submitted_at": "2026-05-11T07:39:08+00:00",
            }
        ),
        encoding="utf-8",
    )

    running_state_path = run_root / "jobs" / "exp-running.json"
    running_state = json.loads(running_state_path.read_text(encoding="utf-8"))
    running_state["status"] = "running"
    running_state["slurm"]["array_job_id"] = "12345"
    running_state["slurm"]["array_task_id"] = "0"
    running_state_path.write_text(json.dumps(running_state), encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[:4] == ["squeue", "-h", "-j", "12345"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("slurm_orchestrator.state.subprocess.run", fake_run)

    collected = collect_run(plan["run_root"])

    assert collected["summary"]["counts"]["failed"] == 2
    assert collected["summary"]["status"] == "failed"
    failed_running = json.loads(running_state_path.read_text(encoding="utf-8"))
    failed_planned = json.loads((run_root / "jobs" / "exp-planned.json").read_text(encoding="utf-8"))
    assert failed_running["status"] == "failed"
    assert "12345_0 is no longer active" in failed_running["last_error"]
    assert failed_planned["status"] == "failed"
    assert failed_planned["slurm"]["array_job_id"] == "12345"
    assert failed_planned["slurm"]["array_task_id"] == "1"


def test_collect_run_preserves_nonterminal_jobs_while_slurm_array_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "experiments": [
                {"id": "exp-running", "model": "model-a", "workload": str(workload)},
                {"id": "exp-planned", "model": "model-b", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-active")
    run_root = tmp_path / "runs" / "run-active"
    (run_root / "resume-submission.json").write_text(
        json.dumps(
            {
                "groups": [{"group_key": "gpu1", "job_id": "12345"}],
                "submitted_at": "2026-05-11T07:39:08+00:00",
            }
        ),
        encoding="utf-8",
    )

    running_state_path = run_root / "jobs" / "exp-running.json"
    running_state = json.loads(running_state_path.read_text(encoding="utf-8"))
    running_state["status"] = "running"
    running_state["slurm"]["array_job_id"] = "12345"
    running_state["slurm"]["array_task_id"] = "0"
    running_state_path.write_text(json.dumps(running_state), encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[:4] == ["squeue", "-h", "-j", "12345"]
        return SimpleNamespace(returncode=0, stdout="12345|12345\n12346|12345\n", stderr="")

    monkeypatch.setattr("slurm_orchestrator.state.subprocess.run", fake_run)

    collected = collect_run(plan["run_root"])

    assert collected["summary"]["counts"]["running"] == 1
    assert collected["summary"]["counts"]["planned"] == 1
    assert collected["summary"]["counts"]["failed"] == 0
    preserved_running = json.loads(running_state_path.read_text(encoding="utf-8"))
    preserved_planned = json.loads((run_root / "jobs" / "exp-planned.json").read_text(encoding="utf-8"))
    assert preserved_running["status"] == "running"
    assert preserved_planned["status"] == "planned"


def test_collect_run_repairs_false_stale_failure_while_slurm_array_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "experiments": [
                {"id": "exp-planned", "model": "model-a", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-repair-active")
    run_root = tmp_path / "runs" / "run-repair-active"
    (run_root / "resume-submission.json").write_text(
        json.dumps(
            {
                "groups": [{"group_key": "gpu1", "job_id": "12345"}],
                "submitted_at": "2026-05-11T07:39:08+00:00",
            }
        ),
        encoding="utf-8",
    )
    state_path = run_root / "jobs" / "exp-planned.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["last_error"] = (
        "Slurm array task 12345_0 is no longer active, but orchestrator state remained planned; "
        "the task likely ended before finalization, for example due to scancel, time limit, or node failure."
    )
    state["slurm"]["array_job_id"] = "12345"
    state["slurm"]["array_task_id"] = "0"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[:4] == ["squeue", "-h", "-j", "12345"]
        return SimpleNamespace(returncode=0, stdout="12345|12345\n", stderr="")

    monkeypatch.setattr("slurm_orchestrator.state.subprocess.run", fake_run)

    collected = collect_run(plan["run_root"])

    assert collected["summary"]["counts"]["planned"] == 1
    assert collected["summary"]["counts"]["failed"] == 0
    repaired = json.loads(state_path.read_text(encoding="utf-8"))
    assert repaired["status"] == "planned"
    assert repaired["last_error"] is None


def test_resume_refreshes_failed_job_plan_and_submits_only_failed_array_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"output_root": str(tmp_path / "runs")},
            "slurm": {"array_concurrency_limit": 4},
            "launch": {"max_model_len": 32768},
            "experiments": [
                {"id": "exp-ok", "model": "model-a", "workload": str(workload)},
                {"id": "exp-failed", "model": "model-b", "workload": str(workload)},
            ],
        },
    )
    plan = materialize_run_plan(load_manifest(manifest_path), "run-resume")
    group_plan_path = Path(plan["groups"][0]["plan_path"])
    group_payload = json.loads(group_plan_path.read_text(encoding="utf-8"))
    success_result_dir = Path(group_payload["jobs"][0]["job"]["result_dir"])
    success_result_dir.mkdir(parents=True, exist_ok=True)
    (success_result_dir / "search_trace.json").write_text("{}", encoding="utf-8")
    (success_result_dir / "final_report.json").write_text("{}", encoding="utf-8")
    finalize_task(group_plan_path, 0, exit_code=0, search_started=True)
    finalize_task(group_plan_path, 1, exit_code=2, search_started=False)

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path / "runs")},
                "slurm": {"array_concurrency_limit": 4},
                "launch": {"max_model_len": 4096},
                "experiments": [
                    {"id": "exp-ok", "model": "model-a", "workload": str(workload)},
                    {"id": "exp-failed", "model": "model-b", "workload": str(workload)},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    refreshed_plan, selected = refresh_run_plan_for_resume(
        load_manifest(manifest_path),
        tmp_path / "runs" / "run-resume",
    )

    failed_state = json.loads(
        (tmp_path / "runs" / "run-resume" / "jobs" / "exp-failed.json").read_text(encoding="utf-8")
    )
    ok_state = json.loads(
        (tmp_path / "runs" / "run-resume" / "jobs" / "exp-ok.json").read_text(encoding="utf-8")
    )
    assert failed_state["max_model_len"] == 4096
    assert failed_state["status"] == "failed"
    assert ok_state["max_model_len"] == 32768
    assert selected == {"gpu1": {1}}

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "12345\n"
            stderr = ""

        return Result()

    monkeypatch.setattr("slurm_orchestrator.planning.subprocess.run", fake_run)
    submission = submit_run_plan_tasks(
        refreshed_plan,
        selected_task_indices_by_group=selected,
        submission_filename="resume-submission.json",
    )

    assert calls == [["sbatch", "--parsable", "--array=1%4", str(Path(plan["groups"][0]["script_path"]))]]
    assert submission["groups"][0]["array_spec"] == "1%4"
    assert (tmp_path / "runs" / "run-resume" / "resume-submission.json").is_file()
