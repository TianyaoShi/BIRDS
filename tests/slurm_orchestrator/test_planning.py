from __future__ import annotations

import json
from pathlib import Path

import yaml

from local_orchestrator.manifest import load_manifest
from slurm_orchestrator.planning import load_run_plan, materialize_run_plan
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
                "cpus_per_task": 32,
                "array_concurrency_limit": 3,
                "modules": ["cuda/12.4"],
                "setup_commands": ["source /venv/bin/activate"],
            },
            "experiments": [
                {"id": "exp-a", "model": "model-a", "workload": str(workload)},
                {"id": "exp-b", "model": "model-b", "workload": str(workload)},
            ],
        },
    )

    plan = materialize_run_plan(load_manifest(manifest_path), "run-b")
    script = Path(plan["groups"][0]["script_path"]).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --array=0-1%3" in script
    assert "#SBATCH --output=" in script
    assert "set -euo pipefail" in script
    assert 'export PYTHONPATH="$PROFILER_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert "module load cuda/12.4" in script
    assert "source /venv/bin/activate" in script
    assert 'setsid "${VLLM_CMD[@]}"' in script
    assert "wait-ready" in script
    assert '"${SEARCH_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"' in script
    assert '"${REPORT_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"' in script
    assert "trap cleanup EXIT" in script


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
    assert (tmp_path / "runs" / "run-c" / "summary.json").is_file()
    assert (tmp_path / "runs" / "run-c" / "summary.md").is_file()
