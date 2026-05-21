from __future__ import annotations

import json
from pathlib import Path

import yaml

from slurm_orchestrator.cli import main as slurm_main
from slurm_orchestrator.quality import (
    _quality_shard_output_dir,
    collect_quality_run,
    finalize_quality_task,
    materialize_quality_run_plan,
    render_quality_task_shell,
)
from output_quality_profiler.manifest import load_quality_manifest


def _write_quality_workload(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "quality" / "shards"
    workload_dir = tmp_path / "quality" / "workload_yamls"
    shard_dir.mkdir(parents=True)
    workload_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard_000.runner.jsonl"
    shard_path.write_text(
        json.dumps(
            {
                "prompt": "Say hello",
                "prompt_len": 2,
                "expected_output_len": 32,
                "metadata": {
                    "request_id": "smoke-000",
                    "source": "smoke",
                    "prompt_length_bucket": "short",
                    "shard_id": "shard_000",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workload_path = workload_dir / "shard_000.yaml"
    workload_path.write_text(
        yaml.safe_dump(
            {
                "name": "quality-smoke-shard-000",
                "dataset": {"type": "jsonl", "path": "../shards/shard_000.runner.jsonl"},
                "sampling": {
                    "seed": 42,
                    "num_requests": 1,
                    "entry_selection": "sequential",
                    "prompt_len": {"mode": "from_dataset"},
                    "output_len": {"mode": "natural_until_eos", "max_tokens": 32},
                },
                "request": {"stream": True, "temperature": 0.6, "ignore_eos": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workload_path


def _write_quality_manifest(
    tmp_path: Path,
    workload: Path,
    generation_overrides: dict[str, object] | None = None,
) -> Path:
    generation = {
        "concurrency_source": "explicit",
        "max_concurrency": 1,
        "include_prompt_text": True,
        "preserve_request_order": True,
        "decoding": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "n": 1,
            "max_tokens": 32768,
            "max_tokens_policy": "model_context_minus_prompt_buffer",
            "prompt_token_buffer": 128,
        },
    }
    generation.update(generation_overrides or {})
    manifest_path = tmp_path / "quality_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "run_id": "quality-slurm-run",
                    "output_root": str(tmp_path / "quality-runs"),
                    "default_endpoint": "/v1/chat/completions",
                    "python_executable": "/venv/bin/python",
                },
                "slurm": {
                    "partition": "ai",
                    "account": "research",
                    "time": "00:05:00",
                    "array_concurrency_limit": 2,
                    "base_port": 9700,
                },
                "launch": {
                    "template": [
                        "python",
                        "-m",
                        "output_quality_profiler.mock_openai_server",
                        "--host",
                        "{host}",
                        "--port",
                        "{port}",
                    ],
                    "readiness_timeout_s": 30,
                    "readiness_interval_s": 1,
                },
                "generation": generation,
                "experiments": [
                    {"id": "mock-quality", "model": "mock-quality-model", "workload": str(workload)}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_materialize_quality_run_plan_renders_script_and_task_shell(tmp_path: Path) -> None:
    workload = _write_quality_workload(tmp_path)
    manifest = load_quality_manifest(_write_quality_manifest(tmp_path, workload))

    run_plan = materialize_quality_run_plan(manifest=manifest, run_id="quality-slurm-run")

    assert run_plan["job_count"] == 1
    assert run_plan["groups"][0]["group_key"] == "gpu1"
    assert run_plan["groups"][0]["array_spec"] == "0-0%2"
    script = Path(run_plan["groups"][0]["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH -t 00:05:00" in script
    assert "emit-quality-task-shell" in script
    assert "mark-quality-task-running" in script
    assert "finalize-quality-task" in script
    assert '"${generation_cmd[@]}" >>"$QUALITY_STDOUT" 2>>"$QUALITY_STDERR"' in script
    assert '"${QUALITY_SUMMARIZE_CMD[@]}" >>"$QUALITY_STDOUT" 2>>"$QUALITY_STDERR"' in script

    task_shell = render_quality_task_shell(run_plan["groups"][0]["plan_path"], 0)
    assert "output_quality_profiler.mock_openai_server" in task_shell
    assert "output_quality_profiler.cli run-live-generation" in task_shell
    assert "--max-concurrency 1" in task_shell
    assert "--load-mode closed_loop" in task_shell
    assert "--temperature 0.6" in task_shell


def test_materialize_quality_run_plan_renders_open_loop_settings(tmp_path: Path) -> None:
    workload = _write_quality_workload(tmp_path)
    manifest = load_quality_manifest(
        _write_quality_manifest(
            tmp_path,
            workload,
            generation_overrides={
                "max_concurrency": 512,
                "load_mode": "open_loop",
                "request_rate": 21.0,
            },
        )
    )

    run_plan = materialize_quality_run_plan(manifest=manifest, run_id="quality-open-loop")
    task_shell = render_quality_task_shell(run_plan["groups"][0]["plan_path"], 0)

    assert "--max-concurrency 512" in task_shell
    assert "--load-mode open_loop" in task_shell
    assert "--request-rate 21.0" in task_shell


def test_quality_shard_output_dirs_are_unique_for_generic_shards(tmp_path: Path) -> None:
    workload = _write_quality_workload(tmp_path)
    sibling = workload.with_name("shard_001.yaml")
    sibling.write_text(workload.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "responses"

    assert _quality_shard_output_dir(output_dir, workload) != _quality_shard_output_dir(output_dir, sibling)


def test_finalize_and_collect_quality_run(tmp_path: Path) -> None:
    workload = _write_quality_workload(tmp_path)
    manifest = load_quality_manifest(_write_quality_manifest(tmp_path, workload))
    run_plan = materialize_quality_run_plan(manifest=manifest, run_id="quality-slurm-run")
    group_plan_path = Path(run_plan["groups"][0]["plan_path"])
    group = json.loads(group_plan_path.read_text(encoding="utf-8"))
    result_dir = Path(group["jobs"][0]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "responses.jsonl").write_text('{"success": true}\n', encoding="utf-8")
    (result_dir / "failed_requests.jsonl").write_text("", encoding="utf-8")
    (result_dir / "summary.json").write_text(
        json.dumps({"total_requests": 1, "successful_requests": 1, "failed_requests": 0}),
        encoding="utf-8",
    )

    state = finalize_quality_task(group_plan_path, 0, exit_code=0, generation_started=True)
    collected = collect_quality_run(run_plan["run_root"])

    assert state["status"] == "succeeded"
    assert collected["summary"]["counts"]["succeeded"] == 1
    assert collected["summary"]["aggregate"]["total_requests"] == 1
    assert (Path(run_plan["run_root"]) / "summary.json").is_file()
    assert (Path(run_plan["run_root"]) / "summary.md").is_file()


def test_quality_plan_cli(tmp_path: Path, capsys) -> None:
    workload = _write_quality_workload(tmp_path)
    manifest_path = _write_quality_manifest(tmp_path, workload)

    assert slurm_main(["quality-plan", "--manifest", str(manifest_path), "--run-id", "quality-cli"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["run_id"] == "quality-cli"
    assert output["job_count"] == 1
