from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from energy_profiler.models import EnergyPlanRounding, EnergyPlanSelection, EnergyPlanSelectionSweep
from energy_profiler.planning import (
    PlanningError,
    generate_plan_from_orchestrator,
    generate_plan_from_orchestrator_runs,
)
from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest


def _write_manifest(
    tmp_path: Path,
    workload: Path,
    *,
    model: str = "Qwen/Qwen3-8B",
    experiment_id: str = "exp-a",
    launch: dict[str, object] | None = None,
) -> Path:
    launch_payload = {
        "executable": "vllm",
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_count": 1,
        "max_model_len": 32768,
        "max_num_seqs": 64,
    }
    if launch:
        launch_payload.update(launch)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "python_executable": "python",
                    "allowed_gpu_ids": [2, 3],
                    "max_active_gpus": 1,
                    "base_port_start": 8100,
                    "base_port_end": 8199,
                    "metrics_port_offset": 2000,
                },
                "slurm": {
                    "partition": "ai",
                    "array_concurrency_limit": 4,
                    "base_port": 8700,
                },
                "launch": launch_payload,
                "search": {
                    "search_mode": "open-loop",
                    "ttft_slo_ms": 2000,
                    "tpot_slo_ms": 80,
                },
                "experiments": [
                    {
                        "id": experiment_id,
                        "model": model,
                        "workload": str(workload),
                        "endpoint": "/v1/chat/completions",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_orchestrator_run(
    tmp_path: Path,
    *,
    mst_rate: float,
    run_id: str = "run-a",
    model: str = "Qwen/Qwen3-8B",
    manifest_name: str = "manifest.yaml",
    workload_name: str = "sharegpt.yaml",
    workload_display_name: str = "sharegpt",
    experiment_id: str = "exp-a",
    launch: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    workload = tmp_path / workload_name
    workload.parent.mkdir(parents=True, exist_ok=True)
    workload.write_text(f"name: {workload_display_name}\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        workload,
        model=model,
        experiment_id=experiment_id,
        launch=launch,
    )
    if manifest_path.name != manifest_name:
        renamed = tmp_path / manifest_name
        manifest_path.rename(renamed)
        manifest_path = renamed
    manifest = load_manifest(manifest_path)
    expanded_job = expand_manifest(manifest)[0]

    run_root = tmp_path / "results" / "orchestrator" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "state.json").write_text(
        json.dumps({"manifest_path": str(manifest_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_dir = Path(expanded_job.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    search_trace = {
        "config": {
            "search_id": "search-a",
            "search_mode": "open-loop",
        },
        "result": {
            "confirmation_trial_id": "trial-confirm",
            "max_no_drift_request_rate": mst_rate,
            "max_slo_satisfying_request_rate": mst_rate,
        },
    }
    (result_dir / "search_trace.json").write_text(
        json.dumps(search_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_dir / "final_report.json").write_text("{}\n", encoding="utf-8")
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "experiment_id": expanded_job.experiment_id,
                        "status": "succeeded",
                        "result_dir": str(result_dir),
                        "max_no_drift_request_rate": mst_rate,
                        "max_slo_satisfying_request_rate": mst_rate,
                        "artifacts": {
                            "search_trace": str(result_dir / "search_trace.json"),
                            "final_report_json": str(result_dir / "final_report.json"),
                        },
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_root, workload


def test_generate_plan_from_orchestrator_rounds_mst_and_preserves_launch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root, workload = _write_orchestrator_run(tmp_path, mst_rate=4.37)

    plan = generate_plan_from_orchestrator(
        orchestrator_run_root=run_root,
        output_plan=tmp_path / "experiments" / "energy" / "sharegpt_l40_energy_000.yaml",
    )

    assert plan.plan.plan_id == "sharegpt_l40_energy_000"
    assert plan.execution.allowed_gpu_ids == (2, 3)
    assert plan.execution.max_active_gpus == 1
    assert plan.execution.base_port_start == 8100
    assert plan.execution.base_port_end == 8199
    assert plan.execution.metrics_port_offset == 2000
    assert plan.slurm.partition == "ai"
    assert plan.slurm.array_concurrency_limit == 4
    assert plan.slurm.base_port == 8700
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.request_rate == pytest.approx(4.37)
    assert job.mst_rate == pytest.approx(4.37)
    assert job.workload == workload.resolve()
    assert job.launch.dtype == "float16"
    assert job.launch.max_model_len == 32768
    assert job.metadata["rounding_policy"] == "floor_decimal"
    assert job.metadata["rounding_step"] == pytest.approx(0.01)
    assert job.metadata["rounding_decimal_places"] == 2
    assert job.metadata["search_id"] == "search-a"
    written = plan.to_dict()
    assert "execution" not in written
    assert "local_execution" in written
    assert written["slurm"]["array_concurrency_limit"] == 4


def test_generate_plan_can_use_legacy_preferred_step_rounding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root, _ = _write_orchestrator_run(tmp_path, mst_rate=4.37)

    plan = generate_plan_from_orchestrator(
        orchestrator_run_root=run_root,
        output_plan=tmp_path / "experiments" / "energy" / "sharegpt_l40_energy_000.yaml",
        rounding=EnergyPlanRounding(mst_mode="floor_preferred"),
    )

    assert plan.jobs[0].request_rate == pytest.approx(4.25)
    assert plan.jobs[0].metadata["rounding_policy"] == "floor_preferred"
    assert plan.jobs[0].metadata["rounding_step"] == pytest.approx(0.25)


def test_generate_sweep_plan_uses_coarser_step_to_respect_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root, _ = _write_orchestrator_run(tmp_path, mst_rate=4.37)

    plan = generate_plan_from_orchestrator(
        orchestrator_run_root=run_root,
        output_plan=tmp_path / "experiments" / "energy" / "sharegpt_sweep.yaml",
        mode="sweep",
        selection=EnergyPlanSelection(
            sweep=EnergyPlanSelectionSweep(enabled=True, max_steps=10),
        ),
    )

    assert [job.request_rate for job in plan.jobs] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert all(job.metadata["rounding_step"] == pytest.approx(0.5) for job in plan.jobs)
    assert [job.metadata["sweep_rate_index"] for job in plan.jobs] == list(range(1, 9))


def test_generate_explicit_plan_validates_requested_filters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root, _ = _write_orchestrator_run(tmp_path, mst_rate=1.84)

    with pytest.raises(PlanningError, match="selection.models value not found"):
        generate_plan_from_orchestrator(
            orchestrator_run_root=run_root,
            output_plan=tmp_path / "experiments" / "energy" / "explicit.yaml",
            mode="explicit",
            selection=EnergyPlanSelection(
                models=("missing-model",),
                explicit_request_rates=(0.5, 1.0),
            ),
        )


def test_generate_plan_from_multiple_orchestrator_roots_uses_later_rerun_for_duplicate_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    main_run, _ = _write_orchestrator_run(
        tmp_path,
        mst_rate=4.37,
        run_id="main-loop",
        model="Qwen/Qwen3-4B-Instruct-2507",
        manifest_name="main.yaml",
    )
    rerun, _ = _write_orchestrator_run(
        tmp_path,
        mst_rate=5.12,
        run_id="targeted-rerun",
        model="Qwen/Qwen3-4B-Instruct-2507",
        manifest_name="rerun.yaml",
        workload_name="sharegpt_mst_anomaly_rerun.yaml",
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(main_run, rerun),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert len(plan.jobs) == 1
    assert plan.jobs[0].mst_rate == pytest.approx(5.12)
    assert plan.jobs[0].metadata["source_orchestrator_run_id"] == "targeted-rerun"
    assert plan.jobs[0].metadata["source_orchestrator_run_root"] == str(rerun)
    assert plan.jobs[0].workload.name == "sharegpt_mst_anomaly_rerun.yaml"


def test_generate_plan_from_multiple_orchestrator_roots_includes_followup_only_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    main_run, _ = _write_orchestrator_run(
        tmp_path,
        mst_rate=4.37,
        run_id="main-loop",
        model="Qwen/Qwen3-4B-Instruct-2507",
        manifest_name="main.yaml",
    )
    rerun, _ = _write_orchestrator_run(
        tmp_path,
        mst_rate=3.21,
        run_id="thinking-rerun",
        model="Qwen/Qwen3-4B-Thinking-2507",
        manifest_name="thinking.yaml",
        experiment_id="exp-thinking",
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(main_run, rerun),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert [job.model for job in plan.jobs] == [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
    ]
    assert {job.metadata["source_orchestrator_run_id"] for job in plan.jobs} == {
        "main-loop",
        "thinking-rerun",
    }


def test_generate_plan_can_exclude_small_models_before_rate_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    small_run, _ = _write_orchestrator_run(
        tmp_path / "small",
        mst_rate=40.0,
        run_id="small-loop",
        model="Qwen/Qwen3-0.6B",
        experiment_id="qwen06",
        manifest_name="small.yaml",
    )
    large_run, _ = _write_orchestrator_run(
        tmp_path / "large",
        mst_rate=4.37,
        run_id="large-loop",
        model="Qwen/Qwen3-8B",
        experiment_id="qwen8",
        manifest_name="large.yaml",
    )
    small_summary = small_run / "summary.json"
    small_payload = json.loads(small_summary.read_text(encoding="utf-8"))
    small_payload["jobs"][0]["max_slo_satisfying_request_rate"] = None
    small_payload["jobs"][0]["max_no_drift_request_rate"] = None
    search_trace_path = Path(small_payload["jobs"][0]["artifacts"]["search_trace"])
    small_trace = json.loads(search_trace_path.read_text(encoding="utf-8"))
    small_trace["result"]["max_slo_satisfying_request_rate"] = None
    small_trace["result"]["max_no_drift_request_rate"] = None
    small_summary.write_text(json.dumps(small_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    search_trace_path.write_text(json.dumps(small_trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(small_run, large_run),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
        selection=EnergyPlanSelection(min_model_size_b=3.0),
    )

    assert [job.model for job in plan.jobs] == ["Qwen/Qwen3-8B"]


def test_generate_plan_from_multiple_roots_preserves_tensor_parallel_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tp1_run, _ = _write_orchestrator_run(
        tmp_path / "tp1",
        mst_rate=4.0,
        run_id="tp1",
        model="Qwen/Qwen3-32B",
        experiment_id="qwen32-tp1",
        manifest_name="tp1.yaml",
    )
    tp2_run, _ = _write_orchestrator_run(
        tmp_path / "tp2",
        mst_rate=7.0,
        run_id="tp2",
        model="Qwen/Qwen3-32B",
        experiment_id="qwen32-tp2",
        manifest_name="tp2.yaml",
        launch={"gpu_count": 2, "tensor_parallel_size": 2},
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(tp1_run, tp2_run),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert sorted((job.launch.tensor_parallel_size, job.mst_rate) for job in plan.jobs) == [
        (1, 4.0),
        (2, 7.0),
    ]


def test_generate_plan_from_multiple_roots_matches_rerun_workload_aliases_and_ignores_mixed_workloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    main_run, _ = _write_orchestrator_run(
        tmp_path / "main",
        mst_rate=4.0,
        run_id="main",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        experiment_id="qwen30-main",
        manifest_name="main.yaml",
        workload_name="wildchat_hf_8192.yaml",
        workload_display_name="wildchat-hf-8192",
    )
    rerun, _ = _write_orchestrator_run(
        tmp_path / "rerun",
        mst_rate=8.0,
        run_id="rerun",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        experiment_id="qwen30-rerun",
        manifest_name="rerun.yaml",
        workload_name="wildchat_hf_8k_mst_anomaly_rerun.yaml",
        workload_display_name="wildchat-hf-8k_mst_anomaly_rerun",
    )
    sharegpt_rerun, _ = _write_orchestrator_run(
        tmp_path / "sharegpt",
        mst_rate=11.0,
        run_id="sharegpt-rerun",
        model="Qwen/Qwen3-4B-Instruct-2507",
        experiment_id="sharegpt-rerun",
        manifest_name="sharegpt.yaml",
        workload_name="sharegpt.yaml",
        workload_display_name="live_sharegpt_workload_context",
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(main_run, rerun, sharegpt_rerun),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert len(plan.jobs) == 1
    assert plan.jobs[0].mst_rate == pytest.approx(8.0)
    assert plan.jobs[0].metadata["source_orchestrator_run_id"] == "rerun"
    assert plan.jobs[0].workload.name == "wildchat_hf_8k_mst_anomaly_rerun.yaml"


def test_generate_plan_from_multiple_roots_matches_multi_shard_rerun_to_original_shard_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    main_run, _ = _write_orchestrator_run(
        tmp_path / "main",
        mst_rate=60.0,
        run_id="main",
        model="Qwen/Qwen3-0.6B",
        experiment_id="code-qwen06-crosscodeeval",
        manifest_name="main.yaml",
        workload_name="crosscodeeval-rg1-unixcoder-cache-realistic-shard-000_mst_anomaly_rerun.yaml",
        workload_display_name="crosscodeeval_rg1_unixcoder_cache_realistic-shard_000_mst_anomaly_rerun",
    )
    rerun, _ = _write_orchestrator_run(
        tmp_path / "rerun",
        mst_rate=39.375,
        run_id="rerun",
        model="Qwen/Qwen3-0.6B",
        experiment_id="code-targeted-qwen06-crosscodeeval",
        manifest_name="rerun.yaml",
        workload_name="a100_crosscodeeval_rg1_unixcoder_cache_realistic_shards_000_001.yaml",
        workload_display_name="crosscodeeval_rg1_unixcoder_cache_realistic-shards_000_001",
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(main_run, rerun),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert len(plan.jobs) == 1
    assert plan.jobs[0].mst_rate == pytest.approx(39.375)
    assert plan.jobs[0].metadata["source_orchestrator_run_id"] == "rerun"
    assert plan.jobs[0].source_experiment_id == "code-targeted-qwen06-crosscodeeval"
    assert plan.jobs[0].workload.name == "a100_crosscodeeval_rg1_unixcoder_cache_realistic_shards_000_001.yaml"
