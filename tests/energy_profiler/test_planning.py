from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from energy_profiler.models import EnergyPlanSelection, EnergyPlanSelectionSweep
from energy_profiler.planning import (
    PlanningError,
    generate_plan_from_orchestrator,
    generate_plan_from_orchestrator_runs,
)
from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest


def _write_manifest(tmp_path: Path, workload: Path, *, model: str = "Qwen/Qwen3-8B") -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "python_executable": "python",
                },
                "launch": {
                    "executable": "vllm",
                    "dtype": "float16",
                    "tensor_parallel_size": 1,
                    "gpu_count": 1,
                    "max_model_len": 32768,
                    "max_num_seqs": 64,
                },
                "search": {
                    "search_mode": "open-loop",
                    "ttft_slo_ms": 2000,
                    "tpot_slo_ms": 80,
                },
                "experiments": [
                    {
                        "id": "exp-a",
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
) -> tuple[Path, Path]:
    workload = tmp_path / "sharegpt.yaml"
    workload.write_text("name: sharegpt\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, workload, model=model)
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
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.request_rate == pytest.approx(4.25)
    assert job.mst_rate == pytest.approx(4.37)
    assert job.workload == workload.resolve()
    assert job.launch.dtype == "float16"
    assert job.launch.max_model_len == 32768
    assert job.metadata["rounding_step"] == pytest.approx(0.25)
    assert job.metadata["search_id"] == "search-a"


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
    )

    plan = generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(main_run, rerun),
        output_plan=tmp_path / "experiments" / "energy" / "merged.yaml",
    )

    assert len(plan.jobs) == 1
    assert plan.jobs[0].mst_rate == pytest.approx(5.12)
    assert plan.jobs[0].metadata["source_orchestrator_run_id"] == "targeted-rerun"
    assert plan.jobs[0].metadata["source_orchestrator_run_root"] == str(rerun)


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
