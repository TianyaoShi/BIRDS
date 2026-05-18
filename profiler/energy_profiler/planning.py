from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping

import yaml

from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest
from local_orchestrator.models import ExpandedExperimentJob
from local_orchestrator.planning import infer_model_size_billions
from local_orchestrator.utils import slugify, stable_hash
from slurm_orchestrator.planning import deserialize_expanded_job

from .models import (
    EnergyLaunchConfig,
    EnergyPlan,
    EnergyPlanDefaults,
    EnergyPlanExecution,
    EnergyPlanHeader,
    EnergyPlanJob,
    EnergyPlanMode,
    EnergyPlanRounding,
    EnergyPlanSelection,
    EnergyPlanSelectionSweep,
    EnergyPlanSlurm,
    EnergyRateSource,
    OrchestratorJobRecord,
)


class PlanningError(RuntimeError):
    pass


MODEL_SIZE_OVERRIDES_B: dict[str, float] = {
    "google/gemma-4-e2b-it": 2.0,
    "google/gemma-4-e4b-it": 4.0,
    "openai/gpt-oss-20b": 20.0,
}


def load_selection_overrides(path: str | Path) -> EnergyPlanSelection:
    payload = _load_yaml_mapping(path)
    selection = payload.get("selection", payload)
    if not isinstance(selection, Mapping):
        raise PlanningError("selection YAML must contain a top-level selection mapping")
    return EnergyPlanSelection.from_dict(selection)


def load_energy_plan(path: str | Path) -> EnergyPlan:
    payload = _load_yaml_mapping(path)
    return EnergyPlan.from_dict(payload)


def write_energy_plan(plan: EnergyPlan, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(plan.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def generate_plan_from_orchestrator(
    *,
    orchestrator_run_root: str | Path,
    output_plan: str | Path,
    rate_source: EnergyRateSource = "max_slo",
    mode: EnergyPlanMode = "mst-rounded",
    selection: EnergyPlanSelection | None = None,
    rounding: EnergyPlanRounding | None = None,
    defaults: EnergyPlanDefaults | None = None,
) -> EnergyPlan:
    return generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=(orchestrator_run_root,),
        output_plan=output_plan,
        rate_source=rate_source,
        mode=mode,
        selection=selection,
        rounding=rounding,
        defaults=defaults,
    )


def generate_plan_from_orchestrator_runs(
    *,
    orchestrator_run_roots: tuple[str | Path, ...] | list[str | Path],
    output_plan: str | Path,
    rate_source: EnergyRateSource = "max_slo",
    mode: EnergyPlanMode = "mst-rounded",
    selection: EnergyPlanSelection | None = None,
    rounding: EnergyPlanRounding | None = None,
    defaults: EnergyPlanDefaults | None = None,
) -> EnergyPlan:
    run_roots = tuple(Path(root) for root in orchestrator_run_roots)
    if not run_roots:
        raise PlanningError("at least one orchestrator run root is required")
    output_plan_path = Path(output_plan)
    resolved_selection = selection or EnergyPlanSelection()
    resolved_rounding = rounding or EnergyPlanRounding()
    resolved_defaults = defaults or EnergyPlanDefaults()
    execution, slurm = _load_runtime_config_from_orchestrator_run(run_roots[-1])

    source_jobs = _apply_source_job_exclusions(
        _load_orchestrator_jobs_from_roots(run_roots),
        selection=resolved_selection,
    )
    succeeded_jobs = [job for job in source_jobs if job.status == "succeeded"]
    if not succeeded_jobs:
        raise PlanningError(
            "no succeeded jobs found in orchestrator summaries: "
            f"{[str(root / 'summary.json') for root in run_roots]}"
        )

    _validate_selection(source_jobs=source_jobs, succeeded_jobs=succeeded_jobs, selection=resolved_selection)

    selected_jobs = _select_source_jobs(
        jobs=succeeded_jobs,
        selection=resolved_selection,
    )
    if not selected_jobs:
        raise PlanningError("selection excluded every succeeded orchestrator job")

    jobs = _build_plan_jobs(
        source_jobs=selected_jobs,
        rate_source=rate_source,
        mode=mode,
        selection=resolved_selection,
        rounding=resolved_rounding,
    )
    header = EnergyPlanHeader(
        plan_id=output_plan_path.stem,
        source_orchestrator_run_root=run_roots[0],
        output_root=Path("results/energy"),
        python_executable=_resolve_python_executable(run_roots[-1]),
        mode=mode,
    )
    return EnergyPlan(
        plan=header,
        selection=resolved_selection,
        defaults=resolved_defaults,
        execution=execution,
        slurm=slurm,
        rounding=resolved_rounding,
        jobs=tuple(jobs),
    )


def render_dry_run(plan: EnergyPlan) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for signature, jobs in _group_jobs_by_server_signature(plan.jobs).items():
        groups.append(
            {
                "server_signature_key": signature,
                "job_count": len(jobs),
                "model": jobs[0].model,
                "workload": str(jobs[0].workload),
                "request_rates": [job.request_rate for job in jobs],
            }
        )
    return {
        "plan_id": plan.plan.plan_id,
        "mode": plan.plan.mode,
        "job_count": len(plan.jobs),
        "local_execution": plan.execution.to_dict(),
        "slurm": plan.slurm.to_dict(),
        "source_orchestrator_run_ids": sorted(
            {
                str(job.metadata.get("source_orchestrator_run_id"))
                for job in plan.jobs
                if job.metadata.get("source_orchestrator_run_id") is not None
            }
        ),
        "groups": groups,
    }


def _load_orchestrator_jobs_from_roots(run_roots: tuple[Path, ...]) -> list[OrchestratorJobRecord]:
    selected_by_key: dict[tuple[str, str, str, str], OrchestratorJobRecord] = {}
    all_jobs: list[OrchestratorJobRecord] = []
    base_workload_keys: set[str] | None = None
    for run_root in run_roots:
        run_jobs = _load_orchestrator_jobs(run_root)
        if base_workload_keys is None:
            base_workload_keys = {_logical_workload_key(job.workload) for job in run_jobs}
        run_jobs = [
            job
            for job in run_jobs
            if _logical_workload_key(job.workload) in base_workload_keys
        ]
        all_jobs.extend(run_jobs)
        for job in run_jobs:
            if job.status != "succeeded":
                continue
            selected_by_key[_decisive_job_key(job)] = job

    decisive_config_jobs = set(id(job) for job in selected_by_key.values())
    merged: list[OrchestratorJobRecord] = []
    for job in all_jobs:
        if job.status == "succeeded" and id(job) not in decisive_config_jobs:
            continue
        merged.append(job)
    return merged


def _load_runtime_config_from_orchestrator_run(run_root: Path) -> tuple[EnergyPlanExecution, EnergyPlanSlurm]:
    state = _load_json_mapping(run_root / "state.json")
    manifest_path = Path(_expect_str(state.get("manifest_path"), "state.json.manifest_path"))
    manifest = load_manifest(manifest_path)
    return EnergyPlanExecution.from_run_config(manifest.run), EnergyPlanSlurm.from_slurm_config(manifest.slurm)


def _decisive_job_key(job: OrchestratorJobRecord) -> tuple[str, str, str, str]:
    return (job.model, _logical_workload_key(job.workload), job.endpoint, job.server_signature_key)


def _logical_workload_key(workload: Path) -> str:
    name = _workload_name(workload)
    normalized = name.strip().lower().replace("_", "-")
    for suffix in ("-mst-anomaly-rerun", "-anomaly-rerun", "-rerun"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    normalized = normalized.replace("-8k", "-8192")
    normalized = normalized.replace("-4k", "-4096")
    return normalized


def _workload_name(workload: Path) -> str:
    try:
        payload = yaml.safe_load(workload.read_text(encoding="utf-8"))
    except OSError:
        return workload.stem
    if isinstance(payload, Mapping) and isinstance(payload.get("name"), str) and payload["name"]:
        return str(payload["name"])
    return workload.stem


def _load_orchestrator_jobs(run_root: Path) -> list[OrchestratorJobRecord]:
    summary = _load_json_mapping(run_root / "summary.json")
    state = _load_json_mapping(run_root / "state.json")
    manifest_path = Path(_expect_str(state.get("manifest_path"), "state.json.manifest_path"))
    manifest = load_manifest(manifest_path)
    expanded_jobs = _load_expanded_jobs(run_root=run_root, manifest=manifest)

    summary_jobs = summary.get("jobs")
    if not isinstance(summary_jobs, list):
        raise PlanningError("orchestrator summary.json must contain jobs[]")

    records: list[OrchestratorJobRecord] = []
    for item in summary_jobs:
        if not isinstance(item, Mapping):
            raise PlanningError("orchestrator summary jobs[] entries must be mappings")
        experiment_id = _expect_str(item.get("experiment_id"), "summary.json.jobs[].experiment_id")
        expanded = expanded_jobs.get(experiment_id)
        if expanded is None:
            raise PlanningError(
                f"orchestrator summary references experiment_id not present in manifest expansion: {experiment_id}"
            )
        raw_result_dir = item.get("result_dir")
        result_dir = Path(str(raw_result_dir)) if isinstance(raw_result_dir, str) and raw_result_dir else expanded.result_dir

        search_trace_path = Path(str(item.get("artifacts", {}).get("search_trace") or result_dir / "search_trace.json"))
        search_trace: Mapping[str, Any] | None = None
        if search_trace_path.is_file():
            search_trace = _load_json_mapping(search_trace_path)
        elif str(item.get("status")) == "succeeded":
            raise PlanningError(f"succeeded orchestrator job is missing search_trace.json: {search_trace_path}")

        search_config = _optional_mapping(search_trace.get("config")) if search_trace else {}
        search_result = _optional_mapping(search_trace.get("result")) if search_trace else {}
        records.append(
            OrchestratorJobRecord(
                source_run_id=run_root.name,
                source_run_root=run_root,
                experiment_id=experiment_id,
                model=expanded.model,
                workload=expanded.workload,
                endpoint=expanded.endpoint,
                result_dir=result_dir,
                status=_expect_str(item.get("status"), "summary.json.jobs[].status"),
                max_no_drift_request_rate=_optional_numeric(
                    item.get("max_no_drift_request_rate"),
                    "summary.json.jobs[].max_no_drift_request_rate",
                )
                or _optional_numeric(
                    search_result.get("max_no_drift_request_rate"),
                    "search_trace.json.result.max_no_drift_request_rate",
                ),
                max_slo_satisfying_request_rate=_optional_numeric(
                    item.get("max_slo_satisfying_request_rate"),
                    "summary.json.jobs[].max_slo_satisfying_request_rate",
                )
                or _optional_numeric(
                    search_result.get("max_slo_satisfying_request_rate"),
                    "search_trace.json.result.max_slo_satisfying_request_rate",
                ),
                model_size_b=infer_model_size_billions(
                    expanded.model,
                    model_size_overrides_b=MODEL_SIZE_OVERRIDES_B,
                ),
                search_id=_optional_string(search_config.get("search_id")),
                search_mode=_optional_string(search_config.get("search_mode")),
                confirmation_trial_id=_optional_string(search_result.get("confirmation_trial_id")),
                launch=EnergyLaunchConfig.from_launch_config(expanded.launch),
                server_signature_key=expanded.server_signature_key,
                server_config_slug=expanded.server_config_slug,
            )
        )
    return records


def _load_expanded_jobs(*, run_root: Path, manifest: Any) -> dict[str, ExpandedExperimentJob]:
    expanded_jobs = {job.experiment_id: job for job in expand_manifest(manifest)}
    expanded_jobs.update(_load_planned_expanded_jobs(run_root=run_root))
    return expanded_jobs


def _load_planned_expanded_jobs(*, run_root: Path) -> dict[str, ExpandedExperimentJob]:
    plan_path = run_root / "plan.json"
    if not plan_path.is_file():
        return {}
    plan_payload = _load_json_mapping(plan_path)
    groups = plan_payload.get("groups")
    if not isinstance(groups, list):
        return {}

    planned_jobs: dict[str, ExpandedExperimentJob] = {}
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            continue
        raw_plan_path = raw_group.get("plan_path")
        if not isinstance(raw_plan_path, str) or not raw_plan_path:
            continue
        group_plan_path = Path(raw_plan_path)
        if not group_plan_path.is_absolute():
            group_plan_path = run_root / group_plan_path
        if not group_plan_path.is_file():
            continue
        group_payload = _load_json_mapping(group_plan_path)
        group_jobs = group_payload.get("jobs")
        if not isinstance(group_jobs, list):
            continue
        for raw_task in group_jobs:
            if not isinstance(raw_task, Mapping):
                continue
            raw_job = raw_task.get("job")
            if not isinstance(raw_job, dict):
                continue
            job = deserialize_expanded_job(raw_job)
            planned_jobs[job.experiment_id] = job
    return planned_jobs


def _apply_source_job_exclusions(
    jobs: list[OrchestratorJobRecord],
    *,
    selection: EnergyPlanSelection,
) -> list[OrchestratorJobRecord]:
    return [job for job in jobs if not _is_excluded_source_job(job, selection)]


def _is_excluded_source_job(job: OrchestratorJobRecord, selection: EnergyPlanSelection) -> bool:
    if selection.exclude_models and job.model in selection.exclude_models:
        return True
    if selection.exclude_experiment_ids and job.experiment_id in selection.exclude_experiment_ids:
        return True
    if selection.exclude_workloads and _workload_matches_any(job.workload, selection.exclude_workloads):
        return True
    if (
        selection.min_model_size_b is not None
        and job.model_size_b is not None
        and job.model_size_b < selection.min_model_size_b
    ):
        return True
    return False


def _select_source_jobs(
    *,
    jobs: list[OrchestratorJobRecord],
    selection: EnergyPlanSelection,
) -> list[OrchestratorJobRecord]:
    selected = [
        job
        for job in jobs
        if _matches_top_level_selection(job, selection)
    ]
    selected.sort(key=lambda job: (job.model, str(job.workload), str(job.result_dir), job.experiment_id))
    return selected


def _matches_top_level_selection(job: OrchestratorJobRecord, selection: EnergyPlanSelection) -> bool:
    if selection.models and job.model not in selection.models:
        return False
    if selection.experiment_ids and job.experiment_id not in selection.experiment_ids:
        return False
    if selection.workloads and not _workload_matches_any(job.workload, selection.workloads):
        return False
    return True


def _validate_selection(
    *,
    source_jobs: list[OrchestratorJobRecord],
    succeeded_jobs: list[OrchestratorJobRecord],
    selection: EnergyPlanSelection,
) -> None:
    all_models = {job.model for job in source_jobs}
    succeeded_models = {job.model for job in succeeded_jobs}
    all_experiment_ids = {job.experiment_id for job in source_jobs}
    succeeded_experiment_ids = {job.experiment_id for job in succeeded_jobs}
    all_workloads = _workload_keyset(source_jobs)
    succeeded_workloads = _workload_keyset(succeeded_jobs)

    _validate_string_filter(selection.models, all_models, succeeded_models, "selection.models")
    _validate_string_filter(
        selection.experiment_ids,
        all_experiment_ids,
        succeeded_experiment_ids,
        "selection.experiment_ids",
    )
    _validate_string_filter(selection.workloads, all_workloads, succeeded_workloads, "selection.workloads")
    _validate_string_filter(selection.sweep.models, all_models, succeeded_models, "selection.sweep.models")
    _validate_string_filter(
        selection.sweep.experiment_ids,
        all_experiment_ids,
        succeeded_experiment_ids,
        "selection.sweep.experiment_ids",
    )


def _validate_string_filter(
    requested: tuple[str, ...],
    available_all: set[str],
    available_succeeded: set[str],
    field_name: str,
) -> None:
    for value in requested:
        if value not in available_all:
            raise PlanningError(f"{field_name} value not found in orchestrator run: {value}")
        if value not in available_succeeded:
            raise PlanningError(f"{field_name} value has no succeeded MST result: {value}")


def _workload_keyset(jobs: list[OrchestratorJobRecord]) -> set[str]:
    values: set[str] = set()
    for job in jobs:
        values.add(str(job.workload))
        values.add(job.workload.name)
        values.add(job.workload.stem)
    return values


def _build_plan_jobs(
    *,
    source_jobs: list[OrchestratorJobRecord],
    rate_source: EnergyRateSource,
    mode: EnergyPlanMode,
    selection: EnergyPlanSelection,
    rounding: EnergyPlanRounding,
) -> list[EnergyPlanJob]:
    jobs: list[EnergyPlanJob] = []
    seen_ids: set[str] = set()

    if mode == "explicit" and not selection.explicit_request_rates:
        raise PlanningError("explicit mode requires selection.explicit_request_rates")

    if mode == "sweep":
        source_jobs = _filter_sweep_jobs(source_jobs, selection.sweep)
        if not source_jobs:
            raise PlanningError("sweep mode selection excluded every succeeded orchestrator job")

    for source_job in sorted(source_jobs, key=lambda job: (job.model, str(job.workload), job.server_config_slug, job.experiment_id)):
        mst_rate, mst_field = _select_mst_rate(source_job, rate_source)
        if mode != "explicit" and mst_rate is None:
            raise PlanningError(
                f"job {source_job.experiment_id} is missing the selected MST rate source {rate_source}"
            )
        if mode == "mst-rounded":
            assert mst_rate is not None
            rounded_rate, step, clamped = round_mst_rate(mst_rate, rounding)
            metadata = _base_job_metadata(source_job, step, clamped)
            metadata["rounding_policy"] = rounding.mst_mode
            metadata["rounded_from_rate"] = mst_rate
            if rounding.mst_mode == "floor_decimal":
                metadata["rounding_decimal_places"] = rounding.mst_decimal_places
            plan_job = _make_plan_job(
                source_job=source_job,
                request_rate=rounded_rate,
                mst_rate=mst_rate,
                mst_rate_source=mst_field,
                metadata=metadata,
                seen_ids=seen_ids,
            )
            jobs.append(plan_job)
            continue

        if mode == "sweep":
            assert mst_rate is not None
            rates, step, clamped = build_sweep_rates(
                mst_rate,
                rounding,
                max_steps=selection.sweep.max_steps,
            )
            for index, request_rate in enumerate(rates, start=1):
                metadata = _base_job_metadata(source_job, step, clamped)
                metadata.update(
                    {
                        "rounding_policy": rounding.sweep_mode,
                        "rounded_from_rate": mst_rate,
                        "sweep_rate_index": index,
                        "sweep_rate_count": len(rates),
                    }
                )
                jobs.append(
                    _make_plan_job(
                        source_job=source_job,
                        request_rate=request_rate,
                        mst_rate=mst_rate,
                        mst_rate_source=mst_field,
                        metadata=metadata,
                        seen_ids=seen_ids,
                    )
                )
            continue

        for request_rate in sorted(selection.explicit_request_rates):
            metadata = _base_job_metadata(source_job, None, False)
            metadata["explicit_request_rate"] = True
            jobs.append(
                _make_plan_job(
                    source_job=source_job,
                    request_rate=request_rate,
                    mst_rate=mst_rate,
                    mst_rate_source=mst_field,
                    metadata=metadata,
                    seen_ids=seen_ids,
                )
            )

    jobs.sort(key=lambda job: (job.server_signature_key, job.model, str(job.workload), job.request_rate, job.id))
    return jobs


def _filter_sweep_jobs(
    jobs: list[OrchestratorJobRecord],
    sweep: EnergyPlanSelectionSweep,
) -> list[OrchestratorJobRecord]:
    selected = jobs
    if sweep.models:
        selected = [job for job in selected if job.model in sweep.models]
    if sweep.experiment_ids:
        selected = [job for job in selected if job.experiment_id in sweep.experiment_ids]
    return selected


def _select_mst_rate(
    source_job: OrchestratorJobRecord,
    rate_source: EnergyRateSource,
) -> tuple[float | None, str]:
    if rate_source == "max_no_drift":
        return source_job.max_no_drift_request_rate, "max_no_drift_request_rate"
    return source_job.max_slo_satisfying_request_rate, "max_slo_satisfying_request_rate"


def _make_plan_job(
    *,
    source_job: OrchestratorJobRecord,
    request_rate: float,
    mst_rate: float | None,
    mst_rate_source: str | None,
    metadata: dict[str, Any],
    seen_ids: set[str],
) -> EnergyPlanJob:
    base_id = f"{slugify(source_job.model, max_length=40)}-{slugify(source_job.workload.stem, max_length=32)}-r{rate_to_id_token(request_rate)}"
    job_id = base_id
    if job_id in seen_ids:
        job_id = f"{base_id}-{stable_hash({'experiment_id': source_job.experiment_id, 'rate': request_rate, 'server': source_job.server_signature_key}, length=8)}"
    seen_ids.add(job_id)
    return EnergyPlanJob(
        id=job_id,
        source_experiment_id=source_job.experiment_id,
        source_result_dir=source_job.result_dir,
        model=source_job.model,
        workload=source_job.workload,
        endpoint=source_job.endpoint,
        request_rate=request_rate,
        mst_rate=mst_rate,
        mst_rate_source=mst_rate_source,
        launch=source_job.launch,
        server_signature_key=source_job.server_signature_key,
        server_config_slug=source_job.server_config_slug,
        metadata=metadata,
    )


def _base_job_metadata(
    source_job: OrchestratorJobRecord,
    rounding_step: float | None,
    clamped: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_orchestrator_run_id": source_job.source_run_id,
        "source_orchestrator_run_root": str(source_job.source_run_root),
        "search_id": source_job.search_id,
        "search_mode": source_job.search_mode,
        "confirmation_trial_id": source_job.confirmation_trial_id,
        "rounding_policy": "floor_preferred",
    }
    if rounding_step is not None:
        metadata["rounding_step"] = rounding_step
    if clamped:
        metadata["clamped_to_minimum_rate"] = True
    return metadata


def choose_display_step(mst_rate: float, rounding: EnergyPlanRounding) -> float:
    if mst_rate < 1.0:
        target = 0.05
    elif mst_rate < 2.0:
        target = 0.1
    elif mst_rate < 5.0:
        target = 0.25
    elif mst_rate < 10.0:
        target = 0.5
    else:
        target = 1.0
    best = min(rounding.preferred_steps, key=lambda step: (abs(step - target), step))
    return float(best)


def round_mst_rate(mst_rate: float, rounding: EnergyPlanRounding) -> tuple[float, float, bool]:
    if mst_rate <= 0.0:
        raise PlanningError(f"MST rate must be positive for rounded profiling, got {mst_rate}")
    if rounding.mst_mode == "floor_decimal":
        step = 10 ** (-rounding.mst_decimal_places)
    else:
        step = choose_display_step(mst_rate, rounding)
    if mst_rate < rounding.minimum_rate:
        return rounding.minimum_rate, step, True
    if rounding.mst_mode == "floor_decimal":
        rounded = _floor_to_decimal_places(mst_rate, rounding.mst_decimal_places)
    else:
        rounded = _floor_to_step(mst_rate, step)
    if rounded < rounding.minimum_rate:
        return rounding.minimum_rate, step, True
    return rounded, step, False


def build_sweep_rates(
    mst_rate: float,
    rounding: EnergyPlanRounding,
    *,
    max_steps: int,
) -> tuple[list[float], float, bool]:
    if mst_rate <= 0.0:
        raise PlanningError(f"MST rate must be positive for sweep profiling, got {mst_rate}")
    if mst_rate < rounding.minimum_rate:
        return [rounding.minimum_rate], choose_display_step(mst_rate, rounding), True

    chosen_step = float(rounding.preferred_steps[-1])
    for step in rounding.preferred_steps:
        point_count = int(_decimal_from_float(mst_rate) / _decimal_from_float(step))
        if point_count <= max_steps:
            chosen_step = float(step)
            break

    rounded_mst = _floor_to_step(mst_rate, chosen_step)
    rates: list[float] = []
    current = _decimal_from_float(chosen_step)
    limit = _decimal_from_float(rounded_mst)
    step_decimal = _decimal_from_float(chosen_step)
    seen: set[float] = set()
    while current <= limit and float(current) > 0.0:
        rate = float(current)
        if rate not in seen:
            rates.append(rate)
            seen.add(rate)
        current += step_decimal
    if not rates:
        rates = [rounding.minimum_rate]
        return rates, chosen_step, True
    return rates, chosen_step, False


def rate_to_id_token(rate: float) -> str:
    text = format(Decimal(str(rate)).normalize(), "f")
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".").replace(".", "_")


def _floor_to_step(value: float, step: float) -> float:
    value_dec = _decimal_from_float(value)
    step_dec = _decimal_from_float(step)
    units = (value_dec / step_dec).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * step_dec)


def _floor_to_decimal_places(value: float, decimal_places: int) -> float:
    quantum = Decimal("1").scaleb(-decimal_places)
    return float(_decimal_from_float(value).quantize(quantum, rounding=ROUND_FLOOR))


def _decimal_from_float(value: float) -> Decimal:
    return Decimal(str(value))


def _group_jobs_by_server_signature(jobs: tuple[EnergyPlanJob, ...]) -> dict[str, list[EnergyPlanJob]]:
    groups: dict[str, list[EnergyPlanJob]] = {}
    for job in jobs:
        groups.setdefault(job.server_signature_key, []).append(job)
    for bucket in groups.values():
        bucket.sort(key=lambda job: job.request_rate)
    return dict(sorted(groups.items()))


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PlanningError(f"YAML file must decode to a mapping: {path}")
    return payload


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanningError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanningError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PlanningError(f"JSON file must decode to a mapping: {path}")
    return payload


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanningError(f"{field_name} must be a mapping")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _expect_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanningError(f"{field_name} must be a non-empty string")
    return value


def _optional_numeric(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanningError(f"{field_name} must be numeric when present")
    numeric = float(value)
    if numeric <= 0.0:
        return None
    return numeric


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    return value


def _workload_matches_any(workload: Path, requested: tuple[str, ...]) -> bool:
    candidates = {str(workload), workload.name, workload.stem}
    return any(value in candidates for value in requested)


def _resolve_python_executable(run_root: Path) -> str:
    state = _load_json_mapping(run_root / "state.json")
    manifest_path = Path(_expect_str(state.get("manifest_path"), "state.json.manifest_path"))
    manifest = load_manifest(manifest_path)
    return manifest.run.python_executable or sys.executable
