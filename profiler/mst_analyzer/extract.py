from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest
from local_orchestrator.models import ExpandedExperimentJob
from local_orchestrator.planning import infer_model_size_billions, load_cached_hf_config
from slurm_orchestrator.planning import deserialize_expanded_job

from .models import MSTRow, TraceInstabilityEvidence, TrialArtifactRef


MODEL_SIZE_OVERRIDES_B: dict[str, float] = {
    "google/gemma-4-e2b-it": 2.0,
    "google/gemma-4-e4b-it": 4.0,
    "openai/gpt-oss-20b": 20.0,
}
_SUSPECT_TERMINATION_REASONS = {
    "max_request_rate_limited",
    "no_confirmed_stable_open_loop_rate",
    "confirmation_inconclusive",
}


@dataclass(slots=True)
class ExtractedRun:
    run_id: str
    run_root: Path
    manifest_path: Path
    manifest_payload: dict[str, Any]
    expanded_jobs: dict[str, ExpandedExperimentJob]
    rows: tuple[MSTRow, ...]
    source_manifest_paths: tuple[Path, ...]


def extract_run(orchestrator_run_root: str | Path) -> ExtractedRun:
    run_root = Path(orchestrator_run_root).resolve()
    summary_payload = _load_json_mapping(run_root / "summary.json")
    state_payload = _load_json_mapping(run_root / "state.json")
    manifest_path = _resolve_general_path(
        _expect_string(state_payload.get("manifest_path"), "state.json.manifest_path"),
        run_root=run_root,
        manifest_path=None,
    )
    manifest_payload = _load_yaml_mapping(manifest_path)
    manifest = load_manifest(manifest_path)
    expanded_jobs = _load_expanded_jobs(run_root=run_root, manifest=manifest, manifest_path=manifest_path)

    summary_jobs = summary_payload.get("jobs")
    if not isinstance(summary_jobs, list):
        raise RuntimeError("orchestrator summary.json must contain jobs[]")

    rows: list[MSTRow] = []
    for item in summary_jobs:
        if not isinstance(item, Mapping):
            raise RuntimeError("orchestrator summary.json jobs[] entries must be mappings")
        if str(item.get("status")) != "succeeded":
            continue
        experiment_id = _expect_string(item.get("experiment_id"), "summary.json.jobs[].experiment_id")
        expanded_job = expanded_jobs.get(experiment_id)
        if expanded_job is None:
            raise RuntimeError(
                "orchestrator summary.json references experiment_id not present in manifest expansion: "
                f"{experiment_id}"
            )
        rows.append(
            _extract_row(
                run_root=run_root,
                manifest_path=manifest_path,
                summary_job=item,
                expanded_job=expanded_job,
            )
        )

    return ExtractedRun(
        run_id=run_root.name,
        run_root=run_root,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        expanded_jobs=expanded_jobs,
        rows=tuple(rows),
        source_manifest_paths=(manifest_path,),
    )


def _load_expanded_jobs(
    *,
    run_root: Path,
    manifest: Any,
    manifest_path: Path,
) -> dict[str, ExpandedExperimentJob]:
    expanded_jobs = {job.experiment_id: job for job in expand_manifest(manifest)}
    planned_jobs = _load_planned_expanded_jobs(run_root=run_root, manifest_path=manifest_path)
    if planned_jobs:
        expanded_jobs.update(planned_jobs)
    return expanded_jobs


def _load_planned_expanded_jobs(*, run_root: Path, manifest_path: Path) -> dict[str, ExpandedExperimentJob]:
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
        group_plan_path = _resolve_general_path(raw_plan_path, run_root=run_root, manifest_path=manifest_path)
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


def extract_runs(orchestrator_run_roots: tuple[str | Path, ...] | list[str | Path]) -> ExtractedRun:
    run_roots = tuple(orchestrator_run_roots)
    if not run_roots:
        raise RuntimeError("at least one orchestrator run root is required")
    extracted_runs = [extract_run(run_root) for run_root in run_roots]

    selected_by_experiment_id: dict[str, MSTRow] = {}
    selected_by_key: dict[tuple[str, str, str, str], MSTRow] = {}
    all_rows: list[MSTRow] = []
    expanded_jobs: dict[str, ExpandedExperimentJob] = {}
    for extracted in extracted_runs:
        expanded_jobs.update(extracted.expanded_jobs)
        all_rows.extend(extracted.rows)
        for row in extracted.rows:
            selected_by_experiment_id[row.experiment_id] = row
            selected_by_key[_decisive_row_key(row)] = row

    decisive_experiment_rows = set(id(row) for row in selected_by_experiment_id.values())
    decisive_config_rows = set(id(row) for row in selected_by_key.values())
    merged_rows = [
        row
        for row in all_rows
        if id(row) in decisive_experiment_rows and id(row) in decisive_config_rows
    ]
    first = extracted_runs[0]
    return ExtractedRun(
        run_id="+".join(extracted.run_id for extracted in extracted_runs),
        run_root=first.run_root,
        manifest_path=first.manifest_path,
        manifest_payload=first.manifest_payload,
        expanded_jobs=expanded_jobs,
        rows=tuple(merged_rows),
        source_manifest_paths=tuple(extracted.manifest_path for extracted in extracted_runs),
    )


def _decisive_row_key(row: MSTRow) -> tuple[str, str, str, str]:
    return (row.model, str(row.workload_path), row.endpoint, row.server_signature_key)


def _extract_row(
    *,
    run_root: Path,
    manifest_path: Path,
    summary_job: Mapping[str, Any],
    expanded_job: ExpandedExperimentJob,
) -> MSTRow:
    result_dir = _resolve_general_path(
        str(summary_job.get("result_dir") or expanded_job.result_dir),
        run_root=run_root,
        manifest_path=manifest_path,
    )
    artifacts = summary_job.get("artifacts")
    artifacts_mapping = artifacts if isinstance(artifacts, Mapping) else {}
    search_trace_path = _resolve_general_path(
        str(artifacts_mapping.get("search_trace") or result_dir / "search_trace.json"),
        run_root=run_root,
        manifest_path=manifest_path,
    )
    search_trace = _load_json_mapping(search_trace_path)
    result_payload = _require_mapping(search_trace.get("result"), "search_trace.json.result")
    bounds_payload = _require_mapping(search_trace.get("bounds", {}), "search_trace.json.bounds")

    final_report_json_path: Path | None = None
    final_report_payload: Mapping[str, Any] = {}
    raw_final_report = artifacts_mapping.get("final_report_json")
    if isinstance(raw_final_report, str) and raw_final_report:
        final_report_json_path = _resolve_general_path(raw_final_report, run_root=run_root, manifest_path=manifest_path)
    else:
        candidate_report = result_dir / "final_report.json"
        if candidate_report.is_file():
            final_report_json_path = candidate_report.resolve()
    if final_report_json_path is not None and final_report_json_path.is_file():
        final_report_payload = _load_json_mapping(final_report_json_path)

    events = search_trace.get("events")
    if not isinstance(events, list):
        raise RuntimeError("search_trace.json.events must be a list")
    event_map: dict[str, Mapping[str, Any]] = {}
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            raise RuntimeError("search_trace.json events[] entries must be mappings")
        trial_id = raw_event.get("trial_id")
        if isinstance(trial_id, str) and trial_id:
            event_map[trial_id] = raw_event

    confirmation_trial_id = _optional_string(result_payload.get("confirmation_trial_id"))
    confirmation_ref, confirmation_summary = _trial_artifact(
        result_dir=result_dir,
        event=event_map.get(confirmation_trial_id) if confirmation_trial_id is not None else None,
        trial_id=confirmation_trial_id,
    )
    high_bound_trial_id = _optional_string(bounds_payload.get("high_trial_id"))
    high_bound_ref, _ = _trial_artifact(
        result_dir=result_dir,
        event=event_map.get(high_bound_trial_id) if high_bound_trial_id is not None else None,
        trial_id=high_bound_trial_id,
    )

    workload_payload = _mapping_or_empty(final_report_payload.get("workload"))
    if not workload_payload and confirmation_summary is not None:
        workload_payload = _workload_payload_from_summary(confirmation_summary)
    workload_name = _optional_string(workload_payload.get("name")) or expanded_job.workload.stem

    stability_policy = _mapping_or_empty(final_report_payload.get("stability_policy"))
    if not stability_policy and confirmation_summary is not None:
        stability_policy = _stability_policy_from_summary(confirmation_summary)
    if not stability_policy:
        stability_policy = {
            "ttft_slo_ms": expanded_job.search.ttft_slo_ms,
            "tpot_slo_ms": expanded_job.search.tpot_slo_ms,
            "ttft_slo_field": expanded_job.search.ttft_slo_field,
            "tpot_slo_field": expanded_job.search.tpot_slo_field,
        }

    server_config = _mapping_or_empty(final_report_payload.get("server_config"))
    if not server_config and confirmation_summary is not None:
        server_config = _server_config_from_summary(confirmation_summary)

    model_config = load_cached_hf_config(expanded_job.model)
    quantization = _infer_quantization(expanded_job.launch.quantization, model_config)
    model_size_b = infer_model_size_billions(
        expanded_job.model,
        model_size_overrides_b=MODEL_SIZE_OVERRIDES_B,
    )
    model_family, model_variant = _infer_model_identity(expanded_job.model)

    confirmation_metrics = _benchmark_metrics_from_summary(confirmation_summary)
    has_slo_signal = _has_slo_signal(
        result_payload=result_payload,
        confirmation_ref=confirmation_ref,
        high_bound_ref=high_bound_ref,
    )

    return MSTRow(
        experiment_id=expanded_job.experiment_id,
        model=expanded_job.model,
        model_family=model_family,
        model_variant=model_variant,
        model_size_b=model_size_b,
        size_bucket=_size_bucket(model_size_b),
        hardware=expanded_job.hardware.name,
        workload_name=workload_name,
        workload_path=expanded_job.workload.resolve(),
        endpoint=expanded_job.endpoint,
        endpoint_type=_endpoint_type(expanded_job.endpoint),
        mst_rps=_optional_numeric(
            result_payload.get("max_slo_satisfying_request_rate"),
            "search_trace.json.result.max_slo_satisfying_request_rate",
        )
        or _optional_numeric(
            result_payload.get("max_no_drift_request_rate"),
            "search_trace.json.result.max_no_drift_request_rate",
        ),
        termination_reason=_optional_string(result_payload.get("termination_reason")),
        bottleneck_class=_optional_string(result_payload.get("bottleneck_class")),
        confidence=_optional_string(result_payload.get("confidence")),
        server_signature_key=expanded_job.server_signature_key,
        max_num_seqs=_optional_numeric(server_config.get("max_num_seqs"), "server_config.max_num_seqs")
        or _float_or_none(expanded_job.launch.max_num_seqs)
        or _float_or_none(expanded_job.search.max_num_seqs),
        max_num_batched_tokens=_optional_numeric(
            server_config.get("max_num_batched_tokens"),
            "server_config.max_num_batched_tokens",
        )
        or _float_or_none(expanded_job.launch.max_num_batched_tokens)
        or _float_or_none(expanded_job.search.max_num_batched_tokens),
        max_model_len=expanded_job.launch.max_model_len,
        tensor_parallel_size=expanded_job.launch.tensor_parallel_size,
        gpu_count=expanded_job.launch.gpu_count,
        dtype=expanded_job.launch.dtype,
        quantization=quantization,
        is_quantized=quantization is not None,
        is_moe=_is_moe(model_config, expanded_job.model),
        ttft_slo_ms=_optional_numeric(stability_policy.get("ttft_slo_ms"), "stability_policy.ttft_slo_ms"),
        tpot_slo_ms=_optional_numeric(stability_policy.get("tpot_slo_ms"), "stability_policy.tpot_slo_ms"),
        ttft_slo_field=_optional_string(stability_policy.get("ttft_slo_field")),
        tpot_slo_field=_optional_string(stability_policy.get("tpot_slo_field")),
        confirmation_trial=confirmation_ref,
        confirmation_successful_completion_rate=confirmation_metrics.get("successful_completion_rate"),
        confirmation_total_token_throughput=confirmation_metrics.get("total_token_throughput"),
        confirmation_generation_token_throughput=confirmation_metrics.get("generation_token_throughput"),
        confirmation_prompt_len_mean=confirmation_metrics.get("prompt_len_mean"),
        confirmation_output_len_mean=confirmation_metrics.get("output_len_mean"),
        high_bound_rate=_optional_numeric(bounds_payload.get("high_rate"), "search_trace.json.bounds.high_rate"),
        high_bound_trial=high_bound_ref,
        result_dir=result_dir,
        search_trace_path=search_trace_path,
        final_report_json_path=final_report_json_path,
        trace_instability=_trace_instability_evidence(
            events=events,
            confidence=_optional_string(result_payload.get("confidence")),
            termination_reason=_optional_string(result_payload.get("termination_reason")),
        ),
        has_slo_signal=has_slo_signal,
    )


def _trial_artifact(
    *,
    result_dir: Path,
    event: Mapping[str, Any] | None,
    trial_id: str | None,
) -> tuple[TrialArtifactRef | None, Mapping[str, Any] | None]:
    if trial_id is None:
        return None, None

    trial_dir: Path | None = None
    event_analysis = _mapping_or_empty(None if event is None else event.get("analysis"))
    if event is not None:
        raw_trial_dir = event.get("trial_dir")
        if isinstance(raw_trial_dir, str) and raw_trial_dir:
            trial_dir = _resolve_trace_trial_dir(result_dir=result_dir, trial_dir_value=raw_trial_dir)
    if trial_dir is None:
        trial_dir = result_dir / "trials" / trial_id

    summary_json = trial_dir / "summary.json"
    analysis_json = trial_dir / "analysis.json"
    summary_payload = _load_json_mapping(summary_json) if summary_json.is_file() else None
    analysis_payload = _load_json_mapping(analysis_json) if analysis_json.is_file() else event_analysis

    trial_validity = _optional_string(None if not analysis_payload else analysis_payload.get("trial_validity"))
    stability = _mapping_or_empty(None if not analysis_payload else analysis_payload.get("stability"))
    stability_status = _optional_string(stability.get("status"))
    reasons: tuple[str, ...] = ()
    if stability:
        reasons = tuple(_string_list(stability.get("reasons"), "analysis.stability.reasons"))
    elif analysis_payload:
        reasons = tuple(_string_list(analysis_payload.get("validity_reasons"), "analysis.validity_reasons"))

    return (
        TrialArtifactRef(
            trial_id=trial_id,
            trial_dir=trial_dir if trial_dir.is_dir() else None,
            summary_json=summary_json if summary_json.is_file() else None,
            analysis_json=analysis_json if analysis_json.is_file() else None,
            trial_validity=trial_validity,
            stability_status=stability_status,
            reasons=reasons,
        ),
        summary_payload,
    )


def _trace_instability_evidence(
    *,
    events: list[Any],
    confidence: str | None,
    termination_reason: str | None,
) -> TraceInstabilityEvidence:
    statuses_by_rate: dict[str, set[str]] = {}
    conflicting_labels: list[str] = []
    uncertain_trials: set[str] = set()
    majority_confirmation_used = False

    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        if raw_event.get("mode") != "open-loop":
            continue
        trial_id = _optional_string(raw_event.get("trial_id")) or ""
        purpose = _optional_string(raw_event.get("purpose")) or ""
        if purpose == "open_loop_confirmation_majority":
            majority_confirmation_used = True
        analysis = _mapping_or_empty(raw_event.get("analysis"))
        stability = _mapping_or_empty(analysis.get("stability"))
        status = _optional_string(stability.get("status"))
        request_rate = _optional_numeric(raw_event.get("request_rate"), "event.request_rate")
        if request_rate is not None and status is not None:
            label = _format_rate(request_rate)
            statuses_by_rate.setdefault(label, set()).add(status)
        if purpose.endswith("extend_uncertain") or status == "uncertain":
            uncertain_trials.add(trial_id)

    for label, statuses in sorted(statuses_by_rate.items()):
        if "stable" in statuses and any(status != "stable" for status in statuses):
            conflicting_labels.append(label)

    return TraceInstabilityEvidence(
        conflicting_rate_labels=tuple(conflicting_labels),
        majority_confirmation_used=majority_confirmation_used,
        uncertain_retry_count=len([trial_id for trial_id in uncertain_trials if trial_id]),
        low_confidence=confidence == "low",
        suspect_termination_reason=termination_reason in _SUSPECT_TERMINATION_REASONS,
    )


def _workload_payload_from_summary(summary_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config_payload = _mapping_or_empty(summary_payload.get("config"))
    metadata = _mapping_or_empty(config_payload.get("metadata"))
    return _mapping_or_empty(metadata.get("workload"))


def _stability_policy_from_summary(summary_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config_payload = _mapping_or_empty(summary_payload.get("config"))
    metadata = _mapping_or_empty(config_payload.get("metadata"))
    return _mapping_or_empty(metadata.get("stability_policy"))


def _server_config_from_summary(summary_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config_payload = _mapping_or_empty(summary_payload.get("config"))
    metadata = _mapping_or_empty(config_payload.get("metadata"))
    server_config = _mapping_or_empty(metadata.get("server_config"))
    if server_config:
        return server_config
    server_metadata = _mapping_or_empty(metadata.get("server_metadata"))
    for key in ("server_config", "vllm_config"):
        nested = _mapping_or_empty(server_metadata.get(key))
        if nested:
            return nested
    return server_metadata


def _benchmark_metrics_from_summary(summary_payload: Mapping[str, Any] | None) -> dict[str, float | None]:
    if summary_payload is None:
        return {
            "successful_completion_rate": None,
            "total_token_throughput": None,
            "generation_token_throughput": None,
            "prompt_len_mean": None,
            "output_len_mean": None,
        }
    summary = _mapping_or_empty(summary_payload.get("summary"))
    benchmark = _mapping_or_empty(summary.get("benchmark_metrics"))
    prompt_lengths = _mapping_or_empty(benchmark.get("prompt_length_summary"))
    output_lengths = _mapping_or_empty(benchmark.get("output_length_summary"))
    return {
        "successful_completion_rate": _optional_numeric(
            summary.get("successful_completion_rate"),
            "summary.successful_completion_rate",
        ),
        "total_token_throughput": _optional_numeric(
            benchmark.get("total_token_throughput"),
            "benchmark_metrics.total_token_throughput",
        ),
        "generation_token_throughput": _optional_numeric(
            benchmark.get("generation_token_throughput"),
            "benchmark_metrics.generation_token_throughput",
        ),
        "prompt_len_mean": _optional_numeric(prompt_lengths.get("mean"), "prompt_length_summary.mean"),
        "output_len_mean": _optional_numeric(output_lengths.get("mean"), "output_length_summary.mean"),
    }


def _has_slo_signal(
    *,
    result_payload: Mapping[str, Any],
    confirmation_ref: TrialArtifactRef | None,
    high_bound_ref: TrialArtifactRef | None,
) -> bool:
    if any(ref is not None and ref.stability_status == "slo_violation" for ref in (confirmation_ref, high_bound_ref)):
        return True
    for raw_reason in _string_list(result_payload.get("reasons"), "search_trace.json.result.reasons"):
        if "slo" in raw_reason.lower():
            return True
    return False


def _endpoint_type(endpoint: str) -> str:
    normalized = endpoint.rstrip("/").lower()
    if normalized.endswith("/chat/completions"):
        return "chat_completions"
    if normalized.endswith("/completions"):
        return "completions"
    return endpoint


def _infer_model_identity(model: str) -> tuple[str, str | None]:
    lowered = model.lower()
    if "qwen3" in lowered:
        family = "qwen3"
    elif "llama" in lowered:
        family = "llama"
    elif "gemma" in lowered:
        family = "gemma"
    elif "gpt-oss" in lowered:
        family = "gpt-oss"
    else:
        family = (model.split("/")[-1].split("-")[0] or "unknown").lower()

    variant: str | None = None
    for candidate in ("thinking", "instruct", "chat", "base"):
        if candidate in lowered:
            variant = candidate
            break
    return family, variant


def _infer_quantization(launch_quantization: str | None, model_config: Mapping[str, Any] | None) -> str | None:
    if launch_quantization:
        return launch_quantization.lower()
    if not isinstance(model_config, Mapping):
        return None
    quantization_config = model_config.get("quantization_config")
    if isinstance(quantization_config, Mapping):
        quant_method = quantization_config.get("quant_method")
        if isinstance(quant_method, str) and quant_method:
            return quant_method.lower()
    return None


def _is_moe(model_config: Mapping[str, Any] | None, model: str) -> bool:
    if "moe" in model.lower() or "mixtral" in model.lower():
        return True
    if not isinstance(model_config, Mapping):
        return False
    for key in ("num_local_experts", "num_experts", "num_experts_per_tok"):
        value = model_config.get(key)
        if isinstance(value, (int, float)) and float(value) > 1.0:
            return True
    return False


def _size_bucket(model_size_b: float | None) -> str | None:
    if model_size_b is None:
        return None
    if model_size_b < 1.0:
        return "tiny"
    if 1.0 <= model_size_b <= 2.0:
        return "small"
    if 3.0 <= model_size_b <= 5.0:
        return "mid"
    if 7.0 <= model_size_b <= 9.0:
        return "large"
    if 13.0 <= model_size_b <= 14.0:
        return "xlarge"
    return None


def _resolve_general_path(
    raw_path: str,
    *,
    run_root: Path,
    manifest_path: Path | None,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        run_root / path,
        run_root.parent / path,
        run_root.parent.parent / path,
    ]
    if manifest_path is not None:
        candidates.append(manifest_path.parent / path)
        if len(manifest_path.parents) >= 2:
            candidates.append(manifest_path.parents[1] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _resolve_trace_trial_dir(*, result_dir: Path, trial_dir_value: str) -> Path:
    trial_dir = Path(trial_dir_value)
    if trial_dir.is_absolute():
        return trial_dir.resolve()
    if trial_dir.exists():
        return trial_dir.resolve()
    return (result_dir / trial_dir).resolve()


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping in {path}")
    return payload


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field_name} must be a mapping")
    return value


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_numeric(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field_name} must be numeric when provided")
    return float(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list when provided")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(f"{field_name} entries must be strings")
        result.append(item)
    return result


def _format_rate(rate: float) -> str:
    text = f"{rate:.4f}"
    return text.rstrip("0").rstrip(".")
