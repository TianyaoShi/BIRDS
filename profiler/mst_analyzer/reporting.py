from __future__ import annotations

import copy
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from local_orchestrator.manifest import load_manifest
from local_orchestrator.models import ExpandedExperimentJob
from local_orchestrator.utils import slugify

from .config import AnalyzerSettings
from .extract import ExtractedRun, extract_run, extract_runs
from .models import AnalysisArtifacts, AnomalyCandidate, BucketSummary, SuggestedRerunPlan, TraceDiagnostic
from .rules import analyze_rows_with_diagnostics


def analyze_orchestrator_run(
    *,
    orchestrator_run_root: str | Path,
    output_dir: str | Path,
    max_rerun_models: int | None = None,
    emit_rerun_manifest: bool = False,
    settings: AnalyzerSettings | None = None,
) -> AnalysisArtifacts:
    extracted = extract_run(orchestrator_run_root)
    return _analyze_extracted_run(
        extracted=extracted,
        output_dir=output_dir,
        max_rerun_models=max_rerun_models,
        emit_rerun_manifest=emit_rerun_manifest,
        settings=settings,
    )


def analyze_orchestrator_runs(
    *,
    orchestrator_run_roots: Sequence[str | Path],
    output_dir: str | Path,
    max_rerun_models: int | None = None,
    emit_rerun_manifest: bool = False,
    settings: AnalyzerSettings | None = None,
) -> AnalysisArtifacts:
    extracted = extract_runs(tuple(orchestrator_run_roots))
    return _analyze_extracted_run(
        extracted=extracted,
        output_dir=output_dir,
        max_rerun_models=max_rerun_models,
        emit_rerun_manifest=emit_rerun_manifest,
        settings=settings,
    )


def _analyze_extracted_run(
    *,
    extracted: ExtractedRun,
    output_dir: str | Path,
    max_rerun_models: int | None,
    emit_rerun_manifest: bool,
    settings: AnalyzerSettings | None,
) -> AnalysisArtifacts:
    resolved_settings = settings or AnalyzerSettings()
    anomalies, buckets, trace_diagnostics = analyze_rows_with_diagnostics(
        extracted.rows,
        settings=resolved_settings,
    )
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rows_json_path = resolved_output_dir / "mst_rows.json"
    rows_json_path.write_text(
        json.dumps([row.to_dict() for row in extracted.rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rerun_plan = None
    if emit_rerun_manifest:
        rerun_plan = _write_suggested_rerun_manifest(
            extracted=extracted,
            anomalies=anomalies,
            output_dir=resolved_output_dir,
            max_rerun_models=max_rerun_models,
        )

    report_payload = _build_report_payload(
        extracted=extracted,
        anomalies=anomalies,
        buckets=buckets,
        trace_diagnostics=trace_diagnostics,
        rows_json_path=rows_json_path,
        rerun_plan=rerun_plan,
        settings=resolved_settings,
    )
    report_json_path = resolved_output_dir / "mst_anomaly_report.json"
    report_md_path = resolved_output_dir / "mst_anomaly_report.md"
    report_json_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md_path.write_text(_render_markdown(report_payload), encoding="utf-8")
    return AnalysisArtifacts(
        rows_json_path=rows_json_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
        rerun_manifest_path=(None if rerun_plan is None else rerun_plan.manifest_path),
        row_count=len(extracted.rows),
        anomaly_count=len(anomalies),
        trace_diagnostic_count=len(trace_diagnostics),
    )


def _build_report_payload(
    *,
    extracted: ExtractedRun,
    anomalies: Sequence[AnomalyCandidate],
    buckets: Sequence[BucketSummary],
    trace_diagnostics: Sequence[TraceDiagnostic],
    rows_json_path: Path,
    rerun_plan: SuggestedRerunPlan | None,
    settings: AnalyzerSettings,
) -> dict[str, Any]:
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for anomaly in anomalies:
        severity_counts[anomaly.severity] += 1
    return {
        "run": {
            "run_id": extracted.run_id,
            "orchestrator_run_root": str(extracted.run_root),
            "manifest_path": str(extracted.manifest_path),
        },
        "summary": {
            "row_count": len(extracted.rows),
            "anomaly_count": len(anomalies),
            "trace_diagnostic_count": len(trace_diagnostics),
            "severity_counts": severity_counts,
            "rows_json_path": str(rows_json_path),
        },
        "analysis_config": settings.to_dict(),
        "buckets": [bucket.to_dict() for bucket in buckets],
        "anomalies": [anomaly.to_dict() for anomaly in anomalies],
        "trace_diagnostics": [diagnostic.to_dict() for diagnostic in trace_diagnostics],
        "suggested_rerun": None if rerun_plan is None else rerun_plan.to_dict(),
    }


def _render_markdown(report_payload: Mapping[str, Any]) -> str:
    run = _require_mapping(report_payload.get("run"), "report.run")
    summary = _require_mapping(report_payload.get("summary"), "report.summary")
    analysis_config = _require_mapping(report_payload.get("analysis_config"), "report.analysis_config")
    anomalies = _require_list(report_payload.get("anomalies"), "report.anomalies")
    trace_diagnostics = _require_list(report_payload.get("trace_diagnostics"), "report.trace_diagnostics")
    buckets = _require_list(report_payload.get("buckets"), "report.buckets")
    rerun = report_payload.get("suggested_rerun")

    lines = [
        f"# MST Anomaly Report: {run['run_id']}",
        "",
        f"- Source run root: `{run['orchestrator_run_root']}`",
        f"- Manifest: `{run['manifest_path']}`",
        f"- Rows analyzed: {summary['row_count']}",
        f"- Anomaly candidates: {summary['anomaly_count']}",
        f"- Trace diagnostics: {summary['trace_diagnostic_count']}",
        f"- Rows JSON: `{summary['rows_json_path']}`",
        f"- Disabled families: {', '.join(analysis_config['suppressions']['disable_families']) or '-'}",
        "",
        "## Summary",
        "",
        "| Severity | Model | MST (rps) | Families | Comparator(s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for anomaly in anomalies:
        anomaly_mapping = _require_mapping(anomaly, "report.anomalies[]")
        comparators = _require_list(anomaly_mapping.get("comparators"), "report.anomalies[].comparators")
        comparator_text = ", ".join(
            _display_model(_require_mapping(comparator, "report.anomalies[].comparators[]"))
            for comparator in comparators[:3]
        ) or "-"
        lines.append(
            "| "
            f"{anomaly_mapping['severity']} ({anomaly_mapping['severity_score']}) | "
            f"{_display_model(anomaly_mapping)} | "
            f"{anomaly_mapping['mst_rps']:.2f} | "
            f"{', '.join(anomaly_mapping['families'])} | "
            f"{comparator_text} |"
        )

    lines.extend(
        [
            "",
            "## Model-Size Buckets",
            "",
            "| Bucket | Comparable Group | Models | Median MST | Median Tok/s | Members |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for bucket in buckets:
        bucket_mapping = _require_mapping(bucket, "report.buckets[]")
        members = bucket_mapping.get("member_labels") or bucket_mapping["models"]
        lines.append(
            "| "
            f"{bucket_mapping['bucket_name']} | "
            f"{bucket_mapping['comparable_group']} | "
            f"{bucket_mapping['model_count']} | "
            f"{bucket_mapping['median_mst_rps']:.2f} | "
            f"{_format_optional_float(bucket_mapping.get('median_total_token_throughput'))} | "
            f"{', '.join(members)} |"
        )

    lines.extend(
        [
            "",
            "## Larger-Model Inversions",
            "",
            "| Model | MST (rps) | Comparator | Comparator MST | Label | Note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for anomaly in anomalies:
        anomaly_mapping = _require_mapping(anomaly, "report.anomalies[]")
        families = set(anomaly_mapping["families"])
        if "larger_model_inversion" not in families:
            continue
        comparator = _first_comparator_of_relation(anomaly_mapping, "larger_model")
        if comparator is None:
            continue
        lines.append(
            "| "
            f"{_display_model(anomaly_mapping)} | "
            f"{anomaly_mapping['mst_rps']:.2f} | "
            f"{_display_model(comparator)} | "
            f"{comparator['mst_rps']:.2f} | "
            f"{comparator['comparison_label']} | "
            f"{comparator['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Trace Instability On Anomalies",
            "",
            "| Model | MST (rps) | Confidence | Confirmation | High Bound | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for anomaly in anomalies:
        anomaly_mapping = _require_mapping(anomaly, "report.anomalies[]")
        families = set(anomaly_mapping["families"])
        if "trace_instability_suspect" not in families:
            continue
        family_reasons = _require_mapping(
            anomaly_mapping.get("family_reasons"),
            "report.anomalies[].family_reasons",
        )
        trace_reasons = _require_list(
            family_reasons.get("trace_instability_suspect", []),
            "report.anomalies[].family_reasons.trace_instability_suspect",
        )
        reason = "; ".join(trace_reasons[:2])
        lines.append(
            "| "
            f"{_display_model(anomaly_mapping)} | "
            f"{anomaly_mapping['mst_rps']:.2f} | "
            f"{anomaly_mapping.get('confidence') or '-'} | "
            f"{anomaly_mapping.get('confirmation_trial_id') or '-'} | "
            f"{anomaly_mapping.get('high_bound_trial_id') or '-'} | "
            f"{reason} |"
        )

    lines.extend(
        [
            "",
            "## Trace-Only Diagnostics",
            "",
            "| Model | MST (rps) | Confidence | Confirmation | High Bound | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for diagnostic in trace_diagnostics:
        diagnostic_mapping = _require_mapping(diagnostic, "report.trace_diagnostics[]")
        reason_list = _require_list(diagnostic_mapping.get("reasons"), "report.trace_diagnostics[].reasons")
        lines.append(
            "| "
            f"{_display_model(diagnostic_mapping)} | "
            f"{_format_optional_float(diagnostic_mapping.get('mst_rps'))} | "
            f"{diagnostic_mapping.get('confidence') or '-'} | "
            f"{diagnostic_mapping.get('confirmation_trial_id') or '-'} | "
            f"{diagnostic_mapping.get('high_bound_trial_id') or '-'} | "
            f"{'; '.join(reason_list[:2]) or '-'} |"
        )

    lines.extend(["", "## Findings", ""])
    for anomaly in anomalies:
        anomaly_mapping = _require_mapping(anomaly, "report.anomalies[]")
        lines.append(f"### {anomaly_mapping['severity'].title()}: {_display_model(anomaly_mapping)}")
        lines.append("")
        lines.append(f"- Summary: {anomaly_mapping['summary']}")
        lines.append(f"- Suggested action: {anomaly_mapping['suggested_action']}")
        lines.append(
            f"- Trials: confirmation={anomaly_mapping.get('confirmation_trial_id') or '-'}, "
            f"high_bound={anomaly_mapping.get('high_bound_trial_id') or '-'}"
        )
        for comparator in anomaly_mapping["comparators"]:
            comparator_mapping = _require_mapping(comparator, "report.anomalies[].comparators[]")
            lines.append(
                "- Comparator: "
                f"{_display_model(comparator_mapping)} ({comparator_mapping['mst_rps']:.2f} rps, "
                f"{comparator_mapping['comparison_label']}, "
                f"ratio={comparator_mapping['rate_ratio_vs_comparator']:.2f}, "
                f"delta={comparator_mapping['absolute_delta_rps']:.2f} rps)"
            )
        lines.append(
            "- Paths: "
            + ", ".join(f"`{path}`" for path in anomaly_mapping["evidence_paths"][:6])
        )
        lines.append("")

    lines.extend(["## Suggested Reruns", ""])
    if rerun is None:
        lines.append("- No rerun manifest was emitted.")
    else:
        rerun_mapping = _require_mapping(rerun, "report.suggested_rerun")
        lines.append(f"- Manifest: `{rerun_mapping['manifest_path']}`")
        lines.append(f"- Models: {', '.join(rerun_mapping['selected_models'])}")
        lines.append(
            "- Workload copies: " + ", ".join(f"`{path}`" for path in rerun_mapping["workload_copies"])
        )
        if rerun_mapping.get("truncated"):
            lines.append("- Model selection was truncated to stay within the requested rerun size cap.")
    lines.append("")
    return "\n".join(lines)


def _write_suggested_rerun_manifest(
    *,
    extracted: ExtractedRun,
    anomalies: Sequence[AnomalyCandidate],
    output_dir: Path,
    max_rerun_models: int | None,
) -> SuggestedRerunPlan | None:
    selected_models: list[str] = []
    selected_experiment_ids: list[str] = []
    selected_workloads: list[Path] = []
    truncated = False

    rows_by_experiment_id = {row.experiment_id: row for row in extracted.rows}
    for anomaly in anomalies:
        row = rows_by_experiment_id.get(anomaly.experiment_id)
        if row is None:
            continue
        required = [row.model]
        required_experiments = [row.experiment_id]
        required_workloads = [row.workload_path]
        for comparator in anomaly.comparators:
            if comparator.relation == "same_size_peer" and comparator.model not in required:
                required.append(comparator.model)
            if comparator.relation == "larger_model" and comparator.model not in required:
                required.append(comparator.model)
            if comparator.experiment_id not in required_experiments:
                required_experiments.append(comparator.experiment_id)

        would_add = [
            model
            for model in required
            if model not in selected_models
        ]
        if max_rerun_models is not None and selected_models and len(selected_models) + len(would_add) > max_rerun_models:
            truncated = True
            continue
        for model in required:
            if model not in selected_models:
                selected_models.append(model)
        for experiment_id in required_experiments:
            if experiment_id not in selected_experiment_ids:
                selected_experiment_ids.append(experiment_id)
        for workload_path in required_workloads:
            if workload_path not in selected_workloads:
                selected_workloads.append(workload_path)

    if not selected_models:
        return None

    manifest_payload = copy.deepcopy(extracted.manifest_payload)
    base_manifest = load_manifest(extracted.manifest_path)
    run_payload = _ensure_mapping(manifest_payload, "run")
    original_run_id = str(run_payload.get("run_id") or extracted.run_id)
    rerun_id = f"{original_run_id}-mst-anomaly-rerun"
    run_payload["run_id"] = rerun_id
    run_payload["output_root"] = str(base_manifest.run.output_root)
    run_payload["mst_output_root"] = str(base_manifest.run.output_root.parent / "mst" / rerun_id)

    search_payload = _ensure_mapping(manifest_payload, "search")
    search_payload["trial_min_duration_s"] = max(_numeric_or_default(search_payload.get("trial_min_duration_s"), 0.0), 180.0)
    search_payload["trial_max_duration_s"] = max(_numeric_or_default(search_payload.get("trial_max_duration_s"), 0.0), 300.0)
    search_payload["final_confirmation_duration_s"] = max(
        _numeric_or_default(search_payload.get("final_confirmation_duration_s"), 0.0),
        300.0,
    )

    workload_copy_map: dict[Path, Path] = {}
    for workload_path in selected_workloads:
        copied = _copy_workload_file(workload_path=workload_path, output_dir=output_dir)
        workload_copy_map[workload_path] = copied

    selected_model_set = set(selected_models)
    selected_experiment_set = set(selected_experiment_ids)
    selected_workload_set = set(selected_workloads)
    experiments = manifest_payload.get("experiments")
    if not isinstance(experiments, list):
        raise RuntimeError("manifest.experiments must be a list")
    jobs_by_source_index: dict[int, list[ExpandedExperimentJob]] = {}
    for job in extracted.expanded_jobs.values():
        jobs_by_source_index.setdefault(job.source_index, []).append(job)

    filtered_experiments: list[dict[str, Any]] = []
    emitted_experiment_ids: set[str] = set()
    for source_index, raw_experiment in enumerate(experiments):
        if not isinstance(raw_experiment, Mapping):
            raise RuntimeError("manifest experiments[] entries must be mappings")
        experiment = dict(raw_experiment)
        models = []
        for key in ("models", "model"):
            raw_models = experiment.get(key)
            if isinstance(raw_models, list):
                models.extend(str(item) for item in raw_models)
            elif isinstance(raw_models, str):
                models.append(raw_models)
        workloads = []
        for key in ("workloads", "workload"):
            raw_workloads = experiment.get(key)
            if isinstance(raw_workloads, list):
                workloads.extend(str(item) for item in raw_workloads)
            elif isinstance(raw_workloads, str):
                workloads.append(raw_workloads)

        source_jobs = _candidate_jobs_for_manifest_experiment(
            jobs=jobs_by_source_index.get(source_index, ()),
            experiment=experiment,
            models=models,
            workloads=workloads,
            manifest_path=extracted.manifest_path,
        )
        if not source_jobs and isinstance(experiment.get("launch"), Mapping):
            source_jobs = _candidate_jobs_for_manifest_experiment(
                jobs=extracted.expanded_jobs.values(),
                experiment=experiment,
                models=models,
                workloads=workloads,
                manifest_path=extracted.manifest_path,
            )
        matched_rows = []
        matched_base_jobs = []
        for base_job in source_jobs:
            if base_job.model not in selected_model_set:
                continue
            if base_job.model not in models:
                continue
            if not _manifest_workload_matches(
                workload_path=base_job.workload,
                manifest_workloads=workloads,
                manifest_path=extracted.manifest_path,
            ):
                continue
            row = _selected_row_for_base_job(
                base_job=base_job,
                rows=extracted.rows,
                selected_experiment_ids=selected_experiment_set,
                selected_workloads=selected_workload_set,
            )
            if row is None:
                continue
            matched_base_jobs.append(base_job)
            matched_rows.append(row)
        if not matched_rows:
            continue

        selected_experiment_models = []
        for base_job in matched_base_jobs:
            if base_job.model not in selected_experiment_models:
                selected_experiment_models.append(base_job.model)
        if not selected_experiment_models:
            continue

        selected_experiment_workloads = []
        for base_job in matched_base_jobs:
            if base_job.workload not in workload_copy_map:
                workload_copy_map[base_job.workload] = _copy_workload_file(
                    workload_path=base_job.workload,
                    output_dir=output_dir,
                )
            copied = workload_copy_map[base_job.workload]
            relative = copied.name
            if relative not in selected_experiment_workloads:
                selected_experiment_workloads.append(relative)

        rate_limited_search = _rerun_search_for_rate_limited_rows(
            matched_rows=matched_rows,
            extracted=extracted,
        )
        if rate_limited_search:
            search = dict(experiment.get("search") or {})
            search.update(rate_limited_search)
            experiment["search"] = search

        emitted_experiment_ids.update(row.experiment_id for row in matched_rows)
        experiment.pop("model", None)
        experiment["models"] = selected_experiment_models
        experiment.pop("workload", None)
        experiment["workloads"] = selected_experiment_workloads
        filtered_experiments.append(experiment)

    missing_experiment_ids = set(selected_experiment_ids) - emitted_experiment_ids
    if missing_experiment_ids:
        selected_experiment_ids = [
            experiment_id
            for experiment_id in selected_experiment_ids
            if experiment_id in emitted_experiment_ids
        ]
    if not filtered_experiments:
        return None

    manifest_payload["experiments"] = filtered_experiments
    _strip_shadowed_search_keys_from_overrides(manifest_payload)
    manifest_path = output_dir / "suggested_rerun_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding="utf-8")
    return SuggestedRerunPlan(
        manifest_path=manifest_path,
        selected_models=tuple(selected_models),
        selected_experiment_ids=tuple(selected_experiment_ids),
        selected_workloads=tuple(selected_workloads),
        workload_copies=tuple(workload_copy_map[path] for path in selected_workloads),
        truncated=truncated,
    )


def _selected_row_for_base_job(
    *,
    base_job: ExpandedExperimentJob,
    rows: Sequence[Any],
    selected_experiment_ids: set[str],
    selected_workloads: set[Path],
) -> Any | None:
    for row in rows:
        if row.experiment_id not in selected_experiment_ids:
            continue
        if _row_matches_base_job(row=row, base_job=base_job):
            return row
    return None


def _candidate_jobs_for_manifest_experiment(
    *,
    jobs: Iterable[ExpandedExperimentJob],
    experiment: Mapping[str, Any],
    models: Sequence[str],
    workloads: Sequence[str],
    manifest_path: Path,
) -> list[ExpandedExperimentJob]:
    candidates: list[ExpandedExperimentJob] = []
    for job in jobs:
        if models and job.model not in models:
            continue
        if workloads and not _manifest_workload_matches(
            workload_path=job.workload,
            manifest_workloads=workloads,
            manifest_path=manifest_path,
        ):
            continue
        if not _job_launch_matches_manifest_experiment(job=job, experiment=experiment):
            continue
        candidates.append(job)
    return candidates


def _job_launch_matches_manifest_experiment(
    *,
    job: ExpandedExperimentJob,
    experiment: Mapping[str, Any],
) -> bool:
    raw_launch = experiment.get("launch")
    if not isinstance(raw_launch, Mapping):
        return True
    launch = job.launch
    comparable = {
        "tensor_parallel_size": launch.tensor_parallel_size,
        "gpu_count": launch.gpu_count,
        "dtype": launch.dtype,
        "quantization": launch.quantization,
        "tokenizer_mode": launch.tokenizer_mode,
        "gpu_memory_utilization": launch.gpu_memory_utilization,
        "max_model_len": launch.max_model_len,
        "max_num_seqs": launch.max_num_seqs,
        "max_num_batched_tokens": launch.max_num_batched_tokens,
        "readiness_timeout_s": launch.readiness_timeout_s,
        "readiness_interval_s": launch.readiness_interval_s,
    }
    for key, actual in comparable.items():
        if key in raw_launch and not _manifest_scalar_matches(actual, raw_launch[key]):
            return False
    raw_env = raw_launch.get("env")
    if isinstance(raw_env, Mapping):
        for key, expected in raw_env.items():
            if str(launch.env.get(str(key))) != str(expected):
                return False
    return True


def _manifest_scalar_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-9
    return actual == expected


def _strip_shadowed_search_keys_from_overrides(manifest_payload: dict[str, Any]) -> None:
    experiments = manifest_payload.get("experiments")
    if not isinstance(experiments, list):
        return
    overrides = manifest_payload.get("overrides")
    if not isinstance(overrides, list):
        return

    for raw_override in overrides:
        if not isinstance(raw_override, dict):
            continue
        search = raw_override.get("search")
        if not isinstance(search, dict):
            continue
        shadowed_keys: set[str] = set()
        for raw_experiment in experiments:
            if not isinstance(raw_experiment, Mapping):
                continue
            experiment_search = raw_experiment.get("search")
            if not isinstance(experiment_search, Mapping):
                continue
            if _raw_override_matches_experiment(raw_override, raw_experiment):
                shadowed_keys.update(str(key) for key in experiment_search)
        for key in shadowed_keys:
            search.pop(key, None)
        if not search:
            raw_override.pop("search", None)


def _raw_override_matches_experiment(raw_override: Mapping[str, Any], raw_experiment: Mapping[str, Any]) -> bool:
    match = raw_override.get("match")
    if not isinstance(match, Mapping):
        return True
    models = _raw_string_values(raw_experiment.get("models")) + _raw_string_values(raw_experiment.get("model"))
    workloads = _raw_string_values(raw_experiment.get("workloads")) + _raw_string_values(raw_experiment.get("workload"))
    model_patterns = _raw_string_values(match.get("models")) + _raw_string_values(match.get("model"))
    workload_patterns = _raw_string_values(match.get("workloads")) + _raw_string_values(match.get("workload"))
    if model_patterns and not any(_matches_any_pattern(model, model_patterns) for model in models):
        return False
    if workload_patterns and not any(_matches_any_pattern(workload, workload_patterns) for workload in workloads):
        return False
    return True


def _raw_string_values(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str)]
    return []


def _matches_any_pattern(value: str, patterns: Sequence[str]) -> bool:
    lowered = value.lower()
    return any(fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def _row_matches_base_job(*, row: Any, base_job: ExpandedExperimentJob) -> bool:
    return (
        row.model == base_job.model
        and row.endpoint == base_job.endpoint
        and row.server_signature_key == base_job.server_signature_key
        and _same_workload_for_rerun(row=row, workload_path=base_job.workload)
    )


def _same_workload_for_rerun(*, row: Any, workload_path: Path) -> bool:
    if row.workload_path == workload_path:
        return True
    return row.workload_name == _workload_name(workload_path)


def _workload_name(workload_path: Path) -> str:
    try:
        payload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    except OSError:
        return workload_path.stem
    if isinstance(payload, Mapping) and isinstance(payload.get("name"), str) and payload["name"]:
        return str(payload["name"])
    return workload_path.stem


def _rerun_search_for_rate_limited_rows(
    *,
    matched_rows: Sequence[Any],
    extracted: ExtractedRun,
) -> dict[str, Any]:
    caps: list[float] = []
    initial_rates: list[float] = []
    for row in matched_rows:
        if row.termination_reason != "max_request_rate_limited" or row.mst_rps is None:
            continue
        job = extracted.expanded_jobs[row.experiment_id]
        current_cap = job.search.max_request_rate
        base = current_cap if current_cap is not None else row.mst_rps
        caps.append(max(float(base) * 2.0, float(row.mst_rps) * 2.0))
        initial_rates.append(max(float(base) * 0.25, job.search.initial_request_rate))
    if not caps:
        return {}
    return {
        "search_mode": "open-loop",
        "initial_request_rate": float(min(initial_rates)),
        "max_request_rate": float(max(caps)),
    }


def _copy_workload_file(*, workload_path: Path, output_dir: Path) -> Path:
    payload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    workload_name = _workload_name(workload_path)
    new_stem = f"{slugify(workload_name, max_length=80)}_mst_anomaly_rerun"
    output_path = output_dir / f"{new_stem}{workload_path.suffix or '.yaml'}"
    if isinstance(payload, Mapping):
        updated = dict(payload)
        name = updated.get("name")
        updated["name"] = f"{name or workload_path.stem}_mst_anomaly_rerun"
        _rewrite_copied_workload_paths(
            workload_payload=updated,
            source_workload_path=workload_path,
            output_workload_path=output_path,
        )
        output_path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
        return output_path
    output_path.write_text(workload_path.read_text(encoding="utf-8"), encoding="utf-8")
    return output_path


def _rewrite_copied_workload_paths(
    *,
    workload_payload: dict[Any, Any],
    source_workload_path: Path,
    output_workload_path: Path,
) -> None:
    dataset = workload_payload.get("dataset")
    if not isinstance(dataset, dict):
        return
    raw_path = dataset.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    dataset_type = dataset.get("type")
    if dataset_type == "hf":
        return
    raw_path_obj = Path(raw_path).expanduser()
    if raw_path_obj.is_absolute():
        return
    source_path = (source_workload_path.parent / raw_path).resolve()
    if dataset_type == "longbench" and not (source_path.exists() or raw_path.endswith(".zip") or raw_path.startswith(".")):
        return
    relative = os.path.relpath(source_path, start=output_workload_path.parent.resolve())
    dataset["path"] = relative


def _manifest_workload_matches(
    *,
    workload_path: Path,
    manifest_workloads: Sequence[str],
    manifest_path: Path,
) -> bool:
    if not manifest_workloads:
        return True
    candidates = {
        str(workload_path.resolve()),
        str(workload_path),
        workload_path.name,
        workload_path.stem,
    }
    for raw_workload in manifest_workloads:
        resolved = Path(raw_workload)
        if not resolved.is_absolute():
            resolved = (manifest_path.parent / resolved).resolve()
        raw_candidates = {
            raw_workload,
            str(resolved),
            Path(raw_workload).name,
            Path(raw_workload).stem,
        }
        if candidates & raw_candidates:
            return True
    return False


def _first_comparator_of_relation(anomaly: Mapping[str, Any], relation: str) -> Mapping[str, Any] | None:
    comparators = _require_list(anomaly.get("comparators"), "anomaly.comparators")
    for comparator in comparators:
        comparator_mapping = _require_mapping(comparator, "anomaly.comparators[]")
        if comparator_mapping.get("relation") == relation:
            return comparator_mapping
    return None


def _ensure_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        payload[key] = {}
        value = payload[key]
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest.{key} must be a mapping")
    return value


def _numeric_or_default(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field_name} must be a mapping")
    return value


def _require_list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list")
    return value


def _format_optional_float(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.2f}"


def _display_model(item: Mapping[str, Any]) -> str:
    model = str(item.get("model") or "unknown")
    serving_label = item.get("serving_config_label")
    if not isinstance(serving_label, str) or not serving_label:
        tp = item.get("tensor_parallel_size")
        gpu_count = item.get("gpu_count")
        parts = []
        if tp is not None:
            parts.append(f"tp={tp}")
        if gpu_count is not None:
            parts.append(f"gpus={gpu_count}")
        serving_label = ", ".join(parts)
    if serving_label:
        return f"{model} ({serving_label})"
    return model
