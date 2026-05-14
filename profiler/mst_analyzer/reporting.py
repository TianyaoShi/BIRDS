from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

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
    if len(set(extracted.source_manifest_paths)) > 1:
        raise RuntimeError(
            "suggested rerun manifest emission requires one source manifest; "
            "run the analyzer per orchestrator root when aggregating different workloads"
        )
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
    run_payload = _ensure_mapping(manifest_payload, "run")
    original_run_id = str(run_payload.get("run_id") or extracted.run_id)
    run_payload["run_id"] = f"{original_run_id}-mst-anomaly-rerun"

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

    filtered_experiments: list[dict[str, Any]] = []
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

        matched_rows = [
            row
            for row in extracted.rows
            if extracted.expanded_jobs[row.experiment_id].source_index == source_index
            if row.experiment_id in selected_experiment_set
            and row.model in selected_model_set
            and row.workload_path in selected_workload_set
            and row.model in models
            and _manifest_workload_matches(
                workload_path=row.workload_path,
                manifest_workloads=workloads,
                manifest_path=extracted.manifest_path,
            )
        ]
        if not matched_rows:
            continue

        selected_experiment_models = [model for model in models if model in selected_model_set]
        if not selected_experiment_models:
            continue

        selected_experiment_workloads = []
        for row in matched_rows:
            copied = workload_copy_map[row.workload_path]
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

        experiment.pop("model", None)
        experiment["models"] = selected_experiment_models
        experiment.pop("workload", None)
        experiment["workloads"] = selected_experiment_workloads
        filtered_experiments.append(experiment)

    manifest_payload["experiments"] = filtered_experiments
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
    new_stem = f"{workload_path.stem}_mst_anomaly_rerun"
    output_path = output_dir / f"{new_stem}{workload_path.suffix or '.yaml'}"
    if isinstance(payload, Mapping):
        updated = dict(payload)
        name = updated.get("name")
        updated["name"] = f"{name or workload_path.stem}_mst_anomaly_rerun"
        output_path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
        return output_path
    output_path.write_text(workload_path.read_text(encoding="utf-8"), encoding="utf-8")
    return output_path


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
