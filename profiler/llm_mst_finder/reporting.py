from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .records import BenchmarkMetrics, BottleneckResult, StabilityResult, TrialAnalysisResult, TrialSummary
from .stability import StabilityConfig, load_window_summaries_csv

try:
    from .plotting import plot_result_comparison, plot_search_results, plot_trial_windows
except ModuleNotFoundError as exc:
    if exc.name not in {"matplotlib", "matplotlib.pyplot"}:
        raise

    def _missing_plotting_dependency(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "plotting requires matplotlib, which is not installed in this environment"
        ) from exc

    plot_result_comparison = _missing_plotting_dependency
    plot_search_results = _missing_plotting_dependency
    plot_trial_windows = _missing_plotting_dependency


def generate_report(
    result_dir: str | Path,
    *,
    compare_result_dirs: Sequence[str | Path] = (),
    plots_enabled: bool = True,
) -> dict[str, object]:
    bundle = _load_result_bundle(Path(result_dir))
    comparison_payload = _build_comparison_payload(
        primary_bundle=bundle,
        compare_result_dirs=compare_result_dirs,
        plots_enabled=plots_enabled,
    )
    report_payload = _build_report_payload(bundle, comparison_payload=comparison_payload)

    if plots_enabled:
        _generate_trial_plots(bundle)
        search_plot_paths = _generate_search_plots(bundle)
        if comparison_payload is not None:
            comparison_plot_paths = _generate_comparison_plots(bundle["result_dir"], comparison_payload)
            comparison_payload["plots"] = comparison_plot_paths
        else:
            comparison_plot_paths = None
        _require_plot_files(bundle, search_plot_paths, comparison_plot_paths)

    final_json_path = Path(bundle["result_dir"]) / "final_report.json"
    final_md_path = Path(bundle["result_dir"]) / "final_report.md"
    final_json_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_md_path.write_text(_render_markdown(report_payload), encoding="utf-8")
    return {
        "final_report_json": str(final_json_path),
        "final_report_md": str(final_md_path),
        "comparison_included": comparison_payload is not None,
    }


def _load_result_bundle(result_dir: Path) -> dict[str, object]:
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result directory not found: {result_dir}")
    trace_path = result_dir / "search_trace.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"result directory is missing search_trace.json: {trace_path}")
    trace_payload = _load_json_mapping(trace_path)
    result_payload = _require_mapping(trace_payload.get("result"), "search_trace.json.result")
    config_payload = _require_mapping(trace_payload.get("config"), "search_trace.json.config")
    raw_events = trace_payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("search_trace.json.events must be a non-empty array")

    trials: list[dict[str, object]] = []
    server_config: dict[str, object] = {}
    for event_idx, raw_event in enumerate(raw_events, start=1):
        event = _require_mapping(raw_event, f"search_trace.json.events[{event_idx}]")
        trial_dir_value = event.get("trial_dir")
        if not isinstance(trial_dir_value, str):
            raise ValueError(f"search_trace.json.events[{event_idx}].trial_dir must be a string")
        trial_dir = _resolve_trace_trial_dir(result_dir, trial_dir_value)
        summary_payload = _load_json_mapping(trial_dir / "summary.json")
        summary = _load_trial_summary_from_mapping(
            _require_mapping(summary_payload.get("summary"), "summary.json.summary")
        )
        analysis = _load_trial_analysis(trial_dir / "analysis.json")
        trace_analysis = _load_trial_analysis_from_mapping(
            _require_mapping(event.get("analysis"), f"search_trace.json.events[{event_idx}].analysis")
        )
        _assert_analysis_matches_trace(analysis, trace_analysis, trial_dir=trial_dir)
        windows = load_window_summaries_csv(trial_dir / "windows.csv")
        trial_server_config = _load_server_config(summary_payload, trial_dir)
        server_config = _merge_server_config(server_config, trial_server_config)
        trials.append(
            {
                "event": dict(event),
                "trial_dir": trial_dir,
                "summary_payload": dict(summary_payload),
                "summary": summary,
                "analysis": analysis,
                "windows": windows,
                "server_config": trial_server_config,
            }
        )

    if not trials:
        raise ValueError("result bundle did not load any trials")
    return {
        "result_dir": result_dir,
        "trace_payload": dict(trace_payload),
        "trace_config": dict(config_payload),
        "search_result": dict(result_payload),
        "trials": trials,
        "server_config": server_config,
    }


def _resolve_trace_trial_dir(result_dir: Path, trial_dir_value: str) -> Path:
    trial_dir = Path(trial_dir_value)
    if trial_dir.is_absolute():
        return trial_dir
    if trial_dir.is_dir():
        return trial_dir
    return result_dir / trial_dir


def _build_report_payload(
    bundle: Mapping[str, object],
    *,
    comparison_payload: dict[str, object] | None,
) -> dict[str, object]:
    trace_payload = _require_mapping(bundle["trace_payload"], "bundle.trace_payload")
    trace_config = _require_mapping(bundle["trace_config"], "bundle.trace_config")
    search_result = _require_mapping(bundle["search_result"], "bundle.search_result")
    trials = _require_trial_list(bundle["trials"])
    workload = _extract_workload_payload(trials, trace_config)
    server_config = _build_server_metadata_payload(_require_mapping(bundle["server_config"], "bundle.server_config"))
    best_trial = _select_headline_trial(trials, search_result)
    bottleneck_payload = _build_bottleneck_payload(best_trial, search_result)
    search_trace_summary = _build_search_trace_summary(trace_payload, trace_config, trials)
    stability_boundary = _build_open_loop_boundary(trials)
    decision_context = _build_search_decision_context(trace_payload, trials)
    limitations = _build_limitations(trials, server_config, workload)
    recommended_action = _recommended_next_action(
        workload=workload,
        trials=trials,
        bottleneck=bottleneck_payload,
    )
    report_payload: dict[str, object] = {
        "result_dir": str(bundle["result_dir"]),
        "workload": workload,
        "stability_policy": _stability_policy_payload(trials),
        "server_config": server_config,
        "search_trace": search_trace_summary,
        "closed_loop": search_result.get("closed_loop"),
        "open_loop_stability_boundary": stability_boundary,
        "decision_context": decision_context,
        "search_result": {
            "search_id": trace_config.get("search_id"),
            "search_mode": trace_config.get("search_mode"),
            "max_no_drift_request_rate": search_result.get("max_no_drift_request_rate"),
            "max_slo_satisfying_request_rate": search_result.get("max_slo_satisfying_request_rate"),
            "rate_precision": search_result.get("rate_precision"),
            "confidence": search_result.get("confidence"),
            "confirmation_trial_id": search_result.get("confirmation_trial_id"),
            "reasons": list(search_result.get("reasons", [])),
        },
        "bottleneck": bottleneck_payload,
        "recommended_next_action": recommended_action,
        "limitations": limitations,
        "trials": [_trial_payload(trial) for trial in trials],
    }
    if comparison_payload is not None:
        report_payload["comparison"] = comparison_payload
    return report_payload


def _build_comparison_payload(
    *,
    primary_bundle: Mapping[str, object],
    compare_result_dirs: Sequence[str | Path],
    plots_enabled: bool,
) -> dict[str, object] | None:
    if not compare_result_dirs:
        return None
    comparison_rows = [_comparison_row_from_bundle(primary_bundle)]
    for raw_path in compare_result_dirs:
        comparison_rows.append(_comparison_row_from_report_dir(Path(raw_path)))
    _validate_comparability(comparison_rows)
    payload = {
        "comparable": True,
        "results": comparison_rows,
        "plots": {} if plots_enabled else None,
        "server_metadata_table": [
            {
                "label": row["label"],
                "server_config": row["server_config"],
            }
            for row in comparison_rows
        ],
    }
    return payload


def _comparison_row_from_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    report_payload = _build_report_payload(bundle, comparison_payload=None)
    search_result = _require_mapping(report_payload["search_result"], "report_payload.search_result")
    closed_loop = report_payload.get("closed_loop")
    if closed_loop is not None:
        closed_loop = _require_mapping(closed_loop, "report_payload.closed_loop")
    return {
        "label": Path(str(bundle["result_dir"])).name,
        "result_dir": str(bundle["result_dir"]),
        "workload": report_payload["workload"],
        "stability_policy": report_payload["stability_policy"],
        "search_trace": report_payload["search_trace"],
        "server_config": report_payload["server_config"],
        "search_result": report_payload["search_result"],
        "bottleneck": report_payload["bottleneck"],
        "max_sustainable_req_s": search_result.get("max_no_drift_request_rate"),
        "max_output_tok_s": _comparison_max_output_tok_s(report_payload, closed_loop),
        "bottleneck_class": _require_mapping(report_payload["bottleneck"], "report_payload.bottleneck").get("class"),
    }


def _comparison_row_from_report_dir(result_dir: Path) -> dict[str, object]:
    report_path = result_dir / "final_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"comparison result directory must already contain final_report.json: {report_path}"
        )
    payload = _load_json_mapping(report_path)
    workload = _require_mapping(payload.get("workload"), "final_report.json.workload")
    stability_policy = _require_mapping(payload.get("stability_policy"), "final_report.json.stability_policy")
    search_trace = _require_mapping(payload.get("search_trace"), "final_report.json.search_trace")
    search_result = _require_mapping(payload.get("search_result"), "final_report.json.search_result")
    server_config = _require_mapping(payload.get("server_config"), "final_report.json.server_config")
    bottleneck = _require_mapping(payload.get("bottleneck"), "final_report.json.bottleneck")
    closed_loop = payload.get("closed_loop")
    if closed_loop is not None:
        closed_loop = _require_mapping(closed_loop, "final_report.json.closed_loop")
    return {
        "label": result_dir.name,
        "result_dir": str(result_dir),
        "workload": dict(workload),
        "stability_policy": dict(stability_policy),
        "search_trace": dict(search_trace),
        "server_config": dict(server_config),
        "search_result": dict(search_result),
        "bottleneck": dict(bottleneck),
        "max_sustainable_req_s": search_result.get("max_no_drift_request_rate"),
        "max_output_tok_s": _comparison_max_output_tok_s(payload, closed_loop),
        "bottleneck_class": bottleneck.get("class"),
    }


def _validate_comparability(rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) < 2:
        return
    reference = rows[0]
    for candidate in rows[1:]:
        for key in ("workload", "stability_policy"):
            if candidate.get(key) != reference.get(key):
                raise ValueError(f"comparison requires matching {key} across all result directories")
        ref_search_trace = _require_mapping(reference["search_trace"], "comparison.search_trace.reference")
        cand_search_trace = _require_mapping(candidate["search_trace"], "comparison.search_trace.candidate")
        for key in ("search_mode", "trial_duration_s", "final_confirmation_duration_s", "rate_precision", "model"):
            if ref_search_trace.get(key) is None or cand_search_trace.get(key) is None:
                raise ValueError(f"comparison requires populated search_trace.{key} in every result directory")
            if cand_search_trace.get(key) != ref_search_trace.get(key):
                raise ValueError(f"comparison requires matching search_trace.{key} across all result directories")
        ref_server = _require_mapping(reference["server_config"], "comparison.server_config.reference")
        cand_server = _require_mapping(candidate["server_config"], "comparison.server_config.candidate")
        if not ref_server or not cand_server:
            raise ValueError("comparison requires declared server metadata in every result directory")


def _generate_trial_plots(bundle: Mapping[str, object]) -> None:
    for trial in _require_trial_list(bundle["trials"]):
        windows = trial["windows"]
        plot_trial_windows(
            trial_dir=trial["trial_dir"],
            x_values=[window.start_s for window in windows],
            arrival_rate=[window.arrival_rate for window in windows],
            completion_rate=[window.completion_rate for window in windows],
            outstanding=[window.outstanding_mean for window in windows],
            ttft_p50_ms=[window.ttft_p50_ms for window in windows],
            ttft_p90_ms=[window.ttft_p90_ms for window in windows],
            ttft_p95_ms=[window.ttft_p95_ms for window in windows],
            ttft_p99_ms=[window.ttft_p99_ms for window in windows],
            tpot_p50_ms=[window.tpot_p50_ms for window in windows],
            tpot_p90_ms=[window.tpot_p90_ms for window in windows],
            tpot_p95_ms=[window.tpot_p95_ms for window in windows],
            tpot_p99_ms=[window.tpot_p99_ms for window in windows],
            output_tok_s=[window.generation_tok_s for window in windows],
            kv_cache_usage=[window.kv_cache_usage_max for window in windows],
            num_running=[window.num_running_mean for window in windows],
            num_waiting=[window.num_waiting_mean for window in windows],
            num_swapped=[window.num_swapped_mean for window in windows],
        )


def _generate_search_plots(bundle: Mapping[str, object]) -> dict[str, str]:
    trials = [trial for trial in _require_trial_list(bundle["trials"]) if trial["summary"].mode == "open-loop"]
    if not trials:
        raise ValueError("report generation requires at least one open-loop trial for search plots")
    request_rates = [
        float(_require_mapping(trial["event"], "trial.event").get("request_rate"))
        for trial in trials
    ]
    classifications = [str(trial["analysis"].trial_validity) + "/" + _stability_label(trial["analysis"]) for trial in trials]
    ttft_p90 = [_max_window_value(trial["windows"], "ttft_p90_ms") for trial in trials]
    tpot_p90 = [_max_window_value(trial["windows"], "tpot_p90_ms") for trial in trials]
    output_tok_s = [trial["summary"].benchmark_metrics.generation_token_throughput for trial in trials]
    queue_drift = [_queue_drift(trial) for trial in trials]
    return plot_search_results(
        output_dir=bundle["result_dir"],
        request_rates=request_rates,
        classifications=classifications,
        ttft_p90_ms=ttft_p90,
        tpot_p90_ms=tpot_p90,
        output_tok_s=output_tok_s,
        queue_drift=queue_drift,
    )


def _generate_comparison_plots(result_dir: Path, comparison_payload: Mapping[str, object]) -> dict[str, str]:
    rows = comparison_payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison payload must contain non-empty results")
    labels = [str(_require_mapping(row, "comparison.results[]").get("label")) for row in rows]
    max_sustainable = [_require_numeric(_require_mapping(row, "comparison.results[]").get("max_sustainable_req_s"), "max_sustainable_req_s") for row in rows]
    max_output = [_require_numeric(_require_mapping(row, "comparison.results[]").get("max_output_tok_s"), "max_output_tok_s") for row in rows]
    bottleneck_classes = [str(_require_mapping(row, "comparison.results[]").get("bottleneck_class")) for row in rows]
    return plot_result_comparison(
        output_dir=result_dir,
        labels=labels,
        max_sustainable_req_s=max_sustainable,
        max_output_tok_s=max_output,
        bottleneck_classes=bottleneck_classes,
    )


def _require_plot_files(
    bundle: Mapping[str, object],
    search_plot_paths: Mapping[str, str],
    comparison_plot_paths: Mapping[str, str] | None,
) -> None:
    for trial in _require_trial_list(bundle["trials"]):
        plots_dir = Path(trial["trial_dir"]) / "plots"
        expected = (
            "arrival_vs_completion_rate.png",
            "outstanding_requests.png",
            "ttft_percentiles.png",
            "tpot_percentiles.png",
            "output_tokens_per_s.png",
            "kv_cache_usage.png",
            "server_queue_state.png",
        )
        for filename in expected:
            if not (plots_dir / filename).is_file():
                raise FileNotFoundError(f"missing required trial plot: {plots_dir / filename}")
    for path in search_plot_paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing required search plot: {path}")
    if comparison_plot_paths is not None:
        for path in comparison_plot_paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(f"missing required comparison plot: {path}")


def _render_markdown(payload: Mapping[str, object]) -> str:
    workload = _require_mapping(payload["workload"], "payload.workload")
    context_policy = _require_mapping(workload.get("context_policy"), "payload.workload.context_policy")
    search_trace = _require_mapping(payload["search_trace"], "payload.search_trace")
    decision_context = _require_mapping(payload["decision_context"], "payload.decision_context")
    search_result = _require_mapping(payload["search_result"], "payload.search_result")
    bottleneck = _require_mapping(payload["bottleneck"], "payload.bottleneck")
    server_config = _require_mapping(payload["server_config"], "payload.server_config")
    stability_policy = _require_mapping(payload["stability_policy"], "payload.stability_policy")
    lines = [
        "# LLM MST Finder Report",
        "",
        "## 1. Workload definition",
        f"- name: {workload.get('name')}",
        f"- dataset_type: {workload.get('dataset_type')}",
        f"- num_requests: {workload.get('num_requests')}",
        "",
        "## 2. Workload/model context compatibility summary",
        f"- tokenizer_source: {context_policy.get('tokenizer_source')}",
        f"- max_model_len: {context_policy.get('max_model_len')}",
        f"- over_limit: {context_policy.get('over_limit')}",
        f"- skipped_samples: {context_policy.get('skipped_samples')}",
        f"- truncated_samples: {context_policy.get('truncated_samples')}",
        "",
        "## 3. Server configuration",
        f"- model: {server_config.get('model')}",
        f"- max_num_seqs: {server_config.get('max_num_seqs')}",
        f"- max_num_batched_tokens: {server_config.get('max_num_batched_tokens')}",
        f"- scheduler_metadata_note: {server_config.get('scheduler_metadata_note')}",
        f"```json\n{json.dumps(server_config, indent=2, sort_keys=True)}\n```",
        "",
        "## 4. Search trace",
        f"- search_mode: {search_trace.get('search_mode')}",
        f"- total_trials: {search_trace.get('total_trials')}",
        f"- open_loop_trials: {search_trace.get('open_loop_trials')}",
        f"- closed_loop_trials: {search_trace.get('closed_loop_trials')}",
        f"- rate_precision: {search_trace.get('rate_precision')}",
        f"- final_low_rate: {search_trace.get('final_low_rate')}",
        f"- final_high_rate: {search_trace.get('final_high_rate')}",
        f"- final_relative_width: {search_trace.get('final_relative_width')}",
        f"- convergence_assessment: {search_trace.get('convergence_assessment')}",
        "",
        "## 5. Closed-loop scouting result",
        f"```json\n{json.dumps(payload.get('closed_loop'), indent=2, sort_keys=True)}\n```",
        "",
        "## 6. Open-loop stability boundary",
        f"```json\n{json.dumps(payload.get('open_loop_stability_boundary'), indent=2, sort_keys=True)}\n```",
        "",
        "## 7. Stability SLOs and decision basis",
        f"- ttft_slo_ms: {stability_policy.get('ttft_slo_ms')}",
        f"- ttft_slo_field: {stability_policy.get('ttft_slo_field')}",
        f"- tpot_slo_ms: {stability_policy.get('tpot_slo_ms')}",
        f"- tpot_slo_field: {stability_policy.get('tpot_slo_field')}",
        f"- decision_subject: {decision_context.get('subject')}",
        f"- decision_trial_id: {decision_context.get('trial_id')}",
        f"- decision_trial_rate: {decision_context.get('request_rate')}",
        f"- decision_stability_status: {decision_context.get('stability_status')}",
        f"- decision_reasoning: {decision_context.get('decision_reasoning')}",
        f"- decision_reason_summary: {decision_context.get('reason_summary')}",
        "",
        "## 8. Max no-drift request rate",
        f"- {search_result.get('max_no_drift_request_rate')}",
        "",
        "## 9. Max SLO-satisfying request rate",
        f"- {search_result.get('max_slo_satisfying_request_rate')}",
        "",
        "## 10. Bottleneck diagnosis",
        f"- class: {bottleneck.get('class')}",
        f"- confidence: {bottleneck.get('confidence')}",
        f"- evidence: {json.dumps(bottleneck.get('evidence'), sort_keys=True)}",
        "",
        "## 11. Recommended next orchestration action",
        f"- {payload.get('recommended_next_action')}",
        "",
        "## 12. Limitations",
    ]
    for item in payload.get("limitations", []):
        lines.append(f"- {item}")
    if "comparison" in payload:
        lines.extend(
            [
                "",
                "## Comparison",
                f"```json\n{json.dumps(payload['comparison'], indent=2, sort_keys=True)}\n```",
            ]
        )
    return "\n".join(lines) + "\n"


def _trial_payload(trial: Mapping[str, object]) -> dict[str, object]:
    summary = trial["summary"]
    analysis = trial["analysis"]
    return {
        "trial_id": summary.trial_id,
        "mode": summary.mode,
        "status": summary.status,
        "request_rate": trial["event"].get("request_rate"),
        "concurrency": trial["event"].get("concurrency"),
        "trial_validity": analysis.trial_validity,
        "stability_status": None if analysis.stability is None else analysis.stability.status,
        "bottleneck_class": None if analysis.bottleneck is None else analysis.bottleneck.bottleneck_class,
        "confidence": None if analysis.bottleneck is None else analysis.bottleneck.confidence,
        "successful_completion_rate": summary.successful_completion_rate,
        "generation_token_throughput": summary.benchmark_metrics.generation_token_throughput,
        "prompt_length_summary": summary.benchmark_metrics.prompt_length_summary,
        "output_length_summary": summary.benchmark_metrics.output_length_summary,
        "validity_reasons": list(analysis.validity_reasons),
    }


def _build_bottleneck_payload(
    trial: Mapping[str, object],
    search_result: Mapping[str, object],
) -> dict[str, object]:
    analysis = trial["analysis"]
    bottleneck = analysis.bottleneck
    evidence = [] if bottleneck is None else list(bottleneck.evidence)
    return {
        "class": search_result.get("bottleneck_class") if search_result.get("bottleneck_class") is not None else (None if bottleneck is None else bottleneck.bottleneck_class),
        "confidence": search_result.get("confidence"),
        "evidence": evidence,
    }


def _build_search_trace_summary(
    trace_payload: Mapping[str, object],
    trace_config: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    open_loop_trials = [trial for trial in trials if trial["summary"].mode == "open-loop"]
    closed_loop_trials = [trial for trial in trials if trial["summary"].mode == "closed-loop"]
    bounds_payload = _require_mapping(trace_payload.get("bounds", {}), "search_trace.json.bounds")
    low_rate = _optional_numeric(bounds_payload.get("low_rate"), "search_trace.json.bounds.low_rate")
    high_rate = _optional_numeric(bounds_payload.get("high_rate"), "search_trace.json.bounds.high_rate")
    relative_width = _relative_width(low_rate=low_rate, high_rate=high_rate)
    precision = _optional_numeric(trace_config.get("rate_precision"), "search_trace.json.config.rate_precision")
    return {
        "total_trials": len(trials),
        "open_loop_trials": len(open_loop_trials),
        "closed_loop_trials": len(closed_loop_trials),
        "search_mode": trace_config.get("search_mode"),
        "trial_duration_s": trace_config.get("trial_duration_s"),
        "final_confirmation_duration_s": trace_config.get("final_confirmation_duration_s"),
        "rate_precision": precision,
        "model": trace_config.get("model"),
        "final_low_rate": low_rate,
        "final_low_trial_id": bounds_payload.get("low_trial_id"),
        "final_high_rate": high_rate,
        "final_high_trial_id": bounds_payload.get("high_trial_id"),
        "final_relative_width": relative_width,
        "convergence_assessment": _convergence_assessment(
            low_rate=low_rate,
            high_rate=high_rate,
            relative_width=relative_width,
            rate_precision=precision,
        ),
    }


def _build_open_loop_boundary(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stable_trials = []
    invalid_workload_trials = []
    for trial in trials:
        summary = trial["summary"]
        analysis = trial["analysis"]
        if summary.mode != "open-loop":
            continue
        if analysis.trial_validity == "invalid_workload":
            invalid_workload_trials.append(summary.trial_id)
            continue
        if analysis.trial_validity != "valid" or analysis.stability is None:
            continue
        stable_trials.append(
            {
                "trial_id": summary.trial_id,
                "request_rate": summary.requested_request_rate,
                "stability_status": analysis.stability.status,
            }
        )
    if not stable_trials:
        raise ValueError("open-loop boundary report requires at least one valid open-loop trial")
    max_stable = max(
        (trial for trial in stable_trials if trial["stability_status"] == "stable"),
        key=lambda item: float(item["request_rate"]),
        default=None,
    )
    if max_stable is None:
        raise ValueError("open-loop boundary report requires at least one stable open-loop trial")
    return {
        "max_stable_trial_id": max_stable["trial_id"],
        "max_stable_request_rate": max_stable["request_rate"],
        "invalid_workload_trial_ids": invalid_workload_trials,
    }


def _build_limitations(
    trials: Sequence[Mapping[str, object]],
    server_config: Mapping[str, object],
    workload: Mapping[str, object],
) -> list[str]:
    limitations: list[str] = []
    if not server_config:
        limitations.append("declared server metadata was not recorded in the result artifacts")
    if server_config.get("max_num_seqs") is None or server_config.get("max_num_batched_tokens") is None:
        limitations.append(
            "max_num_seqs and max_num_batched_tokens are explicit vLLM scheduler config values; "
            "they are not reliably inferred from runtime metrics when missing from saved metadata"
        )
    if any(trial["analysis"].trial_validity != "valid" for trial in trials):
        limitations.append("some trials were excluded from overload interpretation due to invalidity")
    context_policy = workload.get("context_policy")
    if isinstance(context_policy, Mapping) and context_policy.get("metadata_status") == "not_recorded":
        limitations.append(
            "workload context_policy metadata was not recorded; context compatibility summary is unavailable"
        )
    limitations.append("stability thresholds were inferred from the saved default StabilityConfig")
    return limitations


def _recommended_next_action(
    *,
    workload: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
    bottleneck: Mapping[str, object],
) -> str:
    invalid_workload = [
        trial["summary"].trial_id
        for trial in trials
        if trial["analysis"].trial_validity == "invalid_workload"
    ]
    if invalid_workload:
        return (
            "Fix workload/model context compatibility before comparing serving configurations; "
            f"invalid workload trials: {', '.join(invalid_workload)}"
        )
    bottleneck_class = bottleneck.get("class")
    if bottleneck_class == "kv_cache":
        return "Use an external orchestrator to compare serving configurations with more KV capacity or lower sequence pressure."
    if bottleneck_class == "scheduler_cap":
        return "Use an external orchestrator to compare higher max_num_seqs or related scheduler settings for the same workload."
    if bottleneck_class == "decode_bandwidth":
        return "Use an external orchestrator to compare decode-oriented server settings or hardware for the same workload."
    if bottleneck_class == "prefill_compute_or_token_budget":
        return "Use an external orchestrator to compare prefill/token-budget settings for the same workload."
    return (
        "Preserve this workload/context policy and use an external orchestrator for any server-configuration comparison; "
        f"current workload: {workload.get('name')}"
    )


def _extract_workload_payload(
    trials: Sequence[Mapping[str, object]],
    trace_config: Mapping[str, object],
) -> dict[str, object]:
    workload_payload: Mapping[str, Any] | None = None
    for trial in trials:
        metadata = _require_mapping(trial["summary_payload"].get("config"), "summary.json.config").get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("workload")
        if value is None:
            continue
        workload_payload = _require_mapping(value, "summary.json.config.metadata.workload")
        break
    if workload_payload is None:
        raise ValueError("result artifacts are missing workload metadata")
    context_policy = workload_payload.get("context_policy")
    if not isinstance(context_policy, Mapping):
        context_policy = _missing_context_policy_payload(workload_payload)
    benchmark = trials[0]["summary"].benchmark_metrics
    return {
        "name": workload_payload.get("name"),
        "source_path": workload_payload.get("source_path"),
        "dataset_type": workload_payload.get("dataset_type"),
        "num_requests": workload_payload.get("num_requests"),
        "prompt_len_summary": benchmark.prompt_length_summary,
        "output_len_summary": benchmark.output_length_summary,
        "context_policy": dict(context_policy),
        "model": trace_config.get("model"),
    }


def _missing_context_policy_payload(workload_payload: Mapping[str, Any]) -> dict[str, object]:
    num_requests = workload_payload.get("num_requests")
    return {
        "metadata_status": "not_recorded",
        "max_model_len": None,
        "tokenizer_source": None,
        "tokenizer": None,
        "over_limit": None,
        "truncation_side": None,
        "unsafe_allow_workload_tokenizer_for_real_datasets": False,
        "total_samples": num_requests,
        "kept_samples": num_requests,
        "skipped_samples": 0,
        "truncated_samples": 0,
        "skipped_source_indexes": [],
        "truncated_source_indexes": [],
    }


def _stability_policy_payload(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    for trial in trials:
        metadata = _require_mapping(trial["summary_payload"].get("config"), "summary.json.config").get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        policy = metadata.get("stability_policy")
        if policy is not None:
            return dict(_require_mapping(policy, "summary.json.config.metadata.stability_policy"))
    config = StabilityConfig()
    return {
        "warmup_windows": config.warmup_windows,
        "min_eval_windows": config.min_eval_windows,
        "completion_arrival_tolerance": config.completion_arrival_tolerance,
        "max_positive_backlog_slope": config.max_positive_backlog_slope,
        "min_backlog_growth_for_hard_pressure": config.min_backlog_growth_for_hard_pressure,
        "min_backlog_relative_increase": config.min_backlog_relative_increase,
        "backlog_trend_alpha": config.backlog_trend_alpha,
        "min_waiting_queue_mean_for_pressure": config.min_waiting_queue_mean_for_pressure,
        "min_waiting_queue_active_fraction": config.min_waiting_queue_active_fraction,
        "token_throughput_plateau_relative_growth": config.token_throughput_plateau_relative_growth,
        "max_error_rate": config.max_error_rate,
        "ttft_slo_ms": config.ttft_slo_ms,
        "tpot_slo_ms": config.tpot_slo_ms,
        "ttft_slo_field": config.ttft_slo_field,
        "tpot_slo_field": config.tpot_slo_field,
    }


def _load_trial_analysis(path: Path) -> TrialAnalysisResult:
    return _load_trial_analysis_from_mapping(_load_json_mapping(path))


def _load_trial_summary_from_mapping(payload: Mapping[str, Any]) -> TrialSummary:
    benchmark_metrics = BenchmarkMetrics(
        **_require_mapping(payload.get("benchmark_metrics"), "summary.json.summary.benchmark_metrics")
    )
    summary_payload = dict(payload)
    summary_payload["benchmark_metrics"] = benchmark_metrics
    return TrialSummary(**summary_payload)


def _load_trial_analysis_from_mapping(payload: Mapping[str, Any]) -> TrialAnalysisResult:
    stability_payload = payload.get("stability")
    bottleneck_payload = payload.get("bottleneck")
    stability = None
    bottleneck = None
    if stability_payload is not None:
        stability = StabilityResult(**_require_mapping(stability_payload, "analysis.json.stability"))
    if bottleneck_payload is not None:
        bottleneck = BottleneckResult(**_require_mapping(bottleneck_payload, "analysis.json.bottleneck"))
    return TrialAnalysisResult(
        trial_id=_require_string(payload.get("trial_id"), "analysis.json.trial_id"),
        trial_validity=_require_string(payload.get("trial_validity"), "analysis.json.trial_validity"),
        validity_reasons=_require_string_list(payload.get("validity_reasons"), "analysis.json.validity_reasons"),
        stability=stability,
        bottleneck=bottleneck,
    )


def _load_server_config(summary_payload: Mapping[str, Any], trial_dir: Path) -> dict[str, object]:
    config_payload = _require_mapping(summary_payload.get("config"), "summary.json.config")
    config: dict[str, object] = {
        "model": config_payload.get("model"),
    }
    metadata = config_payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("server_metadata", "server_config", "vllm_config"):
            value = metadata.get(key)
            if value is None:
                continue
            config = _merge_server_config(config, _require_mapping(value, f"summary.json.config.metadata.{key}"))
    for filename in ("server_metadata.json", "server_config.json", "vllm_config.json"):
        path = trial_dir / filename
        if path.is_file():
            config = _merge_server_config(config, _load_json_mapping(path))
    return config


def _build_server_metadata_payload(server_config: Mapping[str, object]) -> dict[str, object]:
    payload = dict(server_config)
    payload["max_num_seqs"] = server_config.get("max_num_seqs")
    payload["max_num_batched_tokens"] = server_config.get("max_num_batched_tokens")
    if payload["max_num_seqs"] is None or payload["max_num_batched_tokens"] is None:
        payload["scheduler_metadata_note"] = (
            "max_num_seqs and max_num_batched_tokens were not both recorded. "
            "These are explicit vLLM scheduler configuration values and are not reliably inferred from runtime metrics."
        )
    else:
        payload["scheduler_metadata_note"] = (
            "max_num_seqs and max_num_batched_tokens were recorded directly from saved server metadata."
        )
    return payload


def _merge_server_config(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    merged = dict(left)
    for key, right_value in right.items():
        if key not in merged:
            merged[key] = right_value
            continue
        left_value = merged[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            merged[key] = _merge_server_config(
                _require_mapping(left_value, f"server_config.{key}"),
                _require_mapping(right_value, f"server_config.{key}"),
            )
            continue
        if left_value != right_value:
            raise ValueError(
                f"conflicting recorded server metadata for server_config.{key}: {left_value!r} != {right_value!r}"
            )
    return merged


def _assert_analysis_matches_trace(
    analysis: TrialAnalysisResult,
    trace_analysis: TrialAnalysisResult,
    *,
    trial_dir: Path,
) -> None:
    if analysis.to_dict() != trace_analysis.to_dict():
        raise ValueError(f"analysis.json disagrees with search_trace.json analysis for {trial_dir}")


def _select_headline_trial(
    trials: Sequence[Mapping[str, object]],
    search_result: Mapping[str, object],
) -> Mapping[str, object]:
    confirmation_trial_id = search_result.get("confirmation_trial_id")
    if isinstance(confirmation_trial_id, str):
        for trial in trials:
            if trial["summary"].trial_id == confirmation_trial_id:
                return trial
    stable_open_loop = [
        trial
        for trial in trials
        if trial["summary"].mode == "open-loop"
        and trial["analysis"].trial_validity == "valid"
        and trial["analysis"].stability is not None
        and trial["analysis"].stability.status == "stable"
    ]
    if not stable_open_loop:
        raise ValueError("report generation requires at least one stable open-loop trial")
    return max(
        stable_open_loop,
        key=lambda trial: float(_require_numeric(trial["summary"].requested_request_rate, "requested_request_rate")),
    )


def _comparison_max_output_tok_s(
    payload: Mapping[str, object],
    closed_loop: Mapping[str, object] | None,
) -> float | None:
    if closed_loop is not None and closed_loop.get("peak_output_token_throughput") is not None:
        return _require_numeric(
            closed_loop.get("peak_output_token_throughput"),
            "closed_loop.peak_output_token_throughput",
        )
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        return None
    values = []
    for trial in trials:
        mapping = _require_mapping(trial, "report_payload.trials[]")
        value = mapping.get("generation_token_throughput")
        if value is None:
            continue
        values.append(_require_numeric(value, "trials[].generation_token_throughput"))
    if not values:
        return None
    return max(values)


def _build_search_decision_context(
    trace_payload: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    bounds_payload = _require_mapping(trace_payload.get("bounds", {}), "search_trace.json.bounds")
    result_payload = _require_mapping(trace_payload.get("result"), "search_trace.json.result")
    confirmation_trial_id = result_payload.get("confirmation_trial_id")
    decision_trial = None
    subject = "high_bound"
    if isinstance(confirmation_trial_id, str):
        confirmation_trial = _trial_by_id(trials, confirmation_trial_id)
        if confirmation_trial is not None and _trial_is_failed_decision(confirmation_trial):
            decision_trial = confirmation_trial
            subject = "failed_confirmation"
    if decision_trial is None:
        high_trial_id = bounds_payload.get("high_trial_id")
        if isinstance(high_trial_id, str):
            decision_trial = _trial_by_id(trials, high_trial_id)
            subject = "high_bound"
    if decision_trial is None and result_payload.get("termination_reason") == "max_request_rate_limited":
        low_trial_id = bounds_payload.get("low_trial_id")
        if isinstance(low_trial_id, str):
            decision_trial = _trial_by_id(trials, low_trial_id)
            subject = "max_request_rate_cap"
    if decision_trial is None:
        raise ValueError("report generation requires a high-bound or failed confirmation trial to explain")
    analysis = decision_trial["analysis"]
    status = None if analysis.stability is None else analysis.stability.status
    if subject == "max_request_rate_cap":
        cap = bounds_payload.get("max_request_rate_cap")
        attempted_rate = bounds_payload.get("max_request_rate_cap_attempted_rate")
        reasons = list(result_payload.get("reasons", []))
        return {
            "subject": subject,
            "trial_id": decision_trial["summary"].trial_id,
            "request_rate": decision_trial["summary"].requested_request_rate,
            "stability_status": status,
            "decision_reasoning": "capped_by_max_request_rate",
            "reason_summary": (
                f"next bracketing rate {attempted_rate} exceeded max_request_rate={cap}; "
                "reported MST is the highest measured stable low bound"
            ),
            "reasons": reasons,
            "max_request_rate_cap": cap,
            "max_request_rate_cap_attempted_rate": attempted_rate,
        }
    return {
        "subject": subject,
        "trial_id": decision_trial["summary"].trial_id,
        "request_rate": decision_trial["summary"].requested_request_rate,
        "stability_status": status,
        "decision_reasoning": _decision_reasoning_label(analysis),
        "reason_summary": _decision_reason_summary(analysis),
        "reasons": [] if analysis.stability is None else list(analysis.stability.reasons),
    }


def _trial_by_id(
    trials: Sequence[Mapping[str, object]],
    trial_id: str,
) -> Mapping[str, object] | None:
    for trial in trials:
        if trial["summary"].trial_id == trial_id:
            return trial
    return None


def _trial_is_failed_decision(trial: Mapping[str, object]) -> bool:
    analysis = trial["analysis"]
    if analysis.trial_validity != "valid" or analysis.stability is None:
        return True
    return analysis.stability.status in {"unstable", "slo_violation", "aborted_safety", "uncertain"}


def _decision_reasoning_label(analysis: TrialAnalysisResult) -> str:
    if analysis.trial_validity != "valid":
        return f"trial_validity={analysis.trial_validity}"
    if analysis.stability is None:
        return "missing_stability_result"
    if analysis.stability.status == "slo_violation":
        return "failed_due_to_slo_violation"
    if analysis.stability.status == "unstable":
        return _unstable_reasoning_label(analysis.stability.reasons)
    if analysis.stability.status == "aborted_safety":
        return "failed_due_to_safety_abort"
    if analysis.stability.status == "uncertain":
        return "failed_due_to_uncertain_stability"
    if analysis.stability.status == "stable":
        return "stable"
    raise ValueError(f"unsupported stability status {analysis.stability.status!r}")


def _decision_reason_summary(analysis: TrialAnalysisResult) -> str:
    if analysis.trial_validity != "valid":
        return "; ".join(analysis.validity_reasons)
    if analysis.stability is None:
        raise ValueError("valid decision trial requires stability details")
    if analysis.stability.status == "slo_violation":
        return "The analyzer marked this trial as an SLO violation."
    if analysis.stability.status == "unstable":
        return _unstable_reason_summary(analysis.stability.reasons)
    if analysis.stability.status == "aborted_safety":
        return "The analyzer marked this trial as aborted by the client safety cap."
    if analysis.stability.status == "uncertain":
        return "The analyzer could not make a confident stability decision for this trial."
    if analysis.stability.status == "stable":
        return "The analyzer marked this trial as stable."
    raise ValueError(f"unsupported stability status {analysis.stability.status!r}")


def _relative_width(*, low_rate: float | None, high_rate: float | None) -> float | None:
    if low_rate is None or high_rate is None:
        return None
    if low_rate <= 0.0:
        raise ValueError("final low bound must be positive when computing relative width")
    return (high_rate - low_rate) / low_rate


def _convergence_assessment(
    *,
    low_rate: float | None,
    high_rate: float | None,
    relative_width: float | None,
    rate_precision: float | None,
) -> str:
    if low_rate is None or high_rate is None or relative_width is None or rate_precision is None:
        return "final search bracket was not recorded"
    if relative_width >= rate_precision / 2.0:
        return (
            "precision-limited: final bracket remained loose relative to the configured rate_precision "
            f"({relative_width:.3f} width for target {rate_precision:.3f})"
        )
    return (
        "tightly converged: final bracket was materially narrower than the configured rate_precision "
        f"({relative_width:.3f} width for target {rate_precision:.3f})"
    )


def _unstable_reasoning_label(reasons: Sequence[str]) -> str:
    has_kv_pressure = any("kv cache usage approached saturation" in reason.lower() for reason in reasons)
    has_preemptions = any("preemptions observed after warmup" in reason.lower() for reason in reasons)
    has_latency_drift = any("drifted upward" in reason.lower() for reason in reasons)
    if has_kv_pressure and has_preemptions and has_latency_drift:
        return "failed_due_to_kv_pressure_preemptions_and_latency_drift"
    if has_kv_pressure and has_preemptions:
        return "failed_due_to_kv_pressure_and_preemptions"
    if has_kv_pressure:
        return "failed_due_to_kv_pressure"
    if has_preemptions:
        return "failed_due_to_preemptions"
    if has_latency_drift:
        return "failed_due_to_latency_drift"
    return "failed_due_to_non_slo_instability_evidence"


def _unstable_reason_summary(reasons: Sequence[str]) -> str:
    primary_reasons = [
        reason
        for reason in reasons
        if "confidence lowered" not in reason.lower()
    ]
    if not primary_reasons:
        primary_reasons = list(reasons)
    if not primary_reasons:
        return "The analyzer marked this trial as unstable."
    evidence_text = "; ".join(primary_reasons[:3])
    return f"The analyzer marked this trial as unstable based on: {evidence_text}."


def _queue_drift(trial: Mapping[str, object]) -> float | None:
    stability = trial["analysis"].stability
    if stability is not None and "outstanding_end_slope_per_s" in stability.key_metrics:
        return float(stability.key_metrics["outstanding_end_slope_per_s"])
    windows = trial["windows"]
    if not windows:
        return None
    duration = windows[-1].end_s - windows[0].start_s
    if duration <= 0.0:
        raise ValueError("windows duration must be positive")
    return (windows[-1].outstanding_end - windows[0].outstanding_start) / duration


def _max_window_value(windows: Sequence[object], field_name: str) -> float | None:
    values = [getattr(window, field_name) for window in windows if getattr(window, field_name) is not None]
    if not values:
        return None
    return max(float(value) for value in values)


def _stability_label(analysis: TrialAnalysisResult) -> str:
    return "none" if analysis.stability is None else analysis.stability.status


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, str(path))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a non-empty list of strings")
    return list(value)


def _require_numeric(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _optional_numeric(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _require_numeric(value, label)


def _require_trial_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("bundle.trials must be a non-empty list")
    return [dict(item) for item in value]


__all__ = ["generate_report"]
