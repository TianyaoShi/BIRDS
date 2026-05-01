from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_mst_finder.reporting import generate_report
from llm_mst_finder.records import (
    BenchmarkMetrics,
    BottleneckResult,
    StabilityResult,
    TrialAnalysisResult,
    TrialSummary,
    WindowSummary,
)


def _benchmark_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        successful_requests=20,
        failed_requests=0,
        total_input_tokens=2000,
        total_output_tokens=800,
        request_throughput=10.0,
        successful_request_throughput=10.0,
        prompt_token_throughput=2000.0,
        generation_token_throughput=800.0,
        total_token_throughput=2800.0,
        mean_ttft_ms=100.0,
        median_ttft_ms=90.0,
        std_ttft_ms=10.0,
        percentiles_ttft_ms=[(0.9, 120.0), (0.99, 150.0)],
        mean_tpot_ms=30.0,
        median_tpot_ms=28.0,
        std_tpot_ms=4.0,
        percentiles_tpot_ms=[(0.9, 32.0), (0.99, 36.0)],
        mean_itl_ms=25.0,
        median_itl_ms=25.0,
        std_itl_ms=2.0,
        percentiles_itl_ms=[(0.9, 27.0)],
        mean_e2e_ms=650.0,
        median_e2e_ms=640.0,
        std_e2e_ms=30.0,
        percentiles_e2e_ms=[(0.9, 690.0), (0.99, 720.0)],
        prompt_length_summary={"mean": 100.0, "median": 100.0, "p90": 120.0, "p99": 128.0},
        output_length_summary={"mean": 40.0, "median": 40.0, "p90": 48.0, "p99": 64.0},
    )


def _trial_summary(
    trial_id: str,
    *,
    mode: str,
    request_rate: float | None,
    concurrency: int | None,
    completion_rate: float,
) -> TrialSummary:
    return TrialSummary(
        trial_id=trial_id,
        mode=mode,
        status="completed",
        requested_request_rate=request_rate,
        requested_concurrency=concurrency,
        target_duration_s=6.0,
        wall_time_s=6.0,
        started_requests=20,
        successful_requests=20,
        failed_requests=0,
        actual_send_rate=completion_rate,
        successful_completion_rate=completion_rate,
        error_rate=0.0,
        mean_scheduling_delay_s=0.01,
        max_scheduling_delay_s=0.02,
        max_observed_outstanding=4,
        metrics_sample_count=3,
        abort_reason=None,
        benchmark_metrics=_benchmark_metrics(),
        metadata={},
    )


def _window(trial_id: str, idx: int, *, arrival_rate: float, completion_rate: float, outstanding_end: int, ttft_p90_ms: float, tpot_p90_ms: float) -> WindowSummary:
    return WindowSummary(
        trial_id=trial_id,
        window_idx=idx,
        start_s=float(idx),
        end_s=float(idx + 1),
        arrivals=10,
        completions=10,
        failures=0,
        arrival_rate=arrival_rate,
        completion_rate=completion_rate,
        error_rate=0.0,
        outstanding_start=max(0, outstanding_end - 1),
        outstanding_end=outstanding_end,
        outstanding_mean=float(outstanding_end),
        outstanding_slope=0.1,
        ttft_p50_ms=80.0,
        ttft_p90_ms=ttft_p90_ms,
        ttft_p95_ms=ttft_p90_ms + 10.0,
        ttft_p99_ms=ttft_p90_ms + 20.0,
        tpot_p50_ms=25.0,
        tpot_p90_ms=tpot_p90_ms,
        tpot_p95_ms=tpot_p90_ms + 5.0,
        tpot_p99_ms=tpot_p90_ms + 10.0,
        itl_p50_ms=20.0,
        itl_p90_ms=23.0,
        itl_p95_ms=24.0,
        itl_p99_ms=25.0,
        e2e_p50_ms=550.0,
        e2e_p90_ms=600.0,
        e2e_p95_ms=625.0,
        e2e_p99_ms=650.0,
        prompt_tok_s=1000.0,
        generation_tok_s=400.0,
        total_tok_s=1400.0,
        num_running_mean=3.0,
        num_waiting_mean=0.5,
        num_swapped_mean=0.0,
        kv_cache_usage_mean=0.45,
        kv_cache_usage_max=0.50,
        preemptions_delta=0.0,
    )


def _analysis(trial_id: str, *, status: str, bottleneck_class: str = "scheduler_cap") -> TrialAnalysisResult:
    reason_map = {
        "stable": ["fixture stable"],
        "unstable": ["outstanding requests drifted upward", "TPOT p90 drifted upward"],
        "slo_violation": ["TTFT p90 exceeded configured SLO"],
        "uncertain": ["fixture uncertain"],
        "aborted_safety": ["fixture aborted_safety"],
    }
    return TrialAnalysisResult(
        trial_id=trial_id,
        trial_validity="valid",
        validity_reasons=["fixture valid"],
        stability=StabilityResult(
            status=status,
            confidence="high",
            reasons=reason_map[status],
            key_metrics={"outstanding_end_slope_per_s": 0.1},
        ),
        bottleneck=BottleneckResult(
            bottleneck_class=bottleneck_class,
            confidence="medium",
            evidence=["fixture evidence"],
        ),
    )


def _write_trial(
    result_dir: Path,
    *,
    trial_id: str,
    purpose: str,
    mode: str,
    request_rate: float | None,
    concurrency: int | None,
    analysis: TrialAnalysisResult,
    include_scheduler_metadata: bool = True,
    include_context_policy_metadata: bool = True,
) -> dict[str, object]:
    trial_dir = result_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True)
    summary = _trial_summary(
        trial_id,
        mode=mode,
        request_rate=request_rate,
        concurrency=concurrency,
        completion_rate=8.0 if request_rate is None else request_rate,
    )
    workload_metadata = {
        "name": "fixture-workload",
        "source_path": "fixtures/workload.yaml",
        "dataset_type": "synthetic-fixed",
        "num_requests": 20,
    }
    if include_context_policy_metadata:
        workload_metadata["context_policy"] = {
            "max_model_len": 4096,
            "tokenizer_source": "vllm_model_config",
            "tokenizer": "fake-model",
            "over_limit": "fail",
            "truncation_side": "left",
            "unsafe_allow_workload_tokenizer_for_real_datasets": False,
            "total_samples": 20,
            "kept_samples": 20,
            "skipped_samples": 0,
            "truncated_samples": 0,
            "skipped_source_indexes": [],
            "truncated_source_indexes": [],
        }
    summary_payload = {
        "config": {
            "trial_id": trial_id,
            "mode": mode,
            "duration_s": 6.0,
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/completions",
            "model": "fake-model",
            "request_rate": request_rate,
            "concurrency": concurrency,
            "metadata": {
                "workload": workload_metadata,
                "server_config": (
                    {
                        "max_num_seqs": 32,
                        "max_num_batched_tokens": 4096,
                        "chunked_prefill": True,
                    }
                    if include_scheduler_metadata
                    else {
                        "chunked_prefill": True,
                    }
                ),
            },
        },
        "summary": summary.to_dict(),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (trial_dir / "analysis.json").write_text(json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    windows = [
        _window(trial_id, idx, arrival_rate=8.0 if request_rate is None else request_rate, completion_rate=8.0 if request_rate is None else request_rate, outstanding_end=idx, ttft_p90_ms=100.0 + idx, tpot_p90_ms=30.0 + idx)
        for idx in range(6)
    ]
    with (trial_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].to_dict()))
        writer.writeheader()
        for window in windows:
            writer.writerow(window.to_dict())
    summary_trace = {
        "status": summary.status,
        "requested_request_rate": request_rate,
        "requested_concurrency": concurrency,
        "actual_send_rate": summary.actual_send_rate,
        "successful_completion_rate": summary.successful_completion_rate,
        "error_rate": summary.error_rate,
        "generation_token_throughput": summary.benchmark_metrics.generation_token_throughput,
        "total_token_throughput": summary.benchmark_metrics.total_token_throughput,
        "max_observed_outstanding": summary.max_observed_outstanding,
        "abort_reason": summary.abort_reason,
    }
    return {
        "purpose": purpose,
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "mode": mode,
        "request_rate": request_rate,
        "concurrency": concurrency,
        "summary": summary_trace,
        "analysis": analysis.to_dict(),
    }


def _write_result_dir(
    result_dir: Path,
    *,
    rate_precision: float = 0.05,
    low_rate: float = 8.0,
    high_rate: float = 12.0,
    confirmation_trial_id: str = "trial_001_openloop_r8_0",
    unstable_status: str = "unstable",
    include_scheduler_metadata: bool = True,
    include_context_policy_metadata: bool = True,
) -> None:
    trials_dir = result_dir / "trials"
    trials_dir.mkdir(parents=True)
    closed_loop_event = _write_trial(
        result_dir,
        trial_id="trial_000_closedloop_N4",
        purpose="closed_loop_scout",
        mode="closed-loop",
        request_rate=None,
        concurrency=4,
        analysis=_analysis("trial_000_closedloop_N4", status="stable"),
        include_scheduler_metadata=include_scheduler_metadata,
        include_context_policy_metadata=include_context_policy_metadata,
    )
    stable_event = _write_trial(
        result_dir,
        trial_id="trial_001_openloop_r8_0",
        purpose="open_loop_bracket",
        mode="open-loop",
        request_rate=8.0,
        concurrency=None,
        analysis=_analysis("trial_001_openloop_r8_0", status="stable"),
        include_scheduler_metadata=include_scheduler_metadata,
        include_context_policy_metadata=include_context_policy_metadata,
    )
    unstable_event = _write_trial(
        result_dir,
        trial_id="trial_002_openloop_r12_0",
        purpose="open_loop_bracket",
        mode="open-loop",
        request_rate=12.0,
        concurrency=None,
        analysis=_analysis("trial_002_openloop_r12_0", status=unstable_status),
        include_scheduler_metadata=include_scheduler_metadata,
        include_context_policy_metadata=include_context_policy_metadata,
    )
    trace_payload = {
        "config": {
            "search_id": "fixture-search",
            "search_mode": "hybrid",
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/completions",
            "model": "fake-model",
            "trial_duration_s": 6.0,
            "final_confirmation_duration_s": 6.0,
            "rate_precision": rate_precision,
        },
        "events": [closed_loop_event, stable_event, unstable_event],
        "bounds": {
            "low_rate": low_rate,
            "low_trial_id": "trial_001_openloop_r8_0",
            "high_rate": high_rate,
            "high_trial_id": "trial_002_openloop_r12_0",
        },
        "closed_loop": {
            "peak_request_throughput": 8.0,
            "peak_output_token_throughput": 800.0,
            "plateau_concurrency": 4,
            "stop_reason": "fixture plateau",
        },
        "result": {
            "search_id": "fixture-search",
            "search_mode": "hybrid",
            "max_no_drift_request_rate": 8.0,
            "max_slo_satisfying_request_rate": 8.0,
            "rate_precision": rate_precision,
            "confirmation_trial_id": confirmation_trial_id,
            "closed_loop": {
                "peak_request_throughput": 8.0,
                "peak_output_token_throughput": 800.0,
                "plateau_concurrency": 4,
                "stop_reason": "fixture plateau",
            },
            "bottleneck_class": "scheduler_cap",
            "confidence": "medium",
            "reasons": ["fixture result"],
        },
    }
    (result_dir / "search_trace.json").write_text(json.dumps(trace_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_generate_report_writes_reports_and_plots(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)

    report_result = generate_report(result_dir)

    assert Path(report_result["final_report_json"]).is_file()
    assert Path(report_result["final_report_md"]).is_file()
    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["search_result"]["max_no_drift_request_rate"] == 8.0
    assert payload["bottleneck"]["class"] == "scheduler_cap"
    assert payload["trials"][0]["trial_id"] == "trial_000_closedloop_N4"
    assert payload["search_trace"]["rate_precision"] == 0.05
    assert payload["decision_context"]["decision_reasoning"] == "failed_due_to_latency_drift"
    assert payload["server_config"]["max_num_seqs"] == 32
    assert (result_dir / "plots" / "search_rate_vs_classification.png").is_file()
    assert (result_dir / "trials" / "trial_001_openloop_r8_0" / "plots" / "ttft_percentiles.png").is_file()


def test_generate_report_requires_existing_comparison_report(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)
    comparison_dir = tmp_path / "run_002"
    comparison_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="final_report.json"):
        generate_report(result_dir, compare_result_dirs=[comparison_dir])


def test_generate_report_compares_matching_reports(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)
    generate_report(result_dir)

    comparison_dir = tmp_path / "run_002"
    _write_result_dir(comparison_dir)
    generate_report(comparison_dir)

    generate_report(result_dir, compare_result_dirs=[comparison_dir])

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["comparison"]["comparable"] is True
    assert len(payload["comparison"]["results"]) == 2
    assert (result_dir / "plots" / "comparison_max_sustainable_rate.png").is_file()


def test_generate_report_marks_loose_precision_bracket(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_loose_bracket"
    _write_result_dir(
        result_dir,
        rate_precision=0.2,
        low_rate=3.125,
        high_rate=3.75,
    )

    generate_report(result_dir, plots_enabled=False)

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["search_trace"]["final_low_rate"] == 3.125
    assert payload["search_trace"]["final_high_rate"] == 3.75
    assert payload["search_trace"]["final_relative_width"] == pytest.approx(0.2)
    assert payload["search_trace"]["convergence_assessment"].startswith("precision-limited")
    markdown = (result_dir / "final_report.md").read_text(encoding="utf-8")
    assert "- rate_precision: 0.2" in markdown
    assert "- final_low_rate: 3.125" in markdown
    assert "- final_high_rate: 3.75" in markdown
    assert "- final_relative_width: 0.2" in markdown
    assert "- convergence_assessment: precision-limited" in markdown


def test_generate_report_handles_max_request_rate_limited_result(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_capped"
    _write_result_dir(result_dir)
    trace_path = result_dir / "search_trace.json"
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["bounds"] = {
        "low_rate": 8.0,
        "low_trial_id": "trial_001_openloop_r8_0",
        "high_rate": None,
        "high_trial_id": None,
        "max_request_rate_cap": 12.0,
        "max_request_rate_cap_attempted_rate": 16.0,
    }
    trace_payload["result"]["termination_reason"] = "max_request_rate_limited"
    trace_payload["result"]["confirmation_trial_id"] = None
    trace_payload["result"]["reasons"] = [
        "open-loop bracketing stopped because the next required high-bound rate 16.000 req/s exceeds max_request_rate=12",
        "highest observed stable open-loop rate before the cap was 8.000 req/s",
    ]
    trace_path.write_text(json.dumps(trace_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generate_report(result_dir, plots_enabled=False)

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["decision_context"]["subject"] == "max_request_rate_cap"
    assert payload["decision_context"]["trial_id"] == "trial_001_openloop_r8_0"
    assert payload["decision_context"]["decision_reasoning"] == "capped_by_max_request_rate"
    assert payload["decision_context"]["max_request_rate_cap"] == 12.0
    assert payload["decision_context"]["max_request_rate_cap_attempted_rate"] == 16.0


def test_generate_report_shows_slo_defaults_and_non_slo_instability(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_non_slo_instability"
    _write_result_dir(
        result_dir,
        unstable_status="unstable",
    )

    generate_report(result_dir, plots_enabled=False)

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["stability_policy"]["ttft_slo_ms"] == 2000.0
    assert payload["stability_policy"]["tpot_slo_ms"] == 80.0
    assert all(not key.startswith("e2e_") for key in payload["stability_policy"])
    assert payload["decision_context"]["subject"] == "high_bound"
    assert payload["decision_context"]["stability_status"] == "unstable"
    assert payload["decision_context"]["decision_reasoning"] == "failed_due_to_latency_drift"
    assert "TPOT p90 drifted upward" in payload["decision_context"]["reason_summary"]
    markdown = (result_dir / "final_report.md").read_text(encoding="utf-8")
    assert "- ttft_slo_ms: 2000.0" in markdown
    assert "- tpot_slo_ms: 80.0" in markdown
    assert "e2e_" not in markdown
    assert "- decision_reasoning: failed_due_to_latency_drift" in markdown


def test_generate_report_explains_scheduler_metadata_missing(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_missing_metadata"
    _write_result_dir(
        result_dir,
        include_scheduler_metadata=False,
    )

    generate_report(result_dir, plots_enabled=False)

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["server_config"]["max_num_seqs"] is None
    assert payload["server_config"]["max_num_batched_tokens"] is None
    assert "not reliably inferred from runtime metrics" in payload["server_config"]["scheduler_metadata_note"]
    assert any("max_num_seqs and max_num_batched_tokens are explicit vLLM scheduler config values" in item for item in payload["limitations"])
    markdown = (result_dir / "final_report.md").read_text(encoding="utf-8")
    assert "- max_num_seqs: None" in markdown
    assert "- max_num_batched_tokens: None" in markdown
    assert "not reliably inferred from runtime metrics" in markdown


def test_generate_report_tolerates_missing_context_policy_metadata(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_missing_context_policy"
    _write_result_dir(
        result_dir,
        include_context_policy_metadata=False,
    )

    generate_report(result_dir, plots_enabled=False)

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    context_policy = payload["workload"]["context_policy"]
    assert context_policy["metadata_status"] == "not_recorded"
    assert context_policy["tokenizer_source"] is None
    assert any("context_policy metadata was not recorded" in item for item in payload["limitations"])
    markdown = (result_dir / "final_report.md").read_text(encoding="utf-8")
    assert "- tokenizer_source: None" in markdown
