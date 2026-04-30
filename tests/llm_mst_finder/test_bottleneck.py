from __future__ import annotations

import pytest

from llm_mst_finder.bottleneck import BottleneckConfig, classify_bottleneck
from llm_mst_finder.records import BenchmarkMetrics, RequestRecord, StabilityResult, TrialSummary, WindowSummary


def _window(
    idx: int,
    *,
    arrivals: int = 10,
    completions: int = 10,
    failures: int = 0,
    outstanding_end: int = 0,
    ttft_p90_ms: float | None = 100.0,
    ttft_p99_ms: float | None = 120.0,
    tpot_p90_ms: float | None = 20.0,
    tpot_p99_ms: float | None = 24.0,
    itl_p90_ms: float | None = 18.0,
    e2e_p99_ms: float | None = 700.0,
    prompt_tok_s: float | None = 1000.0,
    generation_tok_s: float | None = 300.0,
    num_running_mean: float | None = 4.0,
    num_waiting_mean: float | None = 0.0,
    num_swapped_mean: float | None = 0.0,
    kv_cache_usage_mean: float | None = 0.40,
    kv_cache_usage_max: float | None = 0.45,
    preemptions_delta: float | None = 0.0,
) -> WindowSummary:
    terminal_events = completions + failures
    return WindowSummary(
        trial_id="trial-bottleneck",
        window_idx=idx,
        start_s=float(idx),
        end_s=float(idx + 1),
        arrivals=arrivals,
        completions=completions,
        failures=failures,
        arrival_rate=float(arrivals),
        completion_rate=float(completions),
        error_rate=failures / terminal_events if terminal_events else 0.0,
        outstanding_start=outstanding_end,
        outstanding_end=outstanding_end,
        outstanding_mean=float(outstanding_end),
        outstanding_slope=0.0,
        ttft_p50_ms=80.0 if ttft_p90_ms is not None else None,
        ttft_p90_ms=ttft_p90_ms,
        ttft_p95_ms=ttft_p90_ms + 10.0 if ttft_p90_ms is not None else None,
        ttft_p99_ms=ttft_p99_ms,
        tpot_p50_ms=15.0 if tpot_p90_ms is not None else None,
        tpot_p90_ms=tpot_p90_ms,
        tpot_p95_ms=tpot_p90_ms + 2.0 if tpot_p90_ms is not None else None,
        tpot_p99_ms=tpot_p99_ms,
        itl_p50_ms=15.0 if itl_p90_ms is not None else None,
        itl_p90_ms=itl_p90_ms,
        itl_p95_ms=itl_p90_ms + 1.0 if itl_p90_ms is not None else None,
        itl_p99_ms=itl_p90_ms + 2.0 if itl_p90_ms is not None else None,
        e2e_p50_ms=550.0 if e2e_p99_ms is not None else None,
        e2e_p90_ms=600.0 if e2e_p99_ms is not None else None,
        e2e_p95_ms=650.0 if e2e_p99_ms is not None else None,
        e2e_p99_ms=e2e_p99_ms,
        prompt_tok_s=prompt_tok_s,
        generation_tok_s=generation_tok_s,
        total_tok_s=(
            None
            if prompt_tok_s is None or generation_tok_s is None
            else prompt_tok_s + generation_tok_s
        ),
        num_running_mean=num_running_mean,
        num_waiting_mean=num_waiting_mean,
        num_swapped_mean=num_swapped_mean,
        kv_cache_usage_mean=kv_cache_usage_mean,
        kv_cache_usage_max=kv_cache_usage_max,
        preemptions_delta=preemptions_delta,
    )


def _stable_windows() -> list[WindowSummary]:
    return [_window(idx) for idx in range(6)]


def test_classifies_scheduler_cap_from_supplied_max_num_seqs() -> None:
    ttft_values = [100.0, 100.0, 100.0, 130.0, 170.0, 230.0]
    windows = [
        _window(
            idx,
            completions=9 if idx >= 2 else 10,
            outstanding_end=max(0, idx - 2),
            ttft_p90_ms=ttft,
            ttft_p99_ms=ttft + 20.0,
            num_running_mean=7.5,
            num_waiting_mean=float(max(0, idx - 2)),
        )
        for idx, ttft in enumerate(ttft_values)
    ]

    result = classify_bottleneck(
        windows,
        server_metadata={"server_config": {"max_num_seqs": 8, "max_num_batched_tokens": 4096}},
    )

    assert result.bottleneck_class == "scheduler_cap"
    assert result.confidence == "high"
    assert any("max_num_seqs=8.000" in item for item in result.evidence)
    assert any("preemptions were absent" in item for item in result.evidence)


def test_does_not_infer_scheduler_cap_without_max_num_seqs_metadata() -> None:
    windows = [
        _window(
            idx,
            num_running_mean=7.5,
        )
        for idx in range(6)
    ]

    result = classify_bottleneck(windows)

    assert result.bottleneck_class == "unknown"
    assert any("scheduler-cap diagnosis requires explicit serving metadata" in item for item in result.evidence)


def test_classifies_kv_cache_high_confidence_without_num_swapped_metric() -> None:
    windows = [
        _window(
            idx,
            kv_cache_usage_mean=0.95,
            kv_cache_usage_max=0.99,
            num_swapped_mean=None,
            preemptions_delta=1.0 if idx == 4 else 0.0,
        )
        for idx in range(6)
    ]

    result = classify_bottleneck(windows)

    assert result.bottleneck_class == "kv_cache"
    assert result.confidence == "high"
    assert any("kv_cache_usage_perc reached saturation" in item for item in result.evidence)
    assert any("num_preemptions increased" in item for item in result.evidence)
    assert not any("num_swapped_mean missing" in item for item in result.evidence)


def test_num_swapped_is_additional_kv_cache_evidence_when_present() -> None:
    windows = [
        _window(
            idx,
            kv_cache_usage_mean=0.95,
            kv_cache_usage_max=0.99,
            num_swapped_mean=1.0 if idx == 4 else 0.0,
            preemptions_delta=1.0 if idx == 4 else 0.0,
        )
        for idx in range(6)
    ]

    result = classify_bottleneck(windows)

    assert result.bottleneck_class == "kv_cache"
    assert result.confidence == "high"
    assert any("legacy swapped-request metric added KV pressure evidence" in item for item in result.evidence)


def test_slo_violation_precedes_server_bottleneck_inference() -> None:
    stability = StabilityResult(
        status="slo_violation",
        confidence="high",
        reasons=["TTFT p90 SLO violated: max=2500.000 ms > 2000.000 ms"],
        key_metrics={},
    )

    result = classify_bottleneck(_stable_windows(), stability_result=stability)

    assert result.bottleneck_class == "slo_limited"
    assert result.confidence == "high"
    assert any("completion rate stayed close" in item for item in result.evidence)


def test_aborted_safety_precedes_server_bottleneck_inference() -> None:
    stability = StabilityResult(
        status="aborted_safety",
        confidence="high",
        reasons=["safety outstanding cap was reached during the trial"],
        key_metrics={},
    )

    result = classify_bottleneck(_stable_windows(), stability_result=stability)

    assert result.bottleneck_class == "client_limited"
    assert result.confidence == "high"
    assert any("safety cap invalidated" in item for item in result.evidence)


def test_open_loop_send_rate_lag_alone_does_not_imply_client_limited() -> None:
    summary = TrialSummary(
        trial_id="trial-bottleneck",
        mode="open-loop",
        status="completed",
        requested_request_rate=2.0,
        requested_concurrency=None,
        target_duration_s=20.0,
        wall_time_s=20.0,
        started_requests=38,
        successful_requests=38,
        failed_requests=0,
        actual_send_rate=1.91,
        successful_completion_rate=1.9,
        error_rate=0.0,
        mean_scheduling_delay_s=0.001,
        max_scheduling_delay_s=0.002,
        max_observed_outstanding=2,
        metrics_sample_count=0,
        abort_reason=None,
        benchmark_metrics=_benchmark_metrics(),
        metadata={},
    )

    result = classify_bottleneck(_stable_windows(), trial_summary=summary)

    assert result.bottleneck_class != "client_limited"


def test_open_loop_send_rate_lag_with_high_scheduling_delay_is_client_limited() -> None:
    summary = TrialSummary(
        trial_id="trial-bottleneck",
        mode="open-loop",
        status="completed",
        requested_request_rate=10.0,
        requested_concurrency=None,
        target_duration_s=20.0,
        wall_time_s=20.0,
        started_requests=160,
        successful_requests=160,
        failed_requests=0,
        actual_send_rate=8.9,
        successful_completion_rate=8.0,
        error_rate=0.0,
        mean_scheduling_delay_s=0.15,
        max_scheduling_delay_s=0.4,
        max_observed_outstanding=10,
        metrics_sample_count=0,
        abort_reason=None,
        benchmark_metrics=_benchmark_metrics(),
        metadata={},
    )

    result = classify_bottleneck(
        _stable_windows(),
        trial_summary=summary,
        config=BottleneckConfig(open_loop_send_rate_tolerance=0.05),
    )

    assert result.bottleneck_class == "client_limited"
    assert any("actual open-loop send rate lagged configured rate" in item for item in result.evidence)
    assert any("client scheduling delay was high" in item for item in result.evidence)


def test_context_length_errors_are_not_bottleneck_evidence() -> None:
    failed_record = RequestRecord(
        request_id="request-1",
        trial_id="trial-bottleneck",
        scheduled_send_ts=0.0,
        actual_send_ts=0.0,
        first_token_ts=None,
        end_ts=0.1,
        success=False,
        error="HTTP 400: prompt is too long for maximum context length",
        prompt_len=4096,
        expected_output_len=512,
        actual_output_len=None,
        ttft_s=None,
        e2e_s=0.1,
        tpot_s=None,
        itl_s=[],
        output_token_timestamps=[],
    )

    result = classify_bottleneck(
        _stable_windows(),
        request_records=[failed_record],
        server_metadata={"max_num_seqs": 8},
    )

    assert result.bottleneck_class == "unknown"
    assert result.confidence == "low"
    assert any("context validation failure" in item for item in result.evidence)
    assert any("not used to infer a server bottleneck" in item for item in result.evidence)


def test_conflicting_server_metadata_fails_fast() -> None:
    with pytest.raises(ValueError, match="conflicting supplied server metadata"):
        classify_bottleneck(
            _stable_windows(),
            server_metadata={
                "max_num_seqs": 8,
                "server_config": {"max_num_seqs": 16},
            },
        )


def _benchmark_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        successful_requests=1,
        failed_requests=0,
        total_input_tokens=10,
        total_output_tokens=2,
        request_throughput=1.0,
        successful_request_throughput=1.0,
        prompt_token_throughput=10.0,
        generation_token_throughput=2.0,
        total_token_throughput=12.0,
        mean_ttft_ms=100.0,
        median_ttft_ms=100.0,
        std_ttft_ms=0.0,
        percentiles_ttft_ms=[(50.0, 100.0)],
        mean_tpot_ms=20.0,
        median_tpot_ms=20.0,
        std_tpot_ms=0.0,
        percentiles_tpot_ms=[(50.0, 20.0)],
        mean_itl_ms=18.0,
        median_itl_ms=18.0,
        std_itl_ms=0.0,
        percentiles_itl_ms=[(50.0, 18.0)],
        mean_e2e_ms=600.0,
        median_e2e_ms=600.0,
        std_e2e_ms=0.0,
        percentiles_e2e_ms=[(50.0, 600.0)],
        prompt_length_summary={"mean": 10.0, "median": 10.0, "p90": 10.0, "p99": 10.0},
        output_length_summary={"mean": 2.0, "median": 2.0, "p90": 2.0, "p99": 2.0},
    )
