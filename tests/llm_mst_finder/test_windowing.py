from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_mst_finder.records import RequestRecord, ServerMetricSample
from llm_mst_finder.windowing import FixedWindowAggregator


def _request_record(
    *,
    request_id: str,
    trial_id: str = "trial-window",
    actual_send_ts: float,
    end_ts: float,
    success: bool = True,
    error: str | None = None,
    first_token_ts: float | None = None,
    actual_output_len: int | None = None,
    ttft_s: float | None = None,
    e2e_s: float | None = None,
    tpot_s: float | None = None,
    itl_s: list[float] | None = None,
    output_token_timestamps: list[float] | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        trial_id=trial_id,
        scheduled_send_ts=actual_send_ts,
        actual_send_ts=actual_send_ts,
        first_token_ts=first_token_ts,
        end_ts=end_ts,
        success=success,
        error=error,
        prompt_len=8,
        expected_output_len=4,
        actual_output_len=actual_output_len,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        tpot_s=tpot_s,
        itl_s=[] if itl_s is None else itl_s,
        output_token_timestamps=[] if output_token_timestamps is None else output_token_timestamps,
        metadata={},
    )


def _metric_sample(
    *,
    ts: float,
    num_running: float | None = None,
    num_waiting: float | None = None,
    num_swapped: float | None = None,
    kv_cache_usage: float | None = None,
    prompt_tokens_total: float | None = None,
    generation_tokens_total: float | None = None,
    preemptions_total: float | None = None,
) -> ServerMetricSample:
    raw: dict[str, object] = {}
    if preemptions_total is not None:
        raw["vllm:num_preemptions_total"] = [
            {"labels": {}, "value": preemptions_total, "timestamp_ms": None}
        ]
    return ServerMetricSample(
        ts=ts,
        raw=raw,
        num_running=num_running,
        num_waiting=num_waiting,
        num_swapped=num_swapped,
        kv_cache_usage=kv_cache_usage,
        prompt_tokens_total=prompt_tokens_total,
        generation_tokens_total=generation_tokens_total,
        request_success_total=None,
        request_abort_total=None,
    )


def test_fixed_window_aggregator_summarizes_requests_and_metrics() -> None:
    aggregator = FixedWindowAggregator(window_s=1.0)
    requests = [
        _request_record(
            request_id="req-0",
            actual_send_ts=0.0,
            first_token_ts=0.2,
            end_ts=0.6,
            actual_output_len=3,
            ttft_s=0.2,
            e2e_s=0.6,
            tpot_s=0.2,
            itl_s=[0.2, 0.2],
            output_token_timestamps=[0.2, 0.4, 0.6],
        ),
        _request_record(
            request_id="req-1",
            actual_send_ts=0.75,
            first_token_ts=1.05,
            end_ts=1.35,
            actual_output_len=2,
            ttft_s=0.3,
            e2e_s=0.6,
            tpot_s=0.3,
            itl_s=[0.3],
            output_token_timestamps=[1.05, 1.35],
        ),
        _request_record(
            request_id="req-2",
            actual_send_ts=1.2,
            end_ts=1.7,
            success=False,
            error="timeout",
            actual_output_len=None,
            e2e_s=0.5,
        ),
    ]
    metrics = [
        _metric_sample(
            ts=0.1,
            num_running=1.0,
            num_waiting=0.0,
            num_swapped=0.0,
            kv_cache_usage=0.4,
            prompt_tokens_total=10.0,
            generation_tokens_total=5.0,
            preemptions_total=0.0,
        ),
        _metric_sample(
            ts=0.9,
            num_running=1.5,
            num_waiting=0.5,
            num_swapped=0.0,
            kv_cache_usage=0.6,
            prompt_tokens_total=18.0,
            generation_tokens_total=9.0,
            preemptions_total=0.0,
        ),
        _metric_sample(
            ts=1.4,
            num_running=2.0,
            num_waiting=1.0,
            num_swapped=0.0,
            kv_cache_usage=0.8,
            prompt_tokens_total=30.0,
            generation_tokens_total=20.0,
            preemptions_total=2.0,
        ),
        _metric_sample(
            ts=1.9,
            num_running=0.0,
            num_waiting=0.0,
            num_swapped=0.0,
            kv_cache_usage=0.3,
            prompt_tokens_total=34.0,
            generation_tokens_total=24.0,
            preemptions_total=2.0,
        ),
    ]

    windows = aggregator.summarize(
        trial_id="trial-window",
        request_records=requests,
        server_metrics=metrics,
    )

    assert len(windows) == 2

    first = windows[0]
    assert first.arrivals == 2
    assert first.completions == 1
    assert first.failures == 0
    assert first.arrival_rate == pytest.approx(2.0)
    assert first.completion_rate == pytest.approx(1.0)
    assert first.error_rate == pytest.approx(0.0)
    assert first.outstanding_start == 0
    assert first.outstanding_end == 1
    assert first.outstanding_mean == pytest.approx(0.85)
    assert first.outstanding_slope == pytest.approx(1.0)
    assert first.ttft_p50_ms == pytest.approx(250.0)
    assert first.ttft_p90_ms == pytest.approx(290.0)
    assert first.tpot_p50_ms == pytest.approx(250.0)
    assert first.itl_p90_ms == pytest.approx(280.0)
    assert first.e2e_p99_ms == pytest.approx(600.0)
    assert first.prompt_tok_s == pytest.approx(8.0)
    assert first.generation_tok_s == pytest.approx(4.0)
    assert first.total_tok_s == pytest.approx(12.0)
    assert first.prompt_len_mean == pytest.approx(8.0)
    assert first.expected_output_len_mean == pytest.approx(4.0)
    assert first.actual_output_len_mean == pytest.approx(2.5)
    assert first.num_running_mean == pytest.approx(1.25)
    assert first.num_waiting_mean == pytest.approx(0.25)
    assert first.kv_cache_usage_mean == pytest.approx(0.5)
    assert first.kv_cache_usage_max == pytest.approx(0.6)
    assert first.preemptions_delta == pytest.approx(0.0)

    second = windows[1]
    assert second.arrivals == 1
    assert second.completions == 1
    assert second.failures == 1
    assert second.error_rate == pytest.approx(0.5)
    assert second.outstanding_start == 1
    assert second.outstanding_end == 0
    assert second.outstanding_mean == pytest.approx(0.85)
    assert second.outstanding_slope == pytest.approx(-1.0)
    assert second.ttft_p50_ms is None
    assert second.prompt_tok_s == pytest.approx(16.0)
    assert second.generation_tok_s == pytest.approx(15.0)
    assert second.total_tok_s == pytest.approx(31.0)
    assert second.prompt_len_mean == pytest.approx(8.0)
    assert second.expected_output_len_mean == pytest.approx(4.0)
    assert second.actual_output_len_mean is None
    assert second.num_running_mean == pytest.approx(1.0)
    assert second.num_waiting_mean == pytest.approx(0.5)
    assert second.kv_cache_usage_mean == pytest.approx(0.55)
    assert second.kv_cache_usage_max == pytest.approx(0.8)
    assert second.preemptions_delta == pytest.approx(2.0)


def test_fixed_window_aggregator_preserves_empty_windows() -> None:
    aggregator = FixedWindowAggregator(window_s=1.0)
    windows = aggregator.summarize(
        trial_id="trial-window",
        request_records=[
            _request_record(
                request_id="req-0",
                actual_send_ts=0.0,
                first_token_ts=0.1,
                end_ts=0.2,
                actual_output_len=2,
                ttft_s=0.1,
                e2e_s=0.2,
                tpot_s=0.1,
                itl_s=[0.1],
                output_token_timestamps=[0.1, 0.2],
            ),
            _request_record(
                request_id="req-1",
                actual_send_ts=3.2,
                first_token_ts=3.3,
                end_ts=3.4,
                actual_output_len=2,
                ttft_s=0.1,
                e2e_s=0.2,
                tpot_s=0.1,
                itl_s=[0.1],
                output_token_timestamps=[3.3, 3.4],
            ),
        ],
        server_metrics=[],
    )

    assert [window.window_idx for window in windows] == [0, 1, 2, 3]
    assert windows[1].arrivals == 0
    assert windows[1].completions == 0
    assert windows[1].failures == 0
    assert windows[1].arrival_rate == pytest.approx(0.0)
    assert windows[1].completion_rate == pytest.approx(0.0)
    assert windows[1].error_rate == pytest.approx(0.0)
    assert windows[1].outstanding_start == 0
    assert windows[1].outstanding_end == 0
    assert windows[1].outstanding_mean == pytest.approx(0.0)
    assert windows[1].outstanding_slope == pytest.approx(0.0)
    assert windows[1].ttft_p50_ms is None
    assert windows[1].prompt_tok_s is None
    assert windows[1].prompt_len_mean is None
    assert windows[1].expected_output_len_mean is None
    assert windows[1].actual_output_len_mean is None


def test_fixed_window_aggregator_writes_csv(tmp_path: Path) -> None:
    aggregator = FixedWindowAggregator(window_s=1.0)
    output_path = tmp_path / "windows.csv"

    written = aggregator.write_outputs(
        trial_id="trial-window",
        request_records=[
            _request_record(
                request_id="req-0",
                actual_send_ts=0.0,
                first_token_ts=0.1,
                end_ts=0.2,
                actual_output_len=2,
                ttft_s=0.1,
                e2e_s=0.2,
                tpot_s=0.1,
                itl_s=[0.1],
                output_token_timestamps=[0.1, 0.2],
            )
        ],
        server_metrics=[],
        output_path=output_path,
    )

    assert written == 1
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["trial_id"] == "trial-window"
    assert rows[0]["window_idx"] == "0"
    assert rows[0]["arrivals"] == "1"


def test_fixed_window_aggregator_rejects_counter_regression() -> None:
    aggregator = FixedWindowAggregator(window_s=1.0)

    with pytest.raises(ValueError, match="counter metric decreased"):
        aggregator.summarize(
            trial_id="trial-window",
            request_records=[
                _request_record(
                    request_id="req-0",
                    actual_send_ts=0.0,
                    first_token_ts=0.1,
                    end_ts=0.2,
                    actual_output_len=2,
                    ttft_s=0.1,
                    e2e_s=0.2,
                    tpot_s=0.1,
                    itl_s=[0.1],
                    output_token_timestamps=[0.1, 0.2],
                )
            ],
            server_metrics=[
                _metric_sample(ts=0.05, prompt_tokens_total=10.0),
                _metric_sample(ts=0.9, prompt_tokens_total=9.0),
            ],
        )


def test_fixed_window_aggregator_accepts_current_vllm_num_preemptions_name() -> None:
    aggregator = FixedWindowAggregator(window_s=1.0)

    windows = aggregator.summarize(
        trial_id="trial-window",
        request_records=[
            _request_record(
                request_id="req-0",
                actual_send_ts=0.0,
                first_token_ts=0.1,
                end_ts=1.2,
                actual_output_len=2,
                ttft_s=0.1,
                e2e_s=1.2,
                tpot_s=1.1,
                itl_s=[1.1],
                output_token_timestamps=[0.1, 1.2],
            )
        ],
        server_metrics=[
            ServerMetricSample(
                ts=0.05,
                raw={
                    "vllm:num_preemptions": [
                        {"labels": {}, "value": 0.0, "timestamp_ms": None}
                    ]
                },
                num_running=1.0,
                num_waiting=0.0,
                num_swapped=None,
                kv_cache_usage=0.99,
                prompt_tokens_total=None,
                generation_tokens_total=None,
                request_success_total=None,
                request_abort_total=None,
            ),
            ServerMetricSample(
                ts=1.05,
                raw={
                    "vllm:num_preemptions": [
                        {"labels": {}, "value": 3.0, "timestamp_ms": None}
                    ]
                },
                num_running=1.0,
                num_waiting=0.0,
                num_swapped=None,
                kv_cache_usage=0.99,
                prompt_tokens_total=None,
                generation_tokens_total=None,
                request_success_total=None,
                request_abort_total=None,
            ),
        ],
    )

    assert windows[1].preemptions_delta == pytest.approx(3.0)
