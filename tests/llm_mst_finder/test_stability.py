from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_mst_finder.records import WindowSummary
from llm_mst_finder.stability import (
    StabilityConfig,
    classify_stability,
    load_window_summaries_csv,
)


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
    tpot_p99_ms: float | None = 25.0,
    e2e_p90_ms: float | None = 600.0,
    num_running_mean: float | None = 8.0,
    num_waiting_mean: float | None = 0.0,
    num_swapped_mean: float | None = 0.0,
    kv_cache_usage_mean: float | None = 0.40,
    kv_cache_usage_max: float | None = 0.45,
    preemptions_delta: float | None = 0.0,
) -> WindowSummary:
    terminal_events = completions + failures
    return WindowSummary(
        trial_id="trial-stability",
        window_idx=idx,
        start_s=float(idx),
        end_s=float(idx + 1),
        arrivals=arrivals,
        completions=completions,
        failures=failures,
        arrival_rate=float(arrivals),
        completion_rate=float(completions),
        error_rate=failures / terminal_events if terminal_events else 0.0,
        outstanding_start=0 if idx == 0 else outstanding_end,
        outstanding_end=outstanding_end,
        outstanding_mean=float(outstanding_end),
        outstanding_slope=0.0,
        ttft_p50_ms=80.0 if ttft_p90_ms is not None else None,
        ttft_p90_ms=ttft_p90_ms,
        ttft_p99_ms=ttft_p99_ms,
        tpot_p50_ms=15.0 if tpot_p90_ms is not None else None,
        tpot_p90_ms=tpot_p90_ms,
        tpot_p99_ms=tpot_p99_ms,
        itl_p90_ms=18.0 if tpot_p90_ms is not None else None,
        e2e_p90_ms=e2e_p90_ms,
        e2e_p99_ms=e2e_p90_ms + 100.0 if e2e_p90_ms is not None else None,
        prompt_tok_s=1000.0,
        generation_tok_s=300.0,
        total_tok_s=1300.0,
        num_running_mean=num_running_mean,
        num_waiting_mean=num_waiting_mean,
        num_swapped_mean=num_swapped_mean,
        kv_cache_usage_mean=kv_cache_usage_mean,
        kv_cache_usage_max=kv_cache_usage_max,
        preemptions_delta=preemptions_delta,
    )


def _stable_windows() -> list[WindowSummary]:
    return [_window(idx) for idx in range(6)]


def test_classifies_stationary_windows_as_stable() -> None:
    result = classify_stability(_stable_windows())

    assert result.status == "stable"
    assert result.confidence == "high"
    assert result.key_metrics["completion_arrival_ratio"] == pytest.approx(1.0)
    assert any("checks passed" in reason for reason in result.reasons)


def test_accepts_negative_within_window_outstanding_slope() -> None:
    windows = _stable_windows()
    windows[3] = WindowSummary(**{**windows[3].to_dict(), "outstanding_slope": -1.0})

    result = classify_stability(windows)

    assert result.status == "stable"


def test_classifies_ttft_drift_as_unstable() -> None:
    windows = _stable_windows()
    ttft_values = [100.0, 100.0, 100.0, 125.0, 160.0, 230.0]
    windows = [
        _window(idx, ttft_p90_ms=ttft, ttft_p99_ms=ttft + 20.0)
        for idx, ttft in enumerate(ttft_values)
    ]

    result = classify_stability(windows)

    assert result.status == "unstable"
    assert result.confidence == "high"
    assert any("TTFT p90 drifted upward" in reason for reason in result.reasons)


def test_classifies_tpot_drift_as_unstable() -> None:
    tpot_values = [20.0, 20.0, 20.0, 24.0, 30.0, 44.0]
    windows = [
        _window(idx, tpot_p90_ms=tpot, tpot_p99_ms=tpot + 5.0)
        for idx, tpot in enumerate(tpot_values)
    ]

    result = classify_stability(windows)

    assert result.status == "unstable"
    assert any("TPOT p90 drifted upward" in reason for reason in result.reasons)


def test_classifies_completion_lag_and_backlog_as_unstable() -> None:
    windows = [_window(0), _window(1)]
    windows.extend(
        _window(idx, completions=8, outstanding_end=idx - 1, num_waiting_mean=float(idx - 1))
        for idx in range(2, 6)
    )

    result = classify_stability(windows)

    assert result.status == "unstable"
    assert any("completion rate lagged arrivals" in reason for reason in result.reasons)
    assert any("outstanding" in reason for reason in result.reasons)


def test_classifies_preemptions_as_unstable() -> None:
    windows = _stable_windows()
    windows[4] = _window(4, preemptions_delta=1.0)

    result = classify_stability(windows)

    assert result.status == "unstable"
    assert any("preemptions observed" in reason for reason in result.reasons)


def test_classifies_stationary_slo_breach_as_slo_violation() -> None:
    windows = [_window(idx, ttft_p90_ms=2500.0, ttft_p99_ms=2600.0) for idx in range(6)]

    result = classify_stability(windows)

    assert result.status == "slo_violation"
    assert result.confidence == "high"
    assert any("TTFT p90 SLO violated" in reason for reason in result.reasons)


def test_insufficient_eval_windows_is_uncertain() -> None:
    result = classify_stability(
        _stable_windows()[:5],
        config=StabilityConfig(warmup_windows=2, min_eval_windows=4),
    )

    assert result.status == "uncertain"
    assert result.confidence == "low"
    assert any("insufficient post-warmup windows" in reason for reason in result.reasons)


def test_missing_server_metrics_lowers_confidence_but_does_not_block_classification() -> None:
    windows = [
        _window(
            idx,
            num_running_mean=None,
            num_waiting_mean=None,
            num_swapped_mean=None,
            kv_cache_usage_mean=None,
            kv_cache_usage_max=None,
            preemptions_delta=None,
        )
        for idx in range(6)
    ]

    result = classify_stability(windows)

    assert result.status == "stable"
    assert result.confidence == "medium"
    assert any("server-side evidence missing" in reason for reason in result.reasons)


def test_missing_latency_metrics_without_other_signal_is_uncertain() -> None:
    windows = [
        _window(
            idx,
            ttft_p90_ms=None,
            ttft_p99_ms=None,
            tpot_p90_ms=None,
            tpot_p99_ms=None,
        )
        for idx in range(6)
    ]

    result = classify_stability(windows)

    assert result.status == "uncertain"
    assert result.confidence == "low"
    assert any("cannot prove stability" in reason for reason in result.reasons)


def test_aborted_safety_status_takes_precedence() -> None:
    result = classify_stability(_stable_windows(), aborted_safety=True)

    assert result.status == "aborted_safety"
    assert result.confidence == "high"


def test_rejects_inconsistent_required_window_fields() -> None:
    bad = _window(0, arrivals=10)
    bad = WindowSummary(**{**bad.to_dict(), "arrival_rate": 9.0})

    with pytest.raises(ValueError, match="arrival_rate"):
        classify_stability([bad, *_stable_windows()[1:]])


def test_load_window_summaries_csv_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "windows.csv"
    windows = _stable_windows()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(windows[0].to_dict())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            writer.writerow(window.to_dict())

    loaded = load_window_summaries_csv(output_path)

    assert len(loaded) == 6
    assert classify_stability(loaded).status == "stable"
