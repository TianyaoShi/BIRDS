from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Literal, Sequence

from scipy import stats

from .records import RequestRecord, StabilityResult, WindowSummary


Status = Literal["stable", "unstable", "slo_violation", "uncertain", "aborted_safety"]
Confidence = Literal["high", "medium", "low"]
TtftSloMode = Literal["static", "length_scaled"]
LongBenchTtftStaticPreset = Literal["default", "tight", "relaxed"]

TTFT_SLO_LENGTH_SCALED: dict[str, dict[str, float]] = {
    "long_output_summarization": {
        "base": 8.0,
        "per_1k_input_tokens": 1.4,
        "cap": 45.0,
    },
    "medium_output_summarization": {
        "base": 6.0,
        "per_1k_input_tokens": 1.2,
        "cap": 40.0,
    },
    "medium_answer_rag_qa": {
        "base": 5.0,
        "per_1k_input_tokens": 0.9,
        "cap": 30.0,
    },
    "short_answer_document_qa": {
        "base": 4.0,
        "per_1k_input_tokens": 0.8,
        "cap": 20.0,
    },
}

TTFT_SLO_LONGBENCH_STATIC_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "long_output_summarization": 35.0,
        "medium_output_summarization": 30.0,
        "medium_answer_rag_qa": 20.0,
        "short_answer_document_qa": 15.0,
    },
    "tight": {
        "long_output_summarization": 25.0,
        "medium_output_summarization": 22.0,
        "medium_answer_rag_qa": 15.0,
        "short_answer_document_qa": 10.0,
    },
    "relaxed": {
        "long_output_summarization": 45.0,
        "medium_output_summarization": 40.0,
        "medium_answer_rag_qa": 30.0,
        "short_answer_document_qa": 20.0,
    },
}


@dataclass(frozen=True, slots=True)
class DriftTestConfig:
    method: Literal["theil_sen"] = "theil_sen"
    min_relative_increase: float = 0.20

    def __post_init__(self) -> None:
        if self.method != "theil_sen":
            raise ValueError(f"unsupported drift test method {self.method!r}")
        _require_non_negative_finite("min_relative_increase", self.min_relative_increase)


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    warmup_windows: int = 2
    min_eval_windows: int = 4
    completion_arrival_tolerance: float = 0.05
    max_positive_backlog_slope: float = 0.10
    min_backlog_growth_for_hard_pressure: float = 2.0
    min_backlog_relative_increase: float = 0.10
    backlog_trend_alpha: float = 0.05
    min_waiting_queue_mean_for_pressure: float = 1.0
    min_waiting_queue_active_fraction: float = 0.5
    token_throughput_plateau_relative_growth: float = 0.05
    max_error_rate: float = 0.03
    ttft_slo_ms: float | None = 2000.0
    tpot_slo_ms: float | None = 80.0
    ttft_slo_field: str = "ttft_p90_ms"
    tpot_slo_field: str = "tpot_p90_ms"
    ttft_slo_mode: TtftSloMode = "static"
    longbench_ttft_static_preset: LongBenchTtftStaticPreset | None = None
    drift_test: DriftTestConfig = field(default_factory=DriftTestConfig)

    def __post_init__(self) -> None:
        _require_non_negative_int("warmup_windows", self.warmup_windows)
        _require_positive_int("min_eval_windows", self.min_eval_windows)
        if self.min_eval_windows < 2:
            raise ValueError("min_eval_windows must be at least 2 for trend estimation")
        _require_non_negative_finite(
            "completion_arrival_tolerance",
            self.completion_arrival_tolerance,
        )
        if self.completion_arrival_tolerance >= 1.0:
            raise ValueError("completion_arrival_tolerance must be less than 1.0")
        _require_non_negative_finite("max_positive_backlog_slope", self.max_positive_backlog_slope)
        _require_non_negative_finite(
            "min_backlog_growth_for_hard_pressure",
            self.min_backlog_growth_for_hard_pressure,
        )
        _require_non_negative_finite(
            "min_backlog_relative_increase",
            self.min_backlog_relative_increase,
        )
        _require_non_negative_finite("backlog_trend_alpha", self.backlog_trend_alpha)
        if self.backlog_trend_alpha <= 0.0 or self.backlog_trend_alpha >= 1.0:
            raise ValueError("backlog_trend_alpha must be between 0 and 1")
        _require_non_negative_finite(
            "min_waiting_queue_mean_for_pressure",
            self.min_waiting_queue_mean_for_pressure,
        )
        if self.min_waiting_queue_mean_for_pressure <= 0.0:
            raise ValueError("min_waiting_queue_mean_for_pressure must be positive")
        _require_non_negative_finite(
            "min_waiting_queue_active_fraction",
            self.min_waiting_queue_active_fraction,
        )
        if self.min_waiting_queue_active_fraction > 1.0:
            raise ValueError("min_waiting_queue_active_fraction must be at most 1.0")
        _require_non_negative_finite(
            "token_throughput_plateau_relative_growth",
            self.token_throughput_plateau_relative_growth,
        )
        _require_non_negative_finite("max_error_rate", self.max_error_rate)
        if self.max_error_rate >= 1.0:
            raise ValueError("max_error_rate must be less than 1.0")
        _require_optional_positive_finite("ttft_slo_ms", self.ttft_slo_ms)
        _require_optional_positive_finite("tpot_slo_ms", self.tpot_slo_ms)
        _require_slo_field("ttft_slo_field", self.ttft_slo_field, {"ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms"})
        _require_slo_field("tpot_slo_field", self.tpot_slo_field, {"tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms"})
        if self.ttft_slo_mode not in {"static", "length_scaled"}:
            raise ValueError(f"unsupported ttft_slo_mode {self.ttft_slo_mode!r}")
        if self.longbench_ttft_static_preset is not None:
            if self.longbench_ttft_static_preset not in TTFT_SLO_LONGBENCH_STATIC_PRESETS:
                raise ValueError(
                    "longbench_ttft_static_preset must be one of: "
                    + ", ".join(sorted(TTFT_SLO_LONGBENCH_STATIC_PRESETS))
                )
            if self.ttft_slo_mode != "static":
                raise ValueError("longbench_ttft_static_preset requires ttft_slo_mode='static'")


@dataclass(frozen=True, slots=True)
class _Trend:
    slope: float
    relative_increase: float
    delta: float
    p_value: float
    slope_ci_low: float
    slope_ci_high: float


def classify_stability(
    windows: Sequence[WindowSummary],
    *,
    config: StabilityConfig | None = None,
    aborted_safety: bool = False,
    request_records: Sequence[RequestRecord] | None = None,
    trial_start_ts: float | None = None,
) -> StabilityResult:
    stability_config = StabilityConfig() if config is None else config
    validated = _validate_windows(windows)
    key_metrics: dict[str, float] = {"total_windows": float(len(validated))}

    if aborted_safety:
        return StabilityResult(
            status="aborted_safety",
            confidence="high",
            reasons=["safety outstanding cap was reached during the trial"],
            key_metrics=key_metrics,
        )

    eval_windows = validated[stability_config.warmup_windows :]
    active_eval_windows = [window for window in eval_windows if window.arrivals > 0]
    key_metrics["eval_windows"] = float(len(eval_windows))
    key_metrics["active_eval_windows"] = float(len(active_eval_windows))
    key_metrics["drain_eval_windows"] = float(len(eval_windows) - len(active_eval_windows))
    if len(eval_windows) < stability_config.min_eval_windows:
        return StabilityResult(
            status="uncertain",
            confidence="low",
            reasons=[
                "insufficient post-warmup windows: "
                f"{len(eval_windows)} < {stability_config.min_eval_windows}"
            ],
            key_metrics=key_metrics,
        )
    if len(active_eval_windows) < stability_config.min_eval_windows:
        return StabilityResult(
            status="uncertain",
            confidence="low",
            reasons=[
                "insufficient post-warmup active-arrival windows: "
                f"{len(active_eval_windows)} < {stability_config.min_eval_windows}"
            ],
            key_metrics=key_metrics,
        )

    total_arrivals = sum(window.arrivals for window in active_eval_windows)
    total_completions = sum(window.completions for window in active_eval_windows)
    total_failures = sum(window.failures for window in active_eval_windows)
    terminal_events = total_completions + total_failures

    if total_arrivals <= 0:
        return StabilityResult(
            status="uncertain",
            confidence="low",
            reasons=["post-warmup windows contain no arrivals"],
            key_metrics=key_metrics,
        )

    completion_arrival_ratio = total_completions / total_arrivals
    aggregate_error_rate = total_failures / terminal_events if terminal_events else 0.0
    key_metrics.update(
        {
            "completion_arrival_ratio": completion_arrival_ratio,
            "aggregate_error_rate": aggregate_error_rate,
            "total_arrivals": float(total_arrivals),
            "total_completions": float(total_completions),
            "total_failures": float(total_failures),
        }
    )

    reasons: list[str] = []
    confidence_penalties = 0
    direct_unstable_reasons: list[str] = []
    capacity_pressure_reasons: list[str] = []
    completion_lag_reasons: list[str] = []
    latency_drift_reasons: list[str] = []
    has_capacity_pressure = False

    if completion_arrival_ratio < 1.0 - stability_config.completion_arrival_tolerance:
        completion_lag_reasons.append(
            "completion rate lagged arrivals: "
            f"completion/arrival={completion_arrival_ratio:.3f} "
            f"< {1.0 - stability_config.completion_arrival_tolerance:.3f}"
        )

    if aggregate_error_rate > stability_config.max_error_rate:
        direct_unstable_reasons.append(
            f"error rate {aggregate_error_rate:.3f} exceeded threshold "
            f"{stability_config.max_error_rate:.3f}"
        )

    outstanding_trend = _trend_for_required(active_eval_windows, "outstanding_end")
    key_metrics["outstanding_end_slope_per_s"] = outstanding_trend.slope
    key_metrics["outstanding_end_delta"] = outstanding_trend.delta
    key_metrics["outstanding_end_relative_increase"] = outstanding_trend.relative_increase
    key_metrics["outstanding_end_mann_kendall_p"] = outstanding_trend.p_value
    key_metrics["outstanding_end_slope_ci_low"] = outstanding_trend.slope_ci_low
    key_metrics["outstanding_end_slope_ci_high"] = outstanding_trend.slope_ci_high
    if _is_clear_positive_backlog_trend(outstanding_trend, stability_config):
        capacity_pressure_reasons.append(
            "outstanding requests drifted upward: "
            f"Theil-Sen slope={outstanding_trend.slope:.3f}/s "
            f"> {stability_config.max_positive_backlog_slope:.3f}/s, "
            f"Mann-Kendall p={outstanding_trend.p_value:.3g} "
            f"< {stability_config.backlog_trend_alpha:.3g}, "
            f"relative increase={outstanding_trend.relative_increase:.3f} "
            f">= {stability_config.min_backlog_relative_increase:.3f}, "
            f"delta={outstanding_trend.delta:.3f}"
        )
        capacity_pressure_reasons.extend(completion_lag_reasons)
        has_capacity_pressure = True

    num_waiting_trend = _trend_for_optional(active_eval_windows, "num_waiting_mean")
    if num_waiting_trend is not None:
        key_metrics["num_waiting_mean_slope_per_s"] = num_waiting_trend.slope
        key_metrics["num_waiting_mean_delta"] = num_waiting_trend.delta
        key_metrics["num_waiting_mean_mann_kendall_p"] = num_waiting_trend.p_value
        waiting_values = _present_values(active_eval_windows, "num_waiting_mean")
        if waiting_values:
            waiting_max = max(waiting_values)
            waiting_mean = sum(waiting_values) / len(waiting_values)
            waiting_positive_fraction = sum(value > 0.0 for value in waiting_values) / len(waiting_values)
            key_metrics["num_waiting_mean_max"] = waiting_max
            key_metrics["num_waiting_mean_mean"] = waiting_mean
            key_metrics["num_waiting_positive_window_fraction"] = waiting_positive_fraction
            sustained_waiting_pressure = (
                waiting_mean >= stability_config.min_waiting_queue_mean_for_pressure
                and waiting_positive_fraction >= stability_config.min_waiting_queue_active_fraction
            )
            rising_waiting_pressure = (
                waiting_max >= stability_config.min_waiting_queue_mean_for_pressure
                and num_waiting_trend.slope > stability_config.max_positive_backlog_slope
                and num_waiting_trend.delta
                >= stability_config.min_backlog_growth_for_hard_pressure
            )
            if sustained_waiting_pressure or rising_waiting_pressure:
                direct_unstable_reasons.append(
                    "server waiting queue showed material pressure after warmup: "
                    f"max num_waiting_mean={waiting_max:.3f}, "
                    f"mean={waiting_mean:.3f}, "
                    f"positive_window_fraction={waiting_positive_fraction:.3f}, "
                    f"Theil-Sen slope={num_waiting_trend.slope:.3f}/s"
                )

    swapped_values = _present_values(active_eval_windows, "num_swapped_mean")
    if swapped_values:
        key_metrics["num_swapped_mean_max"] = max(swapped_values)
        if max(swapped_values) > 0.0:
            direct_unstable_reasons.append(
                f"server reported swapped requests: max num_swapped_mean={max(swapped_values):.3f}"
            )

    kv_values = _present_values(active_eval_windows, "kv_cache_usage_max")
    if kv_values:
        key_metrics["kv_cache_usage_max"] = max(kv_values)
        kv_trend = _trend_for_optional(active_eval_windows, "kv_cache_usage_max")
        if kv_trend is not None:
            key_metrics["kv_cache_usage_max_slope_per_s"] = kv_trend.slope
            if max(kv_values) >= 0.98 and kv_trend.slope > 0.0:
                direct_unstable_reasons.append(
                    "KV cache usage approached saturation while rising: "
                    f"max={max(kv_values):.3f}"
                )

    preemption_values = _present_values(active_eval_windows, "preemptions_delta")
    if preemption_values:
        total_preemptions = sum(preemption_values)
        key_metrics["preemptions_total"] = total_preemptions
        if total_preemptions > 0.0:
            direct_unstable_reasons.append(
                f"preemptions observed after warmup: total={total_preemptions:.3f}"
            )

    generation_tok_s_trend = _trend_for_optional(active_eval_windows, "generation_tok_s")
    generation_plateau = False
    if generation_tok_s_trend is not None:
        key_metrics["generation_tok_s_slope_per_s"] = generation_tok_s_trend.slope
        key_metrics["generation_tok_s_relative_increase"] = generation_tok_s_trend.relative_increase
        generation_plateau = _is_token_throughput_plateau(generation_tok_s_trend, stability_config)
    if capacity_pressure_reasons and generation_plateau:
        direct_unstable_reasons.extend(capacity_pressure_reasons)
        direct_unstable_reasons.append(
            "generation token throughput plateaued while backlog/completion pressure grew: "
            f"relative increase={generation_tok_s_trend.relative_increase:.3f}"
        )

    for field_name, display_name in (
        ("ttft_p90_ms", "TTFT p90"),
        ("ttft_p99_ms", "TTFT p99"),
        ("tpot_p90_ms", "TPOT p90"),
        ("tpot_p99_ms", "TPOT p99"),
    ):
        trend = _trend_for_optional(active_eval_windows, field_name)
        if trend is None:
            continue
        key_metrics[f"{field_name}_slope_per_s"] = trend.slope
        key_metrics[f"{field_name}_relative_increase"] = trend.relative_increase
        key_metrics[f"{field_name}_mann_kendall_p"] = trend.p_value
        if _is_positive_latency_drift(trend, stability_config):
            latency_drift_reasons.append(
                f"{display_name} drifted upward: relative increase="
                f"{trend.relative_increase:.3f}"
            )

    if latency_drift_reasons:
        if direct_unstable_reasons:
            direct_unstable_reasons.extend(latency_drift_reasons)
        elif has_capacity_pressure:
            capacity_pressure_reasons.extend(latency_drift_reasons)
        else:
            confidence_penalties += 1
            reasons.append(
                "latency percentile drift was observed without server/backlog pressure; "
                "not treated as overload evidence"
            )

    missing_server_fields = _missing_optional_server_fields(active_eval_windows)
    if missing_server_fields:
        confidence_penalties += 1
        reasons.append(
            "server-side evidence missing for "
            + ", ".join(missing_server_fields)
            + "; confidence lowered"
        )

    missing_latency_evidence = _missing_latency_evidence(active_eval_windows)
    if missing_latency_evidence:
        confidence_penalties += 1
        reasons.append(
            "latency evidence missing for "
            + ", ".join(missing_latency_evidence)
            + "; confidence lowered"
        )

    slo_reasons = _slo_reasons(
        active_eval_windows,
        stability_config,
        key_metrics,
        request_records=request_records,
        trial_start_ts=trial_start_ts,
    )
    if slo_reasons:
        slo_priority_reasons = list(slo_reasons)
        slo_priority_reasons.extend(reasons)
        if direct_unstable_reasons:
            slo_priority_reasons.extend(direct_unstable_reasons)
        return StabilityResult(
            status="slo_violation",
            confidence=_confidence_after_penalties("high", confidence_penalties),
            reasons=slo_priority_reasons,
            key_metrics=key_metrics,
        )

    if direct_unstable_reasons:
        reasons.extend(direct_unstable_reasons)
        return StabilityResult(
            status="unstable",
            confidence=_confidence_after_penalties("high", confidence_penalties),
            reasons=reasons,
            key_metrics=key_metrics,
        )

    if capacity_pressure_reasons:
        phase_reason = _workload_phase_reason(
            active_eval_windows,
            generation_tok_s_trend=generation_tok_s_trend,
            config=stability_config,
            key_metrics=key_metrics,
        )
        if phase_reason is not None:
            reasons.append(phase_reason)
        reasons.extend(capacity_pressure_reasons)
        reasons.append(
            "capacity pressure was observed, but no waiting queue, saturation signal, SLO violation, "
            "or generation-token throughput plateau confirmed overload"
        )
        return StabilityResult(
            status="uncertain",
            confidence=_confidence_after_penalties("medium", confidence_penalties),
            reasons=reasons,
            key_metrics=key_metrics,
        )

    if missing_latency_evidence:
        return StabilityResult(
            status="uncertain",
            confidence="low",
            reasons=reasons + ["cannot prove stability without post-warmup latency percentiles"],
            key_metrics=key_metrics,
        )

    reasons.append(
        "post-warmup active-window completion rate, backlog, latency drift, error rate, and SLO checks passed"
    )
    return StabilityResult(
        status="stable",
        confidence=_confidence_after_penalties("high", confidence_penalties),
        reasons=reasons,
        key_metrics=key_metrics,
    )


def load_window_summaries_csv(path: str | Path) -> list[WindowSummary]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_window_from_csv_row(row, row_idx=index + 2) for index, row in enumerate(reader)]


def _window_from_csv_row(row: dict[str, str], *, row_idx: int) -> WindowSummary:
    required = {field for field in WindowSummary.__dataclass_fields__} - _BACKCOMPAT_OPTIONAL_WINDOW_FIELDS
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{row_idx}: windows CSV missing columns: {', '.join(missing)}")
    return WindowSummary(
        trial_id=_csv_str(row, "trial_id", row_idx),
        window_idx=_csv_int(row, "window_idx", row_idx),
        start_s=_csv_float(row, "start_s", row_idx),
        end_s=_csv_float(row, "end_s", row_idx),
        arrivals=_csv_int(row, "arrivals", row_idx),
        completions=_csv_int(row, "completions", row_idx),
        failures=_csv_int(row, "failures", row_idx),
        arrival_rate=_csv_float(row, "arrival_rate", row_idx),
        completion_rate=_csv_float(row, "completion_rate", row_idx),
        error_rate=_csv_float(row, "error_rate", row_idx),
        outstanding_start=_csv_int(row, "outstanding_start", row_idx),
        outstanding_end=_csv_int(row, "outstanding_end", row_idx),
        outstanding_mean=_csv_float(row, "outstanding_mean", row_idx),
        outstanding_slope=_csv_float(row, "outstanding_slope", row_idx),
        ttft_p50_ms=_csv_optional_float(row, "ttft_p50_ms", row_idx),
        ttft_p90_ms=_csv_optional_float(row, "ttft_p90_ms", row_idx),
        ttft_p95_ms=_csv_optional_float_missing_ok(row, "ttft_p95_ms", row_idx),
        ttft_p99_ms=_csv_optional_float(row, "ttft_p99_ms", row_idx),
        tpot_p50_ms=_csv_optional_float(row, "tpot_p50_ms", row_idx),
        tpot_p90_ms=_csv_optional_float(row, "tpot_p90_ms", row_idx),
        tpot_p95_ms=_csv_optional_float_missing_ok(row, "tpot_p95_ms", row_idx),
        tpot_p99_ms=_csv_optional_float(row, "tpot_p99_ms", row_idx),
        itl_p50_ms=_csv_optional_float_missing_ok(row, "itl_p50_ms", row_idx),
        itl_p90_ms=_csv_optional_float(row, "itl_p90_ms", row_idx),
        itl_p95_ms=_csv_optional_float_missing_ok(row, "itl_p95_ms", row_idx),
        itl_p99_ms=_csv_optional_float_missing_ok(row, "itl_p99_ms", row_idx),
        e2e_p50_ms=_csv_optional_float_missing_ok(row, "e2e_p50_ms", row_idx),
        e2e_p90_ms=_csv_optional_float(row, "e2e_p90_ms", row_idx),
        e2e_p95_ms=_csv_optional_float_missing_ok(row, "e2e_p95_ms", row_idx),
        e2e_p99_ms=_csv_optional_float(row, "e2e_p99_ms", row_idx),
        prompt_tok_s=_csv_optional_float(row, "prompt_tok_s", row_idx),
        generation_tok_s=_csv_optional_float(row, "generation_tok_s", row_idx),
        total_tok_s=_csv_optional_float(row, "total_tok_s", row_idx),
        num_running_mean=_csv_optional_float(row, "num_running_mean", row_idx),
        num_waiting_mean=_csv_optional_float(row, "num_waiting_mean", row_idx),
        num_swapped_mean=_csv_optional_float(row, "num_swapped_mean", row_idx),
        kv_cache_usage_mean=_csv_optional_float(row, "kv_cache_usage_mean", row_idx),
        kv_cache_usage_max=_csv_optional_float(row, "kv_cache_usage_max", row_idx),
        preemptions_delta=_csv_optional_float(row, "preemptions_delta", row_idx),
        prompt_len_mean=_csv_optional_float_missing_ok(row, "prompt_len_mean", row_idx),
        expected_output_len_mean=_csv_optional_float_missing_ok(
            row, "expected_output_len_mean", row_idx
        ),
        actual_output_len_mean=_csv_optional_float_missing_ok(
            row, "actual_output_len_mean", row_idx
        ),
    )


def _validate_windows(windows: Sequence[WindowSummary]) -> list[WindowSummary]:
    if not windows:
        raise ValueError("windows must be non-empty")
    validated = list(windows)
    trial_id = validated[0].trial_id
    previous_end: float | None = None
    previous_idx: int | None = None
    for window in validated:
        if window.trial_id != trial_id:
            raise ValueError("all windows must have the same trial_id")
        if previous_idx is not None and window.window_idx != previous_idx + 1:
            raise ValueError("windows must have contiguous window_idx values")
        if previous_end is not None and window.start_s < previous_end:
            raise ValueError("windows must be sorted and non-overlapping")
        duration_s = window.end_s - window.start_s
        if duration_s <= 0.0:
            raise ValueError(f"window {window.window_idx} duration must be positive")
        _require_non_negative_int("arrivals", window.arrivals)
        _require_non_negative_int("completions", window.completions)
        _require_non_negative_int("failures", window.failures)
        _require_non_negative_int("outstanding_start", window.outstanding_start)
        _require_non_negative_int("outstanding_end", window.outstanding_end)
        for field_name in (
            "arrival_rate",
            "completion_rate",
            "error_rate",
            "outstanding_mean",
        ):
            _require_non_negative_finite(field_name, float(getattr(window, field_name)))
        _require_finite("outstanding_slope", float(window.outstanding_slope))
        _require_rate_consistency(
            "arrival_rate",
            observed=window.arrival_rate,
            expected=window.arrivals / duration_s,
            window_idx=window.window_idx,
        )
        _require_rate_consistency(
            "completion_rate",
            observed=window.completion_rate,
            expected=window.completions / duration_s,
            window_idx=window.window_idx,
        )
        terminal_events = window.completions + window.failures
        expected_error_rate = window.failures / terminal_events if terminal_events else 0.0
        _require_rate_consistency(
            "error_rate",
            observed=window.error_rate,
            expected=expected_error_rate,
            window_idx=window.window_idx,
        )
        for field_name in _OPTIONAL_FLOAT_FIELDS:
            value = getattr(window, field_name)
            if value is not None:
                _require_non_negative_finite(field_name, float(value))
        previous_end = window.end_s
        previous_idx = window.window_idx
    return validated


def _require_rate_consistency(
    name: str,
    *,
    observed: float,
    expected: float,
    window_idx: int,
) -> None:
    if abs(observed - expected) > 1e-6:
        raise ValueError(
            f"window {window_idx} {name}={observed!r} is inconsistent with expected {expected!r}"
        )


def _trend_for_required(windows: Sequence[WindowSummary], field_name: str) -> _Trend:
    x_values: list[float] = []
    y_values: list[float] = []
    for window in windows:
        x_values.append((window.start_s + window.end_s) / 2.0)
        y_values.append(float(getattr(window, field_name)))
    return _theil_sen_trend(x_values, y_values)


def _trend_for_optional(windows: Sequence[WindowSummary], field_name: str) -> _Trend | None:
    x_values: list[float] = []
    y_values: list[float] = []
    for window in windows:
        value = getattr(window, field_name)
        if value is None:
            continue
        x_values.append((window.start_s + window.end_s) / 2.0)
        y_values.append(float(value))
    if len(y_values) < 2:
        return None
    return _theil_sen_trend(x_values, y_values)


def _theil_sen_trend(x_values: Sequence[float], y_values: Sequence[float]) -> _Trend:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        raise ValueError("Theil-Sen trend requires at least two points")
    for left, right in zip(x_values, x_values[1:]):
        if right <= left:
            raise ValueError("trend x_values must be strictly increasing")

    slope_result = stats.theilslopes(y_values, x_values, alpha=0.95)
    slope = float(slope_result.slope)
    slope_ci_low = float(slope_result.low_slope)
    slope_ci_high = float(slope_result.high_slope)
    kendall_result = stats.kendalltau(x_values, y_values, nan_policy="raise")
    p_value = float(kendall_result.pvalue)

    delta = slope * (x_values[-1] - x_values[0])
    first_half = y_values[: max(1, len(y_values) // 2)]
    baseline = max(abs(median(first_half)), 1e-9)
    relative_increase = delta / baseline
    return _Trend(
        slope=slope,
        relative_increase=relative_increase,
        delta=delta,
        p_value=p_value,
        slope_ci_low=slope_ci_low,
        slope_ci_high=slope_ci_high,
    )


def _is_clear_positive_backlog_trend(trend: _Trend, config: StabilityConfig) -> bool:
    return (
        trend.slope > config.max_positive_backlog_slope
        and trend.p_value < config.backlog_trend_alpha
        and trend.relative_increase >= config.min_backlog_relative_increase
        and trend.delta >= config.min_backlog_growth_for_hard_pressure
    )


def _is_positive_latency_drift(trend: _Trend, config: StabilityConfig) -> bool:
    return (
        trend.slope > 0.0
        and trend.relative_increase >= config.drift_test.min_relative_increase
    )


def _is_token_throughput_plateau(trend: _Trend, config: StabilityConfig) -> bool:
    return trend.relative_increase <= config.token_throughput_plateau_relative_growth


def _workload_phase_reason(
    windows: Sequence[WindowSummary],
    *,
    generation_tok_s_trend: _Trend | None,
    config: StabilityConfig,
    key_metrics: dict[str, float],
) -> str | None:
    phase_reasons: list[str] = []
    for field_name, display_name in (
        ("expected_output_len_mean", "expected output length"),
        ("actual_output_len_mean", "actual output length"),
    ):
        trend = _trend_for_optional(windows, field_name)
        if trend is None:
            continue
        key_metrics[f"{field_name}_slope_per_s"] = trend.slope
        key_metrics[f"{field_name}_relative_increase"] = trend.relative_increase
        if trend.slope > 0.0 and trend.relative_increase >= config.drift_test.min_relative_increase:
            phase_reasons.append(
                f"{display_name} rose across active windows "
                f"(relative increase={trend.relative_increase:.3f})"
            )
    if generation_tok_s_trend is not None and (
        generation_tok_s_trend.relative_increase > config.token_throughput_plateau_relative_growth
    ):
        phase_reasons.append(
            "generation token throughput rose rather than plateaued "
            f"(relative increase={generation_tok_s_trend.relative_increase:.3f})"
        )
    if not phase_reasons:
        return None
    return "workload-phase evidence weakens backlog extrapolation: " + "; ".join(phase_reasons)


def _present_values(windows: Sequence[WindowSummary], field_name: str) -> list[float]:
    return [float(value) for window in windows if (value := getattr(window, field_name)) is not None]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _missing_optional_server_fields(windows: Sequence[WindowSummary]) -> list[str]:
    missing: list[str] = []
    has_kv_pressure_fields = bool(_present_values(windows, "kv_cache_usage_max")) and bool(
        _present_values(windows, "preemptions_delta")
    )
    for field_name in (
        "num_running_mean",
        "num_waiting_mean",
        "num_swapped_mean",
        "kv_cache_usage_mean",
        "kv_cache_usage_max",
        "preemptions_delta",
    ):
        if field_name == "num_swapped_mean" and has_kv_pressure_fields:
            continue
        if all(getattr(window, field_name) is None for window in windows):
            missing.append(field_name)
    return missing


def _missing_latency_evidence(windows: Sequence[WindowSummary]) -> list[str]:
    missing: list[str] = []
    for field_name in ("ttft_p90_ms", "tpot_p90_ms"):
        windows_with_arrivals = [window for window in windows if window.arrivals > 0]
        if windows_with_arrivals and all(
            getattr(window, field_name) is None for window in windows_with_arrivals
        ):
            missing.append(field_name)
    return missing


def _slo_reasons(
    windows: Sequence[WindowSummary],
    config: StabilityConfig,
    key_metrics: dict[str, float],
    *,
    request_records: Sequence[RequestRecord] | None = None,
    trial_start_ts: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    if _uses_request_level_ttft_slo(config):
        reasons.extend(
            _request_level_ttft_slo_reasons(
                windows,
                config,
                key_metrics,
                request_records=request_records,
                trial_start_ts=trial_start_ts,
            )
        )
    elif config.ttft_slo_ms is not None:
        ttft_values = _present_values(windows, config.ttft_slo_field)
        if ttft_values:
            max_ttft = max(ttft_values)
            key_metrics[f"{config.ttft_slo_field}_max"] = max_ttft
            if max_ttft > config.ttft_slo_ms:
                reasons.append(
                    f"{_display_slo_field(config.ttft_slo_field)} SLO violated: max={max_ttft:.3f} ms "
                    f"> {config.ttft_slo_ms:.3f} ms"
                )
    if config.tpot_slo_ms is not None:
        tpot_values = _present_values(windows, config.tpot_slo_field)
        if tpot_values:
            max_tpot = max(tpot_values)
            key_metrics[f"{config.tpot_slo_field}_max"] = max_tpot
            if max_tpot > config.tpot_slo_ms:
                reasons.append(
                    f"{_display_slo_field(config.tpot_slo_field)} SLO violated: max={max_tpot:.3f} ms "
                    f"> {config.tpot_slo_ms:.3f} ms"
                )
    return reasons


def _uses_request_level_ttft_slo(config: StabilityConfig) -> bool:
    return config.ttft_slo_mode == "length_scaled" or config.longbench_ttft_static_preset is not None


def _request_level_ttft_slo_reasons(
    windows: Sequence[WindowSummary],
    config: StabilityConfig,
    key_metrics: dict[str, float],
    *,
    request_records: Sequence[RequestRecord] | None,
    trial_start_ts: float | None,
) -> list[str]:
    if request_records is None:
        raise ValueError("request_records are required for request-level TTFT SLO policies")
    if trial_start_ts is None:
        trial_start_ts = _infer_trial_start_ts(request_records)
    if not isfinite(trial_start_ts):
        raise ValueError("trial_start_ts must be finite for request-level TTFT SLO policies")

    active_records_by_window: list[list[tuple[RequestRecord, float, float]]] = []
    all_thresholds: list[float] = []
    for window in windows:
        window_records: list[tuple[RequestRecord, float, float]] = []
        for record in request_records:
            if record.ttft_s is None or record.actual_send_ts is None:
                continue
            relative_send_s = record.actual_send_ts - trial_start_ts
            if not _record_in_window(relative_send_s, window):
                continue
            threshold_ms = _ttft_threshold_ms_for_record(record, config)
            ratio = (record.ttft_s * 1000.0) / threshold_ms
            window_records.append((record, ratio, threshold_ms))
            all_thresholds.append(threshold_ms)
        if window_records:
            active_records_by_window.append(window_records)

    if not active_records_by_window:
        return []

    percentile = _slo_field_percentile(config.ttft_slo_field)
    window_ratio_values = [
        _percentile([ratio for _record, ratio, _threshold in records], percentile)
        for records in active_records_by_window
    ]
    max_ratio = max(value for value in window_ratio_values if value is not None)
    key_metrics["ttft_slo_ratio_max"] = max_ratio
    key_metrics["ttft_slo_threshold_ms_min"] = min(all_thresholds)
    key_metrics["ttft_slo_threshold_ms_max"] = max(all_thresholds)
    key_metrics["ttft_slo_threshold_ms_mean"] = sum(all_thresholds) / len(all_thresholds)

    if max_ratio <= 1.0:
        return []

    violating_records = [
        (record, ratio, threshold)
        for records in active_records_by_window
        for record, ratio, threshold in records
        if ratio > 1.0
    ]
    worst_record, worst_ratio, worst_threshold_ms = max(
        violating_records,
        key=lambda item: item[1],
    )
    profile = _longbench_profile_for_record(worst_record)
    observed_ms = (worst_record.ttft_s or 0.0) * 1000.0
    return [
        f"{_display_slo_field(config.ttft_slo_field)} request-level TTFT SLO violated: "
        f"max normalized window percentile={max_ratio:.3f} > 1.000 "
        f"(mode={config.ttft_slo_mode}, preset={config.longbench_ttft_static_preset}, "
        f"worst_profile={profile}, worst_prompt_len={worst_record.prompt_len}, "
        f"worst_ttft={observed_ms:.3f} ms > threshold={worst_threshold_ms:.3f} ms)"
    ]


def _record_in_window(relative_send_s: float, window: WindowSummary) -> bool:
    if relative_send_s < window.start_s:
        return False
    if relative_send_s < window.end_s:
        return True
    return relative_send_s == window.end_s and window.arrivals > 0


def _infer_trial_start_ts(records: Sequence[RequestRecord]) -> float:
    send_times = [record.actual_send_ts for record in records if record.actual_send_ts is not None]
    if not send_times:
        raise ValueError("request_records contain no actual_send_ts values")
    return min(send_times)


def _ttft_threshold_ms_for_record(record: RequestRecord, config: StabilityConfig) -> float:
    profile = _longbench_profile_for_record(record)
    if config.ttft_slo_mode == "length_scaled":
        params = TTFT_SLO_LENGTH_SCALED[profile]
        threshold_s = min(
            params["cap"],
            params["base"] + params["per_1k_input_tokens"] * (record.prompt_len / 1000.0),
        )
        return threshold_s * 1000.0
    if config.longbench_ttft_static_preset is not None:
        threshold_s = TTFT_SLO_LONGBENCH_STATIC_PRESETS[config.longbench_ttft_static_preset][profile]
        return threshold_s * 1000.0
    if config.ttft_slo_ms is None:
        raise ValueError("static TTFT SLO policy requires ttft_slo_ms unless a preset is configured")
    return config.ttft_slo_ms


def _longbench_profile_for_record(record: RequestRecord) -> str:
    raw_profile = record.metadata.get("profile")
    if raw_profile is None:
        raw_profile = record.metadata.get("longbench_profile")
    if not isinstance(raw_profile, str) or not raw_profile:
        raise ValueError(
            "request-level LongBench TTFT SLO policies require request metadata.profile"
        )
    if raw_profile not in TTFT_SLO_LENGTH_SCALED:
        raise ValueError(f"unsupported LongBench profile for TTFT SLO policy: {raw_profile!r}")
    return raw_profile


def _slo_field_percentile(field_name: str) -> float:
    if field_name == "ttft_p50_ms":
        return 50.0
    if field_name == "ttft_p90_ms":
        return 90.0
    if field_name == "ttft_p99_ms":
        return 99.0
    raise ValueError(f"unsupported request-level TTFT SLO field {field_name!r}")


def _display_slo_field(field_name: str) -> str:
    metric, percentile, _unit = field_name.split("_", 2)
    return f"{metric.upper()} {percentile}"


def _confidence_after_penalties(base: Confidence, penalties: int) -> Confidence:
    levels: list[Confidence] = ["high", "medium", "low"]
    index = levels.index(base)
    return levels[min(index + penalties, len(levels) - 1)]


def _csv_str(row: dict[str, str], field_name: str, row_idx: int) -> str:
    value = row[field_name]
    if not value:
        raise ValueError(f"{row_idx}: {field_name} must be non-empty")
    return value


def _csv_int(row: dict[str, str], field_name: str, row_idx: int) -> int:
    try:
        return int(row[field_name])
    except ValueError as exc:
        raise ValueError(f"{row_idx}: {field_name} must be an integer") from exc


def _csv_float(row: dict[str, str], field_name: str, row_idx: int) -> float:
    try:
        value = float(row[field_name])
    except ValueError as exc:
        raise ValueError(f"{row_idx}: {field_name} must be a float") from exc
    if not isfinite(value):
        raise ValueError(f"{row_idx}: {field_name} must be finite")
    return value


def _csv_optional_float(row: dict[str, str], field_name: str, row_idx: int) -> float | None:
    raw_value = row[field_name]
    if raw_value == "":
        return None
    return _csv_float(row, field_name, row_idx)


def _csv_optional_float_missing_ok(
    row: dict[str, str],
    field_name: str,
    row_idx: int,
) -> float | None:
    if field_name not in row:
        return None
    return _csv_optional_float(row, field_name, row_idx)


def _require_optional_positive_finite(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _require_slo_field(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")


def _require_non_negative_finite(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


_OPTIONAL_FLOAT_FIELDS = (
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "itl_p90_ms",
    "e2e_p90_ms",
    "e2e_p99_ms",
    "prompt_tok_s",
    "generation_tok_s",
    "total_tok_s",
    "prompt_len_mean",
    "expected_output_len_mean",
    "actual_output_len_mean",
    "num_running_mean",
    "num_waiting_mean",
    "num_swapped_mean",
    "kv_cache_usage_mean",
    "kv_cache_usage_max",
    "preemptions_delta",
)

_BACKCOMPAT_OPTIONAL_WINDOW_FIELDS = {
    "ttft_p95_ms",
    "tpot_p95_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "prompt_len_mean",
    "expected_output_len_mean",
    "actual_output_len_mean",
}


__all__ = [
    "DriftTestConfig",
    "TTFT_SLO_LENGTH_SCALED",
    "TTFT_SLO_LONGBENCH_STATIC_PRESETS",
    "StabilityConfig",
    "classify_stability",
    "load_window_summaries_csv",
]
