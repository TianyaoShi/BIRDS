from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Literal, Sequence

from .records import StabilityResult, WindowSummary


Status = Literal["stable", "unstable", "slo_violation", "uncertain", "aborted_safety"]
Confidence = Literal["high", "medium", "low"]


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
    min_waiting_queue_mean_for_pressure: float = 1.0
    min_waiting_queue_active_fraction: float = 0.5
    token_throughput_plateau_relative_growth: float = 0.05
    max_error_rate: float = 0.03
    ttft_slo_ms: float | None = 2000.0
    tpot_slo_ms: float | None = 80.0
    ttft_slo_field: str = "ttft_p90_ms"
    tpot_slo_field: str = "tpot_p90_ms"
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


@dataclass(frozen=True, slots=True)
class _Trend:
    slope: float
    relative_increase: float
    delta: float


def classify_stability(
    windows: Sequence[WindowSummary],
    *,
    config: StabilityConfig | None = None,
    aborted_safety: bool = False,
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
    latency_drift_reasons: list[str] = []
    has_capacity_pressure = False

    if completion_arrival_ratio < 1.0 - stability_config.completion_arrival_tolerance:
        capacity_pressure_reasons.append(
            "completion rate lagged arrivals: "
            f"completion/arrival={completion_arrival_ratio:.3f} "
            f"< {1.0 - stability_config.completion_arrival_tolerance:.3f}"
        )
        has_capacity_pressure = True

    if aggregate_error_rate > stability_config.max_error_rate:
        direct_unstable_reasons.append(
            f"error rate {aggregate_error_rate:.3f} exceeded threshold "
            f"{stability_config.max_error_rate:.3f}"
        )

    outstanding_trend = _trend_for_required(active_eval_windows, "outstanding_end")
    key_metrics["outstanding_end_slope_per_s"] = outstanding_trend.slope
    key_metrics["outstanding_end_delta"] = outstanding_trend.delta
    if (
        outstanding_trend.slope > stability_config.max_positive_backlog_slope
        and outstanding_trend.delta >= stability_config.min_backlog_growth_for_hard_pressure
    ):
        capacity_pressure_reasons.append(
            "outstanding requests drifted upward: "
            f"Theil-Sen slope={outstanding_trend.slope:.3f}/s "
            f"> {stability_config.max_positive_backlog_slope:.3f}/s, "
            f"delta={outstanding_trend.delta:.3f}"
        )
        has_capacity_pressure = True
    elif _has_repeated_increase([float(window.outstanding_end) for window in active_eval_windows]):
        capacity_pressure_reasons.append("outstanding requests grew across consecutive windows")
        has_capacity_pressure = True

    num_waiting_trend = _trend_for_optional(active_eval_windows, "num_waiting_mean")
    if num_waiting_trend is not None:
        key_metrics["num_waiting_mean_slope_per_s"] = num_waiting_trend.slope
        key_metrics["num_waiting_mean_delta"] = num_waiting_trend.delta
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

    slo_reasons = _slo_reasons(active_eval_windows, stability_config, key_metrics)
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
        ttft_p99_ms=_csv_optional_float(row, "ttft_p99_ms", row_idx),
        tpot_p50_ms=_csv_optional_float(row, "tpot_p50_ms", row_idx),
        tpot_p90_ms=_csv_optional_float(row, "tpot_p90_ms", row_idx),
        tpot_p99_ms=_csv_optional_float(row, "tpot_p99_ms", row_idx),
        itl_p90_ms=_csv_optional_float(row, "itl_p90_ms", row_idx),
        e2e_p90_ms=_csv_optional_float(row, "e2e_p90_ms", row_idx),
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
    slopes: list[float] = []
    for left_idx in range(len(x_values) - 1):
        for right_idx in range(left_idx + 1, len(x_values)):
            dx = x_values[right_idx] - x_values[left_idx]
            if dx <= 0.0:
                raise ValueError("trend x_values must be strictly increasing")
            slopes.append((y_values[right_idx] - y_values[left_idx]) / dx)
    slope = median(slopes)
    delta = slope * (x_values[-1] - x_values[0])
    first_half = y_values[: max(1, len(y_values) // 2)]
    baseline = max(abs(median(first_half)), 1e-9)
    relative_increase = delta / baseline
    return _Trend(slope=slope, relative_increase=relative_increase, delta=delta)


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


def _has_repeated_increase(values: Sequence[float]) -> bool:
    increases = 0
    longest_run = 0
    current_run = 0
    for left, right in zip(values, values[1:]):
        if right > left:
            increases += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return increases >= 2 and longest_run >= 1 and values[-1] > values[0]


def _present_values(windows: Sequence[WindowSummary], field_name: str) -> list[float]:
    return [float(value) for window in windows if (value := getattr(window, field_name)) is not None]


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
) -> list[str]:
    reasons: list[str] = []
    if config.ttft_slo_ms is not None:
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
    "prompt_len_mean",
    "expected_output_len_mean",
    "actual_output_len_mean",
}


__all__ = [
    "DriftTestConfig",
    "StabilityConfig",
    "classify_stability",
    "load_window_summaries_csv",
]
