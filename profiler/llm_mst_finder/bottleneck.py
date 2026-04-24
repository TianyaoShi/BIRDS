from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from statistics import median
from typing import Literal

from .records import BottleneckResult, RequestRecord, StabilityResult, TrialSummary, WindowSummary


BottleneckClass = Literal[
    "scheduler_cap",
    "prefill_compute_or_token_budget",
    "decode_bandwidth",
    "kv_cache",
    "slo_limited",
    "client_limited",
    "mixed",
    "unknown",
]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class BottleneckConfig:
    warmup_windows: int = 2
    min_eval_windows: int = 2
    scheduler_running_fraction: float = 0.90
    kv_cache_saturation_fraction: float = 0.98
    kv_cache_low_fraction: float = 0.85
    min_latency_relative_increase: float = 0.20
    flat_latency_relative_change: float = 0.10
    token_plateau_relative_range: float = 0.15
    completion_arrival_tolerance: float = 0.03
    open_loop_send_rate_tolerance: float = 0.03
    scheduling_delay_warning_s: float = 0.25

    def __post_init__(self) -> None:
        _require_non_negative_int("warmup_windows", self.warmup_windows)
        _require_positive_int("min_eval_windows", self.min_eval_windows)
        for field_name in (
            "scheduler_running_fraction",
            "kv_cache_saturation_fraction",
            "kv_cache_low_fraction",
            "min_latency_relative_increase",
            "flat_latency_relative_change",
            "token_plateau_relative_range",
            "completion_arrival_tolerance",
            "open_loop_send_rate_tolerance",
            "scheduling_delay_warning_s",
        ):
            _require_non_negative_finite(field_name, float(getattr(self, field_name)))
        if self.scheduler_running_fraction <= 0.0 or self.scheduler_running_fraction > 1.0:
            raise ValueError("scheduler_running_fraction must be in (0, 1]")
        if self.kv_cache_saturation_fraction <= 0.0 or self.kv_cache_saturation_fraction > 1.0:
            raise ValueError("kv_cache_saturation_fraction must be in (0, 1]")
        if self.kv_cache_low_fraction >= self.kv_cache_saturation_fraction:
            raise ValueError("kv_cache_low_fraction must be below kv_cache_saturation_fraction")
        if self.completion_arrival_tolerance >= 1.0:
            raise ValueError("completion_arrival_tolerance must be less than 1")
        if self.open_loop_send_rate_tolerance >= 1.0:
            raise ValueError("open_loop_send_rate_tolerance must be less than 1")


@dataclass(frozen=True, slots=True)
class BottleneckClassifier:
    config: BottleneckConfig = field(default_factory=BottleneckConfig)

    def classify(
        self,
        windows: Sequence[WindowSummary],
        *,
        stability_result: StabilityResult | None = None,
        trial_summary: TrialSummary | None = None,
        request_records: Sequence[RequestRecord] = (),
        server_metadata: Mapping[str, object] | None = None,
    ) -> BottleneckResult:
        return classify_bottleneck(
            windows,
            config=self.config,
            stability_result=stability_result,
            trial_summary=trial_summary,
            request_records=request_records,
            server_metadata=server_metadata,
        )


@dataclass(frozen=True, slots=True)
class _Trend:
    slope: float
    relative_increase: float
    delta: float


@dataclass(frozen=True, slots=True)
class _ServerMetadata:
    max_num_seqs: float | None
    max_num_batched_tokens: float | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    bottleneck_class: BottleneckClass
    score: int
    evidence: list[str]


def classify_bottleneck(
    windows: Sequence[WindowSummary],
    *,
    config: BottleneckConfig | None = None,
    stability_result: StabilityResult | None = None,
    trial_summary: TrialSummary | None = None,
    request_records: Sequence[RequestRecord] = (),
    server_metadata: Mapping[str, object] | None = None,
) -> BottleneckResult:
    bottleneck_config = BottleneckConfig() if config is None else config
    validated = _validate_windows(windows)
    eval_windows = validated[bottleneck_config.warmup_windows :]
    if len(eval_windows) < bottleneck_config.min_eval_windows:
        return BottleneckResult(
            bottleneck_class="unknown",
            confidence="low",
            evidence=[
                "insufficient post-warmup windows for bottleneck classification: "
                f"{len(eval_windows)} < {bottleneck_config.min_eval_windows}"
            ],
        )

    context_errors = _context_error_evidence(request_records)
    if context_errors:
        return BottleneckResult(
            bottleneck_class="unknown",
            confidence="low",
            evidence=context_errors
            + ["invalid workload evidence is not used to infer a server bottleneck"],
        )

    client_result = _classify_client_limited(
        stability_result=stability_result,
        trial_summary=trial_summary,
        config=bottleneck_config,
    )
    if client_result is not None:
        return client_result

    if stability_result is not None and stability_result.status == "slo_violation":
        return _classify_slo_limited(eval_windows, stability_result, bottleneck_config)

    metadata = _extract_server_metadata(server_metadata)
    candidates = [
        _kv_cache_candidate(eval_windows, bottleneck_config),
        _scheduler_candidate(eval_windows, metadata, bottleneck_config),
        _prefill_candidate(eval_windows, request_records, metadata, bottleneck_config),
        _decode_candidate(eval_windows, metadata, bottleneck_config),
    ]
    candidates = [candidate for candidate in candidates if candidate.score > 0]

    if not candidates:
        evidence = ["no bottleneck rule matched the observed post-warmup window metrics"]
        evidence.extend(_missing_evidence(eval_windows, metadata, None))
        return BottleneckResult(
            bottleneck_class="unknown",
            confidence="low",
            evidence=evidence,
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    top = candidates[0]
    near_top = [
        candidate
        for candidate in candidates
        if candidate.bottleneck_class != top.bottleneck_class and top.score - candidate.score <= 1
    ]
    if near_top and top.score >= 3:
        evidence = [f"{top.bottleneck_class} evidence: {item}" for item in top.evidence]
        for candidate in near_top[:2]:
            evidence.extend(
                f"{candidate.bottleneck_class} evidence: {item}" for item in candidate.evidence
            )
        missing_evidence = _missing_evidence(eval_windows, metadata, "mixed")
        evidence.extend(missing_evidence)
        return BottleneckResult(
            bottleneck_class="mixed",
            confidence=_confidence_from_score(top.score, missing_evidence),
            evidence=evidence,
        )

    evidence = list(top.evidence)
    missing_evidence = _missing_evidence(eval_windows, metadata, top.bottleneck_class)
    evidence.extend(missing_evidence)
    return BottleneckResult(
        bottleneck_class=top.bottleneck_class,
        confidence=_confidence_from_score(top.score, missing_evidence),
        evidence=evidence,
    )


def _classify_client_limited(
    *,
    stability_result: StabilityResult | None,
    trial_summary: TrialSummary | None,
    config: BottleneckConfig,
) -> BottleneckResult | None:
    evidence: list[str] = []
    if stability_result is not None and stability_result.status == "aborted_safety":
        evidence.append("stability classifier reported aborted_safety")
    if trial_summary is not None and trial_summary.status == "aborted_safety":
        evidence.append(
            "trial summary status is aborted_safety"
            + (f": {trial_summary.abort_reason}" if trial_summary.abort_reason else "")
        )
    if evidence:
        return BottleneckResult(
            bottleneck_class="client_limited",
            confidence="high",
            evidence=evidence + ["safety cap invalidated the configured arrival-rate trial"],
        )

    if trial_summary is None:
        return None
    client_evidence: list[str] = []
    if (
        trial_summary.mode == "open-loop"
        and trial_summary.requested_request_rate is not None
        and trial_summary.actual_send_rate
        < trial_summary.requested_request_rate * (1.0 - config.open_loop_send_rate_tolerance)
    ):
        client_evidence.append(
            "actual open-loop send rate lagged configured rate: "
            f"actual={trial_summary.actual_send_rate:.3f} req/s, "
            f"configured={trial_summary.requested_request_rate:.3f} req/s"
        )
    if (
        trial_summary.max_scheduling_delay_s is not None
        and trial_summary.max_scheduling_delay_s > config.scheduling_delay_warning_s
    ):
        client_evidence.append(
            "client scheduling delay was high: "
            f"max={trial_summary.max_scheduling_delay_s:.3f}s "
            f"> {config.scheduling_delay_warning_s:.3f}s"
        )
    if not client_evidence:
        return None
    return BottleneckResult(
        bottleneck_class="client_limited",
        confidence="high" if len(client_evidence) >= 2 else "medium",
        evidence=client_evidence,
    )


def _classify_slo_limited(
    windows: Sequence[WindowSummary],
    stability_result: StabilityResult,
    config: BottleneckConfig,
) -> BottleneckResult:
    evidence = list(stability_result.reasons)
    if _completion_arrival_ratio(windows) >= 1.0 - config.completion_arrival_tolerance:
        evidence.append("post-warmup completion rate stayed close to arrival rate")
    if not _has_positive_queue_drift(windows, config):
        evidence.append("queue and outstanding metrics were stationary enough for SLO-limited diagnosis")
    return BottleneckResult(
        bottleneck_class="slo_limited",
        confidence=stability_result.confidence,
        evidence=evidence,
    )


def _kv_cache_candidate(
    windows: Sequence[WindowSummary],
    config: BottleneckConfig,
) -> _Candidate:
    evidence: list[str] = []
    kv_max = _max_present(windows, "kv_cache_usage_max")
    swapped_max = _max_present(windows, "num_swapped_mean")
    preemptions_total = _sum_present(windows, "preemptions_delta")
    e2e_trend = _trend_optional(windows, "e2e_p99_ms")
    ttft_trend = _trend_optional(windows, "ttft_p90_ms")
    tpot_trend = _trend_optional(windows, "tpot_p90_ms")

    if kv_max is not None and kv_max >= config.kv_cache_saturation_fraction:
        evidence.append(
            "KV cache usage reached saturation range: "
            f"kv_cache_usage_max={kv_max:.3f} >= {config.kv_cache_saturation_fraction:.3f}"
        )
    if swapped_max is not None and swapped_max > 0.0:
        evidence.append(f"server reported swapped requests: num_swapped_mean max={swapped_max:.3f}")
    if preemptions_total is not None and preemptions_total > 0.0:
        evidence.append(f"preemptions increased after warmup: total={preemptions_total:.3f}")
    if e2e_trend is not None and _is_positive_drift(e2e_trend, config):
        evidence.append(
            "E2E p99 drifted upward while KV pressure signals were present: "
            f"relative increase={e2e_trend.relative_increase:.3f}"
        )
    if (
        ttft_trend is not None
        and tpot_trend is not None
        and _is_positive_drift(ttft_trend, config)
        and _is_positive_drift(tpot_trend, config)
    ):
        evidence.append("both TTFT p90 and TPOT p90 drifted upward")
    return _Candidate("kv_cache", len(evidence), evidence)


def _scheduler_candidate(
    windows: Sequence[WindowSummary],
    metadata: _ServerMetadata,
    config: BottleneckConfig,
) -> _Candidate:
    evidence: list[str] = []
    if metadata.max_num_seqs is None:
        return _Candidate("scheduler_cap", 0, evidence)
    running_max = _max_present(windows, "num_running_mean")
    if running_max is None:
        return _Candidate("scheduler_cap", 0, evidence)
    running_fraction = running_max / metadata.max_num_seqs
    if running_fraction >= config.scheduler_running_fraction:
        evidence.append(
            "num_running_mean reached supplied max_num_seqs: "
            f"max={running_max:.3f}, max_num_seqs={metadata.max_num_seqs:.3f}, "
            f"fraction={running_fraction:.3f}"
        )
    if _has_positive_queue_drift(windows, config):
        evidence.append("server waiting or outstanding requests drifted upward")
    ttft_trend = _trend_optional(windows, "ttft_p90_ms")
    if ttft_trend is not None and _is_positive_drift(ttft_trend, config):
        evidence.append(
            f"TTFT p90 drifted upward: relative increase={ttft_trend.relative_increase:.3f}"
        )
    tpot_trend = _trend_optional(windows, "tpot_p90_ms")
    if tpot_trend is not None and _is_flat(tpot_trend, config):
        evidence.append("TPOT p90 remained flat while queue/TTFT pressure increased")
    kv_mean = _max_present(windows, "kv_cache_usage_mean")
    if kv_mean is not None and kv_mean < config.kv_cache_low_fraction:
        evidence.append(
            "KV cache usage stayed below scheduler-cap exclusion threshold: "
            f"kv_cache_usage_mean max={kv_mean:.3f} < {config.kv_cache_low_fraction:.3f}"
        )
    preemptions_total = _sum_present(windows, "preemptions_delta")
    if preemptions_total is not None and preemptions_total == 0.0:
        evidence.append("preemptions were absent after warmup")
    if running_fraction < config.scheduler_running_fraction:
        return _Candidate("scheduler_cap", 0, evidence)
    return _Candidate("scheduler_cap", len(evidence), evidence)


def _prefill_candidate(
    windows: Sequence[WindowSummary],
    request_records: Sequence[RequestRecord],
    metadata: _ServerMetadata,
    config: BottleneckConfig,
) -> _Candidate:
    evidence: list[str] = []
    score = 0
    ttft_trend = _trend_optional(windows, "ttft_p90_ms")
    tpot_trend = _trend_optional(windows, "tpot_p90_ms")
    prompt_values = _present_values(windows, "prompt_tok_s")

    if ttft_trend is not None and _is_positive_drift(ttft_trend, config):
        score += 1
        evidence.append(
            f"TTFT p90 drifted upward: relative increase={ttft_trend.relative_increase:.3f}"
        )
    if _has_positive_queue_drift(windows, config):
        score += 1
        evidence.append("server waiting or outstanding requests drifted upward")
    if tpot_trend is not None and _is_flat(tpot_trend, config):
        score += 1
        evidence.append("TPOT p90 remained flat while prefill-side pressure increased")
    if prompt_values and _is_plateau(prompt_values, config):
        score += 1
        evidence.append(
            "prompt token throughput plateaued: "
            f"min={min(prompt_values):.3f} tok/s, max={max(prompt_values):.3f} tok/s"
        )
    long_prompt_evidence = _long_prompt_evidence(request_records)
    if long_prompt_evidence is not None:
        score += 1
        evidence.append(long_prompt_evidence)
    if metadata.max_num_batched_tokens is not None:
        evidence.append(
            "max_num_batched_tokens was supplied as serving metadata: "
            f"{metadata.max_num_batched_tokens:.3f}"
        )
    if not (
        (ttft_trend is not None and _is_positive_drift(ttft_trend, config))
        or _has_positive_queue_drift(windows, config)
    ):
        return _Candidate("prefill_compute_or_token_budget", 0, evidence)
    return _Candidate("prefill_compute_or_token_budget", score, evidence)


def _decode_candidate(
    windows: Sequence[WindowSummary],
    metadata: _ServerMetadata,
    config: BottleneckConfig,
) -> _Candidate:
    evidence: list[str] = []
    tpot_trend = _trend_optional(windows, "tpot_p90_ms")
    itl_trend = _trend_optional(windows, "itl_p90_ms")
    generation_values = _present_values(windows, "generation_tok_s")
    running_max = _max_present(windows, "num_running_mean")
    kv_max = _max_present(windows, "kv_cache_usage_max")
    preemptions_total = _sum_present(windows, "preemptions_delta")

    if tpot_trend is not None and _is_positive_drift(tpot_trend, config):
        evidence.append(
            f"TPOT p90 drifted upward: relative increase={tpot_trend.relative_increase:.3f}"
        )
    if itl_trend is not None and _is_positive_drift(itl_trend, config):
        evidence.append(
            f"ITL p90 drifted upward: relative increase={itl_trend.relative_increase:.3f}"
        )
    if generation_values and _is_plateau(generation_values, config):
        evidence.append(
            "generation token throughput plateaued: "
            f"min={min(generation_values):.3f} tok/s, max={max(generation_values):.3f} tok/s"
        )
    if metadata.max_num_seqs is not None and running_max is not None:
        running_fraction = running_max / metadata.max_num_seqs
        if running_fraction >= config.scheduler_running_fraction:
            evidence.append(
                "num_running_mean was high during decode pressure: "
                f"max={running_max:.3f}, max_num_seqs={metadata.max_num_seqs:.3f}, "
                f"fraction={running_fraction:.3f}"
            )
    elif running_max is not None and running_max > 0.0:
        evidence.append(f"num_running_mean was nonzero during decode pressure: max={running_max:.3f}")
    if kv_max is not None and kv_max < config.kv_cache_saturation_fraction:
        evidence.append(
            "KV cache usage did not reach saturation threshold: "
            f"kv_cache_usage_max={kv_max:.3f} < {config.kv_cache_saturation_fraction:.3f}"
        )
    if preemptions_total is not None and preemptions_total == 0.0:
        evidence.append("preemptions were absent after warmup")
    if not (
        (tpot_trend is not None and _is_positive_drift(tpot_trend, config))
        or (itl_trend is not None and _is_positive_drift(itl_trend, config))
    ):
        return _Candidate("decode_bandwidth", 0, evidence)
    return _Candidate("decode_bandwidth", len(evidence), evidence)


def _validate_windows(windows: Sequence[WindowSummary]) -> list[WindowSummary]:
    if not windows:
        raise ValueError("windows must be non-empty")
    validated = list(windows)
    trial_id = validated[0].trial_id
    previous_idx: int | None = None
    previous_end: float | None = None
    for window in validated:
        if window.trial_id != trial_id:
            raise ValueError("all windows must have the same trial_id")
        if previous_idx is not None and window.window_idx != previous_idx + 1:
            raise ValueError("windows must have contiguous window_idx values")
        if previous_end is not None and window.start_s < previous_end:
            raise ValueError("windows must be sorted and non-overlapping")
        for field_name in (
            "start_s",
            "end_s",
            "arrival_rate",
            "completion_rate",
            "error_rate",
            "outstanding_mean",
            "outstanding_slope",
        ):
            _require_finite(field_name, float(getattr(window, field_name)))
        if window.end_s <= window.start_s:
            raise ValueError(f"window {window.window_idx} end_s must be greater than start_s")
        for field_name in (
            "arrivals",
            "completions",
            "failures",
            "outstanding_start",
            "outstanding_end",
        ):
            _require_non_negative_int(field_name, getattr(window, field_name))
        for field_name in _OPTIONAL_FLOAT_FIELDS:
            value = getattr(window, field_name)
            if value is not None:
                _require_non_negative_finite(field_name, float(value))
        previous_idx = window.window_idx
        previous_end = window.end_s
    return validated


def _extract_server_metadata(metadata: Mapping[str, object] | None) -> _ServerMetadata:
    return _ServerMetadata(
        max_num_seqs=_metadata_positive_number(metadata, "max_num_seqs"),
        max_num_batched_tokens=_metadata_positive_number(metadata, "max_num_batched_tokens"),
    )


def _metadata_positive_number(metadata: Mapping[str, object] | None, key: str) -> float | None:
    if metadata is None:
        return None
    values: list[float] = []
    for mapping in _metadata_candidate_mappings(metadata):
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"server metadata {key!r} must be a positive finite number")
        numeric = float(value)
        if not isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"server metadata {key!r} must be a positive finite number")
        values.append(numeric)
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"conflicting supplied server metadata values for {key!r}: {values!r}")
    return first


def _metadata_candidate_mappings(metadata: Mapping[str, object]) -> list[Mapping[str, object]]:
    mappings: list[Mapping[str, object]] = [metadata]
    for nested_key in ("server_config", "server_metadata", "vllm_config"):
        nested = metadata.get(nested_key)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise ValueError(f"server metadata {nested_key!r} must be a mapping when provided")
        mappings.append(nested)
    return mappings


def _context_error_evidence(request_records: Sequence[RequestRecord]) -> list[str]:
    matching_errors: list[str] = []
    for record in request_records:
        if record.error is None:
            continue
        lowered = record.error.lower()
        if any(marker in lowered for marker in _CONTEXT_ERROR_MARKERS):
            matching_errors.append(record.request_id)
    if not matching_errors:
        return []
    return [
        "request errors indicate model context validation failure: "
        f"{len(matching_errors)} request(s), first={matching_errors[0]!r}"
    ]


def _missing_evidence(
    windows: Sequence[WindowSummary],
    metadata: _ServerMetadata,
    bottleneck_class: BottleneckClass | None,
) -> list[str]:
    missing: list[str] = []
    field_names = _missing_fields_for_class(bottleneck_class)
    for field_name in field_names:
        if not _present_values(windows, field_name):
            missing.append(f"{field_name} missing; confidence lowered")
    if bottleneck_class in {None, "scheduler_cap", "decode_bandwidth", "mixed"} and (
        metadata.max_num_seqs is None
    ):
        missing.append("max_num_seqs metadata was not supplied; scheduler-cap evidence limited")
    if bottleneck_class in {None, "prefill_compute_or_token_budget", "mixed"} and (
        metadata.max_num_batched_tokens is None
    ):
        missing.append(
            "max_num_batched_tokens metadata was not supplied; token-budget evidence limited"
        )
    return missing


def _missing_fields_for_class(bottleneck_class: BottleneckClass | None) -> tuple[str, ...]:
    if bottleneck_class == "scheduler_cap":
        return (
            "num_running_mean",
            "num_waiting_mean",
            "kv_cache_usage_mean",
            "preemptions_delta",
            "ttft_p90_ms",
            "tpot_p90_ms",
        )
    if bottleneck_class == "prefill_compute_or_token_budget":
        return ("num_waiting_mean", "ttft_p90_ms", "tpot_p90_ms", "prompt_tok_s")
    if bottleneck_class == "decode_bandwidth":
        return (
            "tpot_p90_ms",
            "itl_p90_ms",
            "generation_tok_s",
            "num_running_mean",
            "kv_cache_usage_max",
            "preemptions_delta",
        )
    if bottleneck_class == "kv_cache":
        return (
            "kv_cache_usage_max",
            "num_swapped_mean",
            "preemptions_delta",
            "e2e_p99_ms",
        )
    return (
        "num_running_mean",
        "num_waiting_mean",
        "num_swapped_mean",
        "kv_cache_usage_mean",
        "kv_cache_usage_max",
        "preemptions_delta",
    )


def _confidence_from_score(score: int, missing_evidence: Sequence[str]) -> Confidence:
    if score >= 4:
        confidence: Confidence = "high"
    elif score >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    if not missing_evidence:
        return confidence
    if confidence == "high":
        return "medium"
    return "low"


def _completion_arrival_ratio(windows: Sequence[WindowSummary]) -> float:
    arrivals = sum(window.arrivals for window in windows)
    if arrivals == 0:
        return 0.0
    completions = sum(window.completions for window in windows)
    return completions / arrivals


def _has_positive_queue_drift(windows: Sequence[WindowSummary], config: BottleneckConfig) -> bool:
    outstanding_trend = _trend_required(windows, "outstanding_end")
    if outstanding_trend.slope > 0.0 and outstanding_trend.relative_increase > 0.0:
        return True
    waiting_trend = _trend_optional(windows, "num_waiting_mean")
    return waiting_trend is not None and waiting_trend.slope > 0.0 and waiting_trend.delta > 0.0


def _trend_required(windows: Sequence[WindowSummary], field_name: str) -> _Trend:
    x_values = [_window_midpoint(window) for window in windows]
    y_values = [float(getattr(window, field_name)) for window in windows]
    return _theil_sen_trend(x_values, y_values)


def _trend_optional(windows: Sequence[WindowSummary], field_name: str) -> _Trend | None:
    x_values: list[float] = []
    y_values: list[float] = []
    for window in windows:
        value = getattr(window, field_name)
        if value is None:
            continue
        x_values.append(_window_midpoint(window))
        y_values.append(float(value))
    if len(y_values) < 2:
        return None
    return _theil_sen_trend(x_values, y_values)


def _theil_sen_trend(x_values: Sequence[float], y_values: Sequence[float]) -> _Trend:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        raise ValueError("trend requires at least two points")
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
    return _Trend(slope=slope, relative_increase=delta / baseline, delta=delta)


def _window_midpoint(window: WindowSummary) -> float:
    return (window.start_s + window.end_s) / 2.0


def _is_positive_drift(trend: _Trend, config: BottleneckConfig) -> bool:
    return trend.slope > 0.0 and trend.relative_increase >= config.min_latency_relative_increase


def _is_flat(trend: _Trend, config: BottleneckConfig) -> bool:
    return abs(trend.relative_increase) <= config.flat_latency_relative_change


def _is_plateau(values: Sequence[float], config: BottleneckConfig) -> bool:
    if len(values) < 2:
        return False
    center = max(abs(median(values)), 1e-9)
    return (max(values) - min(values)) / center <= config.token_plateau_relative_range


def _present_values(windows: Sequence[WindowSummary], field_name: str) -> list[float]:
    return [float(value) for window in windows if (value := getattr(window, field_name)) is not None]


def _max_present(windows: Sequence[WindowSummary], field_name: str) -> float | None:
    values = _present_values(windows, field_name)
    return max(values) if values else None


def _sum_present(windows: Sequence[WindowSummary], field_name: str) -> float | None:
    values = _present_values(windows, field_name)
    return sum(values) if values else None


def _long_prompt_evidence(request_records: Sequence[RequestRecord]) -> str | None:
    if not request_records:
        return None
    prompt_lengths = [float(record.prompt_len) for record in request_records]
    output_lengths = [
        float(record.expected_output_len)
        for record in request_records
        if record.expected_output_len > 0
    ]
    if not output_lengths:
        return None
    prompt_median = median(prompt_lengths)
    output_median = median(output_lengths)
    if prompt_median >= 2.0 * output_median:
        return (
            "prompt lengths dominated requested output lengths: "
            f"median_prompt={prompt_median:.3f}, median_expected_output={output_median:.3f}"
        )
    return None


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_non_negative_finite(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


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
    "num_running_mean",
    "num_waiting_mean",
    "num_swapped_mean",
    "kv_cache_usage_mean",
    "kv_cache_usage_max",
    "preemptions_delta",
)

_CONTEXT_ERROR_MARKERS = (
    "context length",
    "context window",
    "max_model_len",
    "maximum context",
    "maximum sequence length",
    "prompt is too long",
    "prompt too long",
    "tokens exceeds",
    "token count exceeds",
)


__all__ = [
    "BottleneckClass",
    "BottleneckClassifier",
    "BottleneckConfig",
    "classify_bottleneck",
]
