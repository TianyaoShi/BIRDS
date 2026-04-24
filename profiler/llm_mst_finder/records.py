from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int
    extra_body: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        _require_non_negative("prompt_len", self.prompt_len)
        _require_non_negative("expected_output_len", self.expected_output_len)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_index: int
    scheduled_send_ts: float
    sample: SampleRequest

    def __post_init__(self) -> None:
        _require_non_negative("request_index", self.request_index)
        _require_finite("scheduled_send_ts", self.scheduled_send_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index,
            "scheduled_send_ts": self.scheduled_send_ts,
            "sample": self.sample.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: str
    trial_id: str
    scheduled_send_ts: float
    actual_send_ts: float | None
    first_token_ts: float | None
    end_ts: float | None
    success: bool
    error: str | None
    prompt_len: int
    expected_output_len: int
    actual_output_len: int | None
    ttft_s: float | None
    e2e_s: float | None
    tpot_s: float | None
    itl_s: list[float]
    output_token_timestamps: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.trial_id:
            raise ValueError("trial_id must be non-empty")
        _require_finite("scheduled_send_ts", self.scheduled_send_ts)
        _require_non_negative("prompt_len", self.prompt_len)
        _require_non_negative("expected_output_len", self.expected_output_len)
        if self.actual_send_ts is not None:
            _require_finite("actual_send_ts", self.actual_send_ts)
            if self.actual_send_ts < self.scheduled_send_ts:
                raise ValueError("actual_send_ts cannot be earlier than scheduled_send_ts")
        if self.first_token_ts is not None:
            _require_finite("first_token_ts", self.first_token_ts)
            if self.actual_send_ts is None:
                raise ValueError("first_token_ts requires actual_send_ts")
            if self.first_token_ts < self.actual_send_ts:
                raise ValueError("first_token_ts cannot be earlier than actual_send_ts")
        if self.end_ts is not None:
            _require_finite("end_ts", self.end_ts)
            if self.actual_send_ts is None:
                raise ValueError("end_ts requires actual_send_ts")
            if self.end_ts < self.actual_send_ts:
                raise ValueError("end_ts cannot be earlier than actual_send_ts")
        if self.actual_output_len is not None:
            _require_non_negative("actual_output_len", self.actual_output_len)
        for name, value in (
            ("ttft_s", self.ttft_s),
            ("e2e_s", self.e2e_s),
            ("tpot_s", self.tpot_s),
        ):
            if value is not None:
                _require_non_negative(name, value)
        previous_ts: float | None = None
        for token_ts in self.output_token_timestamps:
            _require_finite("output_token_timestamp", token_ts)
            if previous_ts is not None and token_ts < previous_ts:
                raise ValueError("output_token_timestamps must be sorted")
            if self.actual_send_ts is not None and token_ts < self.actual_send_ts:
                raise ValueError("output_token_timestamps cannot precede actual_send_ts")
            previous_ts = token_ts
        for latency in self.itl_s:
            _require_non_negative("itl_s entry", latency)
        if self.success:
            if self.error is not None:
                raise ValueError("successful requests cannot carry an error")
            if self.actual_send_ts is None or self.end_ts is None:
                raise ValueError("successful requests require actual_send_ts and end_ts")
        if self.ttft_s is not None and self.first_token_ts is not None and self.actual_send_ts is not None:
            observed_ttft = self.first_token_ts - self.actual_send_ts
            if abs(observed_ttft - self.ttft_s) > 1e-6:
                raise ValueError("ttft_s does not match first_token_ts - actual_send_ts")
        if self.e2e_s is not None and self.end_ts is not None and self.actual_send_ts is not None:
            observed_e2e = self.end_ts - self.actual_send_ts
            if abs(observed_e2e - self.e2e_s) > 1e-6:
                raise ValueError("e2e_s does not match end_ts - actual_send_ts")

    @property
    def scheduling_delay_s(self) -> float | None:
        if self.actual_send_ts is None:
            return None
        return self.actual_send_ts - self.scheduled_send_ts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ServerMetricSample:
    ts: float
    raw: dict[str, Any]
    num_running: float | None
    num_waiting: float | None
    num_swapped: float | None
    kv_cache_usage: float | None
    prompt_tokens_total: float | None
    generation_tokens_total: float | None
    request_success_total: float | None
    request_abort_total: float | None

    def __post_init__(self) -> None:
        _require_finite("ts", self.ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindowSummary:
    trial_id: str
    window_idx: int
    start_s: float
    end_s: float
    arrivals: int
    completions: int
    failures: int
    arrival_rate: float
    completion_rate: float
    error_rate: float
    outstanding_start: int
    outstanding_end: int
    outstanding_mean: float
    outstanding_slope: float
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    ttft_p99_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    tpot_p99_ms: float | None
    itl_p90_ms: float | None
    e2e_p90_ms: float | None
    e2e_p99_ms: float | None
    prompt_tok_s: float | None
    generation_tok_s: float | None
    total_tok_s: float | None
    num_running_mean: float | None
    num_waiting_mean: float | None
    num_swapped_mean: float | None
    kv_cache_usage_mean: float | None
    kv_cache_usage_max: float | None
    preemptions_delta: float | None

    def __post_init__(self) -> None:
        _require_non_negative("window_idx", self.window_idx)
        _require_finite("start_s", self.start_s)
        _require_finite("end_s", self.end_s)
        if self.end_s <= self.start_s:
            raise ValueError("window end_s must be greater than start_s")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    successful_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    successful_request_throughput: float
    prompt_token_throughput: float
    generation_token_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float | None
    median_ttft_ms: float | None
    std_ttft_ms: float | None
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float | None
    median_tpot_ms: float | None
    std_tpot_ms: float | None
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float | None
    median_itl_ms: float | None
    std_itl_ms: float | None
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2e_ms: float | None
    median_e2e_ms: float | None
    std_e2e_ms: float | None
    percentiles_e2e_ms: list[tuple[float, float]]
    prompt_length_summary: dict[str, float]
    output_length_summary: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StabilityResult:
    status: Literal["stable", "unstable", "slo_violation", "uncertain", "aborted_safety"]
    confidence: Literal["high", "medium", "low"]
    reasons: list[str]
    key_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BottleneckResult:
    bottleneck_class: Literal[
        "scheduler_cap",
        "prefill_compute_or_token_budget",
        "decode_bandwidth",
        "kv_cache",
        "slo_limited",
        "client_limited",
        "mixed",
        "unknown",
    ]
    confidence: Literal["high", "medium", "low"]
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialConfig:
    trial_id: str
    mode: Literal["open-loop", "closed-loop"]
    duration_s: float
    base_url: str
    endpoint: str
    model: str
    request_rate: float | None = None
    concurrency: int | None = None
    think_time_s: float = 0.0
    burstiness: float = 1.0
    request_timeout_s: float = 6 * 60 * 60
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] | None = None
    safety_max_outstanding: int | None = None
    metrics_url: str | None = None
    metrics_interval_s: float = 1.0
    window_s: float = 10.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must be non-empty")
        if self.mode not in {"open-loop", "closed-loop"}:
            raise ValueError(f"unsupported mode {self.mode!r}")
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        if not self.model:
            raise ValueError("model must be non-empty")
        _require_positive("duration_s", self.duration_s)
        _require_positive("request_timeout_s", self.request_timeout_s)
        _require_positive("burstiness", self.burstiness)
        _require_positive("metrics_interval_s", self.metrics_interval_s)
        _require_positive("window_s", self.window_s)
        _require_non_negative("think_time_s", self.think_time_s)
        if self.safety_max_outstanding is not None:
            _require_positive("safety_max_outstanding", self.safety_max_outstanding)
        if self.mode == "open-loop":
            if self.request_rate is None:
                raise ValueError("open-loop trials require request_rate")
            _require_positive("request_rate", self.request_rate)
            if self.concurrency is not None:
                raise ValueError("open-loop trials must not set concurrency")
        if self.mode == "closed-loop":
            if self.concurrency is None:
                raise ValueError("closed-loop trials require concurrency")
            _require_positive("concurrency", self.concurrency)
            if self.request_rate is not None:
                raise ValueError("closed-loop trials must not set request_rate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialSummary:
    trial_id: str
    mode: Literal["open-loop", "closed-loop"]
    status: Literal["completed", "aborted_safety"]
    requested_request_rate: float | None
    requested_concurrency: int | None
    target_duration_s: float
    wall_time_s: float
    started_requests: int
    successful_requests: int
    failed_requests: int
    actual_send_rate: float
    successful_completion_rate: float
    error_rate: float
    mean_scheduling_delay_s: float | None
    max_scheduling_delay_s: float | None
    max_observed_outstanding: int
    metrics_sample_count: int
    abort_reason: str | None
    benchmark_metrics: BenchmarkMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must be non-empty")
        _require_positive("target_duration_s", self.target_duration_s)
        _require_positive("wall_time_s", self.wall_time_s)
        _require_non_negative("started_requests", self.started_requests)
        _require_non_negative("successful_requests", self.successful_requests)
        _require_non_negative("failed_requests", self.failed_requests)
        if self.started_requests != self.successful_requests + self.failed_requests:
            raise ValueError("started_requests must equal successful_requests + failed_requests")
        _require_non_negative("actual_send_rate", self.actual_send_rate)
        _require_non_negative("successful_completion_rate", self.successful_completion_rate)
        _require_non_negative("error_rate", self.error_rate)
        _require_non_negative("max_observed_outstanding", self.max_observed_outstanding)
        _require_non_negative("metrics_sample_count", self.metrics_sample_count)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialAnalysisResult:
    trial_id: str
    trial_validity: Literal["valid", "invalid_workload", "client_limited", "metrics_invalid"]
    validity_reasons: list[str]
    stability: StabilityResult | None
    bottleneck: BottleneckResult | None

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must be non-empty")
        if not self.validity_reasons:
            raise ValueError("validity_reasons must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_validity": self.trial_validity,
            "validity_reasons": list(self.validity_reasons),
            "stability": None if self.stability is None else self.stability.to_dict(),
            "bottleneck": None if self.bottleneck is None else self.bottleneck.to_dict(),
        }
