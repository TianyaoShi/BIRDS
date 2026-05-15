from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol, cast

from .analysis import analyze_trial_dir, write_analysis_artifact
from .loadgen import RequestReusePolicy, RequestSource
from .records import TrialAnalysisResult, TrialConfig
from .trial_runner import TrialRunResult, TrialRunner


SearchMode = Literal["closed-loop", "open-loop", "hybrid"]
Confidence = Literal["high", "medium", "low"]


class SearchError(RuntimeError):
    pass


class InvalidSearchTrial(SearchError):
    pass


class SearchConvergenceError(SearchError):
    pass


class TrialRunnerProtocol(Protocol):
    async def run_trial(
        self,
        config: TrialConfig,
        *,
        request_source: RequestSource,
        output_dir: str | Path,
    ) -> TrialRunResult:
        ...


@dataclass(frozen=True, slots=True)
class SearchConfig:
    search_id: str
    search_mode: SearchMode = "hybrid"
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/v1/completions"
    model: str = ""
    trial_duration_s: float = 60.0
    uncertain_trial_duration_s: float | None = None
    uncertain_trial_duration_multiplier: float = 2.0
    final_confirmation_duration_s: float | None = None
    rate_precision: float = 0.03
    initial_request_rate: float = 1.0
    max_request_rate: float | None = None
    max_binary_steps: int = 24
    max_bracket_trials: int = 16
    open_loop_bracket_growth_factor: float = 2.0
    client_limited_retry_attempts: int = 1
    client_limited_retry_cooldown_s: float = 30.0
    closed_loop_initial_concurrency: int = 1
    closed_loop_min_trials: int = 2
    max_closed_loop_concurrency: int = 128
    closed_loop_plateau_relative_gain: float = 0.05
    closed_loop_start_rate_fraction: float = 0.6
    closed_loop_high_rate_fraction: float = 1.2
    think_time_s: float = 0.0
    burstiness: float = 1.0
    request_timeout_s: float = 6 * 60 * 60
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, object] | None = None
    safety_max_outstanding: int | None = None
    metrics_url: str | None = None
    metrics_interval_s: float = 1.0
    window_s: float = 10.0
    request_reuse_policy: RequestReusePolicy = "no-repeat-across-search"
    request_reuse_strict_unique_threshold: int | None = 4096
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.search_id:
            raise ValueError("search_id must be non-empty")
        if self.search_mode not in {"closed-loop", "open-loop", "hybrid"}:
            raise ValueError(f"unsupported search_mode {self.search_mode!r}")
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        if not self.model:
            raise ValueError("model must be non-empty")
        _require_positive("trial_duration_s", self.trial_duration_s)
        if self.uncertain_trial_duration_s is not None:
            _require_positive("uncertain_trial_duration_s", self.uncertain_trial_duration_s)
            if self.uncertain_trial_duration_s <= self.trial_duration_s:
                raise ValueError("uncertain_trial_duration_s must exceed trial_duration_s")
        _require_positive("uncertain_trial_duration_multiplier", self.uncertain_trial_duration_multiplier)
        if self.uncertain_trial_duration_multiplier <= 1.0:
            raise ValueError("uncertain_trial_duration_multiplier must exceed 1.0")
        if self.final_confirmation_duration_s is not None:
            _require_positive("final_confirmation_duration_s", self.final_confirmation_duration_s)
        _require_positive("rate_precision", self.rate_precision)
        if self.rate_precision >= 1.0:
            raise ValueError("rate_precision must be less than 1.0")
        _require_positive("initial_request_rate", self.initial_request_rate)
        if self.max_request_rate is not None:
            _require_positive("max_request_rate", self.max_request_rate)
            if self.max_request_rate < self.initial_request_rate:
                raise ValueError("max_request_rate must be >= initial_request_rate")
        _require_positive_int("max_binary_steps", self.max_binary_steps)
        _require_positive_int("max_bracket_trials", self.max_bracket_trials)
        _require_positive("open_loop_bracket_growth_factor", self.open_loop_bracket_growth_factor)
        if self.open_loop_bracket_growth_factor <= 1.0:
            raise ValueError("open_loop_bracket_growth_factor must exceed 1.0")
        _require_non_negative_int("client_limited_retry_attempts", self.client_limited_retry_attempts)
        _require_non_negative("client_limited_retry_cooldown_s", self.client_limited_retry_cooldown_s)
        _require_positive_int("closed_loop_initial_concurrency", self.closed_loop_initial_concurrency)
        _require_positive_int("closed_loop_min_trials", self.closed_loop_min_trials)
        _require_positive_int("max_closed_loop_concurrency", self.max_closed_loop_concurrency)
        if self.closed_loop_initial_concurrency > self.max_closed_loop_concurrency:
            raise ValueError("closed_loop_initial_concurrency must be <= max_closed_loop_concurrency")
        _require_non_negative("closed_loop_plateau_relative_gain", self.closed_loop_plateau_relative_gain)
        _require_positive("closed_loop_start_rate_fraction", self.closed_loop_start_rate_fraction)
        _require_positive("closed_loop_high_rate_fraction", self.closed_loop_high_rate_fraction)
        if self.closed_loop_high_rate_fraction <= self.closed_loop_start_rate_fraction:
            raise ValueError("closed_loop_high_rate_fraction must exceed closed_loop_start_rate_fraction")
        _require_non_negative("think_time_s", self.think_time_s)
        _require_positive("burstiness", self.burstiness)
        _require_positive("request_timeout_s", self.request_timeout_s)
        if self.safety_max_outstanding is not None:
            _require_positive_int("safety_max_outstanding", self.safety_max_outstanding)
        _require_positive("metrics_interval_s", self.metrics_interval_s)
        _require_positive("window_s", self.window_s)
        if self.request_reuse_policy not in {
            "cycle",
            "no-repeat-per-trial",
            "no-repeat-across-search",
            "unique-then-cycle",
        }:
            raise ValueError(f"unsupported request_reuse_policy {self.request_reuse_policy!r}")
        if self.request_reuse_strict_unique_threshold is not None:
            _require_positive_int(
                "request_reuse_strict_unique_threshold",
                self.request_reuse_strict_unique_threshold,
            )

    @property
    def confirmation_duration_s(self) -> float:
        return self.final_confirmation_duration_s or self.trial_duration_s

    @property
    def uncertain_retry_duration_s(self) -> float:
        if self.uncertain_trial_duration_s is not None:
            return self.uncertain_trial_duration_s
        return max(
            self.confirmation_duration_s,
            self.trial_duration_s * self.uncertain_trial_duration_multiplier,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClosedLoopScoutResult:
    peak_request_throughput: float | None
    peak_output_token_throughput: float | None
    plateau_concurrency: int | None
    stop_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    search_id: str
    search_mode: SearchMode
    max_no_drift_request_rate: float | None
    max_slo_satisfying_request_rate: float | None
    rate_precision: float
    confirmation_trial_id: str | None
    termination_reason: str
    closed_loop: ClosedLoopScoutResult | None
    bottleneck_class: str
    confidence: Confidence
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.closed_loop is not None:
            payload["closed_loop"] = self.closed_loop.to_dict()
        return payload


@dataclass(slots=True)
class _SearchBounds:
    low_rate: float | None = None
    low_trial_id: str | None = None
    high_rate: float | None = None
    high_trial_id: str | None = None
    max_request_rate_cap: float | None = None
    max_request_rate_cap_attempted_rate: float | None = None

    def has_closed_relative_width(self, precision: float) -> bool:
        if self.low_rate is None or self.high_rate is None:
            return False
        if self.low_rate <= 0.0:
            return False
        return (self.high_rate - self.low_rate) / self.low_rate <= precision


@dataclass(frozen=True, slots=True)
class _StableOpenLoopPoint:
    rate: float
    trial_id: str


class SearchController:
    def __init__(
        self,
        runner: TrialRunnerProtocol | TrialRunner,
        *,
        request_source: RequestSource | None = None,
        request_source_factory: Callable[[], RequestSource] | None = None,
        output_dir: str | Path,
        analyze_trial: Callable[[str | Path], TrialAnalysisResult] = analyze_trial_dir,
        write_analysis: Callable[[str | Path, TrialAnalysisResult], Path] = write_analysis_artifact,
    ) -> None:
        if request_source is None and request_source_factory is None:
            raise ValueError("request_source or request_source_factory is required")
        if request_source is not None and request_source_factory is not None:
            raise ValueError("request_source and request_source_factory are mutually exclusive")
        self._runner = runner
        if request_source_factory is not None:
            self._request_source_factory = request_source_factory
        else:
            assert request_source is not None
            self._request_source_factory = lambda: request_source
        self._output_dir = Path(output_dir)
        self._trials_dir = self._output_dir / "trials"
        self._trace_path = self._output_dir / "search_trace.json"
        self._analyze_trial = analyze_trial
        self._write_analysis = write_analysis
        self._trial_index = 0
        self._trace: dict[str, object] = {}

    async def search(self, config: SearchConfig) -> SearchResult:
        self._prepare_output(config)
        closed_loop_result: ClosedLoopScoutResult | None = None
        if config.search_mode in {"closed-loop", "hybrid"}:
            closed_loop_result = await self._run_closed_loop_sweep(config)

        if config.search_mode == "closed-loop":
            result = self._closed_loop_only_result(config, closed_loop_result)
            self._finish_trace(result)
            return result

        bounds = await self._find_open_loop_bracket(config, closed_loop_result)
        if self._is_max_request_rate_limited(bounds):
            result = self._build_max_request_rate_limited_result(
                config=config,
                bounds=bounds,
                closed_loop_result=closed_loop_result,
            )
            self._finish_trace(result)
            return result
        if self._is_scheduler_config_limited(bounds):
            result = self._build_scheduler_config_limited_result(
                config=config,
                bounds=bounds,
                closed_loop_result=closed_loop_result,
            )
            self._finish_trace(result)
            return result
        bounds = await self._binary_search(config, bounds)
        if self._is_scheduler_config_limited(bounds):
            result = self._build_scheduler_config_limited_result(
                config=config,
                bounds=bounds,
                closed_loop_result=closed_loop_result,
            )
            self._finish_trace(result)
            return result
        result = await self._confirm_result(config, bounds, closed_loop_result)
        self._finish_trace(result)
        return result

    def _prepare_output(self, config: SearchConfig) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._trials_dir.mkdir(parents=True, exist_ok=True)
        if self._trace_path.exists():
            raise FileExistsError(f"refusing to overwrite existing search trace {self._trace_path}")
        self._trace = {
            "config": config.to_dict(),
            "events": [],
            "closed_loop": None,
            "bounds": {"low_rate": None, "low_trial_id": None, "high_rate": None, "high_trial_id": None},
            "result": None,
        }
        self._write_trace()

    async def _run_closed_loop_sweep(self, config: SearchConfig) -> ClosedLoopScoutResult:
        concurrency = config.closed_loop_initial_concurrency
        previous_throughput: float | None = None
        peak_request_throughput: float | None = None
        peak_output_token_throughput: float | None = None
        plateau_concurrency: int | None = None
        stop_reason = "max_closed_loop_concurrency reached"
        completed_trials = 0

        while concurrency <= config.max_closed_loop_concurrency:
            event = await self._run_and_analyze_trial(
                config,
                mode="closed-loop",
                concurrency=concurrency,
                request_rate=None,
                duration_s=config.trial_duration_s,
                purpose="closed_loop_scout",
            )
            analysis = event["analysis_result"]
            summary = event["run_result"].summary
            self._reject_invalid_trial(analysis, event["trial_id"])
            completed_trials += 1

            throughput = summary.successful_completion_rate
            output_tok_s = summary.benchmark_metrics.generation_token_throughput
            if peak_request_throughput is None or throughput > peak_request_throughput:
                peak_request_throughput = throughput
                peak_output_token_throughput = output_tok_s
                plateau_concurrency = concurrency

            if self._should_stop_closed_loop(
                analysis,
                completed_trials=completed_trials,
                config=config,
            ):
                stop_reason = self._closed_loop_stop_reason(analysis)
                break
            if (
                completed_trials >= config.closed_loop_min_trials
                and previous_throughput is not None
                and previous_throughput > 0.0
            ):
                relative_gain = (throughput - previous_throughput) / previous_throughput
                if relative_gain < config.closed_loop_plateau_relative_gain:
                    stop_reason = (
                        "request throughput plateaued: "
                        f"relative_gain={relative_gain:.3f} "
                        f"< {config.closed_loop_plateau_relative_gain:.3f}"
                    )
                    break
            previous_throughput = throughput
            concurrency *= 2

        result = ClosedLoopScoutResult(
            peak_request_throughput=peak_request_throughput,
            peak_output_token_throughput=peak_output_token_throughput,
            plateau_concurrency=plateau_concurrency,
            stop_reason=stop_reason,
        )
        self._trace["closed_loop"] = result.to_dict()
        self._write_trace()
        return result

    async def _find_open_loop_bracket(
        self,
        config: SearchConfig,
        closed_loop_result: ClosedLoopScoutResult | None,
    ) -> _SearchBounds:
        if closed_loop_result is not None and closed_loop_result.peak_request_throughput is not None:
            peak = closed_loop_result.peak_request_throughput
            first_rate = peak * config.closed_loop_start_rate_fraction
            second_rate = peak * config.closed_loop_high_rate_fraction
            if second_rate <= first_rate:
                second_rate = first_rate * config.open_loop_bracket_growth_factor
        else:
            first_rate = config.initial_request_rate
            second_rate = first_rate * config.open_loop_bracket_growth_factor

        bounds = _SearchBounds()
        if self._rate_exceeds_max_rate(config, first_rate):
            assert config.max_request_rate is not None
            bounds.max_request_rate_cap_attempted_rate = first_rate
            await self._test_open_loop_rate(
                config,
                bounds,
                config.max_request_rate,
                purpose="open_loop_bracket_cap",
            )
            if bounds.low_rate is not None and bounds.high_rate is None:
                bounds.max_request_rate_cap = config.max_request_rate
                self._record_bounds(bounds)
                return bounds
            if bounds.high_rate is not None and bounds.low_rate is None:
                await self._shrink_low_bound(config, bounds)
                return bounds

        await self._test_open_loop_rate(config, bounds, first_rate, purpose="open_loop_bracket")
        if bounds.low_rate is not None and bounds.high_rate is None:
            await self._grow_high_bound(config, bounds, second_rate)
        elif bounds.high_rate is not None and bounds.low_rate is None:
            await self._shrink_low_bound(config, bounds)
        if self._is_max_request_rate_limited(bounds):
            return bounds
        if bounds.low_rate is None or bounds.high_rate is None:
            raise SearchConvergenceError("open-loop search failed to establish both low and high bounds")
        return bounds

    async def _grow_high_bound(
        self,
        config: SearchConfig,
        bounds: _SearchBounds,
        start_rate: float,
    ) -> None:
        rate = start_rate
        for _ in range(config.max_bracket_trials):
            if self._rate_exceeds_max_rate(config, rate):
                assert config.max_request_rate is not None
                if bounds.low_rate is not None and bounds.low_rate < config.max_request_rate:
                    bounds.max_request_rate_cap_attempted_rate = rate
                    await self._test_open_loop_rate(
                        config,
                        bounds,
                        config.max_request_rate,
                        purpose="open_loop_bracket_high_cap",
                    )
                    if bounds.high_rate is not None:
                        return
                bounds.max_request_rate_cap = config.max_request_rate
                bounds.max_request_rate_cap_attempted_rate = rate
                self._record_bounds(bounds)
                return
            self._check_max_rate(config, rate)
            await self._test_open_loop_rate(config, bounds, rate, purpose="open_loop_bracket_high")
            if bounds.high_rate is not None:
                return
            rate *= config.open_loop_bracket_growth_factor
        raise SearchConvergenceError(
            f"open-loop bracketing did not find an unstable high bound after "
            f"{config.max_bracket_trials} trials"
        )

    async def _shrink_low_bound(self, config: SearchConfig, bounds: _SearchBounds) -> None:
        if bounds.high_rate is None:
            raise SearchConvergenceError("cannot shrink low bound before a high bound is known")
        rate = bounds.high_rate / config.open_loop_bracket_growth_factor
        for _ in range(config.max_bracket_trials):
            if rate <= 0.0:
                raise SearchConvergenceError("open-loop low-bound search reached a non-positive rate")
            await self._test_open_loop_rate(config, bounds, rate, purpose="open_loop_bracket_low")
            if bounds.low_rate is not None:
                return
            rate /= config.open_loop_bracket_growth_factor
        raise SearchConvergenceError(
            f"open-loop bracketing did not find a stable low bound below "
            f"{bounds.high_rate:.6g} req/s"
        )

    async def _binary_search(self, config: SearchConfig, bounds: _SearchBounds) -> _SearchBounds:
        for _ in range(config.max_binary_steps):
            if bounds.has_closed_relative_width(config.rate_precision):
                return bounds
            if bounds.low_rate is None or bounds.high_rate is None:
                raise SearchConvergenceError("binary search requires both low and high bounds")
            rate = (bounds.low_rate + bounds.high_rate) / 2.0
            await self._test_open_loop_rate(config, bounds, rate, purpose="open_loop_binary")
            if self._is_scheduler_config_limited(bounds):
                return bounds
        raise SearchConvergenceError(
            f"binary search did not reach rate_precision={config.rate_precision:.6g} "
            f"within {config.max_binary_steps} steps"
        )

    async def _confirm_result(
        self,
        config: SearchConfig,
        bounds: _SearchBounds,
        closed_loop_result: ClosedLoopScoutResult | None,
    ) -> SearchResult:
        for _ in range(config.max_binary_steps):
            if bounds.low_rate is None:
                raise SearchConvergenceError("cannot confirm a search without a stable low bound")
            failed_rate = bounds.low_rate
            stable_votes = 1
            rejecting_votes = 0
            stable_event: dict[str, object] | None = None
            stable_analysis: TrialAnalysisResult | None = None
            rejecting_event: dict[str, object] | None = None
            latest_event: dict[str, object] | None = None
            latest_analysis: TrialAnalysisResult | None = None

            for purpose in ("open_loop_confirmation", "open_loop_confirmation_majority"):
                event = await self._run_valid_open_loop_trial(
                    config,
                    duration_s=config.confirmation_duration_s,
                    purpose=purpose,
                    rate=failed_rate,
                )
                analysis = event["analysis_result"]
                decision = self._analysis_decision(analysis)
                latest_event = event
                latest_analysis = analysis
                if decision is True:
                    stable_votes += 1
                    stable_event = event
                    stable_analysis = analysis
                elif decision is False:
                    rejecting_votes += 1
                    rejecting_event = event
                if stable_votes >= 2 or rejecting_votes >= 2:
                    break

            if stable_votes >= 2:
                if stable_event is None or stable_analysis is None:
                    raise RuntimeError("confirmation majority was stable without a stable confirmation event")
                return self._build_search_result(
                    config=config,
                    bounds=bounds,
                    closed_loop_result=closed_loop_result,
                    confirmation_trial_id=str(stable_event["trial_id"]),
                    confirmation_analysis=stable_analysis,
                )

            previous_stable = self._stable_open_loop_point_below(failed_rate)
            if previous_stable is None:
                if latest_event is None or latest_analysis is None:
                    raise RuntimeError("confirmation completed without a trial event")
                termination_reason = (
                    "no_confirmed_stable_open_loop_rate"
                    if rejecting_votes >= 2
                    else "confirmation_inconclusive"
                )
                reported_rate = None if rejecting_votes >= 2 else failed_rate
                return self._build_unconfirmed_result(
                    config=config,
                    bounds=bounds,
                    closed_loop_result=closed_loop_result,
                    reported_rate=reported_rate,
                    confirmation_trial_id=str(latest_event["trial_id"]),
                    confirmation_analysis=latest_analysis,
                    termination_reason=termination_reason,
                    message=(
                        "confirmation majority rejected the selected low-bound rate "
                        "and no lower stable open-loop trial is available"
                        if rejecting_votes >= 2
                        else "confirmation passes did not produce a stable majority "
                        "and no lower stable open-loop trial is available"
                    ),
                )
            bounds.high_rate = failed_rate
            if rejecting_event is not None:
                bounds.high_trial_id = str(rejecting_event["trial_id"])
            elif latest_event is not None:
                bounds.high_trial_id = str(latest_event["trial_id"])
            bounds.low_rate = previous_stable.rate
            bounds.low_trial_id = previous_stable.trial_id
            self._record_bounds(bounds)
            bounds = await self._binary_search(config, bounds)
        raise SearchConvergenceError(
            "confirmation did not converge within max_binary_steps="
            f"{config.max_binary_steps}"
        )

    async def _run_valid_open_loop_trial(
        self,
        config: SearchConfig,
        *,
        duration_s: float,
        purpose: str,
        rate: float,
    ) -> dict[str, object]:
        for attempt_index in range(config.client_limited_retry_attempts + 1):
            retry_suffix = "" if attempt_index == 0 else f"_client_retry{attempt_index}"
            event = await self._run_and_analyze_trial(
                config,
                mode="open-loop",
                concurrency=None,
                request_rate=rate,
                duration_s=duration_s,
                purpose=f"{purpose}{retry_suffix}",
            )
            analysis = event["analysis_result"]
            if not isinstance(analysis, TrialAnalysisResult):
                raise RuntimeError("search trial analysis result has unexpected type")
            if analysis.trial_validity == "valid":
                return event
            if not self._retryable_invalid_trial(analysis):
                self._reject_invalid_trial(analysis, event["trial_id"])
            if attempt_index >= config.client_limited_retry_attempts:
                self._reject_invalid_trial(analysis, event["trial_id"])
            if config.client_limited_retry_cooldown_s > 0.0:
                await asyncio.sleep(config.client_limited_retry_cooldown_s)
        raise RuntimeError("client-limited retry loop exited unexpectedly")

    def _build_unconfirmed_result(
        self,
        *,
        config: SearchConfig,
        bounds: _SearchBounds,
        closed_loop_result: ClosedLoopScoutResult | None,
        reported_rate: float | None,
        confirmation_trial_id: str,
        confirmation_analysis: TrialAnalysisResult,
        termination_reason: str,
        message: str,
    ) -> SearchResult:
        bottleneck_class = "unknown"
        reasons = [message]
        if confirmation_analysis.stability is not None:
            reasons.extend(
                f"confirmation evidence: {reason}"
                for reason in confirmation_analysis.stability.reasons
            )
        if confirmation_analysis.bottleneck is not None:
            bottleneck_class = confirmation_analysis.bottleneck.bottleneck_class
            reasons.extend(
                f"confirmation bottleneck evidence: {item}"
                for item in confirmation_analysis.bottleneck.evidence
            )
        high_bottleneck = self._trace_bottleneck(bounds.high_trial_id)
        if high_bottleneck is not None:
            bottleneck_class = str(high_bottleneck["bottleneck_class"])
            reasons.extend(f"high-bound evidence: {item}" for item in high_bottleneck["evidence"])

        return SearchResult(
            search_id=config.search_id,
            search_mode=config.search_mode,
            max_no_drift_request_rate=reported_rate,
            max_slo_satisfying_request_rate=reported_rate,
            rate_precision=config.rate_precision,
            confirmation_trial_id=confirmation_trial_id,
            termination_reason=termination_reason,
            closed_loop=closed_loop_result,
            bottleneck_class=bottleneck_class,
            confidence="low",
            reasons=reasons,
        )

    def _build_search_result(
        self,
        *,
        config: SearchConfig,
        bounds: _SearchBounds,
        closed_loop_result: ClosedLoopScoutResult | None,
        confirmation_trial_id: str,
        confirmation_analysis: TrialAnalysisResult,
    ) -> SearchResult:
        bottleneck_class = "unknown"
        confidence: Confidence = "low"
        reasons: list[str] = []
        if confirmation_analysis.stability is not None:
            confidence = confirmation_analysis.stability.confidence
            reasons.extend(confirmation_analysis.stability.reasons)
        if confirmation_analysis.bottleneck is not None:
            bottleneck_class = confirmation_analysis.bottleneck.bottleneck_class
            confidence = _min_confidence(confidence, confirmation_analysis.bottleneck.confidence)
            reasons.extend(confirmation_analysis.bottleneck.evidence)
        high_bottleneck = self._trace_bottleneck(bounds.high_trial_id)
        if high_bottleneck is not None:
            bottleneck_class = str(high_bottleneck["bottleneck_class"])
            confidence = _min_confidence(confidence, _as_confidence(high_bottleneck["confidence"]))
            reasons.extend(f"high-bound evidence: {item}" for item in high_bottleneck["evidence"])

        return SearchResult(
            search_id=config.search_id,
            search_mode=config.search_mode,
            max_no_drift_request_rate=bounds.low_rate,
            max_slo_satisfying_request_rate=bounds.low_rate,
            rate_precision=config.rate_precision,
            confirmation_trial_id=confirmation_trial_id,
            termination_reason="confirmed_stable",
            closed_loop=closed_loop_result,
            bottleneck_class=bottleneck_class,
            confidence=confidence,
            reasons=reasons,
        )

    def _build_scheduler_config_limited_result(
        self,
        *,
        config: SearchConfig,
        bounds: _SearchBounds,
        closed_loop_result: ClosedLoopScoutResult | None,
    ) -> SearchResult:
        if bounds.low_rate is None or bounds.high_rate is None:
            raise SearchConvergenceError("scheduler-config-limited result requires closed search bounds")
        high_bottleneck = self._trace_bottleneck(bounds.high_trial_id)
        if high_bottleneck is None:
            raise SearchConvergenceError("scheduler-config-limited result requires high-bound bottleneck evidence")
        reasons = [
            "search stopped early because the high-bound trial identified a high-confidence "
            "scheduler configuration bottleneck; external orchestration should adjust the vLLM "
            "serving configuration before refining the rate bound"
        ]
        reasons.extend(f"high-bound evidence: {item}" for item in high_bottleneck["evidence"])

        return SearchResult(
            search_id=config.search_id,
            search_mode=config.search_mode,
            max_no_drift_request_rate=bounds.low_rate,
            max_slo_satisfying_request_rate=bounds.low_rate,
            rate_precision=config.rate_precision,
            confirmation_trial_id=None,
            termination_reason="scheduler_config_limited",
            closed_loop=closed_loop_result,
            bottleneck_class=str(high_bottleneck["bottleneck_class"]),
            confidence=_as_confidence(high_bottleneck["confidence"]),
            reasons=reasons,
        )

    def _build_max_request_rate_limited_result(
        self,
        *,
        config: SearchConfig,
        bounds: _SearchBounds,
        closed_loop_result: ClosedLoopScoutResult | None,
    ) -> SearchResult:
        if bounds.low_rate is None:
            raise SearchConvergenceError("max-request-rate-limited result requires a stable low bound")
        if bounds.max_request_rate_cap is None or bounds.max_request_rate_cap_attempted_rate is None:
            raise SearchConvergenceError("max-request-rate-limited result requires cap metadata")

        bottleneck_class = "unknown"
        confidence: Confidence = "low"
        reasons = [
            "open-loop bracketing stopped because the next required high-bound "
            f"rate {bounds.max_request_rate_cap_attempted_rate:.6g} req/s exceeds "
            f"max_request_rate={bounds.max_request_rate_cap:.6g}",
            f"highest observed stable open-loop rate before the cap was {bounds.low_rate:.6g} req/s",
            "the true unstable high bound was not measured; increase max_request_rate to refine MST",
        ]
        low_bottleneck = self._trace_bottleneck(bounds.low_trial_id)
        if low_bottleneck is not None:
            bottleneck_class = str(low_bottleneck["bottleneck_class"])
            confidence = _as_confidence(low_bottleneck["confidence"])
            reasons.extend(f"low-bound evidence: {item}" for item in low_bottleneck["evidence"])

        return SearchResult(
            search_id=config.search_id,
            search_mode=config.search_mode,
            max_no_drift_request_rate=bounds.low_rate,
            max_slo_satisfying_request_rate=bounds.low_rate,
            rate_precision=config.rate_precision,
            confirmation_trial_id=None,
            termination_reason="max_request_rate_limited",
            closed_loop=closed_loop_result,
            bottleneck_class=bottleneck_class,
            confidence=confidence,
            reasons=reasons,
        )

    def _closed_loop_only_result(
        self,
        config: SearchConfig,
        closed_loop_result: ClosedLoopScoutResult | None,
    ) -> SearchResult:
        if closed_loop_result is None:
            raise SearchConvergenceError("closed-loop search mode did not produce scout results")
        return SearchResult(
            search_id=config.search_id,
            search_mode=config.search_mode,
            max_no_drift_request_rate=None,
            max_slo_satisfying_request_rate=None,
            rate_precision=config.rate_precision,
            confirmation_trial_id=None,
            termination_reason="closed_loop_only",
            closed_loop=closed_loop_result,
            bottleneck_class="unknown",
            confidence="medium" if closed_loop_result.peak_request_throughput is not None else "low",
            reasons=[
                "closed-loop mode reports scouting throughput only; "
                "run open-loop or hybrid mode for max sustainable external arrival rate"
            ],
        )

    async def _test_open_loop_rate(
        self,
        config: SearchConfig,
        bounds: _SearchBounds,
        rate: float,
        *,
        purpose: str,
    ) -> None:
        self._check_max_rate(config, rate)
        event = await self._run_valid_open_loop_trial(
            config,
            duration_s=config.trial_duration_s,
            purpose=purpose,
            rate=rate,
        )
        analysis = event["analysis_result"]
        decision = self._analysis_decision(analysis)
        if decision is False and self._is_soft_unstable(analysis):
            repeat_event = await self._run_valid_open_loop_trial(
                config,
                duration_s=config.uncertain_retry_duration_s,
                purpose=f"{purpose}_extend_unstable",
                rate=rate,
            )
            repeat_analysis = repeat_event["analysis_result"]
            decision = self._analysis_decision(repeat_analysis)
            event = repeat_event
        if decision is None:
            repeat_event = await self._run_valid_open_loop_trial(
                config,
                duration_s=config.uncertain_retry_duration_s,
                purpose=f"{purpose}_extend_uncertain",
                rate=rate,
            )
            repeat_analysis = repeat_event["analysis_result"]
            decision = self._analysis_decision(repeat_analysis)
            if decision is None:
                if bounds.low_rate is None:
                    raise SearchConvergenceError(
                        f"rate {rate:.6g} req/s remained uncertain and no stable low bound exists"
                    )
                bounds.high_rate = rate
                bounds.high_trial_id = str(repeat_event["trial_id"])
                self._record_bounds(bounds)
                return
            event = repeat_event
        if decision:
            bounds.low_rate = rate
            bounds.low_trial_id = str(event["trial_id"])
        else:
            bounds.high_rate = rate
            bounds.high_trial_id = str(event["trial_id"])
        self._record_bounds(bounds)

    async def _run_and_analyze_trial(
        self,
        config: SearchConfig,
        *,
        mode: Literal["open-loop", "closed-loop"],
        concurrency: int | None,
        request_rate: float | None,
        duration_s: float,
        purpose: str,
    ) -> dict[str, object]:
        trial_id, trial_dir = self._next_trial_identity(mode, concurrency, request_rate)
        trial_config = TrialConfig(
            trial_id=trial_id,
            mode=mode,
            duration_s=duration_s,
            base_url=config.base_url,
            endpoint=config.endpoint,
            model=config.model,
            request_rate=request_rate,
            concurrency=concurrency,
            think_time_s=config.think_time_s,
            burstiness=config.burstiness,
            request_timeout_s=config.request_timeout_s,
            api_key=config.api_key,
            extra_headers=config.extra_headers,
            extra_body=config.extra_body,
            safety_max_outstanding=config.safety_max_outstanding,
            metrics_url=config.metrics_url,
            metrics_interval_s=config.metrics_interval_s,
            window_s=config.window_s,
            metadata=config.metadata,
        )
        run_result = await self._runner.run_trial(
            trial_config,
            request_source=self._request_source_factory(),
            output_dir=trial_dir,
        )
        analysis = self._analyze_trial(trial_dir)
        self._write_analysis(trial_dir, analysis)
        event = {
            "purpose": purpose,
            "trial_id": trial_id,
            "trial_dir": str(trial_dir),
            "mode": mode,
            "request_rate": request_rate,
            "concurrency": concurrency,
            "summary": _summary_trace(run_result),
            "analysis": analysis.to_dict(),
        }
        self._trace_events().append(event)
        self._write_trace()
        return {"trial_id": trial_id, "trial_dir": trial_dir, "run_result": run_result, "analysis_result": analysis}

    def _next_trial_identity(
        self,
        mode: Literal["open-loop", "closed-loop"],
        concurrency: int | None,
        request_rate: float | None,
    ) -> tuple[str, Path]:
        index = self._trial_index
        self._trial_index += 1
        if mode == "closed-loop":
            assert concurrency is not None
            label = f"closedloop_N{concurrency}"
        else:
            assert request_rate is not None
            label = f"openloop_r{_rate_label(request_rate)}"
        trial_id = f"trial_{index:03d}_{label}"
        return trial_id, self._trials_dir / trial_id

    def _record_bounds(self, bounds: _SearchBounds) -> None:
        self._trace["bounds"] = {
            "low_rate": bounds.low_rate,
            "low_trial_id": bounds.low_trial_id,
            "high_rate": bounds.high_rate,
            "high_trial_id": bounds.high_trial_id,
            "max_request_rate_cap": bounds.max_request_rate_cap,
            "max_request_rate_cap_attempted_rate": bounds.max_request_rate_cap_attempted_rate,
        }
        self._write_trace()

    def _finish_trace(self, result: SearchResult) -> None:
        self._trace["result"] = result.to_dict()
        self._write_trace()

    def _trace_events(self) -> list[dict[str, object]]:
        events = self._trace.get("events")
        if not isinstance(events, list):
            raise RuntimeError("search trace events container was corrupted")
        return events

    def _trace_bottleneck(self, trial_id: str | None) -> dict[str, object] | None:
        if trial_id is None:
            return None
        for event in self._trace_events():
            if event.get("trial_id") != trial_id:
                continue
            analysis = event.get("analysis")
            if not isinstance(analysis, dict):
                raise RuntimeError("search trace analysis entry was corrupted")
            bottleneck = analysis.get("bottleneck")
            if bottleneck is None:
                return None
            if not isinstance(bottleneck, dict):
                raise RuntimeError("search trace bottleneck entry was corrupted")
            if bottleneck.get("bottleneck_class") == "unknown":
                return None
            evidence = bottleneck.get("evidence")
            if not isinstance(evidence, list):
                raise RuntimeError("search trace bottleneck evidence was corrupted")
            return bottleneck
        return None

    def _is_scheduler_config_limited(self, bounds: _SearchBounds) -> bool:
        if bounds.low_rate is None or bounds.high_rate is None:
            return False
        high_bottleneck = self._trace_bottleneck(bounds.high_trial_id)
        if high_bottleneck is None:
            return False
        return (
            high_bottleneck.get("bottleneck_class") == "scheduler_cap"
            and high_bottleneck.get("confidence") == "high"
        )

    @staticmethod
    def _is_max_request_rate_limited(bounds: _SearchBounds) -> bool:
        return (
            bounds.low_rate is not None
            and bounds.high_rate is None
            and bounds.max_request_rate_cap is not None
            and bounds.max_request_rate_cap_attempted_rate is not None
        )

    def _stable_open_loop_point_below(self, rate: float) -> _StableOpenLoopPoint | None:
        best: _StableOpenLoopPoint | None = None
        for event in self._trace_events():
            if event.get("mode") != "open-loop":
                continue
            request_rate = event.get("request_rate")
            trial_id = event.get("trial_id")
            if not isinstance(request_rate, (int, float)) or not isinstance(trial_id, str):
                raise RuntimeError("search trace open-loop event was corrupted")
            if request_rate >= rate:
                continue
            analysis = event.get("analysis")
            if not isinstance(analysis, dict):
                raise RuntimeError("search trace analysis entry was corrupted")
            stability = analysis.get("stability")
            if not isinstance(stability, dict) or stability.get("status") != "stable":
                continue
            if best is None or request_rate > best.rate:
                best = _StableOpenLoopPoint(rate=float(request_rate), trial_id=trial_id)
        return best

    def _write_trace(self) -> None:
        self._trace_path.write_text(
            json.dumps(self._trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _reject_invalid_trial(analysis: TrialAnalysisResult, trial_id: object) -> None:
        if analysis.trial_validity == "valid":
            return
        reason_text = "; ".join(analysis.validity_reasons)
        if analysis.trial_validity == "invalid_workload":
            raise InvalidSearchTrial(
                f"trial {trial_id} found invalid workload samples; search bounds were not updated. "
                f"Fix context_policy/tokenizer/max_model_len or workload sampling. Evidence: {reason_text}"
            )
        if analysis.trial_validity == "client_limited":
            raise InvalidSearchTrial(
                f"trial {trial_id} was client-limited; search bounds were not updated. "
                f"Reduce configured rates or client load, or increase client capacity. Evidence: {reason_text}"
            )
        raise InvalidSearchTrial(
            f"trial {trial_id} was not valid for search bounds: "
            f"{analysis.trial_validity}. Evidence: {reason_text}"
        )

    @staticmethod
    def _retryable_invalid_trial(analysis: TrialAnalysisResult) -> bool:
        return analysis.trial_validity in {"client_limited", "metrics_invalid"}

    @staticmethod
    def _analysis_decision(analysis: TrialAnalysisResult) -> bool | None:
        if analysis.stability is None:
            raise ValueError("valid search trial requires a stability result")
        if analysis.stability.status == "stable":
            return True
        if analysis.stability.status in {"unstable", "slo_violation", "aborted_safety"}:
            return False
        if analysis.stability.status == "uncertain":
            return None
        raise ValueError(f"unsupported stability status {analysis.stability.status!r}")

    @staticmethod
    def _closed_loop_stop_reason(analysis: TrialAnalysisResult) -> str:
        if analysis.stability is None:
            raise ValueError("closed-loop stop reason requires stability analysis")
        if analysis.stability.status == "slo_violation":
            return "TPOT/TTFT SLO violation detected during closed-loop scouting"
        if analysis.bottleneck is not None and analysis.bottleneck.bottleneck_class == "kv_cache":
            return "KV/preemption wall detected during closed-loop scouting"
        return f"closed-loop scouting stopped on stability status {analysis.stability.status!r}"

    @staticmethod
    def _is_soft_unstable(analysis: TrialAnalysisResult) -> bool:
        if analysis.stability is None:
            raise ValueError("valid search trial requires a stability result")
        return analysis.stability.status == "unstable"

    @staticmethod
    def _should_stop_closed_loop(
        analysis: TrialAnalysisResult,
        *,
        completed_trials: int,
        config: SearchConfig,
    ) -> bool:
        if analysis.stability is None:
            raise ValueError("closed-loop search trial requires a stability result")
        if analysis.stability.status == "aborted_safety":
            return True
        if completed_trials < config.closed_loop_min_trials:
            return False
        if analysis.stability.status == "slo_violation":
            return True
        if (
            analysis.stability.status != "stable"
            and analysis.bottleneck is not None
            and analysis.bottleneck.bottleneck_class == "kv_cache"
            and analysis.bottleneck.confidence == "high"
        ):
            return True
        return False

    @staticmethod
    def _check_max_rate(config: SearchConfig, rate: float) -> None:
        _require_positive("request_rate", rate)
        if SearchController._rate_exceeds_max_rate(config, rate):
            raise SearchConvergenceError(
                f"required bracketing rate {rate:.6g} req/s exceeds max_request_rate="
                f"{config.max_request_rate:.6g}"
            )

    @staticmethod
    def _rate_exceeds_max_rate(config: SearchConfig, rate: float) -> bool:
        return config.max_request_rate is not None and rate > config.max_request_rate


def _summary_trace(run_result: TrialRunResult) -> dict[str, object]:
    summary = run_result.summary
    return {
        "status": summary.status,
        "requested_request_rate": summary.requested_request_rate,
        "requested_concurrency": summary.requested_concurrency,
        "actual_send_rate": summary.actual_send_rate,
        "successful_completion_rate": summary.successful_completion_rate,
        "error_rate": summary.error_rate,
        "generation_token_throughput": summary.benchmark_metrics.generation_token_throughput,
        "total_token_throughput": summary.benchmark_metrics.total_token_throughput,
        "max_observed_outstanding": summary.max_observed_outstanding,
        "abort_reason": summary.abort_reason,
    }


def _rate_label(rate: float) -> str:
    return f"{rate:.6g}".replace(".", "_")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _require_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _min_confidence(left: Confidence, right: Confidence) -> Confidence:
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order[left] <= order[right] else right


def _as_confidence(value: object) -> Confidence:
    if value not in {"high", "medium", "low"}:
        raise RuntimeError(f"invalid confidence in search trace: {value!r}")
    return cast(Confidence, value)


__all__ = [
    "ClosedLoopScoutResult",
    "InvalidSearchTrial",
    "SearchConfig",
    "SearchController",
    "SearchConvergenceError",
    "SearchError",
    "SearchResult",
]
