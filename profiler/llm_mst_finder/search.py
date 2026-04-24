from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol, cast

from .analysis import analyze_trial_dir, write_analysis_artifact
from .loadgen import RequestSource
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
    final_confirmation_duration_s: float | None = None
    rate_precision: float = 0.03
    initial_request_rate: float = 1.0
    max_request_rate: float | None = None
    max_binary_steps: int = 24
    max_bracket_trials: int = 16
    closed_loop_initial_concurrency: int = 1
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
        _require_positive_int("closed_loop_initial_concurrency", self.closed_loop_initial_concurrency)
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

    @property
    def confirmation_duration_s(self) -> float:
        return self.final_confirmation_duration_s or self.trial_duration_s

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

    def has_closed_relative_width(self, precision: float) -> bool:
        if self.low_rate is None or self.high_rate is None:
            return False
        if self.low_rate <= 0.0:
            return False
        return (self.high_rate - self.low_rate) / self.low_rate <= precision


class SearchController:
    def __init__(
        self,
        runner: TrialRunnerProtocol | TrialRunner,
        *,
        request_source: RequestSource,
        output_dir: str | Path,
        analyze_trial: Callable[[str | Path], TrialAnalysisResult] = analyze_trial_dir,
        write_analysis: Callable[[str | Path, TrialAnalysisResult], Path] = write_analysis_artifact,
    ) -> None:
        self._runner = runner
        self._request_source = request_source
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
        bounds = await self._binary_search(config, bounds)
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

            throughput = summary.successful_completion_rate
            output_tok_s = summary.benchmark_metrics.generation_token_throughput
            if peak_request_throughput is None or throughput > peak_request_throughput:
                peak_request_throughput = throughput
                peak_output_token_throughput = output_tok_s
                plateau_concurrency = concurrency

            decision = self._analysis_decision(analysis)
            if decision is False:
                stop_reason = self._closed_loop_stop_reason(analysis)
                break
            if previous_throughput is not None and previous_throughput > 0.0:
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
            first_rate = max(config.initial_request_rate, peak * config.closed_loop_start_rate_fraction)
            second_rate = peak * config.closed_loop_high_rate_fraction
            if second_rate <= first_rate:
                second_rate = first_rate * 2.0
        else:
            first_rate = config.initial_request_rate
            second_rate = first_rate * 2.0

        bounds = _SearchBounds()
        await self._test_open_loop_rate(config, bounds, first_rate, purpose="open_loop_bracket")
        if bounds.low_rate is not None and bounds.high_rate is None:
            await self._grow_high_bound(config, bounds, second_rate)
        elif bounds.high_rate is not None and bounds.low_rate is None:
            await self._shrink_low_bound(config, bounds)
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
            self._check_max_rate(config, rate)
            await self._test_open_loop_rate(config, bounds, rate, purpose="open_loop_bracket_high")
            if bounds.high_rate is not None:
                return
            rate *= 2.0
        raise SearchConvergenceError(
            f"open-loop bracketing did not find an unstable high bound after "
            f"{config.max_bracket_trials} trials"
        )

    async def _shrink_low_bound(self, config: SearchConfig, bounds: _SearchBounds) -> None:
        if bounds.high_rate is None:
            raise SearchConvergenceError("cannot shrink low bound before a high bound is known")
        rate = bounds.high_rate / 2.0
        for _ in range(config.max_bracket_trials):
            if rate <= 0.0:
                raise SearchConvergenceError("open-loop low-bound search reached a non-positive rate")
            await self._test_open_loop_rate(config, bounds, rate, purpose="open_loop_bracket_low")
            if bounds.low_rate is not None:
                return
            rate /= 2.0
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
        if bounds.low_rate is None:
            raise SearchConvergenceError("cannot confirm a search without a stable low bound")
        event = await self._run_and_analyze_trial(
            config,
            mode="open-loop",
            concurrency=None,
            request_rate=bounds.low_rate,
            duration_s=config.confirmation_duration_s,
            purpose="open_loop_confirmation",
        )
        analysis = event["analysis_result"]
        self._reject_invalid_trial(analysis, event["trial_id"])
        decision = self._analysis_decision(analysis)
        if decision is not True:
            status = None if analysis.stability is None else analysis.stability.status
            raise SearchConvergenceError(
                "final confirmation trial did not prove the selected rate sustainable: "
                f"trial_id={event['trial_id']}, status={status!r}"
            )

        bottleneck_class = "unknown"
        confidence: Confidence = "low"
        reasons: list[str] = []
        if analysis.stability is not None:
            confidence = analysis.stability.confidence
            reasons.extend(analysis.stability.reasons)
        if analysis.bottleneck is not None:
            bottleneck_class = analysis.bottleneck.bottleneck_class
            confidence = _min_confidence(confidence, analysis.bottleneck.confidence)
            reasons.extend(analysis.bottleneck.evidence)
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
            confirmation_trial_id=str(event["trial_id"]),
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
        event = await self._run_and_analyze_trial(
            config,
            mode="open-loop",
            concurrency=None,
            request_rate=rate,
            duration_s=config.trial_duration_s,
            purpose=purpose,
        )
        analysis = event["analysis_result"]
        self._reject_invalid_trial(analysis, event["trial_id"])
        decision = self._analysis_decision(analysis)
        if decision is None:
            repeat_event = await self._run_and_analyze_trial(
                config,
                mode="open-loop",
                concurrency=None,
                request_rate=rate,
                duration_s=config.trial_duration_s,
                purpose=f"{purpose}_repeat_uncertain",
            )
            repeat_analysis = repeat_event["analysis_result"]
            self._reject_invalid_trial(repeat_analysis, repeat_event["trial_id"])
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
            request_source=self._request_source,
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
            return "TPOT/TTFT/E2E SLO violation detected during closed-loop scouting"
        if analysis.bottleneck is not None and analysis.bottleneck.bottleneck_class == "kv_cache":
            return "KV/preemption wall detected during closed-loop scouting"
        return f"closed-loop scouting stopped on stability status {analysis.stability.status!r}"

    @staticmethod
    def _check_max_rate(config: SearchConfig, rate: float) -> None:
        _require_positive("request_rate", rate)
        if config.max_request_rate is not None and rate > config.max_request_rate:
            raise SearchConvergenceError(
                f"required bracketing rate {rate:.6g} req/s exceeds max_request_rate="
                f"{config.max_request_rate:.6g}"
            )


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
