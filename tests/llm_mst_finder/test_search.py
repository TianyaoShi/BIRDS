from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_mst_finder.loadgen import cycling_request_source
from llm_mst_finder.records import (
    BenchmarkMetrics,
    BottleneckResult,
    SampleRequest,
    StabilityResult,
    TrialAnalysisResult,
    TrialConfig,
    TrialSummary,
)
from llm_mst_finder.search import InvalidSearchTrial, SearchConfig, SearchController
from llm_mst_finder.trial_runner import TrialArtifacts, TrialRunResult


def _metrics(*, request_throughput: float, output_tok_s: float) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        successful_requests=10,
        failed_requests=0,
        total_input_tokens=100,
        total_output_tokens=50,
        request_throughput=request_throughput,
        successful_request_throughput=request_throughput,
        prompt_token_throughput=request_throughput * 10.0,
        generation_token_throughput=output_tok_s,
        total_token_throughput=(request_throughput * 10.0) + output_tok_s,
        mean_ttft_ms=100.0,
        median_ttft_ms=100.0,
        std_ttft_ms=0.0,
        percentiles_ttft_ms=[(0.9, 100.0), (0.99, 110.0)],
        mean_tpot_ms=20.0,
        median_tpot_ms=20.0,
        std_tpot_ms=0.0,
        percentiles_tpot_ms=[(0.9, 20.0), (0.99, 22.0)],
        mean_itl_ms=18.0,
        median_itl_ms=18.0,
        std_itl_ms=0.0,
        percentiles_itl_ms=[(0.9, 18.0)],
        mean_e2e_ms=600.0,
        median_e2e_ms=600.0,
        std_e2e_ms=0.0,
        percentiles_e2e_ms=[(0.9, 600.0), (0.99, 650.0)],
        prompt_length_summary={"mean": 10.0, "median": 10.0, "p90": 10.0, "p99": 10.0},
        output_length_summary={"mean": 5.0, "median": 5.0, "p90": 5.0, "p99": 5.0},
    )


def _summary(config: TrialConfig, *, throughput: float, output_tok_s: float) -> TrialSummary:
    return TrialSummary(
        trial_id=config.trial_id,
        mode=config.mode,
        status="completed",
        requested_request_rate=config.request_rate,
        requested_concurrency=config.concurrency,
        target_duration_s=config.duration_s,
        wall_time_s=config.duration_s,
        started_requests=10,
        successful_requests=10,
        failed_requests=0,
        actual_send_rate=throughput,
        successful_completion_rate=throughput,
        error_rate=0.0,
        mean_scheduling_delay_s=0.0,
        max_scheduling_delay_s=0.0,
        max_observed_outstanding=config.concurrency or 1,
        metrics_sample_count=0,
        abort_reason=None,
        benchmark_metrics=_metrics(request_throughput=throughput, output_tok_s=output_tok_s),
        metadata=config.metadata,
    )


def _analysis(
    trial_id: str,
    *,
    validity: str = "valid",
    status: str = "stable",
    bottleneck_class: str = "unknown",
    bottleneck_confidence: str = "medium",
) -> TrialAnalysisResult:
    if validity != "valid":
        return TrialAnalysisResult(
            trial_id=trial_id,
            trial_validity=validity,
            validity_reasons=[f"{validity} fixture"],
            stability=None,
            bottleneck=None,
        )
    return TrialAnalysisResult(
        trial_id=trial_id,
        trial_validity="valid",
        validity_reasons=["fixture valid trial"],
        stability=StabilityResult(
            status=status,
            confidence="high",
            reasons=[f"fixture classified {status}"],
            key_metrics={},
        ),
        bottleneck=BottleneckResult(
            bottleneck_class=bottleneck_class,
            confidence=bottleneck_confidence,
            evidence=["fixture bottleneck evidence"],
        ),
    )


class FakeRunner:
    def __init__(self, *, sustainable_rate: float, closed_loop_peak: float = 12.0) -> None:
        self.sustainable_rate = sustainable_rate
        self.closed_loop_peak = closed_loop_peak
        self.analyses: dict[str, TrialAnalysisResult] = {}
        self.calls: list[TrialConfig] = []

    async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
        del request_source
        self.calls.append(config)
        trial_dir = Path(output_dir)
        trial_dir.mkdir(parents=True)
        if config.mode == "closed-loop":
            assert config.concurrency is not None
            throughput = min(float(config.concurrency * 2), self.closed_loop_peak)
            status = "stable"
        else:
            assert config.request_rate is not None
            throughput = config.request_rate
            status = "stable" if config.request_rate <= self.sustainable_rate else "unstable"
        self.analyses[config.trial_id] = _analysis(config.trial_id, status=status)
        artifacts = TrialArtifacts(
            output_dir=trial_dir,
            request_records_path=trial_dir / "request_records.jsonl",
            summary_path=trial_dir / "summary.json",
            server_metrics_path=None,
            windows_path=trial_dir / "windows.csv",
        )
        return TrialRunResult(
            config=config,
            summary=_summary(config, throughput=throughput, output_tok_s=throughput * 20.0),
            request_records=[],
            server_metrics=[],
            artifacts=artifacts,
        )


def _source():
    return cycling_request_source([SampleRequest(prompt="hello", prompt_len=1, expected_output_len=1)])


def test_hybrid_search_converges_and_writes_trace(tmp_path: Path) -> None:
    async def run() -> None:
        runner = FakeRunner(sustainable_rate=10.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-search",
                search_mode="hybrid",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.05,
                initial_request_rate=1.0,
                max_closed_loop_concurrency=16,
            )
        )

        assert result.max_no_drift_request_rate is not None
        assert 9.5 <= result.max_no_drift_request_rate <= 10.0
        assert result.closed_loop is not None
        assert result.closed_loop.peak_request_throughput == pytest.approx(12.0)
        trace = json.loads((tmp_path / "search" / "search_trace.json").read_text(encoding="utf-8"))
        assert trace["result"]["max_no_drift_request_rate"] == result.max_no_drift_request_rate
        rates = [event["request_rate"] for event in trace["events"] if event["mode"] == "open-loop"]
        assert rates
        assert result.confirmation_trial_id == trace["events"][-1]["trial_id"]

    asyncio.run(run())


def test_hybrid_search_reports_max_request_rate_limited_result(tmp_path: Path) -> None:
    async def run() -> None:
        runner = FakeRunner(sustainable_rate=20.0, closed_loop_peak=12.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-max-rate-limited",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-max-rate-limited",
                search_mode="hybrid",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.05,
                initial_request_rate=1.0,
                max_closed_loop_concurrency=8,
                closed_loop_min_trials=2,
                max_request_rate=8.0,
            )
        )

        assert result.termination_reason == "max_request_rate_limited"
        assert result.max_no_drift_request_rate == pytest.approx(8.0)
        assert result.max_slo_satisfying_request_rate == pytest.approx(8.0)
        assert result.confirmation_trial_id is None
        assert any("max_request_rate=8" in reason for reason in result.reasons)

        open_loop_calls = [call for call in runner.calls if call.mode == "open-loop"]
        assert [call.request_rate for call in open_loop_calls] == [pytest.approx(7.2), pytest.approx(8.0)]
        trace = json.loads(
            (tmp_path / "search-max-rate-limited" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        assert trace["result"]["termination_reason"] == "max_request_rate_limited"
        assert trace["bounds"]["low_rate"] == pytest.approx(8.0)
        assert trace["bounds"]["high_rate"] is None
        assert trace["bounds"]["max_request_rate_cap"] == 8.0
        assert trace["bounds"]["max_request_rate_cap_attempted_rate"] == pytest.approx(14.4)

    asyncio.run(run())


def test_open_loop_search_uses_configured_bracket_growth_factor(tmp_path: Path) -> None:
    async def run() -> None:
        runner = FakeRunner(sustainable_rate=2.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-growth-factor",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        await controller.search(
            SearchConfig(
                search_id="fixture-growth-factor",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.05,
                initial_request_rate=1.0,
                open_loop_bracket_growth_factor=1.5,
            )
        )

        open_loop_calls = [call for call in runner.calls if call.mode == "open-loop"]
        assert [call.request_rate for call in open_loop_calls[:3]] == [
            pytest.approx(1.0),
            pytest.approx(1.5),
            pytest.approx(2.25),
        ]

    asyncio.run(run())


def test_hybrid_search_caps_first_closed_loop_seed_rate(tmp_path: Path) -> None:
    async def run() -> None:
        runner = FakeRunner(sustainable_rate=20.0, closed_loop_peak=20.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-first-rate-capped",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-first-rate-capped",
                search_mode="hybrid",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.05,
                initial_request_rate=1.0,
                max_closed_loop_concurrency=16,
                closed_loop_min_trials=2,
                max_request_rate=8.0,
            )
        )

        assert result.termination_reason == "max_request_rate_limited"
        assert result.max_no_drift_request_rate == pytest.approx(8.0)
        assert result.closed_loop is not None
        assert result.closed_loop.peak_request_throughput == pytest.approx(20.0)

        open_loop_calls = [call for call in runner.calls if call.mode == "open-loop"]
        assert [call.request_rate for call in open_loop_calls] == [pytest.approx(8.0)]
        trace = json.loads(
            (tmp_path / "search-first-rate-capped" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        assert trace["bounds"]["low_rate"] == pytest.approx(8.0)
        assert trace["bounds"]["max_request_rate_cap"] == 8.0
        assert trace["bounds"]["max_request_rate_cap_attempted_rate"] == pytest.approx(12.0)
        assert trace["events"][-1]["purpose"] == "open_loop_bracket_cap"

    asyncio.run(run())


def test_hybrid_search_uses_max_request_rate_as_high_bound_when_cap_is_unstable(tmp_path: Path) -> None:
    async def run() -> None:
        runner = FakeRunner(sustainable_rate=7.5, closed_loop_peak=12.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-cap-as-high-bound",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-cap-as-high-bound",
                search_mode="hybrid",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.05,
                initial_request_rate=1.0,
                max_closed_loop_concurrency=8,
                closed_loop_min_trials=2,
                max_request_rate=8.0,
                max_binary_steps=8,
            )
        )

        assert result.termination_reason == "confirmed_stable"
        assert result.max_no_drift_request_rate is not None
        assert 7.2 <= result.max_no_drift_request_rate < 8.0

        open_loop_calls = [call for call in runner.calls if call.mode == "open-loop"]
        assert open_loop_calls[0].request_rate == pytest.approx(7.2)
        assert open_loop_calls[1].request_rate == pytest.approx(8.0)
        trace = json.loads(
            (tmp_path / "search-cap-as-high-bound" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        assert trace["result"]["termination_reason"] == "confirmed_stable"
        assert trace["bounds"]["high_rate"] < 8.0
        cap_events = [
            event for event in trace["events"]
            if event["purpose"] == "open_loop_bracket_high_cap"
        ]
        assert len(cap_events) == 1
        assert cap_events[0]["request_rate"] == pytest.approx(8.0)

    asyncio.run(run())


def test_closed_loop_does_not_stop_on_low_confidence_stable_kv_label(tmp_path: Path) -> None:
    class LowConfidenceKvRunner(FakeRunner):
        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "closed-loop" and config.concurrency == 1:
                self.analyses[config.trial_id] = _analysis(
                    config.trial_id,
                    status="stable",
                    bottleneck_class="kv_cache",
                    bottleneck_confidence="low",
                )
            return result

    async def run() -> None:
        runner = LowConfidenceKvRunner(sustainable_rate=10.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-closedloop-kv",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-closedloop-kv",
                search_mode="closed-loop",
                model="fake-model",
                trial_duration_s=1.0,
                max_closed_loop_concurrency=2,
            )
        )

        closed_loop_calls = [call for call in runner.calls if call.mode == "closed-loop"]
        assert [call.concurrency for call in closed_loop_calls] == [1, 2]
        assert result.closed_loop is not None
        assert result.closed_loop.peak_request_throughput == pytest.approx(4.0)

    asyncio.run(run())


def test_soft_unstable_open_loop_rate_is_retried_before_rejecting(tmp_path: Path) -> None:
    class SoftUnstableOnceRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=1.0)
            self.rate_calls: dict[float, int] = {}

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate is not None:
                rate = float(config.request_rate)
                self.rate_calls[rate] = self.rate_calls.get(rate, 0) + 1
                if rate == pytest.approx(2.0) and self.rate_calls[rate] == 2:
                    self.analyses[config.trial_id] = _analysis(config.trial_id, status="stable")
            return result

    async def run() -> None:
        runner = SoftUnstableOnceRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-soft-unstable",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        await controller.search(
            SearchConfig(
                search_id="fixture-soft-unstable",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                uncertain_trial_duration_s=3.0,
                rate_precision=0.9,
                initial_request_rate=1.0,
            )
        )

        rate_two_calls = [
            call
            for call in runner.calls
            if call.mode == "open-loop" and call.request_rate == pytest.approx(2.0)
        ]
        assert [call.duration_s for call in rate_two_calls[:2]] == [1.0, 3.0]
        trace = json.loads((tmp_path / "search-soft-unstable" / "search_trace.json").read_text(encoding="utf-8"))
        rate_two_events = [
            event
            for event in trace["events"]
            if event["mode"] == "open-loop" and event["request_rate"] == pytest.approx(2.0)
        ]
        assert [event["purpose"] for event in rate_two_events[:2]] == [
            "open_loop_bracket_high",
            "open_loop_bracket_high_extend_unstable",
        ]

    asyncio.run(run())


def test_invalid_workload_trial_fails_without_updating_high_bound(tmp_path: Path) -> None:
    class InvalidAtHighRunner(FakeRunner):
        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate == pytest.approx(2.0):
                self.analyses[config.trial_id] = _analysis(
                    config.trial_id,
                    validity="invalid_workload",
                )
            return result

    async def run() -> None:
        runner = InvalidAtHighRunner(sustainable_rate=10.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-invalid",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        with pytest.raises(InvalidSearchTrial, match="invalid workload"):
            await controller.search(
                SearchConfig(
                    search_id="fixture-invalid",
                    search_mode="open-loop",
                    model="fake-model",
                    trial_duration_s=1.0,
                    initial_request_rate=1.0,
                    rate_precision=0.1,
                )
            )
        trace = json.loads((tmp_path / "search-invalid" / "search_trace.json").read_text(encoding="utf-8"))
        assert trace["bounds"]["low_rate"] == 1.0
        assert trace["bounds"]["high_rate"] is None
        assert trace["result"] is None

    asyncio.run(run())


def test_client_limited_open_loop_trial_cools_down_and_retries(tmp_path: Path) -> None:
    class ClientLimitedOnceRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=10.0)
            self.rate_two_calls = 0

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate == pytest.approx(2.0):
                self.rate_two_calls += 1
                if self.rate_two_calls == 1:
                    self.analyses[config.trial_id] = _analysis(
                        config.trial_id,
                        validity="client_limited",
                    )
            return result

    async def run() -> None:
        runner = ClientLimitedOnceRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-client-limited-retry",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-client-limited-retry",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.1,
                initial_request_rate=1.0,
                max_request_rate=2.0,
                client_limited_retry_attempts=1,
                client_limited_retry_cooldown_s=0.0,
            )
        )

        assert result.termination_reason == "max_request_rate_limited"
        assert result.max_no_drift_request_rate == pytest.approx(2.0)
        rate_two_calls = [
            call
            for call in runner.calls
            if call.mode == "open-loop" and call.request_rate == pytest.approx(2.0)
        ]
        assert len(rate_two_calls) == 2
        trace = json.loads(
            (tmp_path / "search-client-limited-retry" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        rate_two_events = [
            event
            for event in trace["events"]
            if event["mode"] == "open-loop" and event["request_rate"] == pytest.approx(2.0)
        ]
        assert [event["purpose"] for event in rate_two_events] == [
            "open_loop_bracket_high",
            "open_loop_bracket_high_client_retry1",
        ]
        assert trace["bounds"]["low_rate"] == pytest.approx(2.0)

    asyncio.run(run())


def test_metrics_invalid_open_loop_trial_is_retried(tmp_path: Path) -> None:
    class MetricsInvalidOnceRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=10.0)
            self.rate_two_calls = 0

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate == pytest.approx(2.0):
                self.rate_two_calls += 1
                if self.rate_two_calls == 1:
                    self.analyses[config.trial_id] = _analysis(
                        config.trial_id,
                        validity="metrics_invalid",
                    )
            return result

    async def run() -> None:
        runner = MetricsInvalidOnceRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-metrics-invalid-retry",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-metrics-invalid-retry",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.1,
                initial_request_rate=1.0,
                max_request_rate=2.0,
                client_limited_retry_attempts=1,
                client_limited_retry_cooldown_s=0.0,
            )
        )

        assert result.termination_reason == "max_request_rate_limited"
        rate_two_calls = [
            call
            for call in runner.calls
            if call.mode == "open-loop" and call.request_rate == pytest.approx(2.0)
        ]
        assert len(rate_two_calls) == 2
        trace = json.loads(
            (tmp_path / "search-metrics-invalid-retry" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        rate_two_events = [
            event
            for event in trace["events"]
            if event["mode"] == "open-loop" and event["request_rate"] == pytest.approx(2.0)
        ]
        assert [event["purpose"] for event in rate_two_events] == [
            "open_loop_bracket_high",
            "open_loop_bracket_high_client_retry1",
        ]
        assert trace["bounds"]["low_rate"] == pytest.approx(2.0)

    asyncio.run(run())


def test_uncertain_open_loop_rate_is_retried_with_extended_duration(tmp_path: Path) -> None:
    class UncertainFirstHighRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=1.0)
            self.high_rate_calls = 0

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(
                config,
                request_source=request_source,
                output_dir=output_dir,
            )
            if config.mode == "open-loop" and config.request_rate == pytest.approx(2.0):
                self.high_rate_calls += 1
                if self.high_rate_calls == 1:
                    self.analyses[config.trial_id] = _analysis(config.trial_id, status="uncertain")
            return result

    async def run() -> None:
        runner = UncertainFirstHighRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-uncertain-extended",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        await controller.search(
            SearchConfig(
                search_id="fixture-uncertain-extended",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                uncertain_trial_duration_s=3.0,
                rate_precision=0.9,
                initial_request_rate=1.0,
            )
        )

        high_rate_calls = [
            call
            for call in runner.calls
            if call.mode == "open-loop" and call.request_rate == pytest.approx(2.0)
        ]
        assert [call.duration_s for call in high_rate_calls] == [1.0, 3.0]
        trace = json.loads(
            (tmp_path / "search-uncertain-extended" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        high_rate_events = [
            event
            for event in trace["events"]
            if event["mode"] == "open-loop" and event["request_rate"] == pytest.approx(2.0)
        ]
        assert [event["purpose"] for event in high_rate_events] == [
            "open_loop_bracket_high",
            "open_loop_bracket_high_extend_uncertain",
        ]

    asyncio.run(run())


def test_open_loop_search_stops_early_on_high_confidence_scheduler_cap(tmp_path: Path) -> None:
    class SchedulerCapAtHighRunner(FakeRunner):
        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate is not None and config.request_rate > self.sustainable_rate:
                self.analyses[config.trial_id] = _analysis(
                    config.trial_id,
                    status="slo_violation",
                    bottleneck_class="scheduler_cap",
                    bottleneck_confidence="high",
                )
            return result

    async def run() -> None:
        runner = SchedulerCapAtHighRunner(sustainable_rate=1.0)
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-scheduler-cap",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-scheduler-cap",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.01,
                initial_request_rate=1.0,
            )
        )

        assert result.max_no_drift_request_rate == 1.0
        assert result.bottleneck_class == "scheduler_cap"
        assert result.confidence == "high"
        assert result.confirmation_trial_id is None
        assert result.termination_reason == "scheduler_config_limited"
        open_loop_calls = [call for call in runner.calls if call.mode == "open-loop"]
        assert [call.request_rate for call in open_loop_calls] == [1.0, 2.0]
        trace = json.loads(
            (tmp_path / "search-scheduler-cap" / "search_trace.json").read_text(encoding="utf-8")
        )
        assert trace["result"]["termination_reason"] == "scheduler_config_limited"
        assert trace["bounds"]["low_rate"] == 1.0
        assert trace["bounds"]["high_rate"] == 2.0

    asyncio.run(run())


def test_confirmation_conflict_uses_second_pass_majority(tmp_path: Path) -> None:
    class FlakyConfirmationRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=1.0)
            self.low_rate_calls = 0

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate == pytest.approx(1.0):
                self.low_rate_calls += 1
                if self.low_rate_calls == 2:
                    self.analyses[config.trial_id] = _analysis(
                        config.trial_id,
                        status="slo_violation",
                    )
                elif self.low_rate_calls == 3:
                    self.analyses[config.trial_id] = _analysis(config.trial_id, status="stable")
            return result

    async def run() -> None:
        runner = FlakyConfirmationRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-confirmation-majority",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-confirmation-majority",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.9,
                initial_request_rate=1.0,
            )
        )

        assert result.max_no_drift_request_rate == 1.0
        assert result.termination_reason == "confirmed_stable"
        trace = json.loads(
            (tmp_path / "search-confirmation-majority" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        confirmation_events = [
            event for event in trace["events"] if event["purpose"].startswith("open_loop_confirmation")
        ]
        assert [event["purpose"] for event in confirmation_events] == [
            "open_loop_confirmation",
            "open_loop_confirmation_majority",
        ]
        assert result.confirmation_trial_id == confirmation_events[-1]["trial_id"]
        assert trace["result"]["termination_reason"] == "confirmed_stable"

    asyncio.run(run())


def test_confirmation_majority_rejection_finishes_without_lower_stable_bound(tmp_path: Path) -> None:
    class RejectingConfirmationRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(sustainable_rate=1.0)
            self.low_rate_calls = 0

        async def run_trial(self, config: TrialConfig, *, request_source, output_dir: str | Path):
            result = await super().run_trial(config, request_source=request_source, output_dir=output_dir)
            if config.mode == "open-loop" and config.request_rate == pytest.approx(1.0):
                self.low_rate_calls += 1
                if self.low_rate_calls >= 2:
                    self.analyses[config.trial_id] = _analysis(
                        config.trial_id,
                        status="slo_violation",
                    )
            return result

    async def run() -> None:
        runner = RejectingConfirmationRunner()
        controller = SearchController(
            runner,
            request_source=_source(),
            output_dir=tmp_path / "search-confirmation-rejected",
            analyze_trial=lambda trial_dir: runner.analyses[Path(trial_dir).name],
            write_analysis=lambda trial_dir, result: Path(trial_dir) / "analysis.json",
        )
        result = await controller.search(
            SearchConfig(
                search_id="fixture-confirmation-rejected",
                search_mode="open-loop",
                model="fake-model",
                trial_duration_s=1.0,
                rate_precision=0.9,
                initial_request_rate=1.0,
            )
        )

        assert result.max_no_drift_request_rate is None
        assert result.max_slo_satisfying_request_rate is None
        assert result.termination_reason == "no_confirmed_stable_open_loop_rate"
        assert result.confidence == "low"
        trace = json.loads(
            (tmp_path / "search-confirmation-rejected" / "search_trace.json").read_text(
                encoding="utf-8"
            )
        )
        assert trace["result"]["termination_reason"] == "no_confirmed_stable_open_loop_rate"

    asyncio.run(run())
