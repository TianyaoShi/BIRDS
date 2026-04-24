from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from llm_mst_finder.loadgen import cycling_request_source
from llm_mst_finder.records import RequestRecord, SampleRequest, ScheduledRequest, ServerMetricSample, TrialConfig
from llm_mst_finder.trial_runner import TrialRunner


class StubRequestClient:
    def __init__(self, *, latency_s: float = 0.01) -> None:
        self._latency_s = latency_s

    async def open(self) -> None:
        return None

    async def send_request(
        self,
        sample_request: SampleRequest,
        *,
        request_id: str,
        trial_id: str,
        scheduled_send_ts: float,
    ) -> RequestRecord:
        actual_send_ts = time.perf_counter()
        await asyncio.sleep(self._latency_s)
        first_token_ts = actual_send_ts + min(self._latency_s / 2.0, 0.005)
        end_ts = actual_send_ts + self._latency_s
        return RequestRecord(
            request_id=request_id,
            trial_id=trial_id,
            scheduled_send_ts=scheduled_send_ts,
            actual_send_ts=actual_send_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            success=True,
            error=None,
            prompt_len=sample_request.prompt_len,
            expected_output_len=sample_request.expected_output_len,
            actual_output_len=2,
            ttft_s=first_token_ts - actual_send_ts,
            e2e_s=end_ts - actual_send_ts,
            tpot_s=end_ts - first_token_ts,
            itl_s=[end_ts - first_token_ts],
            output_token_timestamps=[first_token_ts, end_ts],
            metadata=sample_request.metadata,
        )


class StubMetricsPoller:
    async def run(self, *, output_path: Path, stop_event: asyncio.Event, trial_id: str):
        sample = ServerMetricSample(
            ts=time.perf_counter(),
            raw={"trial_id": trial_id},
            num_running=1.0,
            num_waiting=0.0,
            num_swapped=0.0,
            kv_cache_usage=0.25,
            prompt_tokens_total=10.0,
            generation_tokens_total=5.0,
            request_success_total=1.0,
            request_abort_total=0.0,
        )
        output_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
        await stop_event.wait()
        return [sample]


def test_trial_runner_writes_open_loop_artifacts_and_marks_safety_abort(tmp_path: Path) -> None:
    async def run() -> None:
        runner = TrialRunner(
            StubRequestClient(latency_s=0.03),
            metrics_poller=StubMetricsPoller(),
        )
        config = TrialConfig(
            trial_id="trial-open",
            mode="open-loop",
            duration_s=0.05,
            request_rate=100.0,
            base_url="http://127.0.0.1:8000",
            endpoint="/v1/chat/completions",
            model="fake-model",
            safety_max_outstanding=1,
        )
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4)])
        result = await runner.run_trial(config, request_source=source, output_dir=tmp_path / "trial-open")
        assert result.summary.status == "aborted_safety"
        assert result.summary.abort_reason is not None
        assert result.summary.metrics_sample_count == 1
        assert result.artifacts.request_records_path.exists()
        assert result.artifacts.summary_path.exists()
        assert result.artifacts.server_metrics_path is not None
        assert result.artifacts.server_metrics_path.exists()

    asyncio.run(run())


def test_trial_runner_runs_closed_loop_trial(tmp_path: Path) -> None:
    async def run() -> None:
        runner = TrialRunner(StubRequestClient(latency_s=0.01))
        config = TrialConfig(
            trial_id="trial-closed",
            mode="closed-loop",
            duration_s=0.04,
            concurrency=2,
            base_url="http://127.0.0.1:8000",
            endpoint="/v1/completions",
            model="fake-model",
        )
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4)])
        result = await runner.run_trial(config, request_source=source, output_dir=tmp_path / "trial-closed")
        assert result.summary.status == "completed"
        assert result.summary.started_requests >= 2
        assert result.summary.max_observed_outstanding == 2
        assert result.artifacts.request_records_path.exists()
        summary_payload = json.loads(result.artifacts.summary_path.read_text(encoding="utf-8"))
        assert summary_payload["config"]["trial_id"] == "trial-closed"

    asyncio.run(run())


def test_trial_runner_open_loop_send_rate_uses_send_timestamps_not_completion_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenLoopLoadGenerator:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def run(
            self,
            request_source,
            dispatch,
            *,
            duration_s: float,
            start_ts: float | None = None,
            should_abort=None,
        ):
            del duration_s, should_abort
            assert start_ts is not None
            first = ScheduledRequest(
                request_index=0,
                scheduled_send_ts=start_ts,
                sample=request_source(),
            )
            second = ScheduledRequest(
                request_index=1,
                scheduled_send_ts=start_ts + 1.0,
                sample=request_source(),
            )
            return [await dispatch(first), await dispatch(second)]

    class TailHeavyRequestClient:
        async def open(self) -> None:
            return None

        async def send_request(
            self,
            sample_request: SampleRequest,
            *,
            request_id: str,
            trial_id: str,
            scheduled_send_ts: float,
        ) -> RequestRecord:
            actual_send_ts = scheduled_send_ts
            first_token_ts = actual_send_ts + 0.2
            end_ts = actual_send_ts + 5.0
            return RequestRecord(
                request_id=request_id,
                trial_id=trial_id,
                scheduled_send_ts=scheduled_send_ts,
                actual_send_ts=actual_send_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                success=True,
                error=None,
                prompt_len=sample_request.prompt_len,
                expected_output_len=sample_request.expected_output_len,
                actual_output_len=2,
                ttft_s=first_token_ts - actual_send_ts,
                e2e_s=end_ts - actual_send_ts,
                tpot_s=end_ts - first_token_ts,
                itl_s=[end_ts - first_token_ts],
                output_token_timestamps=[first_token_ts, end_ts],
                metadata=sample_request.metadata,
            )

    monkeypatch.setattr("llm_mst_finder.trial_runner.OpenLoopLoadGenerator", FakeOpenLoopLoadGenerator)

    async def run() -> None:
        runner = TrialRunner(TailHeavyRequestClient(), time_fn=lambda: 100.0)
        config = TrialConfig(
            trial_id="trial-rate",
            mode="open-loop",
            duration_s=2.0,
            request_rate=1.0,
            base_url="http://127.0.0.1:8000",
            endpoint="/v1/completions",
            model="fake-model",
        )
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4)])
        result = await runner.run_trial(config, request_source=source, output_dir=tmp_path / "trial-rate")
        assert result.summary.wall_time_s == pytest.approx(6.0)
        assert result.summary.actual_send_rate == pytest.approx(1.0)

    asyncio.run(run())
