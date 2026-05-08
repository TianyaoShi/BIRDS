from __future__ import annotations

import asyncio
import time

from llm_mst_finder.loadgen import (
    ClosedLoopLoadGenerator,
    OpenLoopLoadGenerator,
    RequestSourceExhausted,
    count_unique_request_reuse_keys,
    cycling_request_source,
    request_source_factory_for_reuse_policy,
    unique_request_source,
)
from llm_mst_finder.records import RequestRecord, SampleRequest, ScheduledRequest


def _success_record(scheduled_request: ScheduledRequest, *, latency_s: float = 0.01) -> RequestRecord:
    actual_send_ts = scheduled_request.scheduled_send_ts
    first_token_ts = actual_send_ts + 0.002
    end_ts = actual_send_ts + latency_s
    return RequestRecord(
        request_id=f"req-{scheduled_request.request_index}",
        trial_id="trial-loadgen",
        scheduled_send_ts=scheduled_request.scheduled_send_ts,
        actual_send_ts=actual_send_ts,
        first_token_ts=first_token_ts,
        end_ts=end_ts,
        success=True,
        error=None,
        prompt_len=scheduled_request.sample.prompt_len,
        expected_output_len=scheduled_request.sample.expected_output_len,
        actual_output_len=2,
        ttft_s=first_token_ts - actual_send_ts,
        e2e_s=end_ts - actual_send_ts,
        tpot_s=(end_ts - first_token_ts),
        itl_s=[end_ts - first_token_ts],
        output_token_timestamps=[first_token_ts, end_ts],
        metadata=scheduled_request.sample.metadata,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, duration_s: float) -> None:
        self.now += duration_s


def test_open_loop_deterministic_schedule() -> None:
    async def run() -> list[float]:
        clock = FakeClock()
        generator = OpenLoopLoadGenerator(
            request_rate=2.0,
            deterministic=True,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=3)])
        scheduled = []
        async for request in generator.iter_scheduled_requests(source, duration_s=1.6, start_ts=0.0):
            scheduled.append(request.scheduled_send_ts)
        return scheduled

    scheduled = asyncio.run(run())
    assert scheduled == [0.0, 0.5, 1.0, 1.5]


def test_unique_request_source_uses_content_hash_and_exhausts() -> None:
    samples = [
        SampleRequest(
            prompt=f"prompt-{index}",
            prompt_len=5,
            expected_output_len=3,
            metadata={"content_hash": content_hash},
        )
        for index, content_hash in enumerate(("a", "b", "a"))
    ]
    source = unique_request_source(samples)

    assert source().prompt == "prompt-0"
    assert source().prompt == "prompt-1"
    try:
        source()
    except RequestSourceExhausted as exc:
        assert "no unused request content" in str(exc)
    else:
        raise AssertionError("unique source should exhaust after unique content hashes")
    assert count_unique_request_reuse_keys(samples) == 2


def test_request_source_factory_can_avoid_repeats_across_search() -> None:
    samples = [
        SampleRequest(
            prompt=f"prompt-{index}",
            prompt_len=5,
            expected_output_len=3,
            metadata={"content_hash": str(index)},
        )
        for index in range(4)
    ]
    factory = request_source_factory_for_reuse_policy(
        samples,
        reuse_policy="no-repeat-across-search",
    )

    first_trial = factory()
    second_trial = factory()
    assert [first_trial().prompt for _ in range(2)] == ["prompt-0", "prompt-1"]
    assert [second_trial().prompt for _ in range(2)] == ["prompt-3", "prompt-2"]
    try:
        factory()()
    except RequestSourceExhausted:
        pass
    else:
        raise AssertionError("cross-search unique source should exhaust")


def test_cycling_factory_rotates_trial_start_offset() -> None:
    samples = [
        SampleRequest(prompt=f"prompt-{index}", prompt_len=5, expected_output_len=3)
        for index in range(5)
    ]
    factory = request_source_factory_for_reuse_policy(samples, reuse_policy="cycle")

    assert factory()().prompt == "prompt-0"
    assert factory()().prompt == "prompt-3"


def test_open_loop_abort_stops_scheduling() -> None:
    async def run() -> int:
        clock = FakeClock()
        generator = OpenLoopLoadGenerator(
            request_rate=10.0,
            deterministic=True,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=3)])
        should_abort_checks = 0

        def should_abort() -> bool:
            nonlocal should_abort_checks
            if should_abort_checks >= 2:
                return True
            should_abort_checks += 1
            return False

        async def dispatch(request: ScheduledRequest) -> RequestRecord:
            return _success_record(request)

        records = await generator.run(
            source,
            dispatch,
            duration_s=1.0,
            start_ts=0.0,
            should_abort=should_abort,
        )
        assert len(records) == 2
        return should_abort_checks

    assert asyncio.run(run()) == 2


def test_closed_loop_holds_requested_concurrency() -> None:
    async def run() -> tuple[int, int]:
        generator = ClosedLoopLoadGenerator(concurrency=3)
        source = cycling_request_source([SampleRequest(prompt="hello", prompt_len=5, expected_output_len=3)])
        active = 0
        max_active = 0

        async def dispatch(request: ScheduledRequest) -> RequestRecord:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return _success_record(request)

        records = await generator.run(source, dispatch, duration_s=0.05, start_ts=time.perf_counter())
        return len(records), max_active

    record_count, max_active = asyncio.run(run())
    assert record_count >= 3
    assert max_active == 3
