from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence

from .records import RequestRecord, SampleRequest, ScheduledRequest

RequestSource = Callable[[], SampleRequest]
DispatchCallable = Callable[[ScheduledRequest], Awaitable[RequestRecord]]


def cycling_request_source(requests: Sequence[SampleRequest]) -> RequestSource:
    if not requests:
        raise ValueError("requests must be non-empty")
    iterator: Iterator[SampleRequest] = itertools.cycle(requests)
    return lambda: next(iterator)


class OpenLoopLoadGenerator:
    def __init__(
        self,
        *,
        request_rate: float,
        burstiness: float = 1.0,
        deterministic: bool = False,
        seed: int | None = None,
        time_fn=time.perf_counter,
        sleep_fn=asyncio.sleep,
    ) -> None:
        if request_rate <= 0:
            raise ValueError("request_rate must be positive")
        if burstiness <= 0:
            raise ValueError("burstiness must be positive")
        self._request_rate = request_rate
        self._burstiness = burstiness
        self._deterministic = deterministic
        self._rng = random.Random(seed)
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    def _sample_interval_s(self) -> float:
        if self._deterministic:
            return 1.0 / self._request_rate
        if self._burstiness == 1.0:
            return self._rng.expovariate(self._request_rate)
        shape = 1.0 / (self._burstiness * self._burstiness)
        scale = (self._burstiness * self._burstiness) / self._request_rate
        return self._rng.gammavariate(shape, scale)

    async def iter_scheduled_requests(
        self,
        request_source: RequestSource,
        *,
        duration_s: float,
        start_ts: float | None = None,
    ):
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if start_ts is None:
            start_ts = self._time_fn()
        deadline_ts = start_ts + duration_s
        scheduled_send_ts = start_ts
        request_index = 0
        while scheduled_send_ts < deadline_ts:
            if request_index > 0:
                scheduled_send_ts += self._sample_interval_s()
                if scheduled_send_ts >= deadline_ts:
                    break
                sleep_s = scheduled_send_ts - self._time_fn()
                if sleep_s > 0:
                    await self._sleep_fn(sleep_s)
            yield ScheduledRequest(
                request_index=request_index,
                scheduled_send_ts=scheduled_send_ts,
                sample=request_source(),
            )
            request_index += 1

    async def run(
        self,
        request_source: RequestSource,
        dispatch: DispatchCallable,
        *,
        duration_s: float,
        start_ts: float | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[RequestRecord]:
        tasks: list[Awaitable[RequestRecord]] = []
        async for scheduled_request in self.iter_scheduled_requests(
            request_source,
            duration_s=duration_s,
            start_ts=start_ts,
        ):
            if should_abort is not None and should_abort():
                break
            tasks.append(asyncio.ensure_future(dispatch(scheduled_request)))
            await self._sleep_fn(0)
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))


class ClosedLoopLoadGenerator:
    def __init__(
        self,
        *,
        concurrency: int,
        think_time_s: float = 0.0,
        time_fn=time.perf_counter,
        sleep_fn=asyncio.sleep,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if think_time_s < 0:
            raise ValueError("think_time_s must be non-negative")
        self._concurrency = concurrency
        self._think_time_s = think_time_s
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    async def run(
        self,
        request_source: RequestSource,
        dispatch: DispatchCallable,
        *,
        duration_s: float,
        start_ts: float | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[RequestRecord]:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if start_ts is None:
            start_ts = self._time_fn()
        deadline_ts = start_ts + duration_s
        request_counter = itertools.count()
        records: list[RequestRecord] = []

        async def worker() -> None:
            while self._time_fn() < deadline_ts:
                if should_abort is not None and should_abort():
                    return
                scheduled_send_ts = self._time_fn()
                if scheduled_send_ts >= deadline_ts:
                    return
                scheduled_request = ScheduledRequest(
                    request_index=next(request_counter),
                    scheduled_send_ts=scheduled_send_ts,
                    sample=request_source(),
                )
                records.append(await dispatch(scheduled_request))
                if self._think_time_s > 0:
                    await self._sleep_fn(self._think_time_s)

        await asyncio.gather(*(worker() for _ in range(self._concurrency)))
        records.sort(key=lambda record: record.scheduled_send_ts)
        return records
