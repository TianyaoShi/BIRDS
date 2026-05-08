from __future__ import annotations

import asyncio
import hashlib
import itertools
import random
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Literal

from .records import RequestRecord, SampleRequest, ScheduledRequest

RequestSource = Callable[[], SampleRequest]
DispatchCallable = Callable[[ScheduledRequest], Awaitable[RequestRecord]]
RequestReusePolicy = Literal["cycle", "no-repeat-per-trial", "no-repeat-across-search"]


class RequestSourceExhausted(RuntimeError):
    pass


def request_reuse_key(sample: SampleRequest) -> str:
    metadata = sample.metadata or {}
    for field_name in ("content_hash", "sample_id", "original_sample_id"):
        value = metadata.get(field_name)
        if isinstance(value, str) and value:
            return f"{field_name}:{value}"
    source_index = metadata.get("source_index")
    if isinstance(source_index, int):
        return f"source_index:{source_index}"
    digest = hashlib.sha256(sample.prompt.encode("utf-8")).hexdigest()
    return f"prompt_sha256:{digest}"


def count_unique_request_reuse_keys(requests: Sequence[SampleRequest]) -> int:
    return len({request_reuse_key(sample) for sample in requests})


def cycling_request_source(
    requests: Sequence[SampleRequest],
    *,
    start_offset: int = 0,
) -> RequestSource:
    if not requests:
        raise ValueError("requests must be non-empty")
    iterator: Iterator[SampleRequest] = itertools.cycle(requests)
    for _ in range(start_offset % len(requests)):
        next(iterator)
    return lambda: next(iterator)


def unique_request_source(
    requests: Sequence[SampleRequest],
    *,
    start_offset: int = 0,
    used_keys: set[str] | None = None,
) -> RequestSource:
    if not requests:
        raise ValueError("requests must be non-empty")
    local_used_keys = set() if used_keys is None else used_keys
    index = start_offset % len(requests)

    def next_unique_request() -> SampleRequest:
        nonlocal index
        scanned = 0
        while scanned < len(requests):
            sample = requests[index % len(requests)]
            index += 1
            scanned += 1
            key = request_reuse_key(sample)
            if key in local_used_keys:
                continue
            local_used_keys.add(key)
            return sample
        raise RequestSourceExhausted("no unused request content remains")

    return next_unique_request


def request_source_factory_for_reuse_policy(
    requests: Sequence[SampleRequest],
    *,
    reuse_policy: RequestReusePolicy,
) -> Callable[[], RequestSource]:
    if not requests:
        raise ValueError("requests must be non-empty")
    if reuse_policy not in {"cycle", "no-repeat-per-trial", "no-repeat-across-search"}:
        raise ValueError(f"unsupported request reuse policy {reuse_policy!r}")

    trial_counter = itertools.count()
    start_offset_stride = _start_offset_stride(len(requests))
    shared_used_keys: set[str] = set()

    def make_request_source() -> RequestSource:
        trial_index = next(trial_counter)
        start_offset = trial_index * start_offset_stride
        if reuse_policy == "cycle":
            return cycling_request_source(requests, start_offset=start_offset)
        if reuse_policy == "no-repeat-per-trial":
            return unique_request_source(requests, start_offset=start_offset)
        return unique_request_source(
            requests,
            start_offset=start_offset,
            used_keys=shared_used_keys,
        )

    return make_request_source


def _start_offset_stride(request_count: int) -> int:
    if request_count <= 1:
        return 1
    for candidate in (997, 251, 67, 17):
        if request_count > candidate:
            return candidate
    return max(1, (request_count // 2) + 1)


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
        on_source_exhausted: Callable[[str], None] | None = None,
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
            try:
                sample = request_source()
            except RequestSourceExhausted as exc:
                if on_source_exhausted is not None:
                    on_source_exhausted(str(exc))
                break
            yield ScheduledRequest(
                request_index=request_index,
                scheduled_send_ts=scheduled_send_ts,
                sample=sample,
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
        on_source_exhausted: Callable[[str], None] | None = None,
    ) -> list[RequestRecord]:
        tasks: list[Awaitable[RequestRecord]] = []
        async for scheduled_request in self.iter_scheduled_requests(
            request_source,
            duration_s=duration_s,
            start_ts=start_ts,
            on_source_exhausted=on_source_exhausted,
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
        on_source_exhausted: Callable[[str], None] | None = None,
    ) -> list[RequestRecord]:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if start_ts is None:
            start_ts = self._time_fn()
        deadline_ts = start_ts + duration_s
        request_counter = itertools.count()
        records: list[RequestRecord] = []
        source_exhausted = False

        async def worker() -> None:
            nonlocal source_exhausted
            while self._time_fn() < deadline_ts:
                if should_abort is not None and should_abort():
                    return
                scheduled_send_ts = self._time_fn()
                if scheduled_send_ts >= deadline_ts:
                    return
                try:
                    sample = request_source()
                except RequestSourceExhausted as exc:
                    if on_source_exhausted is not None and not source_exhausted:
                        source_exhausted = True
                        on_source_exhausted(str(exc))
                    return
                scheduled_request = ScheduledRequest(
                    request_index=next(request_counter),
                    scheduled_send_ts=scheduled_send_ts,
                    sample=sample,
                )
                records.append(await dispatch(scheduled_request))
                if self._think_time_s > 0:
                    await self._sleep_fn(self._think_time_s)

        await asyncio.gather(*(worker() for _ in range(self._concurrency)))
        records.sort(key=lambda record: record.scheduled_send_ts)
        return records
