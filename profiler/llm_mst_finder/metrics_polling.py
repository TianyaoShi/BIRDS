from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import aiohttp

from .records import ServerMetricSample
from .request_client import _build_connector

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+(?P<timestamp>-?\d+))?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

_CANONICAL_METRICS: dict[str, tuple[str, ...]] = {
    "num_running": ("vllm:num_requests_running", "vllm_num_requests_running"),
    "num_waiting": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
    "num_swapped": ("vllm:num_requests_swapped", "vllm_num_requests_swapped"),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm_kv_cache_usage_perc"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total", "vllm_prompt_tokens_total"),
    "generation_tokens_total": (
        "vllm:generation_tokens_total",
        "vllm_generation_tokens_total",
    ),
    "request_success_total": (
        "vllm:request_success_total",
        "vllm_request_success_total",
    ),
    "request_abort_total": (
        "vllm:request_abort_total",
        "vllm_request_abort_total",
        "vllm:request_aborted_total",
        "vllm_request_aborted_total",
        "vllm:request_failure_total",
        "vllm_request_failure_total",
        "vllm:request_failed_total",
        "vllm_request_failed_total",
    ),
}


@dataclass(frozen=True, slots=True)
class ParsedMetricSeries:
    name: str
    labels: dict[str, str]
    value: float
    timestamp_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "value": self.value,
            "timestamp_ms": self.timestamp_ms,
        }


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    if raw_labels is None:
        return {}
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw_labels):
        match = _LABEL_RE.match(raw_labels, position)
        if match is None:
            raise ValueError(f"invalid Prometheus labels fragment: {raw_labels!r}")
        key = match.group(1)
        raw_value = match.group(2)
        labels[key] = bytes(raw_value, "utf-8").decode("unicode_escape")
        position = match.end()
        if position == len(raw_labels):
            break
        if raw_labels[position] != ",":
            raise ValueError(f"invalid Prometheus labels fragment: {raw_labels!r}")
        position += 1
    return labels


def parse_prometheus_text(payload: str) -> dict[str, list[ParsedMetricSeries]]:
    metrics: dict[str, list[ParsedMetricSeries]] = defaultdict(list)
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            raise ValueError(f"invalid Prometheus metric line at {line_number}: {raw_line!r}")
        value = float(match.group("value"))
        if not isfinite(value):
            raise ValueError(
                f"Prometheus metric {match.group('name')!r} on line {line_number} must be finite"
            )
        timestamp_group = match.group("timestamp")
        timestamp_ms = int(timestamp_group) if timestamp_group is not None else None
        metric = ParsedMetricSeries(
            name=match.group("name"),
            labels=_parse_labels(match.group("labels")),
            value=value,
            timestamp_ms=timestamp_ms,
        )
        metrics[metric.name].append(metric)
    return dict(metrics)


def _series_to_raw_dict(parsed_metrics: dict[str, list[ParsedMetricSeries]]) -> dict[str, Any]:
    return {
        metric_name: [series.to_dict() for series in series_list]
        for metric_name, series_list in sorted(parsed_metrics.items())
    }


def _sum_metric(
    parsed_metrics: dict[str, list[ParsedMetricSeries]],
    metric_names: tuple[str, ...],
) -> float | None:
    matched_values: list[float] = []
    for metric_name in metric_names:
        matched_values.extend(series.value for series in parsed_metrics.get(metric_name, ()))
    if not matched_values:
        return None
    return sum(matched_values)


def _normalize_kv_cache_usage(value: float | None) -> float | None:
    if value is None:
        return None
    if value < 0 or value > 100:
        raise ValueError(f"kv_cache_usage_perc must be within [0, 100], got {value!r}")
    if value <= 1:
        return value
    return value / 100.0


def build_server_metric_sample(
    *,
    ts: float,
    parsed_metrics: dict[str, list[ParsedMetricSeries]],
) -> ServerMetricSample:
    return ServerMetricSample(
        ts=ts,
        raw=_series_to_raw_dict(parsed_metrics),
        num_running=_sum_metric(parsed_metrics, _CANONICAL_METRICS["num_running"]),
        num_waiting=_sum_metric(parsed_metrics, _CANONICAL_METRICS["num_waiting"]),
        num_swapped=_sum_metric(parsed_metrics, _CANONICAL_METRICS["num_swapped"]),
        kv_cache_usage=_normalize_kv_cache_usage(
            _sum_metric(parsed_metrics, _CANONICAL_METRICS["kv_cache_usage"])
        ),
        prompt_tokens_total=_sum_metric(parsed_metrics, _CANONICAL_METRICS["prompt_tokens_total"]),
        generation_tokens_total=_sum_metric(
            parsed_metrics,
            _CANONICAL_METRICS["generation_tokens_total"],
        ),
        request_success_total=_sum_metric(
            parsed_metrics,
            _CANONICAL_METRICS["request_success_total"],
        ),
        request_abort_total=_sum_metric(parsed_metrics, _CANONICAL_METRICS["request_abort_total"]),
    )


def parse_server_metrics_sample(*, ts: float, payload: str) -> ServerMetricSample:
    return build_server_metric_sample(ts=ts, parsed_metrics=parse_prometheus_text(payload))


class PrometheusMetricsPoller:
    def __init__(
        self,
        *,
        metrics_url: str,
        interval_s: float = 1.0,
        timeout_s: float = 10.0,
        session: aiohttp.ClientSession | None = None,
        time_fn=time.perf_counter,
    ) -> None:
        if not metrics_url:
            raise ValueError("metrics_url must be non-empty")
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        self._metrics_url = metrics_url
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._provided_session = session
        self._session = session
        self._time_fn = time_fn

    async def open(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=_build_connector())

    async def close(self) -> None:
        if self._provided_session is None and self._session is not None:
            await self._session.close()
            self._session = None

    async def _fetch_sample(self) -> ServerMetricSample:
        if self._session is None:
            await self.open()
        assert self._session is not None
        ts = self._time_fn()
        try:
            async with self._session.get(self._metrics_url) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        "Prometheus metrics poll failed: "
                        f"HTTP {response.status}: "
                        f"{body.strip() or response.reason or 'metrics poll failed'}"
                    )
                payload = await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Prometheus metrics poll failed: " f"{type(exc).__name__}: {exc}"
            ) from exc
        return parse_server_metrics_sample(ts=ts, payload=payload)

    async def run(
        self,
        *,
        output_path: Path,
        stop_event: asyncio.Event,
        trial_id: str,
    ) -> list[ServerMetricSample]:
        del trial_id
        await self.open()
        samples: list[ServerMetricSample] = []
        try:
            while True:
                sample = await self._fetch_sample()
                samples.append(sample)
                self._append_jsonl(output_path, sample.to_dict())
                if stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self._interval_s)
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.close()
        return samples

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
