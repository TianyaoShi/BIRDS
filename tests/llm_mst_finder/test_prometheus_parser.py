from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_mst_finder.metrics_polling import (
    PrometheusMetricsPoller,
    parse_prometheus_text,
    parse_server_metrics_sample,
)


def test_parse_server_metrics_sample_normalizes_known_vllm_metrics() -> None:
    payload = """
# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m1"} 2
vllm:num_requests_running{model_name="m2"} 3
vllm:num_requests_waiting 4
vllm:num_requests_swapped 1
vllm:kv_cache_usage_perc 25
vllm:prompt_tokens_total 100
vllm:generation_tokens_total 50
vllm:request_success_total 7
vllm:request_aborted_total 2
vllm:time_to_first_token_seconds_bucket{le="0.1"} 1
"""
    sample = parse_server_metrics_sample(ts=123.0, payload=payload)

    assert sample.ts == 123.0
    assert sample.num_running == 5.0
    assert sample.num_waiting == 4.0
    assert sample.num_swapped == 1.0
    assert sample.kv_cache_usage == 0.25
    assert sample.prompt_tokens_total == 100.0
    assert sample.generation_tokens_total == 50.0
    assert sample.request_success_total == 7.0
    assert sample.request_abort_total == 2.0
    assert "vllm:time_to_first_token_seconds_bucket" in sample.raw


def test_parse_prometheus_text_rejects_invalid_metric_lines() -> None:
    try:
        parse_prometheus_text("not a valid metric line")
    except ValueError as exc:
        assert "invalid Prometheus metric line" in str(exc)
    else:
        raise AssertionError("expected ValueError for malformed Prometheus text")


def test_prometheus_metrics_poller_writes_samples(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        class FakeResponse:
            def __init__(self, *, status: int, text: str, reason: str = "") -> None:
                self.status = status
                self._text = text
                self.reason = reason

            async def text(self) -> str:
                return self._text

            async def __aenter__(self) -> FakeResponse:
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self._responses = [
                    FakeResponse(
                        status=200,
                        text=(
                            "vllm:num_requests_running 2\n"
                            "vllm:num_requests_waiting 1\n"
                            "vllm:kv_cache_usage_perc 50\n"
                            "vllm:request_success_total 9\n"
                        ),
                    ),
                ]
                self._steady_state = FakeResponse(
                    status=200,
                    text=(
                        "vllm:num_requests_running 2\n"
                        "vllm:num_requests_waiting 1\n"
                        "vllm:kv_cache_usage_perc 50\n"
                        "vllm:request_success_total 9\n"
                    ),
                )

            def get(self, _: str) -> FakeResponse:
                if self._responses:
                    return self._responses.pop(0)
                return self._steady_state

        output_path = tmp_path / "server_metrics.jsonl"
        stop_event = asyncio.Event()
        poller = PrometheusMetricsPoller(
            metrics_url="http://127.0.0.1:8000/metrics",
            interval_s=0.01,
            session=FakeSession(),
        )

        async def stop_soon() -> None:
            await asyncio.sleep(0.035)
            stop_event.set()

        stop_task = asyncio.create_task(stop_soon())
        try:
            samples = await poller.run(
                output_path=output_path,
                stop_event=stop_event,
                trial_id="trial-prometheus",
            )
        finally:
            await stop_task

        assert len(samples) >= 1
        assert samples[-1].num_running == 2.0
        assert samples[-1].num_waiting == 1.0
        assert samples[-1].kv_cache_usage == 0.5
        assert samples[-1].request_success_total == 9.0

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(samples)
        decoded = [json.loads(line) for line in lines]
        assert decoded[-1]["num_running"] == 2.0

    asyncio.run(run())


def test_prometheus_metrics_poller_raises_http_failures(tmp_path: Path) -> None:
    async def run() -> None:
        class FakeResponse:
            status = 500
            reason = "Internal Server Error"

            async def text(self) -> str:
                return "server exploded"

            async def __aenter__(self) -> FakeResponse:
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

        class FakeSession:
            def get(self, _: str) -> FakeResponse:
                return FakeResponse()

        output_path = tmp_path / "server_metrics.jsonl"
        stop_event = asyncio.Event()
        poller = PrometheusMetricsPoller(
            metrics_url="http://127.0.0.1:8000/metrics",
            interval_s=0.01,
            session=FakeSession(),
        )

        with pytest.raises(RuntimeError, match="Prometheus metrics poll failed: HTTP 500"):
            await poller.run(
                output_path=output_path,
                stop_event=stop_event,
                trial_id="trial-prometheus",
            )
        assert not output_path.exists()

    asyncio.run(run())
