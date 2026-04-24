from __future__ import annotations

import asyncio
import json

from llm_mst_finder.records import SampleRequest
from llm_mst_finder.request_client import RequestClient


class FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, *, status: int, chunks: list[bytes] | None = None, text: str = "") -> None:
        self.status = status
        self.reason = text
        self.content = FakeStream(chunks or [])
        self._text = text

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def text(self) -> str:
        return self._text


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.requests: list[dict] = []

    def post(self, **kwargs) -> FakeResponse:
        self.requests.append(kwargs)
        return self._response


def test_request_client_parses_streaming_chat_response() -> None:
    async def run() -> None:
        response = FakeResponse(
            status=200,
            chunks=[
                b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
                (
                    b'data: {"choices": [{"delta": {"content": "lo"}}], '
                    b'"usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n'
                ),
                b"data: [DONE]\n\n",
            ],
        )
        session = FakeSession(response)
        async with RequestClient(
            base_url="http://unit-test",
            endpoint="/v1/chat/completions",
            model="fake-model",
            session=session,
        ) as client:
            record = await client.send_request(
                SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4),
                request_id="req-001",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert session.requests[0]["json"]["stream"] is True
        assert record.success is True
        assert record.actual_output_len == 2
        assert len(record.output_token_timestamps) == 2
        assert record.ttft_s is not None
        assert record.tpot_s is not None

    asyncio.run(run())


def test_request_client_returns_failed_record_for_http_error() -> None:
    async def run() -> None:
        session = FakeSession(FakeResponse(status=503, text="busy"))
        async with RequestClient(
            base_url="http://unit-test",
            endpoint="/v1/completions",
            model="fake-model",
            session=session,
        ) as client:
            record = await client.send_request(
                SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4),
                request_id="req-002",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert record.success is False
        assert record.error is not None
        assert "HTTP 503" in record.error

    asyncio.run(run())


def test_request_client_raises_on_malformed_stream_payload() -> None:
    async def run() -> None:
        session = FakeSession(
            FakeResponse(
                status=200,
                chunks=[b"data: {not-json}\n\n", b"data: [DONE]\n\n"],
            )
        )
        async with RequestClient(
            base_url="http://unit-test",
            endpoint="/v1/chat/completions",
            model="fake-model",
            session=session,
        ) as client:
            try:
                await client.send_request(
                    SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4),
                    request_id="req-003",
                    trial_id="trial-001",
                    scheduled_send_ts=0.0,
                )
            except json.JSONDecodeError:
                return
            raise AssertionError("expected malformed stream payload to raise JSONDecodeError")

    asyncio.run(run())
