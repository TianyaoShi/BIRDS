from __future__ import annotations

import asyncio
import json

import pytest

from llm_mst_finder.records import SampleRequest
from llm_mst_finder.request_client import RequestClient
from llm_mst_finder import request_client as request_client_module


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


class OwnedFakeSession(FakeSession):
    def __init__(self, response: FakeResponse) -> None:
        super().__init__(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeClientSessionFactory:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.instances: list[OwnedFakeSession] = []

    def __call__(self, **kwargs) -> OwnedFakeSession:
        del kwargs
        session = OwnedFakeSession(self._response)
        self.instances.append(session)
        return session


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


def test_request_client_parses_streaming_chat_reasoning_tokens() -> None:
    async def run() -> None:
        response = FakeResponse(
            status=200,
            chunks=[
                b'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}\n\n',
                b'data: {"choices": [{"delta": {"reasoning": "think"}}]}\n\n',
                b'data: {"choices": [{"delta": {"reasoning_content": "ing"}}]}\n\n',
                (
                    b'data: {"choices": [], '
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
                request_id="req-reasoning",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert record.success is True
        assert record.actual_output_len == 2
        assert len(record.output_token_timestamps) == 2
        assert record.ttft_s is not None

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


def test_request_client_returns_failed_record_for_stream_error() -> None:
    async def run() -> None:
        response = FakeResponse(
            status=200,
            chunks=[
                b'data: {"choices": [{"delta": {"reasoning": "one"}}]}\n\n',
                (
                    b'data: {"error": {"message": "Harmony parser failed", '
                    b'"type": "server_error", "code": "bad_stream"}}\n\n'
                ),
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
                request_id="req-stream-error",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert record.success is False
        assert record.error is not None
        assert "Harmony parser failed" in record.error
        assert "server_error" in record.error
        assert record.metadata["failure_class"] == "model_server_harmony_stream_error"
        assert len(record.output_token_timestamps) == 1

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


def test_request_client_accepts_usage_only_stream_chunk() -> None:
    async def run() -> None:
        response = FakeResponse(
            status=200,
            chunks=[
                b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
                (
                    b'data: {"choices": [], '
                    b'"usage": {"prompt_tokens": 5, "completion_tokens": 4}}\n\n'
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
                request_id="req-004",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert record.success is True
        assert record.actual_output_len == 4
        assert len(record.output_token_timestamps) == 1

    asyncio.run(run())


def test_request_client_closes_owned_session_on_parser_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        factory = FakeClientSessionFactory(
            FakeResponse(
                status=200,
                chunks=[b"data: {not-json}\n\n", b"data: [DONE]\n\n"],
            )
        )
        monkeypatch.setattr(request_client_module, "_build_connector", lambda: None)
        monkeypatch.setattr(request_client_module.aiohttp, "ClientSession", factory)
        client = RequestClient(
            base_url="http://unit-test",
            endpoint="/v1/chat/completions",
            model="fake-model",
        )
        with pytest.raises(json.JSONDecodeError):
            await client.send_request(
                SampleRequest(prompt="hello", prompt_len=5, expected_output_len=4),
                request_id="req-005",
                trial_id="trial-001",
                scheduled_send_ts=0.0,
            )
        assert len(factory.instances) == 1
        assert factory.instances[0].closed is True
        assert client._session is None

    asyncio.run(run())
