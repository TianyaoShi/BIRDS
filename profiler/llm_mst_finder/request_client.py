from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

import aiohttp

from .records import RequestRecord, SampleRequest
from .vllm_compat import (
    build_openai_payload,
    decode_sse_line,
    detect_endpoint_kind,
    extract_text_from_chunk,
    extract_usage_from_chunk,
    parse_json_payload,
)


def _build_connector() -> aiohttp.TCPConnector:
    connector_kwargs = {
        "limit": 2000,
        "limit_per_host": 1000,
        "ttl_dns_cache": 300,
        "use_dns_cache": True,
        "keepalive_timeout": 60,
        "force_close": False,
        "ssl": False,
    }
    if sys.version_info < (3, 12, 13):
        connector_kwargs["enable_cleanup_closed"] = True
    return aiohttp.TCPConnector(
        **connector_kwargs,
    )


class RequestClient:
    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        model: str,
        timeout_s: float = 6 * 60 * 60,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        session: aiohttp.ClientSession | None = None,
        time_fn=time.perf_counter,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if not endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        if not model:
            raise ValueError("model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._model = model
        self._timeout_s = timeout_s
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._extra_headers = dict(extra_headers or {})
        self._extra_body = dict(extra_body or {})
        self._provided_session = session
        self._session = session
        self._time_fn = time_fn
        detect_endpoint_kind(endpoint)

    @property
    def endpoint_url(self) -> str:
        return f"{self._base_url}{self._endpoint}"

    async def __aenter__(self) -> RequestClient:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=_build_connector())

    async def close(self) -> None:
        if self._provided_session is None and self._session is not None:
            await self._session.close()
            self._session = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._extra_headers)
        return headers

    async def send_request(
        self,
        sample_request: SampleRequest,
        *,
        request_id: str,
        trial_id: str,
        scheduled_send_ts: float,
    ) -> RequestRecord:
        if self._session is None:
            await self.open()
        assert self._session is not None
        payload = build_openai_payload(
            self._endpoint,
            self._model,
            sample_request,
            trial_extra_body=self._extra_body,
        )
        actual_send_ts = self._time_fn()
        first_token_ts: float | None = None
        end_ts: float | None = None
        ttft_s: float | None = None
        e2e_s: float | None = None
        tpot_s: float | None = None
        itl_s: list[float] = []
        output_token_timestamps: list[float] = []
        completion_tokens: int | None = None
        most_recent_token_ts: float | None = None

        try:
            async with self._session.post(
                url=self.endpoint_url,
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    end_ts = self._time_fn()
                    error = f"HTTP {response.status}: {body.strip() or response.reason or 'request failed'}"
                    return RequestRecord(
                        request_id=request_id,
                        trial_id=trial_id,
                        scheduled_send_ts=scheduled_send_ts,
                        actual_send_ts=actual_send_ts,
                        first_token_ts=None,
                        end_ts=end_ts,
                        success=False,
                        error=error,
                        prompt_len=sample_request.prompt_len,
                        expected_output_len=sample_request.expected_output_len,
                        actual_output_len=None,
                        ttft_s=None,
                        e2e_s=end_ts - actual_send_ts,
                        tpot_s=None,
                        itl_s=[],
                        output_token_timestamps=[],
                        metadata=sample_request.metadata,
                    )

                async for raw_chunk in response.content:
                    payload_text = decode_sse_line(raw_chunk)
                    if payload_text is None:
                        continue
                    if payload_text == "[DONE]":
                        end_ts = self._time_fn()
                        break

                    parsed_chunk = parse_json_payload(payload_text)
                    _, chunk_completion_tokens = extract_usage_from_chunk(parsed_chunk)
                    if chunk_completion_tokens is not None:
                        completion_tokens = chunk_completion_tokens

                    text = extract_text_from_chunk(self._endpoint, parsed_chunk)
                    if text is None:
                        continue
                    now = self._time_fn()
                    if first_token_ts is None:
                        first_token_ts = now
                        ttft_s = first_token_ts - actual_send_ts
                    elif most_recent_token_ts is not None:
                        itl_s.append(now - most_recent_token_ts)
                    most_recent_token_ts = now
                    output_token_timestamps.append(now)

                if end_ts is None:
                    end_ts = self._time_fn()
                e2e_s = end_ts - actual_send_ts
                if completion_tokens is None:
                    completion_tokens = len(output_token_timestamps)
                if completion_tokens > 1 and first_token_ts is not None:
                    tpot_s = (end_ts - first_token_ts) / (completion_tokens - 1)
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
                    actual_output_len=completion_tokens,
                    ttft_s=ttft_s,
                    e2e_s=e2e_s,
                    tpot_s=tpot_s,
                    itl_s=itl_s,
                    output_token_timestamps=output_token_timestamps,
                    metadata=sample_request.metadata,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            end_ts = self._time_fn()
            return RequestRecord(
                request_id=request_id,
                trial_id=trial_id,
                scheduled_send_ts=scheduled_send_ts,
                actual_send_ts=actual_send_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                prompt_len=sample_request.prompt_len,
                expected_output_len=sample_request.expected_output_len,
                actual_output_len=completion_tokens,
                ttft_s=ttft_s,
                e2e_s=end_ts - actual_send_ts,
                tpot_s=tpot_s,
                itl_s=itl_s,
                output_token_timestamps=output_token_timestamps,
                metadata=sample_request.metadata,
            )
        except Exception:
            if self._provided_session is None:
                await self.close()
            raise
