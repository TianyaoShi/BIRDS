from __future__ import annotations

import json
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence

from .records import BenchmarkMetrics, RequestRecord, SampleRequest


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def detect_endpoint_kind(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return "chat"
    if normalized.endswith("/completions"):
        return "completions"
    raise ValueError(
        "endpoint must end with '/completions' or '/chat/completions', "
        f"got {endpoint!r}"
    )


def build_openai_payload(
    endpoint: str,
    model: str,
    sample_request: SampleRequest,
    *,
    trial_extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    endpoint_kind = detect_endpoint_kind(endpoint)
    payload: dict[str, Any]
    if endpoint_kind == "chat":
        system_prompt = sample_request.metadata.get("system_prompt")
        messages: list[dict[str, str]] = []
        if isinstance(system_prompt, str) and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": sample_request.prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": sample_request.expected_output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        payload = {
            "model": model,
            "prompt": sample_request.prompt,
            "temperature": 0.0,
            "max_tokens": sample_request.expected_output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    merged_extra_body: dict[str, Any] = {}
    if trial_extra_body:
        merged_extra_body.update(trial_extra_body)
    if sample_request.extra_body:
        merged_extra_body.update(sample_request.extra_body)
    if merged_extra_body:
        payload.update(merged_extra_body)
    return payload


def extract_text_from_chunk(endpoint: str, chunk: dict[str, Any]) -> str | None:
    endpoint_kind = detect_endpoint_kind(endpoint)
    if endpoint_kind == "chat":
        choices = chunk["choices"]
        if not isinstance(choices, list):
            raise TypeError("chat completion chunk choices must be a list")
        if not choices:
            if chunk.get("usage") is not None:
                return None
            raise KeyError("chat completion chunk missing choices[0]")
        delta = choices[0]["delta"]
        content = _first_text_field(delta, ("content", "reasoning_content", "reasoning"))
        if content is None or content == "":
            return None
        return content

    choices = chunk["choices"]
    if not isinstance(choices, list):
        raise TypeError("completion chunk choices must be a list")
    if not choices:
        if chunk.get("usage") is not None:
            return None
        raise KeyError("completion chunk missing choices[0]")
    text = choices[0]["text"]
    if not isinstance(text, str):
        raise TypeError("completion chunk text must be a string")
    return text or None


def _first_text_field(payload: dict[str, Any], field_names: Sequence[str]) -> str | None:
    for field_name in field_names:
        value = payload.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"chat completion delta {field_name} must be a string")
        return value
    return None


def extract_error_from_chunk(chunk: dict[str, Any]) -> str | None:
    error = chunk.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        message = error.get("message")
        error_type = error.get("type")
        code = error.get("code")
        parts = []
        if isinstance(message, str) and message:
            parts.append(message)
        if isinstance(error_type, str) and error_type:
            parts.append(f"type={error_type}")
        if isinstance(code, str) and code:
            parts.append(f"code={code}")
        if parts:
            return "; ".join(parts)
        return json.dumps(error, sort_keys=True)
    return str(error)


def extract_usage_from_chunk(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = chunk.get("usage")
    if usage is None:
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is not None and not isinstance(prompt_tokens, int):
        raise TypeError("usage.prompt_tokens must be an int when present")
    if completion_tokens is not None and not isinstance(completion_tokens, int):
        raise TypeError("usage.completion_tokens must be an int when present")
    return prompt_tokens, completion_tokens


def decode_sse_line(raw_chunk: bytes) -> str | None:
    chunk = raw_chunk.strip()
    if not chunk:
        return None
    text = chunk.decode("utf-8")
    if text.startswith(":"):
        return None
    text = remove_prefix(text, "data: ")
    text = remove_prefix(text, "data:")
    if not text.strip():
        return None
    return text


def parse_json_payload(payload: str) -> dict[str, Any]:
    return json.loads(payload)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires a non-empty sequence")
    if percentile < 0 or percentile > 100:
        raise ValueError(f"percentile must be within [0, 100], got {percentile}")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _optional_stats(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    if len(values) == 1:
        return values[0], values[0], 0.0
    return mean(values), median(values), pstdev(values)


def _length_summary(lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {}
    float_lengths = [float(value) for value in lengths]
    mean_value, median_value, _ = _optional_stats(float_lengths)
    return {
        "mean": mean_value if mean_value is not None else 0.0,
        "median": median_value if median_value is not None else 0.0,
        "p90": _percentile(float_lengths, 90.0),
        "p95": _percentile(float_lengths, 95.0),
        "p99": _percentile(float_lengths, 99.0),
    }


def _percentile_pairs(values: Sequence[float], percentiles: Iterable[float]) -> list[tuple[float, float]]:
    if not values:
        return []
    return [(float(percentile), _percentile(values, float(percentile))) for percentile in percentiles]


def calculate_benchmark_metrics(
    request_records: Sequence[RequestRecord],
    duration_s: float,
    *,
    percentiles: Sequence[float] = (50.0, 90.0, 95.0, 99.0),
) -> BenchmarkMetrics:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s!r}")
    successes = [record for record in request_records if record.success]
    failures = [record for record in request_records if not record.success]
    ttfts_ms = [record.ttft_s * 1000.0 for record in successes if record.ttft_s is not None]
    tpots_ms = [record.tpot_s * 1000.0 for record in successes if record.tpot_s is not None]
    itls_ms = [latency * 1000.0 for record in successes for latency in record.itl_s]
    e2es_ms = [record.e2e_s * 1000.0 for record in successes if record.e2e_s is not None]
    prompt_lengths = [record.prompt_len for record in request_records]
    output_lengths = [
        record.actual_output_len
        for record in successes
        if record.actual_output_len is not None
    ]
    mean_ttft_ms, median_ttft_ms, std_ttft_ms = _optional_stats(ttfts_ms)
    mean_tpot_ms, median_tpot_ms, std_tpot_ms = _optional_stats(tpots_ms)
    mean_itl_ms, median_itl_ms, std_itl_ms = _optional_stats(itls_ms)
    mean_e2e_ms, median_e2e_ms, std_e2e_ms = _optional_stats(e2es_ms)
    total_input_tokens = sum(record.prompt_len for record in request_records)
    total_output_tokens = sum(output_lengths)
    return BenchmarkMetrics(
        successful_requests=len(successes),
        failed_requests=len(failures),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        request_throughput=len(request_records) / duration_s,
        successful_request_throughput=len(successes) / duration_s,
        prompt_token_throughput=total_input_tokens / duration_s,
        generation_token_throughput=total_output_tokens / duration_s,
        total_token_throughput=(total_input_tokens + total_output_tokens) / duration_s,
        mean_ttft_ms=mean_ttft_ms,
        median_ttft_ms=median_ttft_ms,
        std_ttft_ms=std_ttft_ms,
        percentiles_ttft_ms=_percentile_pairs(ttfts_ms, percentiles),
        mean_tpot_ms=mean_tpot_ms,
        median_tpot_ms=median_tpot_ms,
        std_tpot_ms=std_tpot_ms,
        percentiles_tpot_ms=_percentile_pairs(tpots_ms, percentiles),
        mean_itl_ms=mean_itl_ms,
        median_itl_ms=median_itl_ms,
        std_itl_ms=std_itl_ms,
        percentiles_itl_ms=_percentile_pairs(itls_ms, percentiles),
        mean_e2e_ms=mean_e2e_ms,
        median_e2e_ms=median_e2e_ms,
        std_e2e_ms=std_e2e_ms,
        percentiles_e2e_ms=_percentile_pairs(e2es_ms, percentiles),
        prompt_length_summary=_length_summary(prompt_lengths),
        output_length_summary=_length_summary(output_lengths),
    )
