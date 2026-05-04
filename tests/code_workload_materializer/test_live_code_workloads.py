from __future__ import annotations

import json
import os
import time
import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from llm_mst_finder.records import SampleRequest
from llm_mst_finder.vllm_compat import (
    build_openai_payload,
    decode_sse_line,
    extract_error_from_chunk,
    extract_text_from_chunk,
    parse_json_payload,
)
from llm_mst_finder.workload import prepare_workload_for_trial


LIVE_ENV = "CODE_WORKLOAD_MATERIALIZER_RUN_LIVE"
MODEL_ENV = "CODE_WORKLOAD_LIVE_MODEL"
BASE_URL_ENV = "CODE_WORKLOAD_LIVE_BASE_URL"
LOG_PATH_ENV = "CODE_WORKLOAD_LIVE_LOG_PATH"
SAMPLES_PER_WORKLOAD_ENV = "CODE_WORKLOAD_LIVE_SAMPLES_PER_WORKLOAD"
DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SAMPLES_PER_WORKLOAD = 8
WORKLOADS = {
    "crosscodeeval": Path(
        "experiments/code_workloads/"
        "crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls/shard_000.yaml"
    ),
    "repobench_aggregate": Path(
        "experiments/code_workloads/"
        "repobench_python_java_aggregate_cache_realistic/workload_yamls/shard_000.yaml"
    ),
}


pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_ENV) != "1",
    reason=f"set {LIVE_ENV}=1 to run live code-workload smoke tests against localhost:8000",
)


def test_live_code_workload_decoded_responses_are_logged() -> None:
    asyncio.run(_run_live_code_workload_decoded_responses_are_logged())


async def _run_live_code_workload_decoded_responses_are_logged() -> None:
    model = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
    log_path = Path(
        os.environ.get(
            LOG_PATH_ENV,
            "results/live_code_workload_smoke/decoded_responses.jsonl",
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples_per_workload = int(os.environ.get(SAMPLES_PER_WORKLOAD_ENV, DEFAULT_SAMPLES_PER_WORKLOAD))

    samples_by_workload = {
        workload_name: _select_short_samples(
            workload_path,
            model_name=model,
            limit=samples_per_workload,
        )
        for workload_name, workload_path in WORKLOADS.items()
    }
    summary: dict[str, dict[str, int]] = {}
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with log_path.open("w", encoding="utf-8") as handle:
            for workload_name, samples in samples_by_workload.items():
                summary[workload_name] = {
                    "boundary_echo_responses": 0,
                    "empty_responses": 0,
                    "exact_ground_truth_matches": 0,
                    "nonempty_responses": 0,
                    "total": 0,
                }
                for index, sample in enumerate(samples):
                    completion_text = await _send_and_decode_completion(
                        session,
                        base_url=base_url,
                        endpoint="/v1/completions",
                        model=model,
                        sample=sample,
                    )
                    record = {
                        "boundary_echo_response": _is_boundary_echo(completion_text),
                        "completion_text": completion_text,
                        "expected_output_len": sample.expected_output_len,
                        "file_path": sample.metadata.get("file_path"),
                        "ground_truth": sample.metadata.get("ground_truth"),
                        "ground_truth_hash": sample.metadata.get("target_hash"),
                        "ground_truth_match": _ground_truth_matches(
                            completion_text,
                            sample.metadata.get("ground_truth"),
                        ),
                        "language": sample.metadata.get("language"),
                        "prompt": sample.prompt,
                        "prompt_len": sample.prompt_len,
                        "prompt_prefix": sample.prompt[:500],
                        "prompt_suffix": sample.prompt[-500:],
                        "repo_id": sample.metadata.get("repo_id"),
                        "request_index": index,
                        "sample_id": sample.metadata.get("sample_id"),
                        "task": sample.metadata.get("task"),
                        "timestamp": time.time(),
                        "workload": workload_name,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    stats = summary[workload_name]
                    stats["total"] += 1
                    if completion_text.strip():
                        stats["nonempty_responses"] += 1
                    else:
                        stats["empty_responses"] += 1
                    if record["boundary_echo_response"]:
                        stats["boundary_echo_responses"] += 1
                    if record["ground_truth_match"]:
                        stats["exact_ground_truth_matches"] += 1
                    print(
                        json.dumps(
                            {
                                "boundary_echo_response": record["boundary_echo_response"],
                                "completion_text": completion_text,
                                "ground_truth": record["ground_truth"],
                                "ground_truth_match": record["ground_truth_match"],
                                "language": record["language"],
                                "sample_id": record["sample_id"],
                                "task": record["task"],
                                "workload": workload_name,
                            },
                            sort_keys=True,
                        )
                    )
                    assert completion_text.strip()
    summary_path = log_path.with_name("decoded_responses_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logged = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(logged) == len(WORKLOADS) * samples_per_workload


def _select_short_samples(
    workload_path: Path,
    *,
    model_name: str,
    limit: int,
) -> list[SampleRequest]:
    if not workload_path.is_file():
        pytest.skip(f"materialized workload YAML not found: {workload_path}")
    if limit <= 0:
        raise ValueError(f"{SAMPLES_PER_WORKLOAD_ENV} must be positive")
    prepared = prepare_workload_for_trial(workload_path, model_name=model_name)
    source_rows = _load_source_rows(workload_path)
    selected: list[SampleRequest] = []
    for sample in prepared.samples:
        if sample.prompt_len > 1600:
            continue
        if sample.expected_output_len > 32:
            continue
        if sample.expected_output_len < 2:
            continue
        metadata = dict(sample.metadata)
        metadata["ground_truth"] = source_rows[int(sample.metadata["source_index"])]["ground_truth"]
        selected.append(
            SampleRequest(
                prompt=sample.prompt,
                prompt_len=sample.prompt_len,
                expected_output_len=max(16, sample.expected_output_len),
                extra_body=_live_decode_extra_body(sample.extra_body),
                metadata=metadata,
            )
        )
        if len(selected) == limit:
            return selected
    pytest.skip(f"no short live-smoke samples found in {workload_path}")


def _load_source_rows(workload_path: Path) -> list[dict[str, str]]:
    import yaml

    payload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    shard_path = (workload_path.parent / dataset["path"]).resolve()
    rows: list[dict[str, str]] = []
    with shard_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "ground_truth": _extract_ground_truth_from_row(row),
                }
            )
    return rows


def _extract_ground_truth_from_row(row: dict[str, Any]) -> str:
    ground_truth = row["metadata"].get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError(
            "live code-workload diagnostics require materialized rows with "
            "metadata.ground_truth; rematerialize code workloads"
        )
    return ground_truth


def _live_decode_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    body = dict(extra_body or {})
    # The generated profiling workloads may stop after a newline. This smoke test
    # is about decoding and logging visible text, so leave the workload prompt
    # untouched but give the model room to emit a short completion.
    body.pop("stop", None)
    body["ignore_eos"] = True
    return body or None


async def _send_and_decode_completion(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    model: str,
    sample: SampleRequest,
) -> str:
    payload = build_openai_payload(endpoint, model, sample)
    payload["max_tokens"] = min(64, max(16, sample.expected_output_len))
    text_parts: list[str] = []
    async with session.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as response:
        body_prefix = ""
        if response.status != 200:
            body_prefix = await response.text()
            raise AssertionError(f"live request failed: HTTP {response.status}: {body_prefix[:1000]}")
        async for raw_chunk in response.content:
            for payload_text in _decode_sse_payloads(raw_chunk):
                if payload_text == "[DONE]":
                    return "".join(text_parts)
                parsed = parse_json_payload(payload_text)
                error = extract_error_from_chunk(parsed)
                if error is not None:
                    raise AssertionError(f"live stream error: {error}")
                text = extract_text_from_chunk(endpoint, parsed)
                if text is not None:
                    text_parts.append(text)
    return "".join(text_parts)


def _decode_sse_payloads(raw_chunk: bytes) -> list[str]:
    payloads: list[str] = []
    for raw_line in raw_chunk.splitlines():
        payload = decode_sse_line(raw_line)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _is_boundary_echo(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<CURRENT_FILE_PREFIX") or stripped.startswith("</CURRENT_FILE_PREFIX")


def _ground_truth_matches(text: str, ground_truth: Any) -> bool:
    if not isinstance(ground_truth, str) or ground_truth.startswith("<not exported"):
        return False
    return text.strip() == ground_truth.strip()
