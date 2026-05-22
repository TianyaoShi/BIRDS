from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm_mst_finder.records import RequestRecord, SampleRequest
from llm_mst_finder.request_client import (
    CAPTURE_RESPONSE_TEXT_ENV,
    RESPONSE_TEXT_MAX_CHARS_ENV,
    RequestClient,
)
from llm_mst_finder.model_context import resolve_model_context_info
from llm_mst_finder.workload import (
    _normalized_tokenizer_spec,
    _tokenizer_cache_key,
    generate_sample_requests,
    load_workload_config,
    load_workload_samples_for_sampling_only,
    resolve_tokenizer,
)

from .models import QualityDecodingConfig


def run_live_generation(
    *,
    job_id: str,
    output_dir: str | Path,
    workload: str | Path,
    model: str,
    base_url: str,
    endpoint: str,
    request_timeout_s: float,
    max_concurrency: int,
    response_text_max_chars: int,
    decoding: QualityDecodingConfig,
    load_mode: str = "closed_loop",
    request_rate: float | None = None,
    serving_max_model_len: int | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    return asyncio.run(
        run_live_generation_async(
            job_id=job_id,
            output_dir=output_dir,
            workload=workload,
            model=model,
            base_url=base_url,
            endpoint=endpoint,
            request_timeout_s=request_timeout_s,
            max_concurrency=max_concurrency,
            load_mode=load_mode,
            request_rate=request_rate,
            response_text_max_chars=response_text_max_chars,
            decoding=decoding,
            serving_max_model_len=serving_max_model_len,
            run_id=run_id,
            force=force,
        )
    )


def summarize_live_generation_shards(
    *,
    job_id: str,
    output_dir: str | Path,
    shard_output_dirs: Iterable[str | Path],
    model: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [Path(item) for item in shard_output_dirs]
    shard_summaries: list[dict[str, Any]] = []
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    response_text_truncated = 0
    responses_path = resolved_output_dir / "responses.jsonl"
    failed_path = resolved_output_dir / "failed_requests.jsonl"
    with responses_path.open("w", encoding="utf-8") as responses_handle, failed_path.open(
        "w", encoding="utf-8"
    ) as failed_handle:
        for shard_dir in shard_dirs:
            summary_path = shard_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing shard summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            shard_summaries.append(summary)
            total_requests += int(summary.get("total_requests", 0))
            successful_requests += int(summary.get("successful_requests", 0))
            failed_requests += int(summary.get("failed_requests", 0))
            response_text_truncated += int(summary.get("response_text_truncated", 0))
            _copy_jsonl(shard_dir / "responses.jsonl", responses_handle)
            failed_jsonl = shard_dir / "failed_requests.jsonl"
            if failed_jsonl.is_file():
                _copy_jsonl(failed_jsonl, failed_handle)
    aggregate = {
        "run_id": run_id,
        "job_id": job_id,
        "model": model,
        "shard_count": len(shard_dirs),
        "shard_output_dirs": [str(item) for item in shard_dirs],
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "response_text_truncated": response_text_truncated,
        "shards": shard_summaries,
        "artifacts": {
            "responses_jsonl": "responses.jsonl",
            "failed_requests_jsonl": "failed_requests.jsonl",
            "summary_json": "summary.json",
        },
    }
    (resolved_output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


async def run_live_generation_async(
    *,
    job_id: str,
    output_dir: str | Path,
    workload: str | Path,
    model: str,
    base_url: str,
    endpoint: str,
    request_timeout_s: float,
    max_concurrency: int,
    response_text_max_chars: int,
    decoding: QualityDecodingConfig,
    load_mode: str = "closed_loop",
    request_rate: float | None = None,
    serving_max_model_len: int | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if load_mode not in {"closed_loop", "open_loop"}:
        raise ValueError("load_mode must be closed_loop or open_loop")
    if request_rate is not None and request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if load_mode == "open_loop" and request_rate is None:
        raise ValueError("request_rate is required when load_mode=open_loop")
    resolved_output_dir = Path(output_dir)
    if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
        if not force:
            raise RuntimeError(f"refusing to overwrite existing output directory: {resolved_output_dir}")
        for child in resolved_output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    samples, workload_metadata = _prepare_samples(
        workload=Path(workload),
        model=model,
        endpoint=endpoint,
        serving_max_model_len=serving_max_model_len,
        decoding=decoding,
    )
    responses_path = resolved_output_dir / "responses.jsonl"
    failed_path = resolved_output_dir / "failed_requests.jsonl"
    summary_path = resolved_output_dir / "summary.json"

    previous_capture = os.environ.get(CAPTURE_RESPONSE_TEXT_ENV)
    previous_max_chars = os.environ.get(RESPONSE_TEXT_MAX_CHARS_ENV)
    os.environ[CAPTURE_RESPONSE_TEXT_ENV] = "1"
    os.environ[RESPONSE_TEXT_MAX_CHARS_ENV] = str(response_text_max_chars)
    try:
        client = RequestClient(
            base_url=base_url,
            endpoint=endpoint,
            model=model,
            timeout_s=request_timeout_s,
            extra_body=decoding.to_request_extra_body(),
        )
        semaphore = asyncio.Semaphore(max_concurrency)
        async with client:
            if load_mode == "open_loop":
                assert request_rate is not None
                tasks = await _schedule_open_loop_requests(
                    client=client,
                    semaphore=semaphore,
                    samples=samples,
                    job_id=job_id,
                    request_rate=request_rate,
                )
            else:
                tasks = [
                    asyncio.create_task(
                        _send_one(
                            client=client,
                            semaphore=semaphore,
                            sample=sample,
                            request_index=index,
                            job_id=job_id,
                            scheduled_send_ts=float(index),
                        )
                    )
                    for index, sample in enumerate(samples)
                ]
            records = await asyncio.gather(*tasks)
    finally:
        _restore_env(CAPTURE_RESPONSE_TEXT_ENV, previous_capture)
        _restore_env(RESPONSE_TEXT_MAX_CHARS_ENV, previous_max_chars)

    rows = [
        _record_to_response_row(
            record,
            run_id=run_id,
            job_id=job_id,
            model=model,
            workload=str(workload),
            decoding=decoding,
        )
        for record in sorted(records, key=lambda item: _request_index(item))
    ]
    _write_jsonl(responses_path, rows)
    _write_jsonl(failed_path, [row for row in rows if not row["success"]])
    summary = _build_summary(
        rows=rows,
        run_id=run_id,
        job_id=job_id,
        model=model,
        workload=str(workload),
        base_url=base_url,
        endpoint=endpoint,
        max_concurrency=max_concurrency,
        load_mode=load_mode,
        request_rate=request_rate,
        response_text_max_chars=response_text_max_chars,
        decoding=decoding,
        workload_metadata=workload_metadata,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


async def _send_one(
    *,
    client: RequestClient,
    semaphore: asyncio.Semaphore,
    sample: SampleRequest,
    request_index: int,
    job_id: str,
    scheduled_send_ts: float,
) -> RequestRecord:
    async with semaphore:
        return await client.send_request(
            sample,
            request_id=f"{job_id}-{request_index:06d}",
            trial_id=job_id,
            scheduled_send_ts=scheduled_send_ts,
        )


async def _schedule_open_loop_requests(
    *,
    client: RequestClient,
    semaphore: asyncio.Semaphore,
    samples: Sequence[SampleRequest],
    job_id: str,
    request_rate: float,
) -> list[asyncio.Task[RequestRecord]]:
    start_ts = time.perf_counter()
    tasks: list[asyncio.Task[RequestRecord]] = []
    for index, sample in enumerate(samples):
        scheduled_send_ts = start_ts + (index / request_rate)
        sleep_s = scheduled_send_ts - time.perf_counter()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        tasks.append(
            asyncio.create_task(
                _send_one(
                    client=client,
                    semaphore=semaphore,
                    sample=sample,
                    request_index=index,
                    job_id=job_id,
                    scheduled_send_ts=scheduled_send_ts,
                )
            )
        )
        await asyncio.sleep(0)
    return tasks


def _prepare_samples(
    *,
    workload: Path,
    model: str,
    endpoint: str,
    serving_max_model_len: int | None,
    decoding: QualityDecodingConfig,
) -> tuple[list[SampleRequest], dict[str, Any]]:
    config = load_workload_config(workload)
    if config.context_policy is not None:
        fallback_tokenizer_name = config.tokenizer or model
        fallback_tokenizer = resolve_tokenizer(fallback_tokenizer_name)
        fallback_tokenizer_key = _tokenizer_cache_key(
            fallback_tokenizer_name,
            tokenizer=fallback_tokenizer,
        )
        model_context_info = resolve_model_context_info(
            config.context_policy,
            workload_tokenizer=fallback_tokenizer,
            workload_tokenizer_key=fallback_tokenizer_key,
            model_name=model,
            fallback_tokenizer=fallback_tokenizer,
            fallback_tokenizer_key=fallback_tokenizer_key,
            fallback_tokenizer_name=_normalized_tokenizer_spec(fallback_tokenizer_name),
            serving_max_model_len=serving_max_model_len,
        )
        effective_policy = model_context_info.effective_policy(config.context_policy)
        raw_samples = generate_sample_requests(
            config,
            tokenizer=model_context_info.tokenizer,
            tokenizer_key=model_context_info.tokenizer_key,
        )
        samples = []
        skipped_source_indexes: list[int] = []
        for index, sample in enumerate(raw_samples):
            source_index = sample.metadata.get("source_index", index)
            if sample.prompt_len + decoding.prompt_token_buffer >= effective_policy.max_model_len:
                if isinstance(source_index, int):
                    skipped_source_indexes.append(source_index)
                continue
            samples.append(sample)
        if not samples:
            raise ValueError(
                "quality context validation removed all workload samples; no requests remain "
                "after enforcing prompt-only context fit"
            )
        metadata = {
            "workload": {
                "name": config.name,
                "source_path": str(config.source_path),
                "dataset_type": config.dataset.type,
                "num_requests": len(samples),
                "context_policy": {
                    "max_model_len": effective_policy.max_model_len,
                    "tokenizer_source": effective_policy.tokenizer_source,
                    "over_limit": effective_policy.over_limit,
                    "reserve_tokens": decoding.prompt_token_buffer,
                    "total_samples": len(raw_samples),
                    "kept_samples": len(samples),
                    "skipped_samples": len(raw_samples) - len(samples),
                    "truncated_samples": 0,
                    "skipped_source_indexes": skipped_source_indexes,
                    "truncated_source_indexes": [],
                    "quality_validation_mode": "prompt_only_dynamic_output_cap",
                },
                "model_context": model_context_info.to_metadata(),
            }
        }
    else:
        samples = load_workload_samples_for_sampling_only(workload)
        metadata = {
            "workload": {
                "name": config.name,
                "source_path": str(config.source_path),
                "dataset_type": config.dataset.type,
                "num_requests": len(samples),
            }
        }
    return [
        _sample_with_quality_decoding(
            sample,
            decoding=decoding,
            serving_max_model_len=serving_max_model_len,
        )
        for sample in samples
    ], metadata


def _sample_with_quality_decoding(
    sample: SampleRequest,
    *,
    decoding: QualityDecodingConfig,
    serving_max_model_len: int | None,
) -> SampleRequest:
    if decoding.max_tokens_policy == "workload_expected_output_len":
        max_tokens = min(decoding.max_tokens, max(1, int(sample.expected_output_len)))
    else:
        max_tokens = decoding.max_tokens
    if serving_max_model_len is not None:
        max_tokens = min(
            max_tokens,
            max(1, int(serving_max_model_len) - int(sample.prompt_len) - decoding.prompt_token_buffer),
        )
    extra_body = dict(sample.extra_body or {})
    extra_body.update(decoding.to_request_extra_body())
    metadata = dict(sample.metadata)
    metadata["prompt"] = sample.prompt
    return replace(
        sample,
        expected_output_len=max_tokens,
        extra_body=extra_body,
        metadata=metadata,
    )


def _record_to_response_row(
    record: RequestRecord,
    *,
    run_id: str | None,
    job_id: str,
    model: str,
    workload: str,
    decoding: QualityDecodingConfig,
) -> dict[str, Any]:
    metadata = dict(record.metadata or {})
    response_text = metadata.pop("response_text", "")
    response_text_truncated = bool(metadata.pop("response_text_truncated", False))
    return {
        "run_id": run_id,
        "job_id": job_id,
        "model": model,
        "workload": workload,
        "request_id": metadata.get("request_id") or metadata.get("sample_id") or record.request_id,
        "source": metadata.get("source"),
        "prompt_length_bucket": metadata.get("prompt_length_bucket"),
        "prompt": _prompt_from_metadata_or_none(metadata),
        "response_text": response_text,
        "response_text_truncated": response_text_truncated,
        "finish_reason": None,
        "success": record.success,
        "error": record.error,
        "actual_output_len": record.actual_output_len,
        "expected_output_len": record.expected_output_len,
        "decoding": decoding.to_dict(),
        "metadata": metadata,
    }


def _prompt_from_metadata_or_none(metadata: dict[str, Any]) -> str | None:
    prompt = metadata.get("prompt")
    return prompt if isinstance(prompt, str) else None


def _build_summary(
    *,
    rows: Sequence[dict[str, Any]],
    run_id: str | None,
    job_id: str,
    model: str,
    workload: str,
    base_url: str,
    endpoint: str,
    max_concurrency: int,
    load_mode: str,
    request_rate: float | None,
    response_text_max_chars: int,
    decoding: QualityDecodingConfig,
    workload_metadata: dict[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(str(row.get("source")) for row in rows)
    bucket_counts = Counter(str(row.get("prompt_length_bucket")) for row in rows)
    success_count = sum(1 for row in rows if row["success"])
    truncated_count = sum(1 for row in rows if row["response_text_truncated"])
    return {
        "run_id": run_id,
        "job_id": job_id,
        "model": model,
        "workload": workload,
        "base_url": base_url,
        "endpoint": endpoint,
        "max_concurrency": max_concurrency,
        "load_mode": load_mode,
        "request_rate": request_rate,
        "response_text_max_chars": response_text_max_chars,
        "decoding": decoding.to_dict(),
        "total_requests": len(rows),
        "successful_requests": success_count,
        "failed_requests": len(rows) - success_count,
        "response_text_truncated": truncated_count,
        "source_counts": dict(source_counts),
        "prompt_length_bucket_counts": dict(bucket_counts),
        "workload_metadata": workload_metadata,
        "artifacts": {
            "responses_jsonl": "responses.jsonl",
            "failed_requests_jsonl": "failed_requests.jsonl",
            "summary_json": "summary.json",
        },
    }


def _request_index(record: RequestRecord) -> int:
    value = (record.metadata or {}).get("request_index")
    return value if isinstance(value, int) else 0


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _copy_jsonl(path: Path, output_handle: Any) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing jsonl artifact: {path}")
    with path.open("r", encoding="utf-8") as input_handle:
        for line in input_handle:
            output_handle.write(line)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
