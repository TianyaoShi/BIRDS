from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import aiohttp

from llm_mst_finder.records import SampleRequest
from llm_mst_finder.vllm_compat import (
    build_openai_payload,
    decode_sse_line,
    extract_error_from_chunk,
    extract_text_from_chunk,
    parse_json_payload,
)
from llm_mst_finder.workload import prepare_workload_for_trial
from local_orchestrator.lifecycle import VLLMLifecycleManager
from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest
from local_orchestrator.models import ExpandedExperimentJob, PortReservation
from local_orchestrator.utils import runtime_server_signature


DEFAULT_MANIFEST = Path("experiments/single_gpu_cached_models_l40.yaml")
DEFAULT_WORKLOADS = (
    Path("experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls/shard_000.yaml"),
    Path("experiments/code_workloads/repobench_python_java_aggregate_cache_realistic/workload_yamls/shard_000.yaml"),
)
BOUNDARY_RE = re.compile(r"^(```)?\s*</?(CURRENT_FILE_PREFIX|REPOSITORY_CONTEXT|IMPORTS|END_FILE_PREFIX)", re.I)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(_run(args))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="code_workload_materializer.live_model_prompt_loop")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workload", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/live_code_model_prompt_loop"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=8300)
    parser.add_argument("--metrics-port", type=int, default=9300)
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--samples-per-workload", type=int, default=3)
    parser.add_argument("--max-prompt-tokens", type=int, default=1600)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--limit-models", type=int, default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--include-llama2",
        action="store_true",
        help="include Llama-2 models; excluded by default because their context limit is usually too short",
    )
    parser.add_argument("--readiness-timeout-s", type=float, default=600.0)
    parser.add_argument("--readiness-interval-s", type=float, default=3.0)
    parser.add_argument(
        "--preserve-workload-decode",
        action="store_true",
        help="use workload stop/eos settings instead of diagnostic decode overrides",
    )
    parser.add_argument("--keep-server", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


async def _run(args: argparse.Namespace) -> None:
    if args.samples_per_workload <= 0:
        raise ValueError("--samples-per-workload must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    summary_path = output_dir / "summary.json"
    server_logs_dir = output_dir / "server_logs"

    jobs = _select_single_gpu_jobs(
        args.manifest,
        selected_models=tuple(args.model or ()),
        limit_models=args.limit_models,
        include_llama2=args.include_llama2,
    )
    if args.list_models:
        print(json.dumps({"models": [job.model for job in jobs]}, indent=2, sort_keys=True))
        return
    workload_paths = tuple(args.workload or DEFAULT_WORKLOADS)
    lifecycle = VLLMLifecycleManager()
    summary: dict[str, Any] = {
        "models": {},
        "output_dir": str(output_dir),
        "responses_path": str(responses_path),
        "workloads": [str(path) for path in workload_paths],
    }
    try:
        with responses_path.open("w", encoding="utf-8") as handle:
            for job in jobs:
                model_summary = _empty_model_summary()
                summary["models"][job.model] = model_summary
                print(json.dumps({"event": "starting_model", "model": job.model}, sort_keys=True))
                try:
                    server = lifecycle.ensure_server(
                        job=_diagnostic_job(job, endpoint=args.endpoint, args=args),
                        gpu_ids=(args.gpu_id,),
                        ports=PortReservation(base_port=args.base_port, metrics_port=args.metrics_port),
                        runtime_signature=runtime_server_signature(
                            server_signature_key=job.server_signature_key,
                            gpu_ids=(args.gpu_id,),
                            base_port=args.base_port,
                            metrics_port=args.metrics_port,
                        ),
                        logs_dir=server_logs_dir,
                        force_restart=True,
                    )
                    model_summary["server"] = {
                        "base_url": server.base_url,
                        "command": list(server.command),
                        "stderr_log": str(server.stderr_log),
                        "stdout_log": str(server.stdout_log),
                    }
                    samples = _collect_samples(
                        workload_paths,
                        model_name=job.model,
                        samples_per_workload=args.samples_per_workload,
                        max_prompt_tokens=args.max_prompt_tokens,
                    )
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                        for sample_case in samples:
                            for prompt_mode, prompt in _prompt_variants(sample_case["sample"]):
                                request_sample = _diagnostic_sample(
                                    sample_case["sample"],
                                    prompt=prompt,
                                    max_output_tokens=args.max_output_tokens,
                                    preserve_workload_decode=args.preserve_workload_decode,
                                )
                                started = time.time()
                                response_text = await _send_and_decode(
                                    session,
                                    base_url=server.base_url,
                                    endpoint=args.endpoint,
                                    model=job.model,
                                    sample=request_sample,
                                    max_output_tokens=args.max_output_tokens,
                                )
                                record = _response_record(
                                    model=job.model,
                                    workload_name=sample_case["workload_name"],
                                    prompt_mode=prompt_mode,
                                    sample=request_sample,
                                    response_text=response_text,
                                    started=started,
                                )
                                _update_summary(model_summary, record)
                                handle.write(json.dumps(record, sort_keys=True) + "\n")
                                handle.flush()
                                print(
                                    json.dumps(
                                        {
                                            "boundary_echo_response": record["boundary_echo_response"],
                                            "ground_truth_match": record["ground_truth_match"],
                                            "model": job.model,
                                            "prompt_mode": prompt_mode,
                                            "response_prefix": response_text[:80],
                                            "workload": sample_case["workload_name"],
                                        },
                                        sort_keys=True,
                                    )
                                )
                except Exception as exc:
                    model_summary["error"] = f"{type(exc).__name__}: {exc}"
                    print(
                        json.dumps(
                            {"error": model_summary["error"], "event": "model_failed", "model": job.model},
                            sort_keys=True,
                        )
                    )
                finally:
                    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    if not args.keep_server:
                        lifecycle.stop_active_server(reason="model_complete")
    finally:
        if not args.keep_server:
            lifecycle.shutdown()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_single_gpu_jobs(
    manifest_path: Path,
    *,
    selected_models: tuple[str, ...],
    limit_models: int | None,
    include_llama2: bool,
) -> list[ExpandedExperimentJob]:
    manifest = load_manifest(manifest_path)
    jobs_by_model: dict[str, ExpandedExperimentJob] = {}
    for job in expand_manifest(manifest):
        if job.launch.gpu_count != 1 or job.launch.tensor_parallel_size != 1:
            continue
        if not include_llama2 and _is_llama2_model(job.model):
            continue
        if selected_models and job.model not in selected_models:
            continue
        jobs_by_model.setdefault(job.model, job)
    jobs = [jobs_by_model[model] for model in sorted(jobs_by_model)]
    if selected_models:
        missing = sorted(set(selected_models) - set(jobs_by_model))
        if missing:
            raise ValueError(f"requested models are not single-GPU jobs in manifest: {missing}")
        jobs = [jobs_by_model[model] for model in selected_models]
    if limit_models is not None:
        if limit_models <= 0:
            raise ValueError("--limit-models must be positive")
        jobs = jobs[:limit_models]
    if not jobs:
        raise ValueError(f"no single-GPU jobs found in manifest: {manifest_path}")
    return jobs


def _is_llama2_model(model: str) -> bool:
    lowered = model.lower()
    return "llama-2" in lowered or "llama2" in lowered


def _diagnostic_job(
    job: ExpandedExperimentJob,
    *,
    endpoint: str,
    args: argparse.Namespace,
) -> ExpandedExperimentJob:
    launch = replace(
        job.launch,
        gpu_count=1,
        tensor_parallel_size=1,
        readiness_timeout_s=args.readiness_timeout_s,
        readiness_interval_s=args.readiness_interval_s,
    )
    return replace(job, endpoint=endpoint, launch=launch)


def _collect_samples(
    workload_paths: tuple[Path, ...],
    *,
    model_name: str,
    samples_per_workload: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for workload_path in workload_paths:
        prepared = prepare_workload_for_trial(workload_path, model_name=model_name)
        workload_name = _workload_name(workload_path)
        selected = 0
        for sample in prepared.samples:
            if sample.prompt_len > max_prompt_tokens:
                continue
            if sample.expected_output_len < 1 or sample.expected_output_len > 128:
                continue
            cases.append({"sample": sample, "workload_name": workload_name})
            selected += 1
            if selected >= samples_per_workload:
                break
        if selected < samples_per_workload:
            raise ValueError(
                f"only found {selected} short samples in {workload_path}; "
                f"requested {samples_per_workload}"
            )
    return cases


def _workload_name(path: Path) -> str:
    for parent in path.parents:
        if parent.parent.name == "code_workloads":
            return parent.name
    return path.stem


def _prompt_variants(sample: SampleRequest) -> list[tuple[str, str]]:
    if "<CURRENT_FILE_PREFIX>" not in sample.prompt:
        return [("plain_prefix", sample.prompt)]
    return [
        ("current_xml", sample.prompt),
        ("plain_prefix", _plain_prefix_prompt(sample.prompt)),
    ]


def _plain_prefix_prompt(prompt: str) -> str:
    repository_context = _extract_tag(prompt, "REPOSITORY_CONTEXT")
    imports = _extract_tag(prompt, "IMPORTS")
    current_prefix = _extract_tag(prompt, "CURRENT_FILE_PREFIX") or prompt
    parts: list[str] = []
    if repository_context:
        parts.append("Relevant repository context:\n" + repository_context.strip())
    if imports:
        parts.append(imports.strip())
    parts.append(current_prefix.rstrip())
    return "\n\n".join(parts)


def _extract_tag(prompt: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{tag}>\n?(.*?)\n?</{tag}>", re.DOTALL)
    match = pattern.search(prompt)
    if match is None:
        return None
    return match.group(1)


def _diagnostic_sample(
    sample: SampleRequest,
    *,
    prompt: str,
    max_output_tokens: int,
    preserve_workload_decode: bool,
) -> SampleRequest:
    extra_body = dict(sample.extra_body or {})
    if not preserve_workload_decode:
        extra_body.pop("stop", None)
        extra_body["ignore_eos"] = True
    return SampleRequest(
        prompt=prompt,
        prompt_len=sample.prompt_len,
        expected_output_len=max_output_tokens,
        extra_body=extra_body,
        metadata=dict(sample.metadata),
    )


async def _send_and_decode(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    model: str,
    sample: SampleRequest,
    max_output_tokens: int,
) -> str:
    payload = build_openai_payload(endpoint, model, sample)
    payload["max_tokens"] = max_output_tokens
    text_parts: list[str] = []
    async with session.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
        async for raw_chunk in response.content:
            for payload_text in _decode_sse_payloads(raw_chunk):
                if payload_text == "[DONE]":
                    return "".join(text_parts)
                parsed = parse_json_payload(payload_text)
                error = extract_error_from_chunk(parsed)
                if error is not None:
                    raise RuntimeError(f"stream error: {error}")
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


def _response_record(
    *,
    model: str,
    workload_name: str,
    prompt_mode: str,
    sample: SampleRequest,
    response_text: str,
    started: float,
) -> dict[str, Any]:
    ground_truth = sample.metadata.get("ground_truth")
    return {
        "boundary_echo_response": _is_boundary_echo(response_text),
        "completion_text": response_text,
        "elapsed_s": time.time() - started,
        "file_path": sample.metadata.get("file_path"),
        "ground_truth": ground_truth,
        "ground_truth_match": _ground_truth_matches(response_text, ground_truth),
        "language": sample.metadata.get("language"),
        "model": model,
        "prompt": sample.prompt,
        "prompt_mode": prompt_mode,
        "prompt_prefix": sample.prompt[:500],
        "prompt_suffix": sample.prompt[-500:],
        "repo_id": sample.metadata.get("repo_id"),
        "response_prefix": response_text[:500],
        "sample_id": sample.metadata.get("sample_id"),
        "target_hash": sample.metadata.get("target_hash"),
        "task": sample.metadata.get("task"),
        "timestamp": started,
        "workload": workload_name,
    }


def _is_boundary_echo(text: str) -> bool:
    stripped = text.strip()
    return bool(BOUNDARY_RE.match(stripped))


def _ground_truth_matches(text: str, ground_truth: Any) -> bool:
    return isinstance(ground_truth, str) and text.strip() == ground_truth.strip()


def _empty_model_summary() -> dict[str, Any]:
    return {
        "boundary_echo_responses": 0,
        "empty_responses": 0,
        "exact_ground_truth_matches": 0,
        "nonempty_responses": 0,
        "prompt_modes": {},
        "server": None,
        "total": 0,
    }


def _update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["total"] += 1
    mode = record["prompt_mode"]
    prompt_modes = summary["prompt_modes"]
    prompt_modes.setdefault(
        mode,
        {
            "boundary_echo_responses": 0,
            "empty_responses": 0,
            "exact_ground_truth_matches": 0,
            "nonempty_responses": 0,
            "total": 0,
        },
    )
    for target in (summary, prompt_modes[mode]):
        target["total"] += 0 if target is summary else 1
        if record["completion_text"].strip():
            target["nonempty_responses"] += 1
        else:
            target["empty_responses"] += 1
        if record["boundary_echo_response"]:
            target["boundary_echo_responses"] += 1
        if record["ground_truth_match"]:
            target["exact_ground_truth_matches"] += 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
