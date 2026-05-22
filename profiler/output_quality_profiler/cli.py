from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .generation import run_live_generation, summarize_live_generation_shards
from .manifest import load_quality_manifest
from .materialization import load_materialization_config, source_request_counts
from .models import QualityDecodingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="output_quality_profiler.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_materialization = subparsers.add_parser("validate-materialization")
    validate_materialization.add_argument("--config", type=Path, required=True)
    validate_materialization.set_defaults(handler=_validate_materialization)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--force", action="store_true")
    materialize.set_defaults(handler=_materialize)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--manifest", type=Path, required=True)
    dry_run.set_defaults(handler=_dry_run)

    live_generation = subparsers.add_parser("run-live-generation")
    live_generation.add_argument("--job-id", required=True)
    live_generation.add_argument("--output-dir", type=Path, required=True)
    live_generation.add_argument("--workload", type=Path, required=True)
    live_generation.add_argument("--model", required=True)
    live_generation.add_argument("--base-url", required=True)
    live_generation.add_argument("--endpoint", required=True)
    live_generation.add_argument("--request-timeout-s", type=float, required=True)
    live_generation.add_argument("--max-concurrency", type=int, required=True)
    live_generation.add_argument("--load-mode", choices=("closed_loop", "open_loop"), default="closed_loop")
    live_generation.add_argument("--request-rate", type=float, default=None)
    live_generation.add_argument("--response-text-max-chars", type=int, required=True)
    live_generation.add_argument("--serving-max-model-len", type=int, default=None)
    live_generation.add_argument("--run-id", default=None)
    live_generation.add_argument("--temperature", type=float, default=0.6)
    live_generation.add_argument("--top-p", type=float, default=0.95)
    live_generation.add_argument("--top-k", type=int, default=20)
    live_generation.add_argument("--min-p", type=float, default=0.0)
    live_generation.add_argument("--n", type=int, default=1)
    live_generation.add_argument("--max-tokens", type=int, default=32768)
    live_generation.add_argument(
        "--max-tokens-policy",
        default="model_context_minus_prompt_buffer",
    )
    live_generation.add_argument("--prompt-token-buffer", type=int, default=128)
    live_generation.add_argument("--extra-body-json", default=None)
    live_generation.add_argument("--force", action="store_true")
    live_generation.set_defaults(handler=_run_live_generation)

    summarize = subparsers.add_parser("summarize-live-generation")
    summarize.add_argument("--job-id", required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--shard-output-dir", type=Path, action="append", required=True)
    summarize.add_argument("--model", required=True)
    summarize.add_argument("--run-id", default=None)
    summarize.set_defaults(handler=_summarize_live_generation)

    judge_batch = subparsers.add_parser("build-judge-batch")
    judge_batch.add_argument("--responses-root", type=Path, required=True)
    judge_batch.add_argument("--reference-model-slug", required=True)
    judge_batch.add_argument("--candidate-model-slug", action="append", required=True)
    judge_batch.add_argument("--judge-template", type=Path, required=True)
    judge_batch.add_argument("--output-dir", type=Path, required=True)
    judge_batch.add_argument("--evaluator-model", default="gpt-4.1-nano")
    judge_batch.add_argument("--max-comparisons", type=int, default=4)
    judge_batch.add_argument("--seed", type=int, default=20260520)
    judge_batch.add_argument("--endpoint", default="/v1/chat/completions")
    judge_batch.add_argument("--max-tokens", type=int, default=256)
    judge_batch.add_argument("--temperature", type=float, default=0.0)
    judge_batch.add_argument(
        "--shard-id",
        action="append",
        default=[],
        help="restrict judge comparisons to response shards whose directory name matches this shard id; may be repeated",
    )
    judge_batch.set_defaults(handler=_build_judge_batch)

    aggregate_judge = subparsers.add_parser("aggregate-judge-results")
    aggregate_judge.add_argument("--batch-manifest", type=Path, required=True)
    aggregate_judge.add_argument("--judge-results", type=Path, required=True)
    aggregate_judge.add_argument("--output-dir", type=Path, default=None)
    aggregate_judge.set_defaults(handler=_aggregate_judge_results)

    report_judge = subparsers.add_parser("report-judge-results")
    report_judge.add_argument("--judge-responses-dir", type=Path, required=True)
    report_judge.add_argument("--manifest-dir", type=Path, action="append", required=True)
    report_judge.add_argument("--output-dir", type=Path, required=True)
    report_judge.add_argument(
        "--reference-model-slug",
        default="meta-llama-llama-3-1-8b-instruct",
    )
    report_judge.set_defaults(handler=_report_judge_results)

    split_judge = subparsers.add_parser("split-judge-batch-by-candidate")
    split_judge.add_argument("--batch-jsonl", type=Path, required=True)
    split_judge.add_argument("--batch-manifest", type=Path, required=True)
    split_judge.add_argument("--output-dir", type=Path, required=True)
    split_judge.add_argument("--parts-per-candidate", type=int, default=2)
    split_judge.add_argument("--candidate-model-slug", action="append", default=[])
    split_judge.set_defaults(handler=_split_judge_batch_by_candidate)

    preprocess_responses = subparsers.add_parser("preprocess-responses")
    preprocess_responses.add_argument("--responses-root", type=Path, required=True)
    preprocess_responses.add_argument("--output-root", type=Path, required=True)
    preprocess_responses.add_argument("--force", action="store_true")
    preprocess_responses.set_defaults(handler=_preprocess_responses)

    submit_batches = subparsers.add_parser("submit-openai-batches")
    submit_batches.add_argument("--split-manifest", type=Path, required=True)
    submit_batches.add_argument("--api-key-file", type=Path, required=True)
    submit_batches.add_argument("--ledger", type=Path, default=None)
    submit_batches.add_argument("--limit", type=int, default=None)
    submit_batches.add_argument("--wait-for-completion", action="store_true")
    submit_batches.add_argument("--poll-interval-s", type=float, default=60.0)
    submit_batches.add_argument("--completion-window", default="24h")
    submit_batches.set_defaults(handler=_submit_openai_batches)

    select_benchmark = subparsers.add_parser("select-missing-benchmark-scores")
    select_benchmark.add_argument("--scorebook", type=Path, required=True)
    select_benchmark.add_argument("--output-dir", type=Path, required=True)
    select_benchmark.add_argument("--target", action="append", default=None)
    select_benchmark.set_defaults(handler=_select_missing_benchmark_scores)

    build_benchmark_manifest = subparsers.add_parser("build-benchmark-generation-manifest")
    build_benchmark_manifest.add_argument("--missing-plan", type=Path, required=True)
    build_benchmark_manifest.add_argument("--base-manifest", type=Path, required=True)
    build_benchmark_manifest.add_argument("--output", type=Path, required=True)
    build_benchmark_manifest.add_argument("--run-id", default=None)
    build_benchmark_manifest.add_argument("--include-benchmark", action="append", default=None)
    build_benchmark_manifest.set_defaults(handler=_build_benchmark_generation_manifest)

    score_benchmark = subparsers.add_parser("score-benchmark-responses")
    score_benchmark.add_argument("--benchmark", required=True)
    score_benchmark.add_argument("--responses-root", type=Path, required=True)
    score_benchmark.add_argument("--output-dir", type=Path, required=True)
    score_benchmark.set_defaults(handler=_score_benchmark_responses)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _validate_materialization(args: argparse.Namespace) -> int:
    config = load_materialization_config(args.config)
    payload = {
        "path": str(config.path),
        "seed": config.seed,
        "total_requests": config.total_requests,
        "shards": config.shards,
        "source_request_counts": source_request_counts(config),
        "bucket_policy": config.bucket_policy.to_dict(),
        "minimum_per_source_bucket": config.minimum_per_source_bucket,
        "allow_replacement": config.allow_replacement,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _materialize(args: argparse.Namespace) -> int:
    from .materialization import materialize_quality_requests

    result = materialize_quality_requests(args.config, force=bool(args.force))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    manifest = load_quality_manifest(args.manifest)
    jobs = []
    for experiment in manifest.experiments:
        for model in experiment.models:
            jobs.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "model": model,
                    "workloads": [str(workload) for workload in experiment.workloads],
                    "shard_count": len(experiment.workloads),
                    "endpoint": experiment.endpoint,
                    "gpu_count": experiment.launch.gpu_count,
                    "tensor_parallel_size": experiment.launch.tensor_parallel_size,
                    "decoding": experiment.generation.decoding.to_dict(),
                    "load_mode": experiment.generation.load_mode,
                    "request_rate": experiment.generation.request_rate,
                    "max_concurrency": experiment.generation.max_concurrency,
                    "concurrency_source": experiment.generation.concurrency_source,
                    "concurrency_mst_fraction": experiment.generation.concurrency_mst_fraction,
                }
            )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest.manifest_path),
                "run_id": manifest.run.run_id,
                "job_count": len(jobs),
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_live_generation(args: argparse.Namespace) -> int:
    summary = run_live_generation(
        job_id=args.job_id,
        output_dir=args.output_dir,
        workload=args.workload,
        model=args.model,
        base_url=args.base_url,
        endpoint=args.endpoint,
        request_timeout_s=args.request_timeout_s,
        max_concurrency=args.max_concurrency,
        load_mode=args.load_mode,
        request_rate=args.request_rate,
        response_text_max_chars=args.response_text_max_chars,
        serving_max_model_len=args.serving_max_model_len,
        run_id=args.run_id,
        force=bool(args.force),
        decoding=QualityDecodingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            n=args.n,
            max_tokens=args.max_tokens,
            max_tokens_policy=args.max_tokens_policy,
            prompt_token_buffer=args.prompt_token_buffer,
            extra_body=_parse_json_mapping(args.extra_body_json, field_name="--extra-body-json"),
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _summarize_live_generation(args: argparse.Namespace) -> int:
    summary = summarize_live_generation_shards(
        job_id=args.job_id,
        output_dir=args.output_dir,
        shard_output_dirs=args.shard_output_dir,
        model=args.model,
        run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_judge_batch(args: argparse.Namespace) -> int:
    from .judge_batches import build_openai_judge_batch

    result = build_openai_judge_batch(
        responses_root=args.responses_root,
        reference_model_slug=args.reference_model_slug,
        candidate_model_slugs=tuple(args.candidate_model_slug),
        judge_template_path=args.judge_template,
        output_dir=args.output_dir,
        evaluator_model=args.evaluator_model,
        max_comparisons=args.max_comparisons,
        seed=args.seed,
        endpoint=args.endpoint,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        shard_ids=tuple(args.shard_id or ()),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _aggregate_judge_results(args: argparse.Namespace) -> int:
    from .scoring import aggregate_openai_batch_judge_results

    payload = aggregate_openai_batch_judge_results(
        batch_manifest=args.batch_manifest,
        judge_results_jsonl=args.judge_results,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _report_judge_results(args: argparse.Namespace) -> int:
    from .reporting import report_judge_results

    payload = report_judge_results(
        judge_responses_dir=args.judge_responses_dir,
        manifest_dirs=tuple(args.manifest_dir),
        output_dir=args.output_dir,
        reference_model_slug=args.reference_model_slug,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _split_judge_batch_by_candidate(args: argparse.Namespace) -> int:
    from .openai_batch_ops import split_judge_batch_by_candidate

    payload = split_judge_batch_by_candidate(
        batch_jsonl=args.batch_jsonl,
        batch_manifest=args.batch_manifest,
        output_dir=args.output_dir,
        parts_per_candidate=args.parts_per_candidate,
        candidate_model_slugs=tuple(args.candidate_model_slug or ()),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _preprocess_responses(args: argparse.Namespace) -> int:
    from .response_preprocessing import preprocess_response_tree

    payload = preprocess_response_tree(
        responses_root=args.responses_root,
        output_root=args.output_root,
        force=bool(args.force),
    )
    print(json.dumps(payload.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _submit_openai_batches(args: argparse.Namespace) -> int:
    from .openai_batch_ops import submit_openai_batch_parts

    payload = submit_openai_batch_parts(
        split_manifest=args.split_manifest,
        api_key_file=args.api_key_file,
        ledger_path=args.ledger,
        limit=args.limit,
        wait_for_completion=bool(args.wait_for_completion),
        poll_interval_s=args.poll_interval_s,
        completion_window=args.completion_window,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _select_missing_benchmark_scores(args: argparse.Namespace) -> int:
    from .benchmark_selection import select_missing_benchmark_scores

    result = select_missing_benchmark_scores(
        scorebook=args.scorebook,
        output_dir=args.output_dir,
        targets=tuple(args.target) if args.target else None,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _build_benchmark_generation_manifest(args: argparse.Namespace) -> int:
    from .benchmark_selection import build_benchmark_generation_manifest

    result = build_benchmark_generation_manifest(
        missing_plan=args.missing_plan,
        base_manifest=args.base_manifest,
        output_path=args.output,
        run_id=args.run_id,
        include_benchmarks=tuple(args.include_benchmark) if args.include_benchmark else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _score_benchmark_responses(args: argparse.Namespace) -> int:
    from .benchmark_adapters import score_benchmark_responses

    payload = score_benchmark_responses(
        benchmark=args.benchmark,
        responses_root=args.responses_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_json_mapping(raw: str | None, *, field_name: str) -> dict:
    if raw is None or raw == "":
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
