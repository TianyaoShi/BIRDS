from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .generation import run_live_generation
from .manifest import load_quality_manifest
from .materialization import load_materialization_config, source_request_counts
from .models import QualityDecodingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="output_quality_profiler.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_materialization = subparsers.add_parser("validate-materialization")
    validate_materialization.add_argument("--config", type=Path, required=True)
    validate_materialization.set_defaults(handler=_validate_materialization)

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
    live_generation.add_argument("--force", action="store_true")
    live_generation.set_defaults(handler=_run_live_generation)

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


def _dry_run(args: argparse.Namespace) -> int:
    manifest = load_quality_manifest(args.manifest)
    jobs = []
    for experiment in manifest.experiments:
        for model in experiment.models:
            for workload in experiment.workloads:
                jobs.append(
                    {
                        "experiment_id": experiment.experiment_id,
                        "model": model,
                        "workload": str(workload),
                        "endpoint": experiment.endpoint,
                        "gpu_count": experiment.launch.gpu_count,
                        "tensor_parallel_size": experiment.launch.tensor_parallel_size,
                        "decoding": experiment.generation.decoding.to_dict(),
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
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
