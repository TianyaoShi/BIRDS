from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .manifest import load_quality_manifest
from .materialization import load_materialization_config, source_request_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="output_quality_profiler.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_materialization = subparsers.add_parser("validate-materialization")
    validate_materialization.add_argument("--config", type=Path, required=True)
    validate_materialization.set_defaults(handler=_validate_materialization)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--manifest", type=Path, required=True)
    dry_run.set_defaults(handler=_dry_run)

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

