#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_WORKLOAD_ROOT = Path(
    "/scratch/gautschi/shi676/BioLLM/experiments/code_workloads/"
    "repobench_python_java_aggregate_code_chat_completion/workload_yamls"
)

CHAT_EXTRA_BODY = {
    "stop": [
        "</TARGET_FILE>",
        "</COMPLETION>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "\n\n\n",
    ],
    "chat_template_kwargs": {"enable_thinking": False},
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a RepoBench chat-template rerun manifest from the raw-prefix manifest."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/quality/h100_repobench_original_responses.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/quality/h100_repobench_code_chat_completion_rerun.yaml"),
    )
    parser.add_argument(
        "--run-id",
        default="h100-quality-repobench-original-responses-005-chat-rerun",
    )
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit(f"manifest is not a mapping: {args.source}")

    run = _mapping(manifest.setdefault("run", {}), "run")
    run["run_id"] = args.run_id
    run["default_endpoint"] = "/v1/chat/completions"

    workloads = [
        str((args.workload_root / f"shard_{index:03d}.yaml").resolve())
        for index in range(6)
    ]
    for experiment in _sequence(manifest.get("experiments"), "experiments"):
        experiment["endpoint"] = "/v1/chat/completions"
        experiment["workloads"] = workloads
        generation = _mapping(experiment.setdefault("generation", {}), "experiment.generation")
        decoding = _mapping(generation.setdefault("decoding", {}), "experiment.generation.decoding")
        decoding["temperature"] = 0.0
        decoding["top_p"] = 0.95
        decoding["top_k"] = 64
        decoding["min_p"] = 0.0
        decoding["max_tokens"] = 128
        decoding["n"] = 1
        decoding["extra_body"] = dict(CHAT_EXTRA_BODY)

    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output} with {len(manifest.get('experiments', []))} experiments")
    return 0


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{field_name} must be a mapping")
    return value


def _sequence(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SystemExit(f"{field_name} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SystemExit(f"{field_name}[{index}] must be a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
