#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASE_MANIFEST = Path("experiments/quality/h100_supergpqa_hard_original_responses_16k_rerun.yaml")
DEFAULT_SAMPLE_DIR = Path("experiments/reasoning_workloads/mmlu_pro_output_length_sample_1600")
DEFAULT_OUTPUT = Path("experiments/quality/h100_mmlu_pro_output_length_responses.yaml")
DEFAULT_RUN_ID = "h100-quality-mmlu-pro-output-length-responses-000"

PRE_2507_QWEN3_NON_THINKING = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
}

REASONING_SAMPLING_MODELS = {
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build H100 quality response manifest for MMLU-Pro output-length profiling."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    base_manifest = _resolve(args.base_manifest, repo_root=repo_root)
    sample_dir = _resolve(args.sample_dir, repo_root=repo_root)
    output = _resolve(args.output, repo_root=repo_root)
    payload = yaml.safe_load(base_manifest.read_text(encoding="utf-8"))
    workloads = [
        str((sample_dir / "workload_yamls" / f"shard_{index:03d}.yaml").resolve())
        for index in range(4)
    ]

    payload["run"]["run_id"] = args.run_id
    payload["run"]["output_root"] = "../../results/quality"
    payload["run"]["default_endpoint"] = "/v1/chat/completions"
    payload["slurm"]["time"] = "06:00:00"
    payload["slurm"]["array_concurrency_limit"] = 3
    payload["slurm"]["base_port"] = 9900

    payload["generation"]["load_mode"] = "open_loop"
    payload["generation"]["response_text_max_chars"] = 1048576
    payload["generation"]["include_prompt_text"] = True
    payload["generation"]["decoding"]["max_tokens"] = 16384
    payload["generation"]["decoding"]["max_tokens_policy"] = "model_context_minus_prompt_buffer"
    payload["generation"]["decoding"]["prompt_token_buffer"] = 128

    for experiment in payload["experiments"]:
        model = experiment["model"]
        experiment["id"] = _response_id(model)
        experiment["endpoint"] = "/v1/chat/completions"
        experiment["workloads"] = workloads
        generation = experiment.setdefault("generation", {})
        generation["load_mode"] = "open_loop"
        generation["preserve_request_order"] = True
        generation["include_prompt_text"] = True
        generation["response_text_max_chars"] = 1048576
        decoding = generation.setdefault("decoding", {})
        decoding.setdefault("max_tokens", payload["generation"]["decoding"]["max_tokens"])
        decoding["max_tokens_policy"] = "model_context_minus_prompt_buffer"
        decoding["prompt_token_buffer"] = 128
        decoding.setdefault("top_k", 20)
        decoding.setdefault("min_p", 0.0)
        decoding.setdefault("n", 1)
        extra_body = decoding.setdefault("extra_body", {})
        if model in PRE_2507_QWEN3_NON_THINKING:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if model in REASONING_SAMPLING_MODELS:
            decoding["temperature"] = 0.6
            decoding["top_p"] = 0.95
        else:
            decoding["temperature"] = 0.0
            decoding["top_p"] = 1.0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {output}")
    return 0


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _response_id(model: str) -> str:
    slug = (
        model.lower()
        .replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
    )
    return f"{slug}-mmlu-pro-output-length-responses"


if __name__ == "__main__":
    raise SystemExit(main())
