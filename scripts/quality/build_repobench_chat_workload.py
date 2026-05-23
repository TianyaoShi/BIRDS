#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from llm_mst_finder.workload import resolve_tokenizer


SYSTEM_PROMPT = (
    "You are a code completion engine.\n"
    "Return only the exact code continuation after <CURSOR>.\n"
    "Do not return Markdown, code fences, comments, explanations, XML tags, "
    "or natural language.\n"
    "Do not repeat the provided context or prefix.\n"
    "Complete the current statement or line only.\n"
    "Begin immediately with code."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite materialized RepoBench plain-prefix shards into chat-completion prompts."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/code_workloads/repobench_python_java_aggregate_cache_realistic"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/code_workloads/repobench_python_java_aggregate_code_chat_completion"),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    tokenizer = resolve_tokenizer(args.tokenizer)
    source_shards = sorted((args.source / "shards").glob("*.runner.jsonl"))
    if not source_shards:
        raise SystemExit(f"no source runner shards found under {args.source / 'shards'}")

    (args.output / "shards").mkdir(parents=True, exist_ok=True)
    (args.output / "workload_yamls").mkdir(parents=True, exist_ok=True)

    total_rows = 0
    rows_by_shard: list[list[dict[str, Any]]] = []
    language_counts: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    target_tokens: list[int] = []
    content_hashes: set[str] = set()

    for shard_index, source_path in enumerate(source_shards):
        shard_id = f"shard_{shard_index:03d}"
        shard_rows: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total_rows += 1
                rewritten = rewrite_row(json.loads(line), tokenizer=tokenizer)
                meta = rewritten["metadata"]
                shard_rows.append(rewritten)
                language_counts[str(meta.get("language", "unknown"))] += 1
                prompt_tokens.append(int(meta["prompt_token_count"]))
                target_tokens.append(int(meta["target_token_count"]))
                content_hashes.add(str(meta["content_hash"]))
        rows_by_shard.append(shard_rows)
        write_jsonl(args.output / "shards" / f"{shard_id}.runner.jsonl", shard_rows)
        write_workload_yaml(
            args.output / "workload_yamls" / f"{shard_id}.yaml",
            name=f"repobench_python_java_aggregate_code_chat_completion-{shard_id}",
            shard_id=shard_id,
            shard_size=len(shard_rows),
        )

    write_json(
        args.output / "shards_manifest.json",
        {
            "workload_name": "repobench_python_java_aggregate_code_chat_completion",
            "source_workload": str(args.source.resolve()),
            "dataset": "repobench",
            "task": "aggregate",
            "prompt_template": "code_chat_completion",
            "num_shards": len(rows_by_shard),
            "samples_per_shard": 8000,
            "shards": [
                {
                    "shard_id": f"shard_{index:03d}",
                    "num_samples": len(rows),
                    "path": f"shards/shard_{index:03d}.runner.jsonl",
                    "workload_yaml_path": f"workload_yamls/shard_{index:03d}.yaml",
                }
                for index, rows in enumerate(rows_by_shard)
            ],
            "language_counts": dict(language_counts),
            "prompt_tokens": summarize(prompt_tokens),
            "target_tokens": summarize(target_tokens),
            "unique_content_hashes": len(content_hashes),
            "unique_sample_ids": total_rows,
        },
    )
    write_json(
        args.output / "materialization_report.json",
        {
            "workload_name": "repobench_python_java_aggregate_code_chat_completion",
            "source_workload": str(args.source.resolve()),
            "dataset": "repobench",
            "task": "aggregate",
            "prompt_template": "code_chat_completion",
            "output_dir": str(args.output.resolve()),
            "tokenizer": {"name": args.tokenizer, "fallback_used": False},
            "rows": {"total": total_rows, "materialized": total_rows, "drops": {}},
            "language_counts": dict(language_counts),
            "prompt_tokens": summarize(prompt_tokens),
            "target_tokens": summarize(target_tokens),
            "sampling": {
                "unique_sample_ids": total_rows,
                "expanded_sample_count": total_rows,
                "repeat_factor": 1.0,
                "repeat_policy": None,
            },
        },
    )
    (args.output / "materialization_config.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "repobench_python_java_aggregate_code_chat_completion",
                "source_workload": str(args.source.resolve()),
                "prompt_template": "code_chat_completion",
                "tokenization": {"tokenizer": args.tokenizer},
                "sharding": {"output_dir": str(args.output), "samples_per_shard": 8000},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output), "num_samples": total_rows, "num_shards": len(rows_by_shard)}, indent=2))
    return 0


def rewrite_row(row: dict[str, Any], *, tokenizer: Any) -> dict[str, Any]:
    meta = dict(row.get("metadata") or {})
    prompt = str(row.get("prompt") or "")
    repository_context, current_file_prefix = split_repobench_plain_prompt(prompt)
    language = str(meta.get("language") or "unknown")
    file_path = str(meta.get("file_path") or "unknown")
    chat_prompt = render_prompt(
        repository_context=repository_context,
        current_file_prefix=current_file_prefix,
        file_path=file_path,
        language=language,
    )
    prompt_token_count = len(tokenizer.encode(chat_prompt))
    meta["prompt_template"] = "code_chat_completion"
    meta["system_prompt"] = SYSTEM_PROMPT
    meta["current_file_prefix"] = current_file_prefix
    meta["repository_context"] = repository_context
    meta["prompt_token_count"] = prompt_token_count
    meta["content_hash"] = hash_text(chat_prompt)
    rewritten = dict(row)
    rewritten["prompt"] = chat_prompt
    rewritten["prompt_len"] = prompt_token_count
    rewritten["expected_output_len"] = 64
    rewritten["metadata"] = meta
    return rewritten


def split_repobench_plain_prompt(prompt: str) -> tuple[str, str]:
    marker = "Relevant repository context:\n"
    if not prompt.startswith(marker):
        return "", prompt.rstrip()
    body = prompt[len(marker):]
    current_start = body.rfind("\n\n")
    if current_start < 0:
        return "", body.rstrip()
    return body[:current_start].strip(), body[current_start + 2 :].rstrip()


def render_prompt(
    *,
    repository_context: str,
    current_file_prefix: str,
    file_path: str,
    language: str,
) -> str:
    parts = ["Complete the next line of code at <CURSOR>."]
    if repository_context:
        parts.append(
            "<REPOSITORY_CONTEXT>\n"
            f"{repository_context}\n"
            "</REPOSITORY_CONTEXT>"
        )
    parts.append(
        f'<TARGET_FILE path="{file_path}" language="{language}">\n'
        f"{current_file_prefix}<CURSOR>\n"
        "</TARGET_FILE>"
    )
    parts.append("Return only the next line after <CURSOR>.")
    return "\n\n".join(parts)


def write_workload_yaml(path: Path, *, name: str, shard_id: str, shard_size: int) -> None:
    payload = {
        "name": name,
        "dataset": {"type": "jsonl", "path": f"../shards/{shard_id}.runner.jsonl"},
        "sampling": {
            "seed": 42,
            "num_requests": shard_size,
            "entry_selection": "sequential",
            "prompt_len": {"mode": "from_dataset"},
            "output_len": {"mode": "fixed", "value": 64},
        },
        "request": {
            "stream": True,
            "temperature": 0.0,
            "ignore_eos": False,
            "top_p": 0.95,
            "extra_body": {
                "top_k": 64,
                "min_p": 0.0,
                "repetition_penalty": 1.05,
                "frequency_penalty": 0.05,
                "seed": 1,
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
            },
        },
        "context_policy": {
            "max_model_len": 32768,
            "tokenizer_source": "vllm_model_config",
            "unsafe_allow_workload_tokenizer_for_real_datasets": True,
            "over_limit": "truncate_prompt",
            "truncation_side": "left",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def summarize(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": percentile(ordered, 0.5),
        "p90": percentile(ordered, 0.9),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(ordered: list[int], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
