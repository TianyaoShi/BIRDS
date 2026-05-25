#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SOURCE_DIR = Path("experiments/reasoning_workloads/mmlu_pro_reasoning")
DEFAULT_OUTPUT_DIR = Path("experiments/reasoning_workloads/mmlu_pro_output_length_sample_1600")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a stratified MMLU-Pro sample for output-length profiling."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=1600)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_dir = _resolve(args.source_dir, repo_root=repo_root)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"refusing to overwrite non-empty output dir: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    (output_dir / "workload_yamls").mkdir(parents=True, exist_ok=True)

    rows = _load_rows(source_dir)
    sampled_rows, allocation = _stratified_sample(
        rows,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    shards = _balanced_shards(sampled_rows, shard_count=args.shards, seed=args.seed)

    shard_entries = []
    for shard_index, shard_rows in enumerate(shards):
        shard_id = f"shard_{shard_index:03d}"
        shard_path = output_dir / "shards" / f"{shard_id}.runner.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for within_index, row in enumerate(shard_rows):
                payload = dict(row)
                metadata = dict(payload["metadata"])
                metadata["source_mmlu_pro_shard_id"] = metadata.get("shard_id")
                metadata["source_mmlu_pro_sample_id"] = metadata.get("sample_id")
                metadata["shard_id"] = shard_id
                metadata["within_shard_index"] = within_index
                metadata["mmlu_pro_length_sample_seed"] = args.seed
                metadata["mmlu_pro_length_stratum"] = _stratum_key(row)
                metadata["mmlu_pro_prompt_length_bucket"] = _prompt_length_bucket(row)
                metadata["mmlu_pro_difficulty_proxy"] = "prompt_length_bucket"
                payload["metadata"] = metadata
                payload["expected_output_len"] = args.max_output_tokens
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        workload_path = output_dir / "workload_yamls" / f"{shard_id}.yaml"
        workload_path.write_text(
            yaml.safe_dump(
                _workload_yaml(
                    shard_id=shard_id,
                    shard_size=len(shard_rows),
                    max_output_tokens=args.max_output_tokens,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        shard_entries.append(
            {
                "shard_id": shard_id,
                "num_requests": len(shard_rows),
                "path": str(shard_path.relative_to(output_dir)),
                "workload_yaml_path": str(workload_path.relative_to(output_dir)),
                "subject_counts": dict(Counter(row["metadata"].get("subject") for row in shard_rows)),
                "prompt_length_bucket_counts": dict(Counter(_prompt_length_bucket(row) for row in shard_rows)),
            }
        )

    manifest = {
        "name": "mmlu_pro_output_length_sample_1600",
        "source_dir": str(source_dir),
        "sample_size": len(sampled_rows),
        "seed": args.seed,
        "stratification": {
            "primary": "metadata.subject",
            "secondary": "prompt_length_bucket",
            "difficulty_field_available": False,
            "difficulty_proxy": "prompt_length_bucket",
            "prompt_length_buckets": {"short": "<100", "medium": "100-512", "long": ">512"},
        },
        "allocation": allocation,
        "shards": shard_entries,
        "subject_counts": dict(Counter(row["metadata"].get("subject") for row in sampled_rows)),
        "prompt_length_bucket_counts": dict(Counter(_prompt_length_bucket(row) for row in sampled_rows)),
        "stratum_counts": dict(Counter(_stratum_key(row) for row in sampled_rows)),
    }
    (output_dir / "shards_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "materialization_report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "materialization_config.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "mmlu_pro_output_length_sample_1600",
                "source_dir": str(source_dir),
                "sample_size": args.sample_size,
                "seed": args.seed,
                "stratification": ["metadata.subject", "prompt_length_bucket"],
                "difficulty_proxy": "prompt_length_bucket",
                "shards": args.shards,
                "max_output_tokens": args.max_output_tokens,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(sampled_rows)} sampled rows to {output_dir}")
    return 0


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for shard_path in sorted((source_dir / "shards").glob("*.runner.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no source rows found under {source_dir}")
    return rows


def _stratified_sample(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum_key(row)].append(row)
    allocation = _proportional_allocation({key: len(value) for key, value in groups.items()}, sample_size)
    rng = random.Random(seed)
    sampled = []
    for key, group_rows in sorted(groups.items()):
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        sampled.extend(shuffled[: allocation[key]])
    rng.shuffle(sampled)
    return sampled, allocation


def _proportional_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    floors = {key: int(target * count / total) for key, count in counts.items()}
    for key, count in counts.items():
        if count > 0 and floors[key] == 0:
            floors[key] = 1
    while sum(floors.values()) > target:
        key = max((key for key in floors if floors[key] > 1), key=lambda item: floors[item])
        floors[key] -= 1
    remainders = sorted(
        ((target * counts[key] / total - floors[key], key) for key in counts),
        reverse=True,
    )
    index = 0
    while sum(floors.values()) < target:
        _, key = remainders[index % len(remainders)]
        if floors[key] < counts[key]:
            floors[key] += 1
        index += 1
    return dict(sorted(floors.items()))


def _balanced_shards(rows: list[dict[str, Any]], *, shard_count: int, seed: int) -> list[list[dict[str, Any]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[_stratum_key(row)].append(row)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for stratum, stratum_rows in sorted(by_stratum.items()):
        shuffled = list(stratum_rows)
        random.Random(f"{seed}:{stratum}").shuffle(shuffled)
        for index, row in enumerate(shuffled):
            shards[index % shard_count].append(row)
    for shard_index, shard_rows in enumerate(shards):
        random.Random(f"{seed}:shard:{shard_index}").shuffle(shard_rows)
    return shards


def _stratum_key(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return f"{metadata.get('subject', 'unknown')}:{_prompt_length_bucket(row)}"


def _prompt_length_bucket(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    prompt_tokens = int(metadata.get("prompt_token_count") or row.get("prompt_len") or 0)
    if prompt_tokens < 100:
        return "short"
    if prompt_tokens <= 512:
        return "medium"
    return "long"


def _workload_yaml(*, shard_id: str, shard_size: int, max_output_tokens: int) -> dict[str, Any]:
    return {
        "name": f"mmlu_pro_output_length_sample_1600-{shard_id}",
        "dataset": {
            "type": "jsonl",
            "path": f"../shards/{shard_id}.runner.jsonl",
        },
        "sampling": {
            "seed": 42,
            "num_requests": shard_size,
            "entry_selection": "sequential",
            "prompt_len": {"mode": "from_dataset"},
            "output_len": {"mode": "natural_until_eos", "max_tokens": max_output_tokens},
        },
        "request": {
            "stream": True,
            "temperature": 0.0,
            "ignore_eos": False,
        },
        "context_policy": {
            "max_model_len": 32768,
            "tokenizer_source": "vllm_model_config",
            "over_limit": "fail",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
