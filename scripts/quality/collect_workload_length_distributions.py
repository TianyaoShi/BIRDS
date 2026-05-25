#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_TARGETS = (
    "experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic",
    "experiments/code_workloads/repobench_python_java_aggregate_cache_realistic_8k_drop",
    "experiments/longbench_workloads/benchmark_original",
    "experiments/reasoning_workloads/mmlu_pro_reasoning",
    "experiments/reasoning_workloads/supergpqa_reasoning",
    "experiments/reasoning_workloads/supergpqa_hard_reasoning",
)

LONG_BENCH_OFFICIAL_POSTFIX = "_original_official_qwen3_8b"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect cached per-request input/output token lengths from materialized "
            "BioLLM workload shards."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/quality/workload_length_distributions"),
    )
    parser.add_argument(
        "--target",
        action="append",
        help="Workload directory or runner JSONL path. Defaults to the quality benchmark workload set.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    targets = tuple(args.target or DEFAULT_TARGETS)

    rows: list[dict[str, Any]] = []
    for target in targets:
        target_path = _resolve(Path(target), repo_root=repo_root)
        for shard_path in _runner_jsonl_paths(target_path):
            if _workload_group(shard_path) == "longbench":
                workload_name = _workload_dir_for_shard(shard_path).name
                if not workload_name.endswith(LONG_BENCH_OFFICIAL_POSTFIX):
                    continue
            rows.extend(_rows_from_shard(shard_path, repo_root=repo_root))

    if not rows:
        raise SystemExit("no materialized runner rows found")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "request_lengths.csv"
    summary_csv_path = output_dir / "workload_length_summary.csv"
    summary_json_path = output_dir / "workload_length_summary.json"
    manifest_path = output_dir / "manifest.json"

    _write_rows_csv(rows_path, rows)
    summary = _summaries(rows)
    _write_summary_csv(summary_csv_path, summary)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "targets": list(targets),
                "row_count": len(rows),
                "request_lengths_csv": str(rows_path.resolve()),
                "summary_csv": str(summary_csv_path.resolve()),
                "summary_json": str(summary_json_path.resolve()),
                "note": (
                    "Lengths are read from cached materialization metadata: "
                    "metadata.prompt_token_count, metadata.target_token_count, "
                    "and row.expected_output_len. No tokenizer is run by this collector."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {len(rows)} rows to {rows_path}")
    print(f"wrote summary to {summary_csv_path}")
    return 0


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _runner_jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.name.endswith(".runner.jsonl"):
            return [path]
        raise ValueError(f"target file is not a runner JSONL: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.rglob("*.runner.jsonl"))


def _rows_from_shard(shard_path: Path, *, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workload_dir = _workload_dir_for_shard(shard_path)
    workload_name = workload_dir.name
    with shard_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            prompt_tokens = _int_value(metadata.get("prompt_token_count"))
            target_tokens = _int_value(metadata.get("target_token_count"))
            expected_output_len = _int_value(payload.get("expected_output_len"))
            if prompt_tokens is None or target_tokens is None:
                raise ValueError(
                    f"{shard_path}:{index + 1} missing cached prompt/target token counts"
                )
            rows.append(
                {
                    "workload_group": _workload_group(shard_path),
                    "workload_name": workload_name,
                    "shard": shard_path.stem.replace(".runner", ""),
                    "row_index": index,
                    "sample_id": metadata.get("sample_id", ""),
                    "dataset": metadata.get("dataset", ""),
                    "task": metadata.get("task", ""),
                    "profile": metadata.get("profile", ""),
                    "difficulty": metadata.get("difficulty", ""),
                    "language": metadata.get("language", ""),
                    "input_tokens": prompt_tokens,
                    "output_tokens": target_tokens,
                    "expected_output_len": expected_output_len if expected_output_len is not None else "",
                    "cached_prompt_token_count": prompt_tokens,
                    "cached_target_token_count": target_tokens,
                    "source_path": str(shard_path.resolve()),
                    "source_path_relative": _relative_to_repo(shard_path, repo_root=repo_root),
                }
            )
    return rows


def _workload_dir_for_shard(shard_path: Path) -> Path:
    parent = shard_path.parent
    if parent.name == "shards":
        return parent.parent
    return parent


def _workload_group(shard_path: Path) -> str:
    parts = shard_path.parts
    if "code_workloads" in parts:
        return "code"
    if "longbench_workloads" in parts:
        return "longbench"
    if "reasoning_workloads" in parts:
        return "reasoning"
    return "unknown"


def _relative_to_repo(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "workload_group",
        "workload_name",
        "shard",
        "row_index",
        "sample_id",
        "dataset",
        "task",
        "profile",
        "difficulty",
        "language",
        "input_tokens",
        "output_tokens",
        "expected_output_len",
        "cached_prompt_token_count",
        "cached_target_token_count",
        "source_path",
        "source_path_relative",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload_group"]), str(row["workload_name"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (group, workload), values in sorted(grouped.items()):
        input_values = [int(row["input_tokens"]) for row in values]
        output_values = [int(row["output_tokens"]) for row in values]
        expected_values = [
            int(row["expected_output_len"])
            for row in values
            if row["expected_output_len"] != ""
        ]
        summary_rows.append(
            {
                "workload_group": group,
                "workload_name": workload,
                "count": len(values),
                **_stats("input_tokens", input_values),
                **_stats("output_tokens", output_values),
                **_stats("expected_output_len", expected_values),
            }
        )
    return summary_rows


def _stats(prefix: str, values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_p50": "",
            f"{prefix}_p90": "",
            f"{prefix}_p95": "",
            f"{prefix}_p99": "",
            f"{prefix}_max": "",
        }
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean": f"{mean(sorted_values):.6f}",
        f"{prefix}_p50": _percentile(sorted_values, 0.50),
        f"{prefix}_p90": _percentile(sorted_values, 0.90),
        f"{prefix}_p95": _percentile(sorted_values, 0.95),
        f"{prefix}_p99": _percentile(sorted_values, 0.99),
        f"{prefix}_max": sorted_values[-1],
    }


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "workload_group",
        "workload_name",
        "count",
        "input_tokens_mean",
        "input_tokens_p50",
        "input_tokens_p90",
        "input_tokens_p95",
        "input_tokens_p99",
        "input_tokens_max",
        "output_tokens_mean",
        "output_tokens_p50",
        "output_tokens_p90",
        "output_tokens_p95",
        "output_tokens_p99",
        "output_tokens_max",
        "expected_output_len_mean",
        "expected_output_len_p50",
        "expected_output_len_p90",
        "expected_output_len_p95",
        "expected_output_len_p99",
        "expected_output_len_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
