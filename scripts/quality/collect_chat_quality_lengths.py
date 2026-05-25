#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MATERIALIZATION_DIR = Path("experiments/quality/sharegpt_wildchat_10k")
DEFAULT_OUTPUT_DIR = Path("results/quality/workload_length_distributions")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ShareGPT/WildChat quality prompt and reference assistant "
            "lengths. Prompt lengths come from materialized rows; assistant "
            "lengths are recovered from the raw datasets and tokenized."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--source",
        action="append",
        choices=("sharegpt", "wildchat"),
        help="Source to recover. Defaults to both ShareGPT and WildChat.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    materialization_dir = _resolve(args.materialization_dir, repo_root=repo_root)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _read_json(materialization_dir / "materialization_report.json")
    tokenizer_name = args.tokenizer or str(report["tokenizer"])
    selected_rows = _read_selected_rows(materialization_dir)
    tokenizer = _load_tokenizer(tokenizer_name)
    selected_sources = set(args.source or ("sharegpt", "wildchat"))

    sharegpt_lengths = {}
    wildchat_lengths = {}
    if "sharegpt" in selected_sources:
        sharegpt_lengths = _sharegpt_output_lengths(
            selected_rows,
            report=report,
            tokenizer=tokenizer,
        )
    if "wildchat" in selected_sources:
        wildchat_lengths = _wildchat_output_lengths(
            selected_rows,
            report=report,
            tokenizer=tokenizer,
        )
    output_lengths = {**sharegpt_lengths, **wildchat_lengths}

    rows = []
    missing_output_lengths = 0
    for row in selected_rows:
        if row["metadata"].get("source") not in selected_sources:
            continue
        key = _row_key(row)
        output_tokens = output_lengths.get(key)
        if output_tokens is None:
            missing_output_lengths += 1
        metadata = row["metadata"]
        rows.append(
            {
                "workload_group": "chat_quality",
                "workload_name": "sharegpt_wildchat_10k",
                "source": metadata.get("source", ""),
                "prompt_length_bucket": metadata.get("prompt_length_bucket", ""),
                "shard": metadata.get("shard_id", ""),
                "within_shard_index": metadata.get("within_shard_index", ""),
                "sample_id": metadata.get("sample_id", ""),
                "request_id": metadata.get("request_id", ""),
                "session_id": metadata.get("session_id", ""),
                "turn_index": metadata.get("turn_index", ""),
                "source_row_index": metadata.get("source_row_index", ""),
                "input_tokens": int(row["prompt_len"]),
                "output_tokens": output_tokens if output_tokens is not None else "",
                "expected_output_len": row.get("expected_output_len", ""),
                "tokenizer": tokenizer_name,
            }
        )

    rows_path = output_dir / "chat_quality_request_lengths.csv"
    summary_path = output_dir / "chat_quality_length_summary.csv"
    manifest_path = output_dir / "chat_quality_length_manifest.json"
    _write_rows_csv(rows_path, rows)
    _write_summary_csv(summary_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "materialization_dir": str(materialization_dir),
                "tokenizer": tokenizer_name,
                "sources": sorted(selected_sources),
                "row_count": len(rows),
                "missing_output_length_count": missing_output_lengths,
                "request_lengths_csv": str(rows_path.resolve()),
                "summary_csv": str(summary_path.resolve()),
                "note": (
                    "input_tokens are cached materialized prompt_len values; "
                    "output_tokens are tokenized raw assistant turns recovered by "
                    "source_row_index from the original ShareGPT/WildChat sources."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {len(rows)} rows to {rows_path}")
    print(f"missing output lengths: {missing_output_lengths}")
    return 0 if missing_output_lengths == 0 else 2


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_selected_rows(materialization_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for shard_path in sorted((materialization_dir / "shards").glob("*.runner.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload.get("metadata"), dict):
                    raise ValueError(f"{shard_path} row missing metadata")
                rows.append(payload)
    if not rows:
        raise ValueError(f"no runner rows found under {materialization_dir}")
    return rows


def _load_tokenizer(tokenizer_name: str) -> Any:
    from llm_mst_finder.workload import resolve_tokenizer

    return resolve_tokenizer(tokenizer_name)


def _sharegpt_output_lengths(
    selected_rows: list[dict[str, Any]],
    *,
    report: dict[str, Any],
    tokenizer: Any,
) -> dict[tuple[str, int, str], int]:
    selected = [
        row
        for row in selected_rows
        if row["metadata"].get("source") == "sharegpt"
    ]
    if not selected:
        return {}
    dataset_path = Path(report["sources"]["sharegpt"]["dataset_path"])
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_source_index = {int(row["metadata"]["source_row_index"]) for row in selected}
    lengths = {}
    for source_index in sorted(by_source_index):
        row_payload = payload[source_index]
        turns = _conversation_turns(row_payload.get("conversations"))
        if len(turns) < 2 or turns[0][0] != "user" or turns[1][0] != "assistant":
            continue
        assistant = turns[1][1]
        output_len = len(tokenizer.encode(assistant))
        for selected_row in selected:
            metadata = selected_row["metadata"]
            if int(metadata["source_row_index"]) == source_index:
                lengths[_row_key(selected_row)] = output_len
    return lengths


def _wildchat_output_lengths(
    selected_rows: list[dict[str, Any]],
    *,
    report: dict[str, Any],
    tokenizer: Any,
) -> dict[tuple[str, int, str], int]:
    selected = [
        row
        for row in selected_rows
        if row["metadata"].get("source") == "wildchat"
    ]
    if not selected:
        return {}
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to recover WildChat output lengths") from exc

    source_report = report["sources"]["wildchat"]
    dataset_path = source_report["dataset_path"]
    dataset_split = source_report["dataset_split"]
    scan_limit = int(source_report["scanned_rows"])
    selected_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_index[int(row["metadata"]["source_row_index"])].append(row)
    max_needed_index = max(selected_by_index)

    lengths = {}
    rows = load_dataset(dataset_path, split=dataset_split, streaming=True)
    for row_index, row_payload in enumerate(rows):
        if row_index > max_needed_index or row_index >= scan_limit:
            break
        if row_index not in selected_by_index:
            continue
        conversations = row_payload.get("conversation")
        turns = _conversation_turns(conversations)
        if len(turns) < 2 or turns[0][0] != "user" or turns[1][0] != "assistant":
            continue
        assistant = turns[1][1]
        output_len = len(tokenizer.encode(assistant))
        for selected_row in selected_by_index[row_index]:
            lengths[_row_key(selected_row)] = output_len
    return lengths


def _conversation_turns(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    turns: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("from", item.get("role"))
        text = item.get("value", item.get("content", item.get("text")))
        if not isinstance(role, str) or not isinstance(text, str) or not text:
            continue
        normalized_role = _normalize_role(role)
        if normalized_role is None:
            continue
        turns.append((normalized_role, text))
    return turns


def _normalize_role(role: str) -> str | None:
    lowered = role.lower()
    if lowered in {"human", "user"}:
        return "user"
    if lowered in {"gpt", "assistant"}:
        return "assistant"
    if lowered == "system":
        return "system"
    return None


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    metadata = row["metadata"]
    return (
        str(metadata.get("source")),
        int(metadata.get("source_row_index")),
        str(metadata.get("content_hash")),
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "workload_group",
        "workload_name",
        "source",
        "prompt_length_bucket",
        "shard",
        "within_shard_index",
        "sample_id",
        "request_id",
        "session_id",
        "turn_index",
        "source_row_index",
        "input_tokens",
        "output_tokens",
        "expected_output_len",
        "tokenizer",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["prompt_length_bucket"]))].append(row)
    fieldnames = [
        "source",
        "prompt_length_bucket",
        "count",
        "missing_output_len_count",
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (source, bucket), values in sorted(grouped.items()):
            input_values = [int(row["input_tokens"]) for row in values]
            output_values = [
                int(row["output_tokens"])
                for row in values
                if row["output_tokens"] != ""
            ]
            writer.writerow(
                {
                    "source": source,
                    "prompt_length_bucket": bucket,
                    "count": len(values),
                    "missing_output_len_count": len(values) - len(output_values),
                    **_stats("input_tokens", input_values),
                    **(_stats("output_tokens", output_values) if output_values else _empty_stats("output_tokens")),
                }
            )


def _stats(prefix: str, values: list[int]) -> dict[str, Any]:
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean": f"{mean(sorted_values):.6f}",
        f"{prefix}_p50": _percentile(sorted_values, 0.50),
        f"{prefix}_p90": _percentile(sorted_values, 0.90),
        f"{prefix}_p95": _percentile(sorted_values, 0.95),
        f"{prefix}_p99": _percentile(sorted_values, 0.99),
        f"{prefix}_max": sorted_values[-1],
    }


def _empty_stats(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": "",
        f"{prefix}_p50": "",
        f"{prefix}_p90": "",
        f"{prefix}_p95": "",
        f"{prefix}_p99": "",
        f"{prefix}_max": "",
    }


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


if __name__ == "__main__":
    raise SystemExit(main())
