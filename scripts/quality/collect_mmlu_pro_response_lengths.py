#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RUN_ROOT = Path("results/quality/h100-quality-mmlu-pro-output-length-responses-000")
DEFAULT_OUTPUT_DIR = Path("results/quality/workload_length_distributions")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect real MMLU-Pro response output token length stats."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = _resolve(args.run_root, repo_root=repo_root)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _model_response_dirs(run_root)
    records = _collect_records(model_dirs)
    if not records:
        raise SystemExit(f"no response rows found under {run_root / 'responses'}")

    by_request = _group_by_request(records)
    skipped_dirs = _incomplete_response_dirs(run_root, set(model_dirs))

    request_jsonl = output_dir / "mmlu_pro_output_lengths_by_request.jsonl"
    summary_by_model_csv = output_dir / "mmlu_pro_output_length_summary_by_model.csv"
    summary_by_subject_csv = output_dir / "mmlu_pro_output_length_summary_by_model_subject.csv"
    summary_by_bucket_csv = (
        output_dir / "mmlu_pro_output_length_summary_by_model_prompt_bucket.csv"
    )
    manifest_path = output_dir / "mmlu_pro_output_length_manifest.json"

    _write_jsonl(request_jsonl, by_request)
    _write_summary_csv(
        summary_by_model_csv,
        records,
        group_fields=["model_slug", "model"],
    )
    _write_summary_csv(
        summary_by_subject_csv,
        records,
        group_fields=["model_slug", "model", "subject"],
    )
    _write_summary_csv(
        summary_by_bucket_csv,
        records,
        group_fields=["model_slug", "model", "prompt_length_bucket"],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "responses_dir": str(run_root / "responses"),
                "finished_model_count": len(model_dirs),
                "finished_models": sorted(model_dirs),
                "incomplete_or_pending_response_dirs": skipped_dirs,
                "response_rows": len(records),
                "request_rows": len(by_request),
                "expected_requests_per_finished_model": _mode_count(
                    [count for count in _counts_by(records, "model_slug").values()]
                ),
                "length_field": "actual_output_len",
                "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
                "outputs": {
                    "request_jsonl": str(request_jsonl.resolve()),
                    "summary_by_model_csv": str(summary_by_model_csv.resolve()),
                    "summary_by_subject_csv": str(summary_by_subject_csv.resolve()),
                    "summary_by_prompt_bucket_csv": str(summary_by_bucket_csv.resolve()),
                },
                "note": (
                    "This file reflects only model response directories that already "
                    "have a top-level responses.jsonl. Rerun this script after more "
                    "MMLU-Pro output-length jobs finish."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"wrote {len(by_request)} request rows to {request_jsonl}")
    print(f"wrote model summary for {len(model_dirs)} models to {summary_by_model_csv}")
    if skipped_dirs:
        print(f"skipped {len(skipped_dirs)} incomplete/pending response dirs")
    return 0


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _model_response_dirs(run_root: Path) -> dict[str, Path]:
    responses_dir = run_root / "responses"
    if not responses_dir.is_dir():
        raise FileNotFoundError(responses_dir)
    model_dirs = {}
    for child in sorted(responses_dir.iterdir()):
        if child.is_dir() and (child / "responses.jsonl").is_file():
            model_dirs[child.name] = child
    if not model_dirs:
        raise FileNotFoundError(f"no responses.jsonl files under {responses_dir}")
    return model_dirs


def _incomplete_response_dirs(run_root: Path, complete_model_slugs: set[str]) -> list[str]:
    responses_dir = run_root / "responses"
    if not responses_dir.is_dir():
        return []
    return [
        child.name
        for child in sorted(responses_dir.iterdir())
        if child.is_dir() and child.name not in complete_model_slugs
    ]


def _collect_records(model_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    records = []
    for model_slug, model_dir in sorted(model_dirs.items()):
        responses_path = model_dir / "responses.jsonl"
        with responses_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                metadata = payload.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                prompt_bucket = (
                    metadata.get("mmlu_pro_prompt_length_bucket")
                    or payload.get("prompt_length_bucket")
                    or ""
                )
                records.append(
                    {
                        "model_slug": model_slug,
                        "model": payload.get("model", ""),
                        "request_id": payload.get("request_id")
                        or metadata.get("sample_id")
                        or metadata.get("record_id")
                        or f"{model_slug}:{line_number}",
                        "sample_id": metadata.get("sample_id", ""),
                        "record_id": metadata.get("record_id", ""),
                        "source_index": metadata.get("source_index", ""),
                        "subject": metadata.get("subject", ""),
                        "prompt_length_bucket": prompt_bucket,
                        "length_stratum": metadata.get("mmlu_pro_length_stratum", ""),
                        "answer_label": metadata.get("answer_label", ""),
                        "prompt_token_count": _int_value(metadata.get("prompt_token_count")),
                        "target_token_count": _int_value(metadata.get("target_token_count")),
                        "actual_output_len": _int_value(payload.get("actual_output_len")),
                        "expected_output_len": _int_value(payload.get("expected_output_len")),
                        "finish_reason": payload.get("finish_reason"),
                        "success": payload.get("success"),
                        "response_text_truncated": payload.get("response_text_truncated"),
                        "error": payload.get("error"),
                        "source_path": str(responses_path.resolve()),
                    }
                )
    return records


def _group_by_request(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["request_id"])].append(record)

    rows = []
    for request_id, values in sorted(grouped.items()):
        first = values[0]
        rows.append(
            {
                "benchmark": "mmlu_pro",
                "request_id": request_id,
                "sample_id": first["sample_id"],
                "record_id": first["record_id"],
                "source_index": first["source_index"],
                "subject": first["subject"],
                "prompt_length_bucket": first["prompt_length_bucket"],
                "length_stratum": first["length_stratum"],
                "answer_label": first["answer_label"],
                "prompt_token_count": first["prompt_token_count"],
                "target_token_count": first["target_token_count"],
                "model_count": len(values),
                "outputs": [
                    {
                        "model_slug": value["model_slug"],
                        "model": value["model"],
                        "actual_output_len": value["actual_output_len"],
                        "expected_output_len": value["expected_output_len"],
                        "finish_reason": value["finish_reason"],
                        "success": value["success"],
                        "response_text_truncated": value["response_text_truncated"],
                        "error": value["error"],
                    }
                    for value in sorted(values, key=lambda item: str(item["model_slug"]))
                ],
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_summary_csv(
    path: Path, records: list[dict[str, Any]], *, group_fields: list[str]
) -> None:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    total_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    missing_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    cap_hits: dict[tuple[Any, ...], int] = defaultdict(int)
    truncated_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    error_counts: dict[tuple[Any, ...], int] = defaultdict(int)

    for record in records:
        key = tuple(record.get(field, "") for field in group_fields)
        total_counts[key] += 1
        if record.get("error"):
            error_counts[key] += 1
        if record.get("response_text_truncated"):
            truncated_counts[key] += 1
        actual_output_len = record.get("actual_output_len")
        if actual_output_len is None:
            missing_counts[key] += 1
            continue
        actual_output_len = int(actual_output_len)
        grouped[key].append(actual_output_len)
        expected_output_len = record.get("expected_output_len")
        if expected_output_len is not None and actual_output_len >= int(expected_output_len):
            cap_hits[key] += 1

    fieldnames = [
        *group_fields,
        "response_count",
        "valid_output_len_count",
        "missing_output_len_count",
        "error_count",
        "response_text_truncated_count",
        "expected_cap_hits",
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
        for key in sorted(total_counts):
            lengths = grouped.get(key, [])
            stats = _stats("output_tokens", lengths) if lengths else _empty_stats("output_tokens")
            writer.writerow(
                {
                    **dict(zip(group_fields, key, strict=True)),
                    "response_count": total_counts[key],
                    "valid_output_len_count": len(lengths),
                    "missing_output_len_count": missing_counts[key],
                    "error_count": error_counts[key],
                    "response_text_truncated_count": truncated_counts[key],
                    "expected_cap_hits": cap_hits[key],
                    **stats,
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


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _counts_by(records: list[dict[str, Any]], field: str) -> dict[Any, int]:
    counts: dict[Any, int] = defaultdict(int)
    for record in records:
        counts[record.get(field, "")] += 1
    return counts


def _mode_count(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


if __name__ == "__main__":
    raise SystemExit(main())
