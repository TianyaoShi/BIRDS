#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_HARD_RUN_ROOT = Path(
    "results/quality/h100-quality-supergpqa-hard-original-responses-002-16k"
)
DEFAULT_EASY_MEDIUM_RUN_ROOT = Path(
    "results/quality/h100-quality-supergpqa-easy-medium-original-responses-001"
)
DEFAULT_OUTPUT_DIR = Path("results/quality/workload_length_distributions")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect real SuperGPQA response output token lengths by request."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--hard-run-root", type=Path, default=DEFAULT_HARD_RUN_ROOT)
    parser.add_argument(
        "--easy-medium-run-root", type=Path, default=DEFAULT_EASY_MEDIUM_RUN_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    hard_run_root = _resolve(args.hard_run_root, repo_root=repo_root)
    easy_medium_run_root = _resolve(args.easy_medium_run_root, repo_root=repo_root)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_models = _model_response_dirs(hard_run_root)
    easy_medium_models = _model_response_dirs(easy_medium_run_root)
    common_model_slugs = sorted(set(hard_models) & set(easy_medium_models))
    if not common_model_slugs:
        raise SystemExit("no models have both hard and easy-medium responses")

    hard_records = _collect_records(
        model_dirs=hard_models,
        benchmark="supergpqa_hard",
        split="hard",
        allowed_model_slugs=None,
    )
    full_records = []
    full_records.extend(
        _collect_records(
            model_dirs=hard_models,
            benchmark="supergpqa_full",
            split="hard",
            allowed_model_slugs=set(common_model_slugs),
        )
    )
    full_records.extend(
        _collect_records(
            model_dirs=easy_medium_models,
            benchmark="supergpqa_full",
            split="easy_medium",
            allowed_model_slugs=set(common_model_slugs),
        )
    )

    hard_rows = _group_by_request(hard_records, benchmark="supergpqa_hard")
    full_rows = _group_by_request(full_records, benchmark="supergpqa_full")

    hard_jsonl = output_dir / "supergpqa_hard_real_output_lengths_by_request.jsonl"
    full_jsonl = output_dir / "supergpqa_full_real_output_lengths_by_request.jsonl"
    summary_csv = output_dir / "supergpqa_real_output_length_summary.csv"
    manifest_path = output_dir / "supergpqa_real_output_length_manifest.json"

    _write_jsonl(hard_jsonl, hard_rows)
    _write_jsonl(full_jsonl, full_rows)
    _write_summary_csv(summary_csv, hard_records + full_records)
    manifest_path.write_text(
        json.dumps(
            {
                "hard_run_root": str(hard_run_root),
                "easy_medium_run_root": str(easy_medium_run_root),
                "hard_models": sorted(hard_models),
                "full_models": common_model_slugs,
                "hard_request_count": len(hard_rows),
                "full_request_count": len(full_rows),
                "hard_jsonl": str(hard_jsonl.resolve()),
                "full_jsonl": str(full_jsonl.resolve()),
                "summary_csv": str(summary_csv.resolve()),
                "length_field": "actual_output_len",
                "note": (
                    "SuperGPQA-hard includes all model response directories in the hard run. "
                    "SuperGPQA-full includes only models present in both the hard and "
                    "easy-medium run roots, and combines both splits."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"wrote {len(hard_rows)} hard request rows to {hard_jsonl}")
    print(f"wrote {len(full_rows)} full request rows to {full_jsonl}")
    print(f"full SuperGPQA model count: {len(common_model_slugs)}")
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


def _collect_records(
    *,
    model_dirs: dict[str, Path],
    benchmark: str,
    split: str,
    allowed_model_slugs: set[str] | None,
) -> list[dict[str, Any]]:
    records = []
    for model_slug, model_dir in sorted(model_dirs.items()):
        if allowed_model_slugs is not None and model_slug not in allowed_model_slugs:
            continue
        responses_path = model_dir / "responses.jsonl"
        with responses_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                actual_output_len = _int_value(payload.get("actual_output_len"))
                metadata = payload.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                records.append(
                    {
                        "benchmark": benchmark,
                        "split": split,
                        "request_id": payload.get("request_id")
                        or metadata.get("sample_id")
                        or metadata.get("record_id"),
                        "sample_id": metadata.get("sample_id", ""),
                        "record_id": metadata.get("record_id", ""),
                        "difficulty": metadata.get("difficulty", ""),
                        "subject": metadata.get("subject", ""),
                        "language": metadata.get("language", ""),
                        "answer_label": metadata.get("answer_label", ""),
                        "model_slug": model_slug,
                        "model": payload.get("model", ""),
                        "actual_output_len": actual_output_len,
                        "expected_output_len": _int_value(payload.get("expected_output_len")),
                        "max_tokens": _int_value(
                            payload.get("decoding", {}).get("max_tokens")
                            if isinstance(payload.get("decoding"), dict)
                            else None
                        ),
                        "finish_reason": payload.get("finish_reason"),
                        "success": payload.get("success"),
                        "response_text_truncated": payload.get("response_text_truncated"),
                        "error": payload.get("error"),
                        "source_path": str(responses_path.resolve()),
                    }
                )
    return records


def _group_by_request(
    records: list[dict[str, Any]], *, benchmark: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["split"]), str(record["request_id"]))].append(record)

    rows = []
    for (split, request_id), values in sorted(grouped.items()):
        first = values[0]
        rows.append(
            {
                "benchmark": benchmark,
                "split": split,
                "request_id": request_id,
                "sample_id": first["sample_id"],
                "record_id": first["record_id"],
                "difficulty": first["difficulty"],
                "subject": first["subject"],
                "language": first["language"],
                "answer_label": first["answer_label"],
                "model_count": len(values),
                "outputs": [
                    {
                        "model_slug": value["model_slug"],
                        "model": value["model"],
                        "actual_output_len": value["actual_output_len"],
                        "expected_output_len": value["expected_output_len"],
                        "max_tokens": value["max_tokens"],
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


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    cap_hits: dict[tuple[str, str, str, str], int] = defaultdict(int)
    missing_lengths: dict[tuple[str, str, str, str], int] = defaultdict(int)
    model_names: dict[tuple[str, str, str, str], str] = {}
    for record in records:
        key = (
            str(record["benchmark"]),
            str(record["split"]),
            str(record["model_slug"]),
            str(record["model"]),
        )
        model_names[key] = str(record["model"])
        if record["actual_output_len"] is None:
            missing_lengths[key] += 1
            continue
        grouped[key].append(int(record["actual_output_len"]))
        if (
            record["expected_output_len"] is not None
            and record["actual_output_len"] >= record["expected_output_len"]
        ):
            cap_hits[key] += 1

    fieldnames = [
        "benchmark",
        "split",
        "model_slug",
        "model",
        "response_count",
        "missing_output_len_count",
        "output_tokens_mean",
        "output_tokens_p50",
        "output_tokens_p90",
        "output_tokens_p95",
        "output_tokens_p99",
        "output_tokens_max",
        "expected_cap_hits",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(set(grouped) | set(missing_lengths)):
            lengths = grouped.get(key, [])
            stats = _stats("output_tokens", lengths) if lengths else _empty_stats("output_tokens")
            benchmark, split, model_slug, model = key
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "split": split,
                    "model_slug": model_slug,
                    "model": model_names[key] or model,
                    "response_count": len(lengths),
                    "missing_output_len_count": missing_lengths[key],
                    **stats,
                    "expected_cap_hits": cap_hits[key],
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


if __name__ == "__main__":
    raise SystemExit(main())
