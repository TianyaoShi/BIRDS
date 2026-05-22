#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


BUCKET_COLUMNS = (
    "long_output_summarization",
    "medium_output_summarization",
    "medium_answer_rag_qa",
    "short_answer_document_qa",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a sorted LongBench CSV using the best score per model across score roots."
    )
    parser.add_argument(
        "--score-root",
        action="append",
        required=True,
        help="Score root as LABEL=PATH. May be repeated; immediate child directories are scanned for score.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = [_parse_labeled_path(raw) for raw in args.score_root]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    labels: list[str] = []
    for label, root in roots:
        labels.append(label)
        for score_path in sorted(root.glob("*/score.json")):
            payload = json.loads(score_path.read_text())
            model = str(payload.get("model") or score_path.parent.name)
            score = _as_float(payload.get("overall_score"))
            if score is None:
                continue
            record = _record_from_score(label=label, score_path=score_path, payload=payload)
            existing = grouped.setdefault(model, {}).get(label)
            if existing is None or score > float(existing["overall_score"]):
                grouped[model][label] = record

    rows = []
    for model, by_label in grouped.items():
        selected = max(by_label.values(), key=lambda item: float(item["overall_score"]))
        row = {
            "model": model,
            "selected_run": selected["source_label"],
            "overall_score": selected["overall_score"],
            "item_weighted_score": selected["item_weighted_score"],
            "scored_items": selected["scored_items"],
            "failed_generations": selected["failed_generations"],
            "invalid_items": selected["invalid_items"],
            "raw_reference_mismatches": selected["raw_reference_mismatches"],
            "score_dir": selected["score_dir"],
        }
        for bucket in BUCKET_COLUMNS:
            row[bucket] = selected.get(bucket)
        for label in labels:
            candidate = by_label.get(label)
            row[f"{label}_overall_score"] = None if candidate is None else candidate["overall_score"]
        rows.append(row)

    rows.sort(key=lambda item: _sort_score(item["overall_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "model",
        "selected_run",
        "overall_score",
        "item_weighted_score",
        *BUCKET_COLUMNS,
        "scored_items",
        "failed_generations",
        "invalid_items",
        "raw_reference_mismatches",
        *[f"{label}_overall_score" for label in labels],
        "score_dir",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--score-root must be LABEL=PATH, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label or not label.replace("_", "").replace("-", "").isalnum():
        raise SystemExit(f"score-root label must be alphanumeric plus _/-, got: {label!r}")
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"score root does not exist or is not a directory: {root}")
    return label, root


def _record_from_score(*, label: str, score_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    by_bucket = payload.get("by_bucket") or {}
    record = {
        "source_label": label,
        "overall_score": _format_float(payload.get("overall_score")),
        "item_weighted_score": _format_float(payload.get("item_weighted_score")),
        "scored_items": payload.get("scored_items"),
        "failed_generations": payload.get("failed_generations"),
        "invalid_items": payload.get("invalid_items"),
        "raw_reference_mismatches": payload.get("raw_reference_mismatches"),
        "score_dir": str(score_path.parent),
    }
    for bucket in BUCKET_COLUMNS:
        record[bucket] = _format_float((by_bucket.get(bucket) or {}).get("score"))
    return record


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _format_float(value: Any) -> str | None:
    result = _as_float(value)
    if result is None:
        return None
    return f"{result:.6f}"


def _sort_score(value: Any) -> float:
    result = _as_float(value)
    return float("-inf") if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
