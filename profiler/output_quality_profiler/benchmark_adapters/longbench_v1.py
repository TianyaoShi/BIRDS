from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    normalize_text,
    write_score_artifacts,
)


SUMMARIZATION_TASKS = {"gov_report", "gov_report_e", "multi_news", "multi_news_e", "qmsum", "vcsum"}


def score_longbench_v1_responses(*, responses_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = load_response_rows(responses_root)
    per_item: list[dict[str, Any]] = []
    failed = invalid = scored = 0
    scores: list[float] = []
    by_task: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        task = str(meta.get("longbench_task") or row.get("dataset_task") or "unknown")
        if not row.get("success", False):
            failed += 1
            per_item.append({"request_id": request_id, "task": task, "invalid_reason": "failed_generation"})
            continue
        references = ground_truth_values(row)
        if not references:
            invalid += 1
            per_item.append({"request_id": request_id, "task": task, "invalid_reason": "missing_ground_truth"})
            continue
        prediction = str(row.get("response_text") or "")
        item_score = _score_longbench_item(prediction, references=references, task=task)
        scored += 1
        scores.append(item_score)
        by_task.setdefault(task, []).append(item_score)
        per_item.append(
            {
                "request_id": request_id,
                "task": task,
                "prediction": prediction,
                "ground_truth": references,
                "score": item_score,
            }
        )
    score = {
        "benchmark": "LongBench-v1-covered",
        "model": model_name_from_rows(rows),
        "adapter": "longbench_v1_covered_compat_v1",
        "metric": "covered_task_mean_percent",
        "overall_score": None if not scores else 100.0 * sum(scores) / len(scores),
        "total_items": len(rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "by_task": {
            task: {
                "count": len(values),
                "score": 100.0 * sum(values) / len(values) if values else None,
            }
            for task, values in sorted(by_task.items())
        },
        "is_full_benchmark": False,
        "compatibility_note": (
            "Scores only the materialized LongBench v1 covered-task subset. "
            "Summarization uses ROUGE-L F1; QA-style covered tasks use token F1."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title="LongBench v1 Covered Score",
    )


def _score_longbench_item(prediction: str, *, references: Sequence[str], task: str) -> float:
    if task in SUMMARIZATION_TASKS:
        return max(rouge_l_f1(prediction, reference) for reference in references)
    return max(token_f1(prediction, reference) for reference in references)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _tokens(value: str) -> list[str]:
    normalized = normalize_text(value)
    return re.findall(r"[\w]+", normalized)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
