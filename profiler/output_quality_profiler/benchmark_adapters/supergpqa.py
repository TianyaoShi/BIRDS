from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    normalize_text,
    write_score_artifacts,
)


_ANSWER_PATTERNS = (
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?\s*([A-Z])\s*\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([A-Z])\s*\)?(?:[).:\s]|$)", re.IGNORECASE),
)


def score_supergpqa_responses(
    *,
    responses_root: str | Path,
    output_dir: str | Path,
    benchmark_name: str = "SuperGPQA",
    is_full_benchmark: bool = True,
) -> dict[str, Any]:
    rows = load_response_rows(responses_root)
    per_item: list[dict[str, Any]] = []
    correct = scored = failed = invalid = 0
    by_subject: dict[str, list[bool]] = {}
    by_difficulty: dict[str, list[bool]] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        if not row.get("success", False):
            failed += 1
            per_item.append({"request_id": request_id, "correct": False, "invalid_reason": "failed_generation"})
            continue
        truth_values = ground_truth_values(row)
        if not truth_values:
            invalid += 1
            per_item.append({"request_id": request_id, "correct": False, "invalid_reason": "missing_ground_truth"})
            continue
        ground_truth = truth_values[0].strip()
        response_text = str(row.get("response_text") or "")
        prediction = extract_supergpqa_answer(response_text, ground_truth=ground_truth)
        if prediction is None:
            invalid += 1
            per_item.append(
                {
                    "request_id": request_id,
                    "ground_truth": ground_truth,
                    "response_text": response_text,
                    "correct": False,
                    "invalid_reason": "could_not_extract_answer",
                }
            )
            continue
        is_correct = _answer_matches(
            prediction,
            ground_truth=ground_truth,
            ground_truth_text=str(meta.get("ground_truth_text") or ""),
        )
        scored += 1
        correct += int(is_correct)
        subject = str(meta.get("subject") or meta.get("discipline") or "unknown")
        difficulty = str(meta.get("difficulty") or "unknown")
        by_subject.setdefault(subject, []).append(is_correct)
        by_difficulty.setdefault(difficulty, []).append(is_correct)
        per_item.append(
            {
                "request_id": request_id,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "subject": subject,
                "difficulty": difficulty,
            }
        )
    score = {
        "benchmark": benchmark_name,
        "model": model_name_from_rows(rows),
        "adapter": "supergpqa_compat_v1",
        "metric": "accuracy",
        "overall_score": None if scored == 0 else correct / scored,
        "correct": correct,
        "total_items": len(rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "by_subject": _accuracy_breakdown(by_subject),
        "by_difficulty": _accuracy_breakdown(by_difficulty),
        "is_full_benchmark": is_full_benchmark,
        "compatibility_note": (
            "Uses local answer-label extraction against materialized SuperGPQA ground truth. "
            "Run the official evaluator too if strict benchmark parity is required."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title=f"{benchmark_name} Score",
    )


def extract_supergpqa_answer(response_text: str, *, ground_truth: str) -> str | None:
    text = response_text.strip()
    if not text:
        return None
    if len(ground_truth.strip()) == 1 and ground_truth.strip().isalpha():
        for pattern in _ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        options = sorted(set(re.findall(r"\b[A-Z]\b", text.upper())))
        if len(options) == 1:
            return options[0]
        return None
    return text


def _answer_matches(prediction: str, *, ground_truth: str, ground_truth_text: str) -> bool:
    if normalize_text(prediction) == normalize_text(ground_truth):
        return True
    if ground_truth_text and normalize_text(prediction) == normalize_text(ground_truth_text):
        return True
    return False


def _accuracy_breakdown(groups: dict[str, list[bool]]) -> dict[str, dict[str, Any]]:
    return {
        key: {"count": len(values), "accuracy": sum(values) / len(values) if values else None}
        for key, values in sorted(groups.items())
    }
