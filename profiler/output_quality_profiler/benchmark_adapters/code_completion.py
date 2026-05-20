from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    strip_code_fences,
    write_score_artifacts,
)


def score_code_completion_responses(
    *,
    responses_root: str | Path,
    output_dir: str | Path,
    benchmark_name: str,
) -> dict[str, Any]:
    rows = load_response_rows(responses_root)
    per_item: list[dict[str, Any]] = []
    exact = normalized_exact = failed = invalid = scored = 0
    similarities: list[float] = []
    by_language: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        if not row.get("success", False):
            failed += 1
            per_item.append({"request_id": request_id, "invalid_reason": "failed_generation"})
            continue
        truth_values = ground_truth_values(row)
        if not truth_values:
            invalid += 1
            per_item.append({"request_id": request_id, "invalid_reason": "missing_ground_truth"})
            continue
        target = truth_values[0]
        prediction = normalize_code_completion(str(row.get("response_text") or ""))
        target_normalized = normalize_code_completion(target)
        exact_match = prediction == target
        normalized_match = prediction == target_normalized
        similarity = SequenceMatcher(None, prediction, target_normalized).ratio()
        language = str(meta.get("language") or "unknown")
        scored += 1
        exact += int(exact_match)
        normalized_exact += int(normalized_match)
        similarities.append(similarity)
        by_language.setdefault(language, []).append(similarity)
        per_item.append(
            {
                "request_id": request_id,
                "language": language,
                "prediction": prediction,
                "ground_truth": target,
                "exact_match": exact_match,
                "normalized_exact_match": normalized_match,
                "similarity": similarity,
            }
        )
    score = {
        "benchmark": benchmark_name,
        "model": model_name_from_rows(rows),
        "adapter": "code_completion_compat_v1",
        "metric": "normalized_exact_match_rate",
        "overall_score": None if scored == 0 else normalized_exact / scored,
        "mean_similarity": None if not similarities else sum(similarities) / len(similarities),
        "exact_match_rate": None if scored == 0 else exact / scored,
        "normalized_exact_match_rate": None if scored == 0 else normalized_exact / scored,
        "total_items": len(rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "by_language": {
            language: {
                "count": len(values),
                "mean_similarity": sum(values) / len(values) if values else None,
            }
            for language, values in sorted(by_language.items())
        },
        "is_full_benchmark": True,
        "compatibility_note": (
            "Uses a lightweight local code-completion compatibility metric. "
            "Vendor or call the original evaluator before treating this as an official score."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title=f"{benchmark_name} Score",
    )


def normalize_code_completion(value: str) -> str:
    text = strip_code_fences(value)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()
