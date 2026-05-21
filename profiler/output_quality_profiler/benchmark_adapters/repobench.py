from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    write_score_artifacts,
)


try:  # pragma: no cover - depends on optional benchmark packages.
    from fuzzywuzzy import fuzz as _fuzz
except ImportError:  # pragma: no cover - covered through dependency report.
    _fuzz = None

try:  # pragma: no cover - depends on optional benchmark packages.
    from codebleu import calc_codebleu as _calc_codebleu
except ImportError:  # pragma: no cover - covered through dependency report.
    _calc_codebleu = None


REPOBENCH_TASKS = ("cross_file_first", "cross_file_random", "in_file")


def score_repobench_responses(
    *,
    responses_root: str | Path,
    output_dir: str | Path,
    benchmark_name: str = "RepoBench",
) -> dict[str, Any]:
    rows = load_response_rows(responses_root)
    unique_rows, duplicate_items = _deduplicate_rows(rows)
    per_item: list[dict[str, Any]] = []
    failed = invalid = scored = exact_matches = 0
    edit_scores: list[float] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_token_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_language: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, row in enumerate(unique_rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        task = _repo_task(meta)
        language = str(meta.get("language") or "unknown")
        token_level = str(meta.get("level") or "unknown")
        if not row.get("success", False):
            failed += 1
            per_item.append(
                {
                    "request_id": request_id,
                    "task": task,
                    "language": language,
                    "token_level": token_level,
                    "invalid_reason": "failed_generation",
                }
            )
            continue
        truth_values = ground_truth_values(row)
        if not truth_values:
            invalid += 1
            per_item.append(
                {
                    "request_id": request_id,
                    "task": task,
                    "language": language,
                    "token_level": token_level,
                    "invalid_reason": "missing_ground_truth",
                }
            )
            continue
        ground_truth = truth_values[0]
        prediction = str(row.get("response_text") or "")
        exact_match = repobench_exact_match(prediction, ground_truth)
        edit_similarity = repobench_edit_similarity(prediction, ground_truth)
        item = {
            "request_id": request_id,
            "sample_id": meta.get("sample_id"),
            "task": task,
            "language": language,
            "token_level": token_level,
            "repo_id": meta.get("repo_id"),
            "file_path": meta.get("file_path"),
            "sequence_index": meta.get("sequence_index"),
            "prediction": prediction,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "edit_similarity": edit_similarity,
        }
        scored += 1
        exact_matches += int(exact_match)
        edit_scores.append(edit_similarity)
        by_task[task].append(item)
        by_language[language].append(item)
        by_token_level[token_level].append(item)
        by_task_language[f"{task}:{language}"].append(item)
        per_item.append(item)

    codebleu_by_language, codebleu_errors = _codebleu_by_language(per_item)
    score = {
        "benchmark": benchmark_name,
        "model": model_name_from_rows(rows),
        "adapter": "repobench_official_metrics_v1",
        "metric": "edit_similarity_percent",
        "overall_score": None if not edit_scores else sum(edit_scores) / len(edit_scores),
        "exact_match_percent": None if scored == 0 else 100.0 * exact_matches / scored,
        "edit_similarity_percent": None if not edit_scores else sum(edit_scores) / len(edit_scores),
        "codebleu_percent": _weighted_codebleu(codebleu_by_language),
        "codebleu_errors": codebleu_errors,
        "total_items": len(rows),
        "unique_items": len(unique_rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "duplicate_items": duplicate_items,
        "by_task": _group_payload(by_task),
        "by_language": _group_payload(by_language, codebleu_by_language=codebleu_by_language),
        "by_token_level": _group_payload(by_token_level),
        "by_task_language": _group_payload(by_task_language),
        "is_full_benchmark": True,
        "dependency_status": repobench_dependency_status(),
        "score_interpretation": (
            "RepoBench official evaluation reports EM, ES, and CodeBLEU. overall_score is "
            "the weighted Edit Similarity percent over all scored samples; use by_task for "
            "cross_file_first, cross_file_random, and in_file breakdowns."
        ),
        "compatibility_note": (
            "Implements the official RepoBench v1.1 EM and ES definitions locally: EM compares "
            "whitespace-tokenized predictions and references, while ES is fuzz.ratio-compatible "
            "edit similarity on raw strings. CodeBLEU is emitted only when the optional codebleu "
            "package is installed."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title=f"{benchmark_name} Score",
    )


def repobench_exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction.split() == ground_truth.split()


def repobench_edit_similarity(prediction: str, ground_truth: str) -> float:
    if _fuzz is not None:
        return float(_fuzz.ratio(prediction, ground_truth))
    return round(100.0 * SequenceMatcher(None, prediction, ground_truth).ratio(), 5)


def repobench_dependency_status() -> dict[str, bool]:
    return {
        "fuzzywuzzy": _fuzz is not None,
        "codebleu": _calc_codebleu is not None,
    }


def _repo_task(meta: dict[str, Any]) -> str:
    value = str(meta.get("task") or meta.get("repobench_task") or "unknown")
    return value if value in REPOBENCH_TASKS else value


def _deduplicate_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for index, row in enumerate(rows):
        meta = metadata(row)
        key = _dedupe_key(row, meta, fallback_index=index)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _dedupe_key(row: dict[str, Any], meta: dict[str, Any], *, fallback_index: int) -> tuple[Any, ...]:
    sample_id = meta.get("sample_id") or row.get("request_id")
    if sample_id:
        return ("sample_id", sample_id)
    language = meta.get("language")
    task = meta.get("task")
    sequence_index = meta.get("sequence_index")
    if language is not None and task is not None and sequence_index is not None:
        return ("sequence", language, task, sequence_index)
    return ("row", fallback_index)


def _group_payload(
    groups: dict[str, list[dict[str, Any]]],
    *,
    codebleu_by_language: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key, values in sorted(groups.items()):
        exact_count = sum(1 for item in values if item["exact_match"])
        edit_scores = [float(item["edit_similarity"]) for item in values]
        entry = {
            "count": len(values),
            "exact_match_percent": None if not values else 100.0 * exact_count / len(values),
            "edit_similarity_percent": None if not edit_scores else sum(edit_scores) / len(edit_scores),
        }
        if codebleu_by_language and key in codebleu_by_language:
            entry["codebleu_percent"] = codebleu_by_language[key]["codebleu_percent"]
        payload[key] = entry
    return payload


def _codebleu_by_language(
    per_item: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if _calc_codebleu is None:
        return {}, {}
    grouped: dict[str, dict[str, list[str]]] = {}
    for item in per_item:
        if "edit_similarity" not in item:
            continue
        language = str(item.get("language") or "unknown")
        if language not in {"python", "java"}:
            continue
        bucket = grouped.setdefault(language, {"predictions": [], "ground_truths": []})
        bucket["predictions"].append(str(item.get("prediction") or "").replace("\r", ""))
        bucket["ground_truths"].append(str(item.get("ground_truth") or "").replace("\r", ""))
    scores: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for language, values in grouped.items():
        if not values["predictions"]:
            continue
        try:
            result = _calc_codebleu(
                values["ground_truths"],
                values["predictions"],
                language,
                [0.25, 0.25, 0.25, 0.25],
                tokenizer=None,
            )
        except Exception as exc:  # pragma: no cover - depends on optional parser packages.
            errors[language] = str(exc)
            continue
        scores[language] = {
            "count": len(values["predictions"]),
            "codebleu_percent": 100.0 * float(result["codebleu"]),
        }
    return scores, errors


def _weighted_codebleu(codebleu_by_language: dict[str, dict[str, Any]]) -> float | None:
    total = sum(int(item["count"]) for item in codebleu_by_language.values())
    if total == 0:
        return None
    return sum(float(item["codebleu_percent"]) * int(item["count"]) for item in codebleu_by_language.values()) / total
