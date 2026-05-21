from __future__ import annotations

import json
from pathlib import Path

import pytest

from output_quality_profiler.benchmark_adapters import score_benchmark_responses
from output_quality_profiler.benchmark_adapters.longbench_v1 import (
    rouge_l_f1,
    score_longbench_v1_responses,
    token_f1,
)
from output_quality_profiler.benchmark_adapters.supergpqa import extract_supergpqa_answer


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_supergpqa_adapter_scores_extracted_answer(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "q1",
                "success": True,
                "response_text": "The final answer is (B).",
                "metadata": {
                    "ground_truth": "B",
                    "ground_truth_text": "choice b",
                    "subject": "biology",
                    "difficulty": "hard",
                },
            },
            {
                "model": "org/model",
                "request_id": "q2",
                "success": True,
                "response_text": "Answer: A",
                "metadata": {"ground_truth": "C"},
            },
        ],
    )

    score = score_benchmark_responses(
        benchmark="SuperGPQA",
        responses_root=responses,
        output_dir=tmp_path / "score",
    )

    assert extract_supergpqa_answer("Answer: D", ground_truth="D") == "D"
    assert score["overall_score"] == pytest.approx(0.5)
    assert score["correct"] == 1
    assert (tmp_path / "score" / "per_item.jsonl").is_file()


def test_code_completion_adapter_reports_exact_and_similarity(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "c1",
                "success": True,
                "response_text": "```python\nreturn value\n```",
                "metadata": {"ground_truth": "return value", "language": "python"},
            },
            {
                "model": "org/model",
                "request_id": "c2",
                "success": True,
                "response_text": "return other",
                "metadata": {"ground_truth": "return value", "language": "python"},
            },
        ],
    )

    score = score_benchmark_responses(
        benchmark="RepoBench",
        responses_root=responses,
        output_dir=tmp_path / "code-score",
    )

    assert score["normalized_exact_match_rate"] == pytest.approx(0.5)
    assert score["mean_similarity"] < 1.0
    assert score["by_language"]["python"]["count"] == 2


def test_longbench_adapter_scores_covered_subset(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "l1",
                "success": True,
                "response_text": "alpha beta gamma",
                "metadata": {"ground_truth": "alpha beta gamma", "longbench_task": "qasper"},
            },
            {
                "model": "org/model",
                "request_id": "l2",
                "success": True,
                "response_text": "summary alpha beta",
                "metadata": {"ground_truth": "summary alpha beta", "longbench_task": "gov_report"},
            },
        ],
    )

    score = score_benchmark_responses(
        benchmark="LongBench-v1-covered",
        responses_root=responses,
        output_dir=tmp_path / "long-score",
    )

    assert token_f1("a b", "a b c") == pytest.approx(0.8)
    assert rouge_l_f1("a b", "a b c") == pytest.approx(0.8)
    assert score["overall_score"] == pytest.approx(100.0)
    assert score["is_full_benchmark"] is False


def test_longbench_adapter_resolves_raw_answer_lists(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "longbench"
    _write_rows(
        raw_root / "qasper_e.jsonl",
        [
            {
                "_id": "row-0",
                "answers": ["alpha", "beta"],
                "all_classes": [],
                "length": 1234,
            }
        ],
    )
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "l1",
                "success": True,
                "response_text": "beta",
                "metadata": {
                    "ground_truth": "alpha",
                    "longbench_dataset": "qasper_e",
                    "longbench_id": "row-0",
                    "longbench_row_index": 0,
                },
            },
        ],
    )

    score = score_longbench_v1_responses(
        responses_root=responses,
        output_dir=tmp_path / "long-score-raw",
        raw_data_root=raw_root,
    )

    assert score["overall_score"] == pytest.approx(100.0)
    assert score["raw_reference_matches"] == 1
    assert score["by_task"]["qasper_e"]["metric"] == "qa_f1"
