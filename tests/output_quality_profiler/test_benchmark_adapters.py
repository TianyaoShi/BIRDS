from __future__ import annotations

import json
from pathlib import Path

import pytest

from output_quality_profiler.benchmark_adapters import score_benchmark_responses
from output_quality_profiler.benchmark_adapters.code_completion import (
    crosscodeeval_edit_similarity,
    crosscodeeval_exact_match,
    crosscodeeval_postprocess_prediction,
    extract_identifiers,
)
from output_quality_profiler.benchmark_adapters.longbench_v1 import (
    rouge_l_f1,
    score_longbench_v1_responses,
    token_f1,
)
from output_quality_profiler.benchmark_adapters.repobench import (
    repobench_edit_similarity,
    repobench_exact_match,
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


def test_repobench_adapter_reports_official_em_and_es(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "c1",
                "success": True,
                "response_text": "return    value",
                "metadata": {
                    "ground_truth": "return value",
                    "language": "python",
                    "task": "in_file",
                    "level": "2k",
                    "sample_id": "repobench-python-000001",
                },
            },
            {
                "model": "org/model",
                "request_id": "c2",
                "success": True,
                "response_text": "return other",
                "metadata": {
                    "ground_truth": "return value",
                    "language": "python",
                    "task": "cross_file_first",
                    "level": "4k",
                    "sample_id": "repobench-python-000002",
                },
            },
            {
                "model": "org/model",
                "request_id": "c2-dup",
                "success": True,
                "response_text": "return other",
                "metadata": {
                    "ground_truth": "return value",
                    "language": "python",
                    "task": "cross_file_first",
                    "level": "4k",
                    "sample_id": "repobench-python-000002",
                },
            },
        ],
    )

    score = score_benchmark_responses(
        benchmark="RepoBench",
        responses_root=responses,
        output_dir=tmp_path / "code-score",
    )

    assert repobench_exact_match("return    value", "return value")
    assert repobench_edit_similarity("abc", "abc") == pytest.approx(100.0)
    assert score["exact_match_percent"] == pytest.approx(50.0)
    assert score["edit_similarity_percent"] < 100.0
    assert score["overall_score"] == score["edit_similarity_percent"]
    assert score["duplicate_items"] == 1
    assert score["by_language"]["python"]["count"] == 2
    assert score["by_task"]["cross_file_first"]["count"] == 1


def test_crosscodeeval_adapter_reports_official_code_and_identifier_metrics(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    _write_rows(
        responses,
        [
            {
                "model": "org/model",
                "request_id": "cc1",
                "success": True,
                "prompt": "function f() {\n  ",
                "response_text": "return helper(value); // trailing\n}\nextra();",
                "metadata": {
                    "ground_truth": "return helper(value);",
                    "language": "typescript",
                    "sample_id": "crosscodeeval-typescript-000001",
                },
            },
            {
                "model": "org/model",
                "request_id": "cc2",
                "success": True,
                "prompt": "x = ",
                "response_text": "other_value\nprint(other_value)",
                "metadata": {
                    "ground_truth": "target_value",
                    "language": "python",
                    "sample_id": "crosscodeeval-python-000002",
                },
            },
        ],
    )

    score = score_benchmark_responses(
        benchmark="CrossCodeEval",
        responses_root=responses,
        output_dir=tmp_path / "cceval-score",
    )

    assert crosscodeeval_postprocess_prediction(
        prompt="function f() {\n  ",
        completion="return helper(value); // trailing\n}",
        language="typescript",
    ) == "return helper(value);"
    assert crosscodeeval_exact_match(" return helper(value); ", "return helper(value);")
    assert crosscodeeval_edit_similarity("abc", "abc") == pytest.approx(100.0)
    assert extract_identifiers('return helper("value", actual_value);', "typescript") == [
        "helper",
        "actual_value",
    ]
    assert score["adapter"] == "crosscodeeval_official_metrics_v1"
    assert score["overall_score"] == pytest.approx(50.0)
    assert score["exact_match_percent"] == pytest.approx(50.0)
    assert score["edit_similarity_percent"] < 100.0
    assert score["identifier_f1_percent"] < 100.0
    assert score["by_language"]["typescript"]["exact_match_percent"] == pytest.approx(100.0)


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
                "metadata": {
                    "ground_truth": "alpha beta gamma",
                    "longbench_task": "qasper",
                    "profile": "short_answer_document_qa",
                },
            },
            {
                "model": "org/model",
                "request_id": "l2",
                "success": True,
                "response_text": "summary alpha beta",
                "metadata": {
                    "ground_truth": "summary alpha beta",
                    "longbench_task": "gov_report",
                    "profile": "long_output_summarization",
                },
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
    assert score["by_bucket"]["short_answer_document_qa"]["score"] == pytest.approx(100.0)
    assert score["by_bucket"]["short_answer_document_qa"]["metric_family"] == "F1-style QA"
    assert score["by_bucket"]["long_output_summarization"]["score"] == pytest.approx(100.0)
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
                    "profile": "short_answer_document_qa",
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
    assert score["by_bucket"]["short_answer_document_qa"]["by_task"]["qasper_e"]["score"] == pytest.approx(100.0)
