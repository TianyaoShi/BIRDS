from __future__ import annotations

import json

import pytest

from output_quality_profiler.scoring import (
    JudgeResult,
    aggregate_openai_batch_judge_results,
    compute_pairwise_score,
)


def test_compute_pairwise_score_inverts_randomized_ab_order() -> None:
    score = compute_pairwise_score(
        [
            JudgeResult("a", "A_BETTER", candidate_is_a=True),
            JudgeResult("b", "A_BETTER", candidate_is_a=False),
            JudgeResult("c", "B_BETTER", candidate_is_a=True),
            JudgeResult("d", "B_BETTER", candidate_is_a=False),
            JudgeResult("e", "TIE", candidate_is_a=True),
            JudgeResult("f", "INVALID", candidate_is_a=False),
        ]
    )

    assert score.wins == 2
    assert score.ties == 1
    assert score.losses == 2
    assert score.invalid == 1
    assert score.valid == 5
    assert score.q_chat == pytest.approx(0.5)


def test_compute_pairwise_score_returns_none_when_all_invalid() -> None:
    score = compute_pairwise_score([JudgeResult("a", "INVALID", candidate_is_a=True)])

    assert score.valid == 0
    assert score.q_chat is None


def test_aggregate_openai_batch_judge_results_inverts_ab_and_reports_positions(tmp_path) -> None:
    manifest = {
        "comparisons": [
            {
                "custom_id": "one",
                "candidate_model_slug": "candidate",
                "reference_model_slug": "reference",
                "candidate_is_a": True,
                "source": "sharegpt",
                "prompt_length_bucket": "short",
            },
            {
                "custom_id": "two",
                "candidate_model_slug": "candidate",
                "reference_model_slug": "reference",
                "candidate_is_a": False,
                "source": "wildchat",
                "prompt_length_bucket": "medium",
            },
        ]
    }
    results = [
        {
            "custom_id": "one",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"content": json.dumps({"winner": "A", "reason": "better"})}}
                    ]
                },
            },
            "error": None,
        },
        {
            "custom_id": "two",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"content": json.dumps({"winner": "A", "reason": "better"})}}
                    ]
                },
            },
            "error": None,
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "score"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results_path.write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")

    payload = aggregate_openai_batch_judge_results(
        batch_manifest=manifest_path,
        judge_results_jsonl=results_path,
        output_dir=output_dir,
    )

    aggregate = payload["aggregates"][0]
    assert aggregate["overall"]["wins"] == 1
    assert aggregate["overall"]["losses"] == 1
    assert aggregate["overall"]["q_chat"] == pytest.approx(0.5)
    assert set(aggregate["by_candidate_position"]) == {"A", "B"}
    assert (output_dir / "judge_score_summary.json").is_file()
    assert (output_dir / "judge_score_summary.md").is_file()
