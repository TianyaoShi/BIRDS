from __future__ import annotations

import pytest

from output_quality_profiler.scoring import JudgeResult, compute_pairwise_score


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

