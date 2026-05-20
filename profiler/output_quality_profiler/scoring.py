from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .models import JudgeLabel


CandidateOutcome = Literal["win", "tie", "loss", "invalid"]


@dataclass(frozen=True, slots=True)
class JudgeResult:
    comparison_id: str
    judge_label: JudgeLabel
    candidate_is_a: bool
    source: str | None = None
    prompt_length_bucket: str | None = None


@dataclass(frozen=True, slots=True)
class PairwiseScore:
    wins: int
    ties: int
    losses: int
    invalid: int
    valid: int
    q_chat: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "invalid": self.invalid,
            "valid": self.valid,
            "q_chat": self.q_chat,
        }


def judge_label_to_candidate_outcome(label: JudgeLabel, *, candidate_is_a: bool) -> CandidateOutcome:
    if label == "INVALID":
        return "invalid"
    if label == "TIE":
        return "tie"
    if label == "A_BETTER":
        return "win" if candidate_is_a else "loss"
    if label == "B_BETTER":
        return "loss" if candidate_is_a else "win"
    raise ValueError(f"unsupported judge label {label!r}")


def compute_pairwise_score(results: Iterable[JudgeResult]) -> PairwiseScore:
    wins = ties = losses = invalid = 0
    for result in results:
        outcome = judge_label_to_candidate_outcome(
            result.judge_label,
            candidate_is_a=result.candidate_is_a,
        )
        if outcome == "win":
            wins += 1
        elif outcome == "tie":
            ties += 1
        elif outcome == "loss":
            losses += 1
        else:
            invalid += 1
    valid = wins + ties + losses
    q_chat = None if valid == 0 else (wins + 0.5 * ties) / valid
    return PairwiseScore(
        wins=wins,
        ties=ties,
        losses=losses,
        invalid=invalid,
        valid=valid,
        q_chat=q_chat,
    )

