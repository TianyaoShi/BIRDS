from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

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


@dataclass(frozen=True, slots=True)
class JudgeAggregationResult:
    candidate_model_slug: str
    reference_model_slug: str
    overall: PairwiseScore
    by_source: dict[str, PairwiseScore]
    by_prompt_length_bucket: dict[str, PairwiseScore]
    by_candidate_position: dict[str, PairwiseScore]
    invalid_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_model_slug": self.candidate_model_slug,
            "reference_model_slug": self.reference_model_slug,
            "overall": self.overall.to_dict(),
            "by_source": {key: value.to_dict() for key, value in sorted(self.by_source.items())},
            "by_prompt_length_bucket": {
                key: value.to_dict() for key, value in sorted(self.by_prompt_length_bucket.items())
            },
            "by_candidate_position": {
                key: value.to_dict() for key, value in sorted(self.by_candidate_position.items())
            },
            "invalid_results": self.invalid_results,
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


def aggregate_openai_batch_judge_results(
    *,
    batch_manifest: str | Path,
    judge_results_jsonl: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(batch_manifest).read_text(encoding="utf-8"))
    comparison_by_id = _comparison_index(manifest)
    parsed = _parse_openai_batch_results(judge_results_jsonl, comparison_by_id=comparison_by_id)
    grouped: dict[tuple[str, str], list[JudgeResult]] = defaultdict(list)
    invalid_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result, invalid_payload in parsed:
        comparison = comparison_by_id[result.comparison_id]
        key = (str(comparison["candidate_model_slug"]), str(comparison["reference_model_slug"]))
        grouped[key].append(result)
        if invalid_payload is not None:
            invalid_rows[key].append(invalid_payload)

    aggregates = []
    for (candidate_slug, reference_slug), results in sorted(grouped.items()):
        aggregates.append(
            JudgeAggregationResult(
                candidate_model_slug=candidate_slug,
                reference_model_slug=reference_slug,
                overall=compute_pairwise_score(results),
                by_source=_score_by(results, lambda item: item.source or "unknown"),
                by_prompt_length_bucket=_score_by(
                    results,
                    lambda item: item.prompt_length_bucket or "unknown",
                ),
                by_candidate_position=_score_by(
                    results,
                    lambda item: "A" if item.candidate_is_a else "B",
                ),
                invalid_results=invalid_rows.get((candidate_slug, reference_slug), []),
            ).to_dict()
        )
    payload = {
        "batch_manifest": str(Path(batch_manifest).resolve()),
        "judge_results_jsonl": str(Path(judge_results_jsonl).resolve()),
        "expected_results": len(comparison_by_id),
        "parsed_results": sum(item["overall"]["wins"] + item["overall"]["ties"] + item["overall"]["losses"] + item["overall"]["invalid"] for item in aggregates),
        "aggregates": aggregates,
        "position_bias_note": (
            "A/B order is randomized when composing judge requests. If position effects are "
            "observed in by_candidate_position or external analysis, run position-swap judging "
            "for the affected comparison set."
        ),
    }
    if output_dir is not None:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "judge_score_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_path / "judge_score_summary.md").write_text(
            _render_score_markdown(payload),
            encoding="utf-8",
        )
    return payload


def _parse_openai_batch_results(
    judge_results_jsonl: str | Path,
    *,
    comparison_by_id: dict[str, dict[str, Any]],
) -> list[tuple[JudgeResult, dict[str, Any] | None]]:
    seen: set[str] = set()
    parsed: list[tuple[JudgeResult, dict[str, Any] | None]] = []
    with Path(judge_results_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str):
                raise ValueError(f"line {line_number}: missing string custom_id")
            if custom_id not in comparison_by_id:
                raise ValueError(f"line {line_number}: custom_id not found in manifest: {custom_id}")
            if custom_id in seen:
                raise ValueError(f"line {line_number}: duplicate custom_id: {custom_id}")
            seen.add(custom_id)
            comparison = comparison_by_id[custom_id]
            label, invalid_payload = _extract_judge_label(row, custom_id=custom_id)
            parsed.append(
                (
                    JudgeResult(
                        comparison_id=custom_id,
                        judge_label=label,
                        candidate_is_a=bool(comparison["candidate_is_a"]),
                        source=comparison.get("source"),
                        prompt_length_bucket=comparison.get("prompt_length_bucket"),
                    ),
                    invalid_payload,
                )
            )
    missing = sorted(set(comparison_by_id) - seen)
    if missing:
        raise ValueError(f"judge results are missing {len(missing)} custom_id entries, first={missing[0]}")
    return parsed


def _comparison_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("batch manifest must contain a non-empty comparisons list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise ValueError(f"manifest comparison {index} must be a mapping")
        custom_id = comparison.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"manifest comparison {index} missing custom_id")
        if custom_id in indexed:
            raise ValueError(f"duplicate manifest custom_id: {custom_id}")
        indexed[custom_id] = comparison
    return indexed


def _extract_judge_label(row: dict[str, Any], *, custom_id: str) -> tuple[JudgeLabel, dict[str, Any] | None]:
    if row.get("error") is not None:
        return "INVALID", {"custom_id": custom_id, "reason": "batch_request_error", "error": row.get("error")}
    response = row.get("response")
    status_code = response.get("status_code") if isinstance(response, dict) else None
    if status_code != 200:
        return "INVALID", {"custom_id": custom_id, "reason": "non_200_status", "status_code": status_code}
    body = response.get("body") if isinstance(response, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return "INVALID", {"custom_id": custom_id, "reason": "missing_choices"}
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return "INVALID", {"custom_id": custom_id, "reason": "missing_content"}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return "INVALID", {"custom_id": custom_id, "reason": "invalid_json_content", "content": content, "error": str(exc)}
    winner = parsed.get("winner")
    if winner == "A":
        return "A_BETTER", None
    if winner == "B":
        return "B_BETTER", None
    if winner == "Tie":
        return "TIE", None
    return "INVALID", {"custom_id": custom_id, "reason": "unsupported_winner", "winner": winner, "content": parsed}


def _score_by(results: Iterable[JudgeResult], key_fn: Any) -> dict[str, PairwiseScore]:
    groups: dict[str, list[JudgeResult]] = defaultdict(list)
    for result in results:
        groups[str(key_fn(result))].append(result)
    return {key: compute_pairwise_score(value) for key, value in groups.items()}


def _render_score_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Judge Score Summary",
        "",
        f"- Expected results: {payload['expected_results']}",
        f"- Parsed results: {payload['parsed_results']}",
        f"- Position-bias note: {payload['position_bias_note']}",
        "",
    ]
    for aggregate in payload["aggregates"]:
        overall = aggregate["overall"]
        lines.extend(
            [
                f"## {aggregate['candidate_model_slug']} vs {aggregate['reference_model_slug']}",
                "",
                f"- Qchat: {overall['q_chat']}",
                f"- Wins/ties/losses/invalid: {overall['wins']}/{overall['ties']}/{overall['losses']}/{overall['invalid']}",
                "- Candidate position breakdown:",
            ]
        )
        for position, score in aggregate["by_candidate_position"].items():
            lines.append(
                f"  - {position}: q_chat={score['q_chat']}, "
                f"wins/ties/losses/invalid={score['wins']}/{score['ties']}/{score['losses']}/{score['invalid']}"
            )
        lines.append("")
    return "\n".join(lines)
