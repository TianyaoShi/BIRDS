from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .scoring import _comparison_index, _extract_judge_label, judge_label_to_candidate_outcome


DEFAULT_REFERENCE_MODEL_SLUG = "meta-llama-llama-3-1-8b-instruct"


def report_judge_results(
    *,
    judge_responses_dir: str | Path,
    manifest_dirs: Sequence[str | Path],
    output_dir: str | Path,
    reference_model_slug: str = DEFAULT_REFERENCE_MODEL_SLUG,
) -> dict[str, Any]:
    responses_path = Path(judge_responses_dir).resolve()
    manifest_paths = [Path(item).resolve() for item in manifest_dirs]
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows, validation = _load_judge_rows(
        judge_responses_dir=responses_path,
        manifest_dirs=manifest_paths,
    )
    model_summary = _summarize_models(rows, reference_model_slug=reference_model_slug)
    position_bias = _summarize_position_bias(rows)

    _write_json(output_path / "validation.json", validation)
    _write_json(output_path / "model_qscores.json", model_summary)
    _write_json(output_path / "position_bias.json", position_bias)
    (output_path / "judge_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(output_path / "model_qscores.csv", model_summary)
    (output_path / "summary.md").write_text(
        _render_markdown(
            validation=validation,
            rows=rows,
            model_summary=model_summary,
            position_bias=position_bias,
        ),
        encoding="utf-8",
    )
    plot_path = output_path / "model_size_vs_qscore.png"
    histogram_path = output_path / "qscore_distribution.png"
    _write_plots(
        model_summary=model_summary,
        reference_model_slug=reference_model_slug,
        plot_path=plot_path,
        histogram_path=histogram_path,
    )
    return {
        "judge_responses_dir": str(responses_path),
        "manifest_dirs": [str(item) for item in manifest_paths],
        "output_dir": str(output_path),
        "judge_output_files": len(validation),
        "parsed_rows": len(rows),
        "candidate_models_scored": len(model_summary),
        "invalid_rows": sum(1 for row in rows if row["outcome"] == "invalid"),
        "plot": str(plot_path),
        "histogram": str(histogram_path),
        "summary_md": str(output_path / "summary.md"),
        "model_qscores_csv": str(output_path / "model_qscores.csv"),
        "position_bias": position_bias,
    }


def _load_judge_rows(
    *,
    judge_responses_dir: Path,
    manifest_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for output_file in sorted(judge_responses_dir.glob("*.output.jsonl")):
        manifest_file = _matching_manifest(output_file, manifest_dirs=manifest_dirs)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        comparison_by_id = _comparison_index(manifest)
        seen: set[str] = set()
        invalid = 0
        with output_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if custom_id not in comparison_by_id:
                    raise ValueError(
                        f"{output_file.name}:{line_number}: custom_id missing from manifest: {custom_id}"
                    )
                if custom_id in seen:
                    raise ValueError(f"{output_file.name}:{line_number}: duplicate custom_id: {custom_id}")
                seen.add(custom_id)
                comparison = comparison_by_id[custom_id]
                label, invalid_payload = _extract_judge_label(row, custom_id=custom_id)
                outcome = judge_label_to_candidate_outcome(
                    label,
                    candidate_is_a=bool(comparison["candidate_is_a"]),
                )
                usage = (((row.get("response") or {}).get("body") or {}).get("usage") or {})
                rows.append(
                    {
                        "custom_id": custom_id,
                        "candidate_model_slug": comparison["candidate_model_slug"],
                        "reference_model_slug": comparison["reference_model_slug"],
                        "request_id": comparison.get("request_id"),
                        "candidate_is_a": bool(comparison["candidate_is_a"]),
                        "judge_label": label,
                        "outcome": outcome,
                        "source": comparison.get("source") or "unknown",
                        "prompt_length_bucket": comparison.get("prompt_length_bucket") or "unknown",
                        "output_file": str(output_file),
                        "manifest_file": str(manifest_file),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "invalid_reason": None if invalid_payload is None else invalid_payload.get("reason"),
                    }
                )
                invalid += int(invalid_payload is not None)
        missing = sorted(set(comparison_by_id) - seen)
        if missing:
            raise ValueError(f"{output_file.name}: missing {len(missing)} results, first={missing[0]}")
        validation.append(
            {
                "output_file": output_file.name,
                "manifest_file": manifest_file.name,
                "expected": len(comparison_by_id),
                "parsed": len(seen),
                "missing": len(missing),
                "invalid": invalid,
                "candidate_model_slug": manifest.get("candidate_model_slug"),
            }
        )
    if not rows:
        raise ValueError(f"no *.output.jsonl files found in {judge_responses_dir}")
    return rows, validation


def _matching_manifest(output_file: Path, *, manifest_dirs: Sequence[Path]) -> Path:
    if not output_file.name.endswith(".output.jsonl"):
        raise ValueError(f"unexpected judge output filename: {output_file.name}")
    stem = output_file.name[: -len(".output.jsonl")]
    matches = [directory / f"{stem}.manifest.json" for directory in manifest_dirs]
    matches = [path for path in matches if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        output_ids = _custom_ids_in_output(output_file)
        exact_matches = []
        for manifest_path in matches:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_ids = set(_comparison_index(manifest))
            if output_ids == manifest_ids:
                exact_matches.append(manifest_path)
        if len(exact_matches) == 1:
            return exact_matches[0]
        raise ValueError(f"{output_file.name}: expected exactly one matching manifest, found {matches}")
    raise ValueError(f"{output_file.name}: expected exactly one matching manifest, found {matches}")


def _custom_ids_in_output(output_file: Path) -> set[str]:
    custom_ids: set[str] = set()
    with output_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str):
                raise ValueError(f"{output_file.name}:{line_number}: missing string custom_id")
            custom_ids.add(custom_id)
    return custom_ids


def _summarize_models(
    rows: Sequence[dict[str, Any]],
    *,
    reference_model_slug: str,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["candidate_model_slug"]].append(row)
    summaries: list[dict[str, Any]] = []
    for slug, model_rows in sorted(by_model.items()):
        overall = _score_counts(model_rows)
        pos_a = _score_counts([row for row in model_rows if row["candidate_is_a"]])
        pos_b = _score_counts([row for row in model_rows if not row["candidate_is_a"]])
        labels = Counter(row["judge_label"] for row in model_rows)
        summaries.append(
            {
                "candidate_model_slug": slug,
                "label": _label_for_slug(slug, reference_model_slug=reference_model_slug),
                "size_b": _infer_size_b(slug, reference_model_slug=reference_model_slug),
                **overall,
                "q_chat_candidate_A": pos_a["q_chat"],
                "valid_candidate_A": pos_a["valid"],
                "q_chat_candidate_B": pos_b["q_chat"],
                "valid_candidate_B": pos_b["valid"],
                "position_q_delta_A_minus_B": (
                    None
                    if pos_a["q_chat"] is None or pos_b["q_chat"] is None
                    else pos_a["q_chat"] - pos_b["q_chat"]
                ),
                "judge_A_BETTER": labels.get("A_BETTER", 0),
                "judge_B_BETTER": labels.get("B_BETTER", 0),
                "judge_TIE": labels.get("TIE", 0),
                "judge_INVALID": labels.get("INVALID", 0),
            }
        )
    summaries.sort(key=lambda item: item["q_chat"] if item["q_chat"] is not None else -1, reverse=True)
    return summaries


def _summarize_position_bias(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["judge_label"] for row in rows)
    candidate_positions = Counter("A" if row["candidate_is_a"] else "B" for row in rows)
    valid_rows = [row for row in rows if row["outcome"] != "invalid"]
    pos_a = _score_counts([row for row in valid_rows if row["candidate_is_a"]])
    pos_b = _score_counts([row for row in valid_rows if not row["candidate_is_a"]])
    judged_wins = labels.get("A_BETTER", 0) + labels.get("B_BETTER", 0)
    return {
        "judge_label_counts": dict(labels),
        "judge_A_minus_B_winner_rate": None
        if judged_wins == 0
        else (labels.get("A_BETTER", 0) - labels.get("B_BETTER", 0)) / judged_wins,
        "candidate_position_counts": dict(candidate_positions),
        "candidate_A_score": pos_a,
        "candidate_B_score": pos_b,
        "candidate_position_q_delta_A_minus_B": None
        if pos_a["q_chat"] is None or pos_b["q_chat"] is None
        else pos_a["q_chat"] - pos_b["q_chat"],
    }


def _score_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for row in rows if row["outcome"] == "win")
    ties = sum(1 for row in rows if row["outcome"] == "tie")
    losses = sum(1 for row in rows if row["outcome"] == "loss")
    invalid = sum(1 for row in rows if row["outcome"] == "invalid")
    valid = wins + ties + losses
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "invalid": invalid,
        "valid": valid,
        "q_chat": None if valid == 0 else (wins + 0.5 * ties) / valid,
    }


def _infer_size_b(slug: str, *, reference_model_slug: str) -> float | None:
    if slug == reference_model_slug:
        return 8.0
    normalized = slug.lower()
    explicit = {
        "qwen-qwen3-0-6b": 0.6,
        "qwen-qwen3-1-7b": 1.7,
        "google-gemma-4-e2b-it": 2.0,
        "google-gemma-4-e4b-it": 4.0,
        "openai-gpt-oss-20b": 20.0,
        "openai-gpt-oss-120b": 120.0,
    }
    if normalized in explicit:
        return explicit[normalized]
    sizes = []
    for token in re.split(r"[-_]", normalized):
        if re.fullmatch(r"\d+(?:\.\d+)?b", token):
            sizes.append(float(token[:-1]))
    return max(sizes) if sizes else None


def _label_for_slug(slug: str, *, reference_model_slug: str) -> str:
    labels = {
        reference_model_slug: "Llama-3.1-8B ref",
        "meta-llama-llama-3-1-70b-instruct": "Llama-3.1-70B",
        "meta-llama-llama-3-2-1b-instruct": "Llama-3.2-1B",
        "meta-llama-llama-3-2-3b-instruct": "Llama-3.2-3B",
        "meta-llama-llama-2-7b-chat-hf": "Llama-2-7B",
        "meta-llama-llama-2-13b-chat-hf": "Llama-2-13B",
        "meta-llama-llama-2-70b-chat-hf": "Llama-2-70B",
        "qwen-qwen3-0-6b": "Qwen3-0.6B",
        "qwen-qwen3-1-7b": "Qwen3-1.7B",
        "qwen-qwen3-4b-instruct-2507": "Qwen3-4B-I",
        "qwen-qwen3-4b-thinking-2507": "Qwen3-4B-T",
        "qwen-qwen3-8b": "Qwen3-8B",
        "qwen-qwen3-14b": "Qwen3-14B",
        "qwen-qwen3-30b-a3b-instruct-2507": "Qwen3-30B-I",
        "qwen-qwen3-30b-a3b-thinking-2507": "Qwen3-30B-T",
        "qwen-qwen3-32b": "Qwen3-32B",
        "qwen-qwen3-235b-a22b-instruct-2507": "Qwen3-235B-I",
        "qwen-qwen3-235b-a22b-thinking-2507": "Qwen3-235B-T",
        "google-gemma-4-e2b-it": "Gemma-2B",
        "google-gemma-4-e4b-it": "Gemma-4B",
        "google-gemma-4-26b-a4b-it": "Gemma-26B",
        "google-gemma-4-31b-it": "Gemma-31B",
        "openai-gpt-oss-20b": "GPT-OSS-20B",
        "openai-gpt-oss-120b": "GPT-OSS-120B",
    }
    return labels.get(slug, slug)


def _render_markdown(
    *,
    validation: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    model_summary: Sequence[dict[str, Any]],
    position_bias: dict[str, Any],
) -> str:
    labels = Counter(row["judge_label"] for row in rows)
    positions = Counter("A" if row["candidate_is_a"] else "B" for row in rows)
    lines = [
        "# Current Judge Qscore Report",
        "",
        f"- Judge output files: {len(validation)}",
        f"- Parsed judge rows: {len(rows)}",
        f"- Candidate models scored: {len(model_summary)}",
        f"- Invalid judge rows: {sum(1 for row in rows if row['outcome'] == 'invalid')}",
        "",
        "## Position Bias",
        "",
        (
            f"- Judge labels: A_BETTER={labels.get('A_BETTER', 0)}, "
            f"B_BETTER={labels.get('B_BETTER', 0)}, TIE={labels.get('TIE', 0)}, "
            f"INVALID={labels.get('INVALID', 0)}"
        ),
        f"- Candidate position counts: A={positions.get('A', 0)}, B={positions.get('B', 0)}",
        f"- Candidate-position Q delta A-B: {position_bias['candidate_position_q_delta_A_minus_B']:.4f}",
        f"- Raw judge A-vs-B winner-rate delta: {position_bias['judge_A_minus_B_winner_rate']:.4f}",
        "",
        "## Qscore Distribution",
        "",
        "| rank | model | size_B | q_chat | valid | W/T/L | q_A | q_B | q_A-q_B |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for index, row in enumerate(model_summary, start=1):
        lines.append(
            f"| {index} | {row['label']} | {'' if row['size_b'] is None else row['size_b']} | "
            f"{row['q_chat']:.4f} | {row['valid']} | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['q_chat_candidate_A']:.4f} | {row['q_chat_candidate_B']:.4f} | "
            f"{row['position_q_delta_A_minus_B']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _write_plots(
    *,
    model_summary: Sequence[dict[str, Any]],
    reference_model_slug: str,
    plot_path: Path,
    histogram_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [
        row for row in model_summary if row["size_b"] is not None and row["q_chat"] is not None
    ]
    plot_rows.append(
        {
            "candidate_model_slug": reference_model_slug,
            "label": _label_for_slug(reference_model_slug, reference_model_slug=reference_model_slug),
            "size_b": 8.0,
            "q_chat": 0.5,
        }
    )
    plt.figure(figsize=(13, 8))
    for row in plot_rows:
        is_reference = row["candidate_model_slug"] == reference_model_slug
        plt.scatter(
            row["size_b"],
            row["q_chat"],
            s=130 if is_reference else 70,
            marker="*" if is_reference else "o",
            color="black" if is_reference else None,
            zorder=3,
        )
        plt.annotate(row["label"], (row["size_b"], row["q_chat"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Model size (B parameters, log scale)")
    plt.ylabel("Qchat vs Llama-3.1-8B reference")
    plt.title("Chat Quality Qscore by Model Size")
    plt.ylim(0, 1)
    plt.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    scores = [row["q_chat"] for row in model_summary if row["q_chat"] is not None]
    plt.hist(scores, bins=10, edgecolor="black")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Qchat")
    plt.ylabel("Model count")
    plt.title("Current Qscore distribution")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=180)
    plt.close()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
