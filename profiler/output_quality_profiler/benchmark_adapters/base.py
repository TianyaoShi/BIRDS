from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_response_rows(responses_root: str | Path) -> list[dict[str, Any]]:
    root = Path(responses_root).resolve()
    paths = list(response_jsonl_paths(root))
    if not paths:
        raise FileNotFoundError(f"no response JSONL files found under {root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: response row must be a mapping")
                rows.append(row)
    return rows


def response_jsonl_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for name in ("aggregate_responses.jsonl", "responses.jsonl"):
        path = root / name
        if path.is_file():
            yield path
            return
    shards = root / "shards"
    if shards.is_dir():
        yield from sorted(shards.glob("*/responses.jsonl"))
        yield from sorted(shards.glob("*.responses.jsonl"))
        return
    yield from sorted(root.rglob("responses.jsonl"))


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def ground_truth_values(row: dict[str, Any]) -> list[str]:
    meta = metadata(row)
    raw = meta.get("ground_truth", row.get("ground_truth"))
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item is not None]
    return [str(raw)]


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"\s+", " ", value.strip()).split()).lower()


def strip_code_fences(value: str) -> str:
    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def write_score_artifacts(
    *,
    output_dir: str | Path,
    score: dict[str, Any],
    per_item: Sequence[dict[str, Any]],
    markdown_title: str,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "score.json"
    per_item_path = output / "per_item.jsonl"
    markdown_path = output / "score.md"
    score_payload = dict(score)
    score_payload.setdefault("artifacts", {})
    score_payload["artifacts"].update(
        {
            "score_json": str(score_path),
            "per_item_jsonl": str(per_item_path),
            "score_md": str(markdown_path),
        }
    )
    score_path.write_text(json.dumps(score_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with per_item_path.open("w", encoding="utf-8") as handle:
        for row in per_item:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    markdown_path.write_text(render_score_markdown(markdown_title, score_payload), encoding="utf-8")
    return score_payload


def render_score_markdown(title: str, score: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Benchmark: {score.get('benchmark')}",
        f"- Model: {score.get('model')}",
        f"- Overall score: {score.get('overall_score')}",
        f"- Total items: {score.get('total_items')}",
        f"- Scored items: {score.get('scored_items')}",
        f"- Failed generations: {score.get('failed_generations')}",
        f"- Invalid items: {score.get('invalid_items')}",
    ]
    limitation = score.get("compatibility_note")
    if limitation:
        lines.append(f"- Compatibility note: {limitation}")
    return "\n".join(lines) + "\n"


def model_name_from_rows(rows: Sequence[dict[str, Any]]) -> str | None:
    for row in rows:
        model = row.get("model")
        if isinstance(model, str) and model:
            return model
    return None
