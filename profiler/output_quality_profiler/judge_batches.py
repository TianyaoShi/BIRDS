from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


THINK_BLOCK_RE = re.compile(r"^\s*<think>\s*.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class JudgeBatchResult:
    output_jsonl: Path
    manifest_path: Path
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_jsonl": str(self.output_jsonl),
            "manifest_path": str(self.manifest_path),
            "request_count": self.request_count,
        }


def build_openai_judge_batch(
    *,
    responses_root: str | Path,
    reference_model_slug: str,
    candidate_model_slugs: Sequence[str],
    judge_template_path: str | Path,
    output_dir: str | Path,
    evaluator_model: str,
    max_comparisons: int,
    seed: int = 20260520,
    endpoint: str = "/v1/chat/completions",
    max_tokens: int = 256,
    temperature: float = 0.0,
    shard_ids: Sequence[str] = (),
) -> JudgeBatchResult:
    if max_comparisons <= 0:
        raise ValueError("max_comparisons must be positive")
    if not candidate_model_slugs:
        raise ValueError("at least one candidate model slug is required")
    responses_root_path = Path(responses_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    template = Path(judge_template_path).read_text(encoding="utf-8")

    reference_rows = _load_successful_rows(
        responses_root_path / reference_model_slug,
        shard_ids=shard_ids,
    )
    if not reference_rows:
        raise ValueError(f"no successful reference responses found for {reference_model_slug}")

    rng = random.Random(seed)
    comparison_records: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    request_counts_by_candidate: dict[str, int] = {}
    remaining = max_comparisons
    for candidate_slug in candidate_model_slugs:
        if remaining <= 0:
            break
        candidate_rows = _load_successful_rows(
            responses_root_path / candidate_slug,
            shard_ids=shard_ids,
        )
        shared_request_ids = sorted(set(reference_rows) & set(candidate_rows))
        rng.shuffle(shared_request_ids)
        candidate_count = 0
        for request_id in shared_request_ids[:remaining]:
            candidate = candidate_rows[request_id]
            reference = reference_rows[request_id]
            candidate_is_a = bool(rng.getrandbits(1))
            candidate_response = _judge_visible_response_text(candidate["response_text"])
            reference_response = _judge_visible_response_text(reference["response_text"])
            response_a = candidate_response if candidate_is_a else reference_response
            response_b = reference_response if candidate_is_a else candidate_response
            prompt = candidate.get("prompt") or reference.get("prompt") or ""
            custom_id = _custom_id(
                candidate_slug=candidate_slug,
                reference_slug=reference_model_slug,
                request_id=request_id,
                index=len(batch_rows),
            )
            user_content = _render_template(
                template,
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
            )
            batch_rows.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": endpoint,
                    "body": {
                        "model": evaluator_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_content,
                            }
                        ],
                        "temperature": temperature,
                        _max_tokens_field(evaluator_model): max_tokens,
                        "response_format": {"type": "json_object"},
                    },
                }
            )
            comparison_records.append(
                {
                    "custom_id": custom_id,
                    "candidate_model_slug": candidate_slug,
                    "reference_model_slug": reference_model_slug,
                    "request_id": request_id,
                    "candidate_is_a": candidate_is_a,
                    "candidate_model": candidate.get("model"),
                    "reference_model": reference.get("model"),
                    "source": candidate.get("source") or reference.get("source"),
                    "prompt_length_bucket": candidate.get("prompt_length_bucket")
                    or reference.get("prompt_length_bucket"),
                    "candidate_response_preprocessing": _response_preprocessing_metadata(
                        original=candidate["response_text"],
                        rendered=candidate_response,
                    ),
                    "reference_response_preprocessing": _response_preprocessing_metadata(
                        original=reference["response_text"],
                        rendered=reference_response,
                    ),
                }
            )
            candidate_count += 1
        request_counts_by_candidate[candidate_slug] = candidate_count
        remaining = max_comparisons - len(batch_rows)

    if not batch_rows:
        raise ValueError("no overlapping successful responses found for requested comparison set")

    output_jsonl = output_dir_path / "batch_000.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in batch_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = output_dir_path / "batch_manifest.json"
    manifest = {
        "format": "openai_batch_chat_completions",
        "endpoint": endpoint,
        "evaluator_model": evaluator_model,
        "reference_model_slug": reference_model_slug,
        "candidate_model_slugs": list(candidate_model_slugs),
        "seed": seed,
        "request_count": len(batch_rows),
        "request_counts_by_candidate": request_counts_by_candidate,
        "output_jsonl": str(output_jsonl),
        "judge_template_path": str(Path(judge_template_path).resolve()),
        "shard_ids": list(shard_ids),
        "max_tokens_field": _max_tokens_field(evaluator_model),
        "response_preprocessing": {
            "strip_leading_think_blocks": True,
            "policy": "remove a completed leading <think>...</think> block before judge prompt composition",
        },
        "comparisons": comparison_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return JudgeBatchResult(
        output_jsonl=output_jsonl,
        manifest_path=manifest_path,
        request_count=len(batch_rows),
    )


def _load_successful_rows(
    model_dir: Path,
    *,
    shard_ids: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in _response_jsonl_paths(model_dir, shard_ids=shard_ids):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("success", False):
                    continue
                request_id = row.get("request_id")
                response_text = row.get("response_text")
                if not isinstance(request_id, str) or not isinstance(response_text, str):
                    continue
                rows.setdefault(request_id, row)
    return rows


def _render_template(
    template: str,
    *,
    prompt: str,
    response_a: str,
    response_b: str,
) -> str:
    return (
        template.replace("{prompt}", prompt)
        .replace("{response_a}", response_a)
        .replace("{response_b}", response_b)
    )


def _judge_visible_response_text(text: str) -> str:
    normalized = text.lstrip().lower()
    if normalized.startswith("<think>"):
        stripped = THINK_BLOCK_RE.sub("", text, count=1)
        if stripped != text:
            return stripped.strip()
        return ""
    orphan_close_index = normalized.find("</think>")
    if orphan_close_index >= 0:
        leading_ws = len(text) - len(text.lstrip())
        return text[leading_ws + orphan_close_index + len("</think>") :].strip()
    return text


def _response_preprocessing_metadata(*, original: str, rendered: str) -> dict[str, Any]:
    removed_chars = len(original) - len(rendered)
    normalized = original.lstrip().lower()
    starts_with_think = normalized.startswith("<think>")
    stripped_orphan_close = not starts_with_think and "</think>" in normalized and removed_chars > 0
    return {
        "stripped_leading_think_block": bool(removed_chars > 0 and starts_with_think),
        "stripped_unclosed_leading_think_block": bool(starts_with_think and rendered == ""),
        "stripped_orphan_think_close_prefix": bool(stripped_orphan_close),
        "original_chars": len(original),
        "rendered_chars": len(rendered),
        "removed_chars": max(0, removed_chars),
    }


def _max_tokens_field(model: str) -> str:
    normalized = model.lower()
    if normalized.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def _response_jsonl_paths(model_dir: Path, *, shard_ids: Sequence[str] = ()) -> Iterable[Path]:
    aggregate = model_dir / "responses.jsonl"
    if aggregate.is_file() and not shard_ids:
        yield aggregate
        return
    shards_dir = model_dir / "shards"
    if shards_dir.is_dir():
        for path in sorted(shards_dir.glob("*/responses.jsonl")):
            if shard_ids and not _shard_matches(path.parent.name, shard_ids):
                continue
            yield path


def _shard_matches(shard_name: str, shard_ids: Sequence[str]) -> bool:
    normalized = _normalize_shard_id(shard_name)
    return any(normalized == _normalize_shard_id(shard_id) for shard_id in shard_ids)


def _normalize_shard_id(value: str) -> str:
    return value.replace("_", "-").lower()


def _custom_id(
    *,
    candidate_slug: str,
    reference_slug: str,
    request_id: str,
    index: int,
) -> str:
    safe_request_id = "".join(ch if ch.isalnum() else "-" for ch in request_id).strip("-")
    safe_request_id = safe_request_id[:48] or f"request-{index:06d}"
    return f"judge-{index:06d}-{candidate_slug}-vs-{reference_slug}-{safe_request_id}"[:512]
