from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THINK_BLOCK_RE = re.compile(r"^\s*<think>\s*.*?</think>\s*", re.DOTALL | re.IGNORECASE)
REASONING_JOINED_FINAL_ANSWER_BOUNDARY_RE = re.compile(
    r"[.!?][\"'”’)\]]?(?=(?:[#*_`>|\\[]|[A-Za-z]|[^\x00-\x7F]))",
)
REASONING_START_RE = re.compile(
    r"^\s*(?:we\s+need|the\s+user|user\s+(?:asks|says|wants)|the\s+prompt|need\s+to|okay[, ]|"
    r"we\s+should|i\s+need|we\s+are\s+given|this\s+is)\b",
    re.IGNORECASE,
)
REASONING_FINAL_CUE_RE = re.compile(
    r"(?:^|[\n.!?]\s+)(?:"
    r"(?:final\s+)?(?:answer|response|reply|output|tweet|summary)\s*:"
    r"|(?:let'?s|we\s+can|i'?ll)\s+(?:craft|write|produce|provide|give)\s+"
    r"(?:the\s+)?(?:final\s+)?(?:answer|response|reply|output|tweet|summary|code)[^\n]*"
    r"|(?:let'?s|we\s+can|i'?ll)\s+(?:craft|write|produce|provide|give)\s*\.?"
    r"|(?:so|thus|therefore)\s*:"
    r")\s*\n{2,}",
    re.IGNORECASE,
)
LOWERCASE_FINAL_STARTERS = (
    "a ",
    "an ",
    "as ",
    "below ",
    "because ",
    "for ",
    "from ",
    "hello",
    "here",
    "hi",
    "i ",
    "if ",
    "in ",
    "it ",
    "let ",
    "my ",
    "no",
    "our ",
    "since ",
    "sure",
    "that ",
    "the ",
    "these ",
    "this ",
    "those ",
    "to ",
    "we ",
    "when ",
    "yes",
    "you ",
    "your ",
)
PROTECTED_ABBREVIATIONS = {
    "e.g",
    "i.e",
    "vs",
    "etc",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
}


@dataclass(frozen=True, slots=True)
class ResponsePreprocessingResult:
    input_root: Path
    output_root: Path
    response_files: int
    rows: int
    changed_rows: int
    removed_chars: int
    model_summaries: dict[str, dict[str, Any]]
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "response_files": self.response_files,
            "rows": self.rows,
            "changed_rows": self.changed_rows,
            "removed_chars": self.removed_chars,
            "model_summaries": self.model_summaries,
            "summary_path": str(self.summary_path),
        }


def visible_response_text(text: str, *, model_slug: str = "") -> str:
    """Return benchmark/judge-visible answer text with leading reasoning removed."""

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
    if _should_strip_untagged_reasoning(model_slug):
        stripped = _strip_untagged_reasoning_prefix(text)
        if stripped != text:
            return stripped.strip()
    return text


def response_preprocessing_metadata(*, original: str, rendered: str, model_slug: str = "") -> dict[str, Any]:
    removed_chars = len(original) - len(rendered)
    normalized = original.lstrip().lower()
    starts_with_think = normalized.startswith("<think>")
    stripped_orphan_close = not starts_with_think and "</think>" in normalized and removed_chars > 0
    stripped_untagged_reasoning = (
        _should_strip_untagged_reasoning(model_slug)
        and removed_chars > 0
        and REASONING_START_RE.search(original) is not None
    )
    return {
        "stripped_leading_think_block": bool(removed_chars > 0 and starts_with_think),
        "stripped_unclosed_leading_think_block": bool(starts_with_think and rendered == ""),
        "stripped_orphan_think_close_prefix": bool(stripped_orphan_close),
        "stripped_untagged_reasoning_prefix": bool(stripped_untagged_reasoning),
        "stripped_gpt_oss_untagged_reasoning_prefix": bool(
            _is_gpt_oss_model_slug(model_slug) and stripped_untagged_reasoning
        ),
        "original_chars": len(original),
        "rendered_chars": len(rendered),
        "removed_chars": max(0, removed_chars),
        "original_sha256": _sha256_text(original),
        "rendered_sha256": _sha256_text(rendered),
    }


def preprocess_response_tree(
    *,
    responses_root: str | Path,
    output_root: str | Path,
    force: bool = False,
) -> ResponsePreprocessingResult:
    input_root = Path(responses_root).resolve()
    output = Path(output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"responses root does not exist: {input_root}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output root already exists: {output}")
        shutil.rmtree(output)
    if input_root.is_file():
        output.mkdir(parents=True, exist_ok=True)
        response_paths = [input_root]
    else:
        shutil.copytree(input_root, output, ignore=shutil.ignore_patterns("responses.jsonl"))
        response_paths = list(_response_jsonl_paths(input_root))
    if not response_paths:
        raise FileNotFoundError(f"no responses.jsonl files found under {input_root}")

    model_summaries: dict[str, dict[str, Any]] = {}
    total_files = 0
    total_rows = 0
    total_changed = 0
    total_removed = 0
    for source_path in response_paths:
        relative_path = source_path.name if input_root.is_file() else source_path.relative_to(input_root)
        target_path = output / relative_path
        model_slug = _model_slug_for_path(source_path, input_root)
        summary = _preprocess_response_file(source_path, target_path, model_slug=model_slug)
        model_summary = model_summaries.setdefault(
            model_slug,
            {
                "response_files": 0,
                "rows": 0,
                "changed_rows": 0,
                "removed_chars": 0,
                "metadata_counts": {},
            },
        )
        model_summary["response_files"] += 1
        model_summary["rows"] += summary["rows"]
        model_summary["changed_rows"] += summary["changed_rows"]
        model_summary["removed_chars"] += summary["removed_chars"]
        for key, value in summary["metadata_counts"].items():
            model_summary["metadata_counts"][key] = model_summary["metadata_counts"].get(key, 0) + value
        total_files += 1
        total_rows += summary["rows"]
        total_changed += summary["changed_rows"]
        total_removed += summary["removed_chars"]

    result_payload = {
        "input_root": str(input_root),
        "output_root": str(output),
        "response_files": total_files,
        "rows": total_rows,
        "changed_rows": total_changed,
        "removed_chars": total_removed,
        "policy": {
            "strip_leading_think_blocks": True,
            "strip_orphan_think_close_prefix": True,
            "strip_untagged_reasoning_prefix_for_models": ["openai-gpt-oss-*", "*thinking*"],
        },
        "model_summaries": model_summaries,
    }
    summary_path = output / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ResponsePreprocessingResult(
        input_root=input_root,
        output_root=output,
        response_files=total_files,
        rows=total_rows,
        changed_rows=total_changed,
        removed_chars=total_removed,
        model_summaries=model_summaries,
        summary_path=summary_path,
    )


def _preprocess_response_file(source_path: Path, target_path: Path, *, model_slug: str) -> dict[str, Any]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed_rows = 0
    removed_chars = 0
    metadata_counts: dict[str, int] = {}
    with source_path.open("r", encoding="utf-8") as source, target_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source_path}:{line_number}: response row must be a mapping")
            original = row.get("response_text")
            if isinstance(original, str):
                rendered = visible_response_text(original, model_slug=model_slug)
                metadata = response_preprocessing_metadata(
                    original=original,
                    rendered=rendered,
                    model_slug=model_slug,
                )
                row["response_text"] = rendered
                row_metadata = row.get("metadata")
                if not isinstance(row_metadata, dict):
                    row_metadata = {}
                row_metadata["response_preprocessing"] = metadata
                row["metadata"] = row_metadata
                if rendered != original:
                    changed_rows += 1
                    removed_chars += max(0, len(original) - len(rendered))
                for key, value in metadata.items():
                    if isinstance(value, bool) and value:
                        metadata_counts[key] = metadata_counts.get(key, 0) + 1
            rows += 1
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": rows,
        "changed_rows": changed_rows,
        "removed_chars": removed_chars,
        "metadata_counts": metadata_counts,
    }


def _strip_untagged_reasoning_prefix(text: str) -> str:
    stripped = text
    for _ in range(8):
        if not REASONING_START_RE.search(stripped):
            break
        boundary_end = _final_cue_boundary_end(stripped)
        if boundary_end is None:
            boundary_end = _joined_final_boundary_end(stripped)
        if boundary_end is None:
            break
        next_text = stripped[boundary_end:].lstrip()
        if not next_text or next_text == stripped:
            break
        stripped = next_text
    return stripped


def _should_strip_untagged_reasoning(model_slug: str) -> bool:
    normalized = model_slug.lower()
    return _is_gpt_oss_model_slug(normalized) or "thinking" in normalized


def _is_gpt_oss_model_slug(model_slug: str) -> bool:
    return model_slug.lower().startswith("openai-gpt-oss-")


def _joined_final_boundary_end(text: str) -> int | None:
    for match in REASONING_JOINED_FINAL_ANSWER_BOUNDARY_RE.finditer(text):
        punctuation_index = match.start()
        punctuation = text[punctuation_index]
        next_index = match.end()
        if next_index >= len(text):
            continue
        next_char = text[next_index]
        previous_char = text[punctuation_index - 1] if punctuation_index > 0 else ""
        if punctuation == "." and previous_char.isdigit() and next_char.isdigit():
            continue
        if punctuation == "." and previous_char.isalpha() and next_char.isalpha():
            previous_token = _previous_alpha_token(text, punctuation_index)
            if _looks_like_protected_abbreviation(text, punctuation_index, previous_token):
                continue
            if next_char.islower() and not _looks_like_lowercase_final_start(text[next_index:]):
                continue
        return next_index
    return None


def _final_cue_boundary_end(text: str) -> int | None:
    boundary_end = None
    for match in REASONING_FINAL_CUE_RE.finditer(text):
        suffix = text[match.end() :].strip()
        if suffix:
            boundary_end = match.end()
    return boundary_end


def _looks_like_lowercase_final_start(text: str) -> bool:
    normalized = text.lstrip().lower()
    return any(normalized.startswith(starter) for starter in LOWERCASE_FINAL_STARTERS)


def _looks_like_protected_abbreviation(text: str, punctuation_index: int, previous_token: str) -> bool:
    lowered = previous_token.lower()
    if lowered in PROTECTED_ABBREVIATIONS:
        return True
    fragment_start = max(0, punctuation_index - 3)
    if text[fragment_start : punctuation_index + 1].lower() in {"e.g.", "i.e."}:
        return True
    return False


def _previous_alpha_token(text: str, end_index: int) -> str:
    index = end_index - 1
    while index >= 0 and text[index].isalpha():
        index -= 1
    return text[index + 1 : end_index]


def _response_jsonl_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("responses.jsonl")):
        if path.is_file():
            yield path


def _model_slug_for_path(path: Path, input_root: Path) -> str:
    if input_root.is_file():
        return input_root.parent.name
    relative = path.relative_to(input_root)
    if len(relative.parts) == 1:
        return input_root.name
    return relative.parts[0]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
