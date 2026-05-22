from __future__ import annotations

import ast
import keyword
import re
from pathlib import Path
from typing import Any

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    strip_code_fences,
    write_score_artifacts,
)

try:  # pragma: no cover - depends on benchmark packages.
    from fuzzywuzzy import fuzz as _fuzz
except ImportError:  # pragma: no cover - covered through dependency report.
    _fuzz = None

try:  # pragma: no cover - depends on benchmark packages.
    from nltk.tokenize import RegexpTokenizer as _RegexpTokenizer
except ImportError:  # pragma: no cover - covered through dependency report.
    _RegexpTokenizer = None

try:  # pragma: no cover - depends on benchmark packages.
    from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International as _TokenizerV14International
except ImportError:  # pragma: no cover - covered through dependency report.
    _TokenizerV14International = None

try:  # pragma: no cover - depends on benchmark packages.
    import timeout_decorator as _timeout_decorator
except ImportError:  # pragma: no cover - covered through dependency report.
    _timeout_decorator = None


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_STRING_RE = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
_LINE_COMMENT_RE = re.compile(r"#.*|//.*")
_CODE_TOKENIZER = _RegexpTokenizer(r"\w+") if _RegexpTokenizer is not None else None
_STRING_TOKENIZER = _TokenizerV14International() if _TokenizerV14International is not None else None

_LANGUAGE_KEYWORDS: dict[str, set[str]] = {
    "python": set(keyword.kwlist),
    "java": {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",
        "true",
        "false",
        "null",
    },
    "typescript": {
        "abstract",
        "any",
        "as",
        "async",
        "await",
        "boolean",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "constructor",
        "continue",
        "debugger",
        "declare",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "from",
        "function",
        "get",
        "if",
        "implements",
        "import",
        "in",
        "infer",
        "instanceof",
        "interface",
        "is",
        "keyof",
        "let",
        "module",
        "namespace",
        "never",
        "new",
        "null",
        "number",
        "of",
        "private",
        "protected",
        "public",
        "readonly",
        "require",
        "return",
        "set",
        "static",
        "string",
        "super",
        "switch",
        "symbol",
        "this",
        "throw",
        "true",
        "try",
        "type",
        "typeof",
        "undefined",
        "var",
        "void",
        "while",
        "with",
        "yield",
    },
    "csharp": {
        "abstract",
        "as",
        "base",
        "bool",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "checked",
        "class",
        "const",
        "continue",
        "decimal",
        "default",
        "delegate",
        "do",
        "double",
        "else",
        "enum",
        "event",
        "explicit",
        "extern",
        "false",
        "finally",
        "fixed",
        "float",
        "for",
        "foreach",
        "goto",
        "if",
        "implicit",
        "in",
        "int",
        "interface",
        "internal",
        "is",
        "lock",
        "long",
        "namespace",
        "new",
        "null",
        "object",
        "operator",
        "out",
        "override",
        "params",
        "private",
        "protected",
        "public",
        "readonly",
        "ref",
        "return",
        "sbyte",
        "sealed",
        "short",
        "sizeof",
        "stackalloc",
        "static",
        "string",
        "struct",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "uint",
        "ulong",
        "unchecked",
        "unsafe",
        "ushort",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    },
}


def score_code_completion_responses(
    *,
    responses_root: str | Path,
    output_dir: str | Path,
    benchmark_name: str,
) -> dict[str, Any]:
    require_crosscodeeval_dependencies()
    rows = load_response_rows(responses_root)
    per_item: list[dict[str, Any]] = []
    failed = invalid = scored = exact_matches = id_exact_matches = 0
    edit_similarities: list[float] = []
    id_precisions: list[float] = []
    id_recalls: list[float] = []
    id_f1s: list[float] = []
    by_language: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        language = str(meta.get("language") or "unknown")
        if not row.get("success", False):
            failed += 1
            per_item.append(
                {"request_id": request_id, "language": language, "invalid_reason": "failed_generation"}
            )
            continue
        truth_values = ground_truth_values(row)
        if not truth_values:
            invalid += 1
            per_item.append(
                {"request_id": request_id, "language": language, "invalid_reason": "missing_ground_truth"}
            )
            continue
        target = truth_values[0]
        prompt = str(
            meta.get("current_file_prefix")
            or row.get("prompt")
            or meta.get("prompt")
            or ""
        )
        completion = str(row.get("response_text") or "")
        if _is_gemma_completion_row(row, meta):
            completion = sanitize_gemma_completion(completion)
        prediction = crosscodeeval_postprocess_prediction(
            prompt=prompt,
            completion=completion,
            language=language,
        )
        target_processed = remove_code_comments(target)
        exact_match = crosscodeeval_exact_match(prediction, target_processed)
        edit_similarity = crosscodeeval_edit_similarity(prediction, target_processed)
        pred_ids = extract_identifiers(prediction, language)
        target_ids = extract_identifiers(target_processed, language)
        id_exact_match = pred_ids == target_ids
        id_precision, id_recall, id_f1 = identifier_prf(pred_ids, target_ids)
        item = {
            "request_id": request_id,
            "sample_id": meta.get("sample_id"),
            "language": language,
            "repo_id": meta.get("repo_id"),
            "file_path": meta.get("file_path"),
            "sequence_index": meta.get("sequence_index"),
            "prediction": prediction,
            "ground_truth": target_processed,
            "raw_ground_truth": target,
            "exact_match": exact_match,
            "edit_similarity": edit_similarity,
            "identifier_exact_match": id_exact_match,
            "identifier_precision": id_precision,
            "identifier_recall": id_recall,
            "identifier_f1": id_f1,
            "pred_identifiers": pred_ids,
            "target_identifiers": target_ids,
        }
        scored += 1
        exact_matches += int(exact_match)
        id_exact_matches += int(id_exact_match)
        edit_similarities.append(edit_similarity)
        id_precisions.append(id_precision)
        id_recalls.append(id_recall)
        id_f1s.append(id_f1)
        by_language.setdefault(language, []).append(item)
        per_item.append(item)
    score = {
        "benchmark": benchmark_name,
        "model": model_name_from_rows(rows),
        "adapter": "crosscodeeval_official_metrics_v1",
        "metric": "exact_match_percent",
        "overall_score": None if scored == 0 else 100.0 * exact_matches / scored,
        "exact_match_percent": None if scored == 0 else 100.0 * exact_matches / scored,
        "edit_similarity_percent": None if not edit_similarities else sum(edit_similarities) / len(edit_similarities),
        "identifier_exact_match_percent": None if scored == 0 else 100.0 * id_exact_matches / scored,
        "identifier_precision_percent": _mean_percent(id_precisions),
        "identifier_recall_percent": _mean_percent(id_recalls),
        "identifier_f1_percent": _mean_percent(id_f1s),
        "total_items": len(rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "by_language": _crosscodeeval_group_payload(by_language),
        "is_full_benchmark": True,
        "dependency_status": crosscodeeval_dependency_status(),
        "score_interpretation": (
            "CrossCodeEval official evaluation reports Code Matching EM/ES and Identifier "
            "Matching ID-EM/ID-F1. overall_score is Code Matching exact-match percent."
        ),
        "compatibility_note": (
            "Implements the CrossCodeEval metric semantics locally: generated completions are "
            "truncated to one statement, comments are removed, EM compares stripped nonempty "
            "lines, ES is fuzzywuzzy fuzz.ratio-compatible edit similarity, and identifier "
            "metrics use regex identifiers after string/comment removal."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title=f"{benchmark_name} Score",
    )


def normalize_code_completion(value: str) -> str:
    text = strip_code_fences(value)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def crosscodeeval_postprocess_prediction(*, prompt: str, completion: str, language: str) -> str:
    text = strip_code_fences(completion)
    if language in {"java", "csharp", "typescript"}:
        text = _first_bracket_language_statement(text)
    elif language == "python":
        text = _first_python_statement(prompt, text)
    return remove_code_comments(text)


def sanitize_gemma_completion(value: str) -> str:
    text = value
    for marker in (
        "<|channel>thought\n<channel|>",
        "<|channel>thought",
        "<channel|>",
        "<|turn>model",
        "<|turn>assistant",
        "<turn|>",
    ):
        text = text.replace(marker, "")
    for marker in ("<COMPLETION>", "<CURSOR>"):
        if marker in text:
            text = text.split(marker, 1)[-1]
    for marker in ("</COMPLETION>", "</TARGET_FILE>", "</REPOSITORY_CONTEXT>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip("\r\n ")


def _is_gemma_completion_row(row: dict[str, Any], meta: dict[str, Any]) -> bool:
    model = str(row.get("model") or "").lower()
    prompt_template = str(meta.get("prompt_template") or "")
    return "gemma" in model or prompt_template == "gemma_chat_completion"


def _first_bracket_language_statement(completion: str) -> str:
    for index, character in enumerate(completion):
        if character in {";", "}", "{"}:
            return completion[: index + 1]
    return completion


def _first_python_statement(prompt: str, completion: str) -> str:
    for index, character in enumerate(completion):
        if character != "\n":
            continue
        candidate = completion[:index].rstrip()
        if not candidate:
            continue
        if _python_prompt_plus_completion_parseable(prompt, candidate):
            return candidate
    return completion


def _python_prompt_plus_completion_parseable(prompt: str, completion: str) -> bool:
    if _timeout_decorator is None:
        require_crosscodeeval_dependencies()

    @_timeout_decorator.timeout(5)
    def parse() -> bool:
        try:
            ast.parse(prompt + completion)
        except SyntaxError:
            return False
        return True

    try:
        return parse()
    except Exception:
        return False


def remove_code_comments(code: str) -> str:
    return _LINE_COMMENT_RE.sub("", code)


def crosscodeeval_exact_match(prediction: str, ground_truth: str) -> bool:
    pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    gt_lines = [line.strip() for line in ground_truth.splitlines() if line.strip()]
    return pred_lines == gt_lines


def crosscodeeval_edit_similarity(prediction: str, ground_truth: str) -> float:
    if _fuzz is None:
        require_crosscodeeval_dependencies()
    prediction = prediction.strip()
    ground_truth = ground_truth.strip()
    return float(_fuzz.ratio(prediction, ground_truth))


def extract_identifiers(source_code: str, language: str) -> list[str]:
    if _CODE_TOKENIZER is None:
        require_crosscodeeval_dependencies()
    without_strings = _STRING_RE.sub("", source_code)
    keywords = _LANGUAGE_KEYWORDS.get(language, set())
    return [
        token
        for token in _CODE_TOKENIZER.tokenize(without_strings)
        if _IDENTIFIER_RE.match(token) and token not in keywords
    ]


def identifier_prf(pred_ids: list[str], target_ids: list[str]) -> tuple[float, float, float]:
    pred = set(pred_ids)
    target = set(target_ids)
    tp = len(pred & target)
    fp = len(pred - target)
    fn = len(target - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return precision, recall, f1


def crosscodeeval_dependency_status() -> dict[str, bool]:
    return {
        "fuzzywuzzy": _fuzz is not None,
        "nltk": _RegexpTokenizer is not None,
        "sacrebleu": _TokenizerV14International is not None,
        "timeout_decorator": _timeout_decorator is not None,
        "string_tokenizer_initialized": _STRING_TOKENIZER is not None,
    }


def require_crosscodeeval_dependencies() -> None:
    missing = [
        name
        for name, installed in crosscodeeval_dependency_status().items()
        if name != "string_tokenizer_initialized" and not installed
    ]
    if missing:
        raise RuntimeError(
            "CrossCodeEval official metric dependencies are missing: "
            + ", ".join(missing)
            + ". Please install the original CCEval metric dependencies before scoring."
        )


def _mean_percent(values: list[float]) -> float | None:
    return None if not values else 100.0 * sum(values) / len(values)


def _crosscodeeval_group_payload(groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for language, values in sorted(groups.items()):
        edit_scores = [float(item["edit_similarity"]) for item in values]
        id_precisions = [float(item["identifier_precision"]) for item in values]
        id_recalls = [float(item["identifier_recall"]) for item in values]
        id_f1s = [float(item["identifier_f1"]) for item in values]
        payload[language] = {
            "count": len(values),
            "exact_match_percent": (
                None if not values else 100.0 * sum(item["exact_match"] for item in values) / len(values)
            ),
            "edit_similarity_percent": None if not edit_scores else sum(edit_scores) / len(edit_scores),
            "identifier_exact_match_percent": (
                None
                if not values
                else 100.0 * sum(item["identifier_exact_match"] for item in values) / len(values)
            ),
            "identifier_precision_percent": _mean_percent(id_precisions),
            "identifier_recall_percent": _mean_percent(id_recalls),
            "identifier_f1_percent": _mean_percent(id_f1s),
        }
    return payload
