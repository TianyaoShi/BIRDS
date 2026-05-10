from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


SUPPORTED_DATASET_KINDS: dict[str, str] = {
    "crosscodeeval": "code_completion",
    "aime": "reasoning_qa",
    "gpqa": "reasoning_qa",
    "longbench": "long_context_nlp",
    "mmlu": "reasoning_qa",
    "mmlu_pro": "reasoning_qa",
    "natural_reasoning": "reasoning_qa",
    "repobench": "code_completion",
    "supergpqa": "reasoning_qa",
}


def dataset_kind(dataset_name: str) -> str:
    try:
        return SUPPORTED_DATASET_KINDS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_DATASET_KINDS))
        raise ValueError(f"supported dataset.name values: {supported}") from exc


def config_raw_path(dataset: dict[str, Any], *, base_dir: Path) -> Path | None:
    raw_path = dataset.get("raw_path")
    if raw_path is None:
        return None
    return resolve_path(expect_string(raw_path, "dataset.raw_path"), base_dir=base_dir)


def prompt_template(value: Any) -> str:
    template = optional_string(value, "prompt_template") or "plain_prefix"
    if template not in {"plain_prefix", "xml_tags"}:
        raise ValueError("prompt_template must be one of: plain_prefix, xml_tags")
    return template


def resolve_path(path: str, *, base_dir: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return optional_mapping(payload[key], key)


def optional_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def required_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return expect_string(payload[key], key)


def optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return expect_string(value, field_name)


def expect_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [expect_string(item, f"{field_name}[]") for item in value]


def int_setting(payload: dict[str, Any], key: str, default: int) -> int:
    return positive_int(payload.get(key, default), key)


def positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def language_filter(payload: Any) -> dict[str, set[str]]:
    languages = optional_mapping(payload, "filtering.languages")
    return {
        "include": set(string_list(languages.get("include", []), "filtering.languages.include")),
        "exclude": set(string_list(languages.get("exclude", []), "filtering.languages.exclude")),
    }


def language_allowed(language: str, language_filter_payload: dict[str, set[str]]) -> bool:
    include = language_filter_payload["include"]
    exclude = language_filter_payload["exclude"]
    return (not include or language in include) and language not in exclude


def dedup_content_hash(payload: Any) -> bool:
    dedup = optional_mapping(payload, "filtering.dedup")
    value = dedup.get("content_hash", True)
    if not isinstance(value, bool):
        raise ValueError("filtering.dedup.content_hash must be a boolean")
    return value


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_files(raw_path: Path) -> list[Path]:
    if raw_path.is_file():
        if raw_path.suffix != ".jsonl":
            raise ValueError(f"raw_path file must be .jsonl: {raw_path}")
        return [raw_path]
    if raw_path.is_dir():
        files = sorted(raw_path.rglob("*.jsonl"))
        if files:
            return files
        raise FileNotFoundError(f"raw_path directory has no .jsonl files: {raw_path}")
    raise FileNotFoundError(f"raw_path not found: {raw_path}")
