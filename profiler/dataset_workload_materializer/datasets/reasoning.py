from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from llm_mst_finder.workload import PromptTokenizer

from ..common import (
    expect_string,
    hash_text,
    jsonl_files,
    optional_mapping,
    optional_string,
    string_list,
)
from ..models import DatasetLoadResult, MaterializationContext, MaterializedSample
from .code import lookup_alias, metadata_alias


DEFAULT_REASONING_FIELD_ALIASES: dict[str, list[str]] = {
    "question": ["question", "Question", "prompt", "input", "problem", "Problem"],
    "choices": ["choices", "options", "answer_choices", "Options"],
    "answer": [
        "answer_letter",
        "answer",
        "Answer",
        "reference_answer",
        "target",
        "correct_answer",
        "Correct Answer",
        "label",
        "answer_idx",
        "answer_index",
        "gold",
        "final_answer",
    ],
    "subject": [
        "subject",
        "category",
        "discipline",
        "field",
        "subfield",
        "subdomain",
        "Subject",
        "Subdomain",
        "High-level domain",
    ],
    "record_id": ["id", "_id", "uuid", "question_id", "problem_id", "record_id", "Record ID", "problem_idx"],
    "year": ["year", "contest_year"],
    "difficulty": ["difficulty", "Difficulty", "level"],
}

GPQA_INCORRECT_ALIASES = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHOICE_PREFIX_RE = re.compile(r"^\s*([A-Z])[\).:]\s+")


def load_reasoning_dataset(dataset: dict[str, Any], ctx: MaterializationContext) -> DatasetLoadResult:
    if ctx.raw_path is None:
        raise ValueError(f"{ctx.dataset_name} materialization requires dataset.raw_path")
    task = (
        optional_string(dataset.get("task"), "dataset.task")
        or optional_string(dataset.get("mode"), "dataset.mode")
        or ctx.dataset_name
    )
    prompt_template_name = reasoning_prompt_template(dataset.get("prompt_template"))
    aliases = reasoning_field_aliases(dataset.get("field_aliases", {}))
    difficulty_filter = reasoning_difficulty_filter(dataset)
    samples = load_reasoning_samples(
        ctx.raw_path,
        aliases=aliases,
        difficulty_filter=difficulty_filter,
        dataset_name=ctx.dataset_name,
        dataset_kind=ctx.dataset_kind,
        split=ctx.split,
        task=task,
        prompt_template=prompt_template_name,
        tokenizer=ctx.tokenizer,
        filtering=ctx.filtering,
        counters=ctx.counters,
        seed=ctx.sampling.seed,
    )
    return DatasetLoadResult(samples=samples, task=task, prompt_template=prompt_template_name)


def load_reasoning_samples(
    raw_path: Path,
    *,
    aliases: dict[str, list[str]],
    difficulty_filter: set[str] | None,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    tokenizer: PromptTokenizer,
    filtering,
    counters,
    seed: int,
) -> list[MaterializedSample]:
    samples: list[MaterializedSample] = []
    seen_content_hashes: set[str] = set()
    global_index = 0
    for file_path in jsonl_files(raw_path):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                counters.total_rows += 1
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{file_path}:{line_index + 1} must be a JSON object")
                sample = reasoning_row_to_sample(
                    row,
                    row_index=global_index,
                    input_path=file_path,
                    aliases=aliases,
                    difficulty_filter=difficulty_filter,
                    dataset_name=dataset_name,
                    dataset_kind=dataset_kind,
                    split=split,
                    task=task,
                    prompt_template=prompt_template,
                    tokenizer=tokenizer,
                    filtering=filtering,
                    seen_content_hashes=seen_content_hashes,
                    counters=counters,
                    seed=seed,
                    source=f"{file_path}:{line_index + 1}",
                )
                global_index += 1
                if sample is not None:
                    samples.append(sample)
    counters.materialized_rows = len(samples)
    return samples


def reasoning_row_to_sample(
    row: dict[str, Any],
    *,
    row_index: int,
    input_path: Path,
    aliases: dict[str, list[str]],
    difficulty_filter: set[str] | None,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    tokenizer: PromptTokenizer,
    filtering,
    seen_content_hashes: set[str],
    counters,
    seed: int,
    source: str,
) -> MaterializedSample | None:
    question = string_alias(row, aliases["question"], field_name="question", source=source)
    if question is None:
        counters.drops["missing_empty_prompt"] += 1
        return None
    difficulty = metadata_alias(row, aliases["difficulty"], "")
    difficulty_text = "" if difficulty in (None, "") else str(difficulty)
    if difficulty_filter is not None and difficulty_text.strip().lower() not in difficulty_filter:
        counters.drops["difficulty_not_selected"] += 1
        return None

    choices, answer_text_from_choices = reasoning_choices(row, aliases=aliases, row_index=row_index, seed=seed)
    answer_value = value_alias(row, aliases["answer"])
    answer = normalize_answer(
        answer_value,
        choices=choices,
        answer_text_from_choices=answer_text_from_choices,
        source=source,
    )
    if answer is None:
        counters.drops["missing_empty_target"] += 1
        return None

    prompt = render_reasoning_prompt(
        question,
        choices=choices,
        prompt_template=prompt_template,
    )
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(answer.ground_truth))
    if prompt_token_count < filtering.min_prompt_tokens:
        counters.drops["prompt_too_short"] += 1
        return None
    if prompt_token_count > filtering.max_prompt_tokens:
        counters.drops["prompt_too_long"] += 1
        return None
    if target_token_count < filtering.min_target_tokens:
        counters.drops["missing_empty_target"] += 1
        return None
    if target_token_count > filtering.max_target_tokens:
        counters.drops["target_too_long"] += 1
        return None

    source_metadata = optional_mapping(row.get("metadata"), f"{source}.metadata")
    subject = str(metadata_alias(row, aliases["subject"], task))
    record_id = metadata_alias(row, aliases["record_id"], f"{row_index:06d}")
    year = metadata_alias(row, aliases["year"], "")
    content_hash = hash_text(prompt)
    if filtering.dedup_content_hash and content_hash in seen_content_hashes:
        counters.drops["duplicate_content_hash"] += 1
        return None
    seen_content_hashes.add(content_hash)
    sample_id = f"{dataset_name}-{task}-{row_index:06d}"
    metadata = {
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "profile": "long_reasoning",
        "workload_type": "problem_solving",
        "output_regime": "natural_until_eos",
        "prompt_template": prompt_template,
        "split": split,
        "language": "en",
        "subject": subject,
        "file_path": input_path.as_posix(),
        "session_id": f"{dataset_name}::{task}::{subject}",
        "sample_id": sample_id,
        "sequence_index": row_index,
        "content_hash": content_hash,
        "ground_truth": answer.ground_truth,
        "ground_truth_text": answer.ground_truth_text,
        "target_hash": hash_text(answer.ground_truth),
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "record_id": record_id,
    }
    metadata.update(source_metadata)
    if choices:
        metadata["choices"] = [{"label": label, "text": text} for label, text in choices]
        metadata["answer_label"] = answer.ground_truth
    if year not in (None, ""):
        metadata["year"] = year
    if difficulty_text:
        metadata["difficulty"] = difficulty_text
    return MaterializedSample(
        sample_id=sample_id,
        prompt=prompt,
        target=answer.ground_truth,
        expected_output_len=max(1, target_token_count),
        metadata=metadata,
    )


class NormalizedAnswer:
    def __init__(self, *, ground_truth: str, ground_truth_text: str) -> None:
        self.ground_truth = ground_truth
        self.ground_truth_text = ground_truth_text


def reasoning_choices(
    row: dict[str, Any],
    *,
    aliases: dict[str, list[str]],
    row_index: int,
    seed: int,
) -> tuple[list[tuple[str, str]], str | None]:
    correct_answer = string_value(row.get("Correct Answer"))
    incorrect_answers = [string_value(row.get(alias)) for alias in GPQA_INCORRECT_ALIASES]
    if correct_answer is not None and all(answer is not None for answer in incorrect_answers):
        texts = [correct_answer, *[answer for answer in incorrect_answers if answer is not None]]
        rng = random.Random(f"{seed}:{row_index}:{correct_answer}")
        rng.shuffle(texts)
        choices = [(CHOICE_LABELS[index], text) for index, text in enumerate(texts)]
        return choices, correct_answer

    choices_value = value_alias(row, aliases["choices"])
    if choices_value is not None:
        return normalize_choices_payload(choices_value), None

    field_choices: list[tuple[str, str]] = []
    for label in CHOICE_LABELS[:10]:
        value = string_value(row.get(label))
        if value is not None:
            field_choices.append((label, strip_choice_prefix(value)))
    return field_choices, None


def normalize_choices_payload(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        choices = []
        for key in sorted(value):
            label = str(key).strip().upper()
            if len(label) != 1 or label not in CHOICE_LABELS:
                raise ValueError(f"choice label must be A-Z, got {key!r}")
            text = string_value(value[key])
            if text is None:
                raise ValueError(f"choice {key!r} must be a non-empty string")
            choices.append((label, strip_choice_prefix(text)))
        if not choices:
            raise ValueError("choices mapping must not be empty")
        return choices
    if isinstance(value, list):
        choices = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                label = string_value(item.get("label")) or CHOICE_LABELS[index]
                text = string_value(item.get("text") or item.get("value") or item.get("choice"))
            else:
                label = CHOICE_LABELS[index]
                text = string_value(item)
            if text is None:
                raise ValueError(f"choices[{index}] must contain non-empty text")
            label = label.strip().upper()
            if len(label) != 1 or label not in CHOICE_LABELS:
                raise ValueError(f"choices[{index}].label must be A-Z")
            choices.append((label, strip_choice_prefix(text)))
        if not choices:
            raise ValueError("choices list must not be empty")
        return choices
    raise ValueError("choices must be a list or mapping")


def normalize_answer(
    value: Any,
    *,
    choices: list[tuple[str, str]],
    answer_text_from_choices: str | None,
    source: str,
) -> NormalizedAnswer | None:
    if choices:
        if answer_text_from_choices is not None:
            label = label_for_choice_text(answer_text_from_choices, choices)
            return NormalizedAnswer(ground_truth=label, ground_truth_text=answer_text_from_choices)
        if value is None or value == "":
            return None
        if isinstance(value, int):
            if value < 0 or value >= len(choices):
                raise ValueError(f"{source} answer index out of range")
            label, text = choices[value]
            return NormalizedAnswer(ground_truth=label, ground_truth_text=text)
        answer_text = str(value).strip()
        if answer_text.isdigit():
            index = int(answer_text)
            if index < 0 or index >= len(choices):
                raise ValueError(f"{source} answer index out of range")
            label, text = choices[index]
            return NormalizedAnswer(ground_truth=label, ground_truth_text=text)
        upper = answer_text.upper()
        choice_labels = {label for label, _ in choices}
        if upper in choice_labels:
            return NormalizedAnswer(
                ground_truth=upper,
                ground_truth_text=dict(choices)[upper],
            )
        try:
            label = label_for_choice_text(answer_text, choices)
        except ValueError as exc:
            raise ValueError(f"{source} answer does not match any choice: {answer_text!r}") from exc
        return NormalizedAnswer(ground_truth=label, ground_truth_text=answer_text)

    answer_text = string_value(value)
    if answer_text is None:
        return None
    return NormalizedAnswer(ground_truth=answer_text, ground_truth_text=answer_text)


def label_for_choice_text(answer_text: str, choices: list[tuple[str, str]]) -> str:
    normalized = normalize_text(answer_text)
    for label, text in choices:
        if normalize_text(text) == normalized:
            return label
    raise ValueError("answer text does not match choices")


def render_reasoning_prompt(
    question: str,
    *,
    choices: list[tuple[str, str]],
    prompt_template: str,
) -> str:
    if prompt_template not in {"reasoning_auto", "reasoning_mcq", "reasoning_free_response"}:
        raise ValueError(f"unsupported reasoning prompt_template: {prompt_template}")
    if choices and prompt_template != "reasoning_free_response":
        choices_text = "\n".join(f"{label}. {text}" for label, text in choices)
        return (
            "Question:\n"
            f"{question.strip()}\n\n"
            "Choices:\n"
            f"{choices_text}\n\n"
            "Think step by step, then put the final answer on its own line as "
            "'Answer: <letter>'."
        )
    return (
        "Problem:\n"
        f"{question.strip()}\n\n"
        "Think step by step, then put the final answer on its own line as "
        "'Answer: <answer>'."
    )


def reasoning_prompt_template(value: Any) -> str:
    template = optional_string(value, "dataset.prompt_template") or "reasoning_auto"
    if template not in {"reasoning_auto", "reasoning_mcq", "reasoning_free_response"}:
        raise ValueError(
            "dataset.prompt_template must be one of: reasoning_auto, "
            "reasoning_mcq, reasoning_free_response"
        )
    return template


def reasoning_difficulty_filter(dataset: dict[str, Any]) -> set[str] | None:
    raw_value = dataset.get("difficulties", dataset.get("difficulty"))
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        values = [raw_value]
    else:
        values = string_list(raw_value, "dataset.difficulties")
    selected = {value.strip().lower() for value in values if value.strip()}
    if not selected:
        raise ValueError("dataset.difficulties must contain at least one non-empty value")
    return selected


def reasoning_field_aliases(payload: Any) -> dict[str, list[str]]:
    aliases = {key: list(value) for key, value in DEFAULT_REASONING_FIELD_ALIASES.items()}
    overrides = optional_mapping(payload, "dataset.field_aliases")
    for key, value in overrides.items():
        if key not in aliases:
            raise ValueError(f"dataset.field_aliases has unknown key: {key}")
        if not isinstance(value, list) or not value:
            raise ValueError(f"dataset.field_aliases.{key} must be a non-empty list")
        aliases[key] = [expect_string(item, f"dataset.field_aliases.{key}[]") for item in value]
    return aliases


def value_alias(row: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        found, value = lookup_alias(row, alias)
        if found and value not in (None, ""):
            return value
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            found, value = lookup_alias(metadata, alias)
            if found and value not in (None, ""):
                return value
    return None


def string_alias(
    row: dict[str, Any],
    aliases: list[str],
    *,
    field_name: str,
    source: str,
) -> str | None:
    value = value_alias(row, aliases)
    if value in (None, ""):
        return None
    text = string_value(value)
    if text is None:
        raise ValueError(f"{source} field for {field_name} must be a string")
    return text


def string_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        value = value["text"]
    if not isinstance(value, str):
        return str(value)
    return value


def strip_choice_prefix(text: str) -> str:
    return CHOICE_PREFIX_RE.sub("", text.strip())


def normalize_text(text: str) -> str:
    return " ".join(strip_choice_prefix(text).lower().split())
