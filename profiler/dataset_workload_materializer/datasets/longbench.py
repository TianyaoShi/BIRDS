from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from ..common import expect_string, hash_text, language_allowed, optional_string, required_string, string_list
from ..models import DatasetLoadResult, LongBenchProfileSpec, MaterializationContext, MaterializedSample
from ..outputs import sample_order_key


LONGBENCH_PROFILE_SPECS: dict[str, LongBenchProfileSpec] = {
    "long_output_summarization": LongBenchProfileSpec(
        tasks=("gov_report", "gov_report_e"),
        workload_type="summarization",
        output_regime="long",
    ),
    "medium_output_summarization": LongBenchProfileSpec(
        tasks=("multi_news", "multi_news_e", "qmsum", "vcsum"),
        workload_type="summarization",
        output_regime="medium",
    ),
    "medium_answer_rag_qa": LongBenchProfileSpec(
        tasks=("dureader",),
        workload_type="rag_qa",
        output_regime="medium",
    ),
    "short_answer_document_qa": LongBenchProfileSpec(
        tasks=("multifieldqa_en", "multifieldqa_en_e", "multifieldqa_zh", "qasper", "qasper_e"),
        workload_type="document_qa",
        output_regime="short",
    ),
}

LONGBENCH_EXCLUDED_TASKS: frozenset[str] = frozenset(
    {
        "2wikimqa",
        "2wikimqa_e",
        "hotpotqa",
        "hotpotqa_e",
        "musique",
        "triviaqa",
        "triviaqa_e",
        "trec",
        "trec_e",
        "lsht",
        "samsum",
        "samsum_e",
        "passage_count",
        "passage_count_e",
        "passage_retrieval_en",
        "passage_retrieval_en_e",
        "passage_retrieval_zh",
        "narrativeqa",
        "lcc",
        "lcc_e",
        "repobench-p",
        "repobench-p_e",
    }
)


def load_longbench_dataset(dataset: dict[str, Any], ctx: MaterializationContext) -> DatasetLoadResult:
    if ctx.raw_path is None:
        raise ValueError("longbench materialization requires dataset.raw_path")
    effective_policy = ctx.sampling.policy or "task_uniform"
    if effective_policy != "task_uniform":
        raise ValueError("longbench materialization only supports sampling.policy=task_uniform")
    if ctx.sampling.samples_per_task is None:
        raise ValueError("longbench materialization requires sampling.samples_per_task")
    profile, profile_spec, selected_tasks = longbench_selection(dataset)
    samples = load_longbench_samples(
        ctx.raw_path,
        dataset_name=ctx.dataset_name,
        dataset_kind=ctx.dataset_kind,
        split=ctx.split,
        profile=profile,
        profile_spec=profile_spec,
        selected_tasks=selected_tasks,
        tokenizer=ctx.tokenizer,
        filtering=ctx.filtering,
        counters=ctx.counters,
        seed=ctx.sampling.seed,
        samples_per_task=ctx.sampling.samples_per_task,
    )
    return DatasetLoadResult(
        samples=samples,
        task=profile,
        prompt_template="longbench_context_task",
        profile=profile,
        selected_tasks=selected_tasks,
    )


def load_longbench_samples(
    raw_path: Path,
    *,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    selected_tasks: list[str],
    tokenizer,
    filtering,
    counters,
    seed: int,
    samples_per_task: int,
) -> list[MaterializedSample]:
    samples: list[MaterializedSample] = []
    for task_name in selected_tasks:
        seen_content_hashes: set[str] = set()
        task_samples: list[MaterializedSample] = []
        for row_index, row, source in iter_longbench_rows(raw_path, task_name):
            counters.total_rows += 1
            sample = longbench_row_to_sample(
                row,
                row_index=row_index,
                dataset_name=dataset_name,
                dataset_kind=dataset_kind,
                split=split,
                task=task_name,
                profile=profile,
                profile_spec=profile_spec,
                tokenizer=tokenizer,
                filtering=filtering,
                language_filter_payload=filtering.language_filter,
                seen_content_hashes=seen_content_hashes,
                counters=counters,
                source=source,
            )
            if sample is not None:
                task_samples.append(sample)
        if not task_samples:
            raise ValueError(f"LongBench task {task_name!r} produced no usable rows from {raw_path}")
        selected = select_longbench_task_samples(
            task_samples,
            task_name=task_name,
            seed=seed,
            limit=samples_per_task,
        )
        counters.drops["not_selected_by_sampling"] += len(task_samples) - len(selected)
        samples.extend(selected)
    counters.materialized_rows = len(samples)
    return samples


def iter_longbench_rows(raw_path: Path, task_name: str) -> list[tuple[int, dict[str, Any], str]]:
    if raw_path.is_file():
        if raw_path.suffix != ".zip":
            raise ValueError(f"LongBench raw_path file must be .zip: {raw_path}")
        member_name = f"data/{task_name}.jsonl"
        with zipfile.ZipFile(raw_path) as archive:
            names = set(archive.namelist())
            if member_name not in names:
                raise FileNotFoundError(f"LongBench task {task_name!r} not found in {raw_path}")
            rows: list[tuple[int, dict[str, Any], str]] = []
            with archive.open(member_name) as handle:
                for row_index, raw_line in enumerate(handle):
                    stripped = raw_line.decode("utf-8").strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    if not isinstance(row, dict):
                        raise ValueError(f"{member_name}:{row_index + 1} must be a JSON object")
                    rows.append((row_index, row, f"{raw_path}:{member_name}:{row_index + 1}"))
            return rows
    if raw_path.is_dir():
        zip_path = raw_path / "data.zip"
        if zip_path.is_file():
            return iter_longbench_rows(zip_path, task_name)
        candidate_paths = [raw_path / "data" / f"{task_name}.jsonl", raw_path / f"{task_name}.jsonl"]
        for candidate in candidate_paths:
            if not candidate.is_file():
                continue
            rows = []
            with candidate.open("r", encoding="utf-8") as handle:
                for row_index, raw_line in enumerate(handle):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    if not isinstance(row, dict):
                        raise ValueError(f"{candidate}:{row_index + 1} must be a JSON object")
                    rows.append((row_index, row, f"{candidate}:{row_index + 1}"))
            return rows
        raise FileNotFoundError(
            f"LongBench task {task_name!r} not found under {raw_path}; expected data.zip or task jsonl files"
        )
    raise FileNotFoundError(f"LongBench raw_path not found: {raw_path}")


def longbench_row_to_sample(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    tokenizer,
    filtering,
    language_filter_payload: dict[str, set[str]],
    seen_content_hashes: set[str],
    counters,
    source: str,
) -> MaterializedSample | None:
    raw_task_input = row.get("input")
    if not isinstance(raw_task_input, str):
        raise ValueError(f"{source}.input must be a string")
    target = longbench_reference_answer(row.get("answers"))
    task_input, task_input_source = resolved_longbench_task_input(
        raw_task_input,
        target=target,
        profile_spec=profile_spec,
    )
    if task_input is None:
        counters.drops["missing_empty_task_input"] += 1
        return None
    prompt = render_longbench_prompt(row, task_input=task_input)
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(target))
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
    language = optional_string(row.get("language"), f"{source}.language") or default_longbench_language(task)
    if not language_allowed(language, language_filter_payload):
        counters.drops["unsupported_language"] += 1
        return None
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
        "profile": profile,
        "workload_type": profile_spec.workload_type,
        "output_regime": profile_spec.output_regime,
        "prompt_template": "longbench_context_task",
        "split": split,
        "language": language,
        "file_path": f"data/{task}.jsonl",
        "session_id": f"{dataset_name}::{profile}::{task}",
        "sample_id": sample_id,
        "sequence_index": row_index,
        "content_hash": content_hash,
        "ground_truth": target,
        "target_hash": hash_text(target),
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "longbench_id": row.get("_id"),
        "longbench_row_index": row_index,
        "longbench_length": row.get("length"),
        "longbench_dataset": row.get("dataset"),
        "longbench_task_input_source": task_input_source,
    }
    return MaterializedSample(
        sample_id=sample_id,
        prompt=prompt,
        target=target,
        expected_output_len=max(1, target_token_count),
        metadata=metadata,
    )


def render_longbench_prompt(row: dict[str, Any], *, task_input: str | None = None) -> str:
    context = expect_string(row.get("context"), "longbench row.context")
    resolved_task_input = task_input if task_input is not None else expect_string(
        row.get("input"),
        "longbench row.input",
    )
    all_classes = row.get("all_classes")
    class_text = ""
    if isinstance(all_classes, list) and all_classes:
        labels = [str(item) for item in all_classes if str(item)]
        if labels:
            class_text = "\nCandidate labels/classes: " + ", ".join(labels)
    language = row.get("language")
    language_text = f"\nLanguage: {language}" if isinstance(language, str) and language else ""
    return (
        "Context:\n"
        f"{context}\n\n"
        "Task:\n"
        f"{resolved_task_input}"
        f"{class_text}"
        f"{language_text}\n\n"
        "Answer:"
    )


def resolved_longbench_task_input(
    raw_task_input: str,
    *,
    target: str,
    profile_spec: LongBenchProfileSpec,
) -> tuple[str | None, str]:
    if raw_task_input:
        return raw_task_input, "dataset"
    if profile_spec.workload_type == "summarization":
        return synthesized_summarization_task_input(target), "synthesized_summarization_instruction"
    return None, "missing"


def synthesized_summarization_task_input(target: str) -> str:
    target_words = [word for word in target.split() if word]
    approximate_words = rounded_word_target(len(target_words))
    return f"Summarize the document in about {approximate_words} words."


def rounded_word_target(word_count: int) -> int:
    if word_count <= 0:
        return 100
    rounded = int(25 * round(word_count / 25))
    return max(25, rounded)


def longbench_reference_answer(answers: Any) -> str:
    if isinstance(answers, list):
        text_answers = [str(item) for item in answers if str(item)]
        if text_answers:
            return max(text_answers, key=len)
        return " "
    if answers is None:
        return " "
    return str(answers)


def select_longbench_task_samples(
    samples: list[MaterializedSample],
    *,
    task_name: str,
    seed: int,
    limit: int,
) -> list[MaterializedSample]:
    if len(samples) <= limit:
        return sorted(samples, key=sample_order_key)
    ranked = sorted(
        samples,
        key=lambda sample: (
            hash_text(f"{seed}:{task_name}:{sample.sample_id}:{sample.metadata['content_hash']}"),
            sample.sample_id,
        ),
    )
    return sorted(ranked[:limit], key=sample_order_key)


def longbench_selection(dataset: dict[str, Any]) -> tuple[str, LongBenchProfileSpec, list[str]]:
    profile = required_string(dataset, "profile")
    try:
        profile_spec = LONGBENCH_PROFILE_SPECS[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(LONGBENCH_PROFILE_SPECS))
        raise ValueError(f"supported LongBench dataset.profile values: {supported}") from exc
    requested = (
        string_list(dataset.get("configs"), "dataset.configs")
        if dataset.get("configs") is not None
        else list(profile_spec.tasks)
    )
    if not requested:
        raise ValueError("dataset.configs must not be empty when provided")
    excluded = sorted(task for task in requested if task in LONGBENCH_EXCLUDED_TASKS)
    if excluded:
        raise ValueError("LongBench realistic-NL profiles exclude tasks: " + ", ".join(excluded))
    allowed = set(profile_spec.tasks)
    invalid = sorted(task for task in requested if task not in allowed)
    if invalid:
        raise ValueError(
            f"LongBench profile {profile!r} only supports tasks {list(profile_spec.tasks)}; got {invalid}"
        )
    selected_tasks: list[str] = []
    seen: set[str] = set()
    for task_name in requested:
        if task_name in seen:
            continue
        selected_tasks.append(task_name)
        seen.add(task_name)
    return profile, profile_spec, selected_tasks


def default_longbench_language(task_name: str) -> str:
    if task_name.endswith("_zh") or task_name == "dureader":
        return "zh"
    return "en"
