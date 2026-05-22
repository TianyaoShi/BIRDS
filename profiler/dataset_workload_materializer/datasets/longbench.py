from __future__ import annotations

import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..common import (
    expect_string,
    hash_text,
    language_allowed,
    optional_mapping,
    optional_string,
    required_string,
    resolve_path,
    string_list,
)
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

LONGBENCH_EXTERNAL_DATASETS: dict[str, tuple[str, ...]] = {
    "long_output_summarization": ("gov_report_original",),
    "medium_output_summarization": ("multi_news_original", "qmsum_original", "meetingbank"),
    "medium_answer_rag_qa": ("dureader_full",),
    "short_answer_document_qa": ("qasper_full",),
}

LONGBENCH_EXTERNAL_HF_DATASETS: dict[str, str] = {
    "gov_report_original": "launch/gov_report",
    "multi_news_original": "alexfabbri/multi_news",
    "qmsum_original": "mattercalm/qmsum",
    "meetingbank": "huuuyeah/meetingbank",
    "dureader_full": "baidu/DuReader",
    "qasper_full": "allenai/qasper",
}

EXTERNAL_SUMMARIZATION_PROMPT = "Summarize the following document.\n\n{document}"
EXTERNAL_QUERY_FOCUSED_SUMMARIZATION_PROMPT = (
    "Given the meeting transcript below, answer the query with a concise summary.\n\n"
    "Query:\n{query}\n\nTranscript:\n{transcript}"
)
EXTERNAL_RAG_QA_PROMPT = (
    "Answer the question using the provided documents.\n\n"
    "Question:\n{question}\n\nDocuments:\n{documents}"
)
EXTERNAL_DOCUMENT_QA_PROMPT = (
    "Answer the question based on the document.\n\n"
    "Question:\n{question}\n\nDocument:\n{document}"
)
LONGBENCH_OFFICIAL_PROMPTS: dict[str, str] = {
    "qasper": (
        "You are given a scientific article and a question.\n"
        "Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        'If the question cannot be answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can, using a single phrase "
        "or sentence if possible.\n"
        'If the question cannot be answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n'
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n"
        "{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
}
LONGBENCH_OFFICIAL_MAX_NEW_TOKENS: dict[str, int] = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
}
LONGBENCH_OFFICIAL_ANSWER_STYLE_PROMPTS: dict[str, str] = {
    "qasper": (
        "You are given a scientific article and a question.\n"
        "Answer with only the final answer text. Use a single short phrase or sentence whenever possible. "
        'If the article does not answer the question, write exactly "unanswerable". '
        'If the question is yes/no, write exactly "yes", "no", or "unanswerable". '
        "Do not include reasoning, citations, section titles, bullet points, markdown, or explanations.\n\n"
        "Article: {context}\n\n"
        "Question: {input}\n\n"
        "Final answer only:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer the question.\n\n"
        "{context}\n\n"
        "Return only the answer span or shortest possible answer phrase. "
        "Do not write a full sentence unless the answer itself requires it. "
        "Do not include reasoning, source discussion, markdown, or any extra words.\n\n"
        "Question: {input}\n"
        "Final answer only:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并回答问题。\n\n"
        "{context}\n\n"
        "只输出最终答案本身，尽量使用最短的答案短语。"
        "不要解释，不要引用来源，不要使用 Markdown，不要输出多余文字。\n\n"
        "问题：{input}\n"
        "最终答案："
    ),
    "dureader": (
        "请基于给定的文章回答问题。\n\n"
        "文章：{context}\n\n"
        "只输出最终答案本身，尽量使用最短的答案短语。"
        "不要解释，不要引用来源，不要使用 Markdown，不要输出多余文字。\n\n"
        "问题：{input}\n"
        "最终答案："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Write only the final one-page summary. Start directly with the summary content. "
        "Do not mention that you are summarizing, do not include planning text, headings, markdown, or bullet points.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction.\n\n"
        "Transcript:\n{context}\n\n"
        "Answer the query directly in one or more sentences. "
        "Do not include reasoning, planning text, headings, markdown, bullet points, or source discussion.\n\n"
        "Query: {input}\n"
        "Final answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\n"
        "Write only the final one-page summary. Start directly with the summary content. "
        "Do not mention that you are summarizing, do not include planning text, headings, markdown, or bullet points.\n\n"
        "Summary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后写一段总结。\n"
        "会议记录：\n{context}\n\n"
        "只输出最终会议总结正文。不要解释，不要写计划过程，不要使用标题、Markdown 或项目符号。\n\n"
        "会议总结："
    ),
}


def load_longbench_dataset(dataset: dict[str, Any], ctx: MaterializationContext) -> DatasetLoadResult:
    if ctx.raw_path is None:
        raise ValueError("longbench materialization requires dataset.raw_path")
    effective_policy = ctx.sampling.policy or "task_uniform"
    if effective_policy != "task_uniform":
        raise ValueError("longbench materialization only supports sampling.policy=task_uniform")
    profile, profile_spec, selected_tasks = longbench_selection(dataset)
    prompt_template = optional_string(dataset.get("prompt_template"), "dataset.prompt_template")
    if prompt_template is not None and prompt_template not in {
        "longbench_context_task",
        "longbench_official",
        "longbench_official_answer_style",
    }:
        raise ValueError(
            "dataset.prompt_template must be one of: longbench_context_task, "
            "longbench_official, longbench_official_answer_style"
        )
    prompt_template = prompt_template or "longbench_context_task"
    external_sources = longbench_external_sources(dataset, base_dir=ctx.base_dir, profile=profile)
    if ctx.sampling.samples_per_task is None and not external_sources:
        raise ValueError("longbench materialization requires sampling.samples_per_task")
    samples: list[MaterializedSample] = []
    if ctx.sampling.samples_per_task is not None:
        samples.extend(
            load_longbench_samples(
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
                prompt_template=prompt_template,
            )
        )
    elif selected_tasks:
        raise ValueError("sampling.samples_per_task is required when dataset.configs selects LongBench tasks")
    if external_sources:
        samples.extend(
            load_longbench_external_samples(
                external_sources,
                dataset_name=ctx.dataset_name,
                dataset_kind=ctx.dataset_kind,
                profile=profile,
                profile_spec=profile_spec,
                tokenizer=ctx.tokenizer,
                filtering=ctx.filtering,
                counters=ctx.counters,
                seed=ctx.sampling.seed,
                limit_per_dataset=ctx.sampling.external_samples_per_dataset,
                max_group_reuse=ctx.sampling.max_external_group_reuse,
            )
        )
    if not samples:
        raise ValueError("LongBench materialization produced no usable rows")
    ctx.counters.materialized_rows = len(samples)
    selected_external = [source["name"] for source in external_sources]
    selected = selected_tasks + selected_external
    report_prompt_template = (
        f"{prompt_template}_plus_external_sources" if external_sources else prompt_template
    )
    return DatasetLoadResult(
        samples=samples,
        task=profile,
        prompt_template=report_prompt_template,
        profile=profile,
        selected_tasks=selected,
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
    prompt_template: str,
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
                prompt_template=prompt_template,
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


def longbench_external_sources(
    dataset: dict[str, Any],
    *,
    base_dir: Path,
    profile: str,
) -> list[dict[str, Any]]:
    raw_sources = dataset.get("external_datasets")
    if raw_sources is None:
        return []
    if not isinstance(raw_sources, list):
        raise ValueError("dataset.external_datasets must be a list")
    allowed = set(LONGBENCH_EXTERNAL_DATASETS[profile])
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        source = optional_mapping(raw_source, f"dataset.external_datasets[{index}]")
        name = expect_string(source.get("name"), f"dataset.external_datasets[{index}].name")
        if name not in allowed:
            raise ValueError(
                f"LongBench profile {profile!r} only supports external datasets {sorted(allowed)}; got {name!r}"
            )
        if name in seen:
            raise ValueError(f"duplicate LongBench external dataset: {name}")
        raw_path = source.get("raw_path")
        if raw_path is None:
            raise ValueError(f"dataset.external_datasets[{index}].raw_path is required")
        hf_dataset = optional_string(
            source.get("hf_dataset"),
            f"dataset.external_datasets[{index}].hf_dataset",
        ) or LONGBENCH_EXTERNAL_HF_DATASETS[name]
        sources.append(
            {
                "name": name,
                "raw_path": resolve_path(
                    expect_string(raw_path, f"dataset.external_datasets[{index}].raw_path"),
                    base_dir=base_dir,
                ),
                "split": optional_string(source.get("split"), f"dataset.external_datasets[{index}].split")
                or "train",
                "hf_dataset": hf_dataset,
            }
        )
        seen.add(name)
    return sources


def load_longbench_external_samples(
    sources: list[dict[str, Any]],
    *,
    dataset_name: str,
    dataset_kind: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    tokenizer,
    filtering,
    counters,
    seed: int,
    limit_per_dataset: int | None,
    max_group_reuse: int,
) -> list[MaterializedSample]:
    samples: list[MaterializedSample] = []
    seen_content_hashes: set[str] = set()
    for source_config in sources:
        external_name = str(source_config["name"])
        source_samples: list[MaterializedSample] = []
        group_counts: Counter[str] = Counter()
        rows = iter_external_rows(Path(source_config["raw_path"]))
        for row_index, row, source in progress_iter(
            rows,
            total=len(rows),
            desc=f"materialize {external_name}",
        ):
            counters.total_rows += 1
            row_samples = external_longbench_row_to_samples(
                row,
                row_index=row_index,
                source=source,
                external_name=external_name,
                hf_dataset=str(source_config["hf_dataset"]),
                split=str(source_config["split"]),
                dataset_name=dataset_name,
                dataset_kind=dataset_kind,
                profile=profile,
                profile_spec=profile_spec,
                tokenizer=tokenizer,
                filtering=filtering,
                language_filter_payload=filtering.language_filter,
                seen_content_hashes=seen_content_hashes,
                counters=counters,
            )
            source_samples.extend(row_samples)
        if not source_samples:
            raise ValueError(
                f"LongBench external dataset {external_name!r} produced no usable rows from "
                f"{source_config['raw_path']}"
            )
        selected = select_longbench_external_samples(
            source_samples,
            source_name=external_name,
            seed=seed,
            limit=limit_per_dataset,
            max_group_reuse=max_group_reuse,
            group_counts=group_counts,
        )
        counters.drops["external_not_selected_by_sampling"] += len(source_samples) - len(selected)
        samples.extend(selected)
    return samples


def iter_external_rows(raw_path: Path) -> list[tuple[int, dict[str, Any], str]]:
    if raw_path.is_file():
        return list(iter_external_rows_from_file(raw_path))
    if raw_path.is_dir():
        files = sorted(
            path
            for path in raw_path.rglob("*")
            if path.is_file() and path.suffix in {".jsonl", ".json"}
        )
        if not files:
            raise FileNotFoundError(f"external LongBench source has no .jsonl or .json files: {raw_path}")
        rows: list[tuple[int, dict[str, Any], str]] = []
        for file_path in files:
            rows.extend(iter_external_rows_from_file(file_path))
        return rows
    raise FileNotFoundError(f"external LongBench source not found: {raw_path}")


def iter_external_rows_from_file(path: Path) -> list[tuple[int, dict[str, Any], str]]:
    if path.suffix == ".jsonl":
        rows: list[tuple[int, dict[str, Any], str]] = []
        with path.open("r", encoding="utf-8") as handle:
            line_iter = enumerate(handle)
            for row_index, raw_line in progress_iter(
                line_iter,
                total=None,
                desc=f"read {path.name}",
                enable=path.stat().st_size >= 10 * 1024 * 1024,
            ):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{row_index + 1} must be a JSON object")
                rows.append((row_index, row, f"{path}:{row_index + 1}"))
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                payload = payload["data"]
            elif isinstance(payload.get("rows"), list):
                payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list or a mapping with a data/rows list")
        rows = []
        for row_index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"{path}[{row_index}] must be a JSON object")
            rows.append((row_index, row, f"{path}:{row_index + 1}"))
        return rows
    raise ValueError(f"external LongBench source file must be .jsonl or .json: {path}")


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
    prompt_template: str,
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
    prompt = render_longbench_prompt(row, task_input=task_input, prompt_template=prompt_template)
    if prompt_fails_char_filter(prompt, filtering=filtering, counters=counters):
        return None
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(target))
    expected_output_len = (
        official_longbench_max_new_tokens(task)
        if prompt_template in {"longbench_official", "longbench_official_answer_style"}
        else max(1, target_token_count)
    )
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
        "prompt_template": prompt_template,
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
        "benchmark_max_new_tokens": official_longbench_max_new_tokens(task),
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
        expected_output_len=expected_output_len,
        metadata=metadata,
    )


def external_longbench_row_to_samples(
    row: dict[str, Any],
    *,
    row_index: int,
    source: str,
    external_name: str,
    hf_dataset: str,
    split: str,
    dataset_name: str,
    dataset_kind: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    tokenizer,
    filtering,
    language_filter_payload: dict[str, set[str]],
    seen_content_hashes: set[str],
    counters,
) -> list[MaterializedSample]:
    normalized_rows = normalize_external_longbench_row(row, external_name=external_name, row_index=row_index)
    if not normalized_rows:
        counters.drops["missing_empty_task_input"] += 1
        return []
    samples: list[MaterializedSample] = []
    for normalized_index, normalized in enumerate(normalized_rows):
        prompt = expect_string(normalized.get("prompt"), f"{source}.prompt")
        target = optional_string(normalized.get("target"), f"{source}.target") or " "
        if prompt_fails_char_filter(prompt, filtering=filtering, counters=counters):
            continue
        prompt_token_count = len(tokenizer.encode(prompt))
        target_token_count = len(tokenizer.encode(target))
        if prompt_token_count < filtering.min_prompt_tokens:
            counters.drops["prompt_too_short"] += 1
            continue
        if prompt_token_count > filtering.max_prompt_tokens:
            counters.drops["prompt_too_long"] += 1
            continue
        if target_token_count < filtering.min_target_tokens:
            counters.drops["missing_empty_target"] += 1
            continue
        if target_token_count > filtering.max_target_tokens:
            counters.drops["target_too_long"] += 1
            continue
        language = expect_string(normalized.get("language"), f"{source}.language")
        if not language_allowed(language, language_filter_payload):
            counters.drops["unsupported_language"] += 1
            continue
        content_hash = hash_text(prompt)
        if filtering.dedup_content_hash and content_hash in seen_content_hashes:
            counters.drops["duplicate_content_hash"] += 1
            continue
        seen_content_hashes.add(content_hash)
        sample_id = f"{dataset_name}-{external_name}-{row_index:06d}-{normalized_index:02d}"
        group_id = expect_string(normalized.get("group_id"), f"{source}.group_id")
        metadata = {
            "dataset": dataset_name,
            "dataset_kind": dataset_kind,
            "task": external_name,
            "profile": profile,
            "workload_type": profile_spec.workload_type,
            "output_regime": profile_spec.output_regime,
            "prompt_template": str(normalized["prompt_template"]),
            "split": split,
            "language": language,
            "file_path": source.split(":", 1)[0],
            "session_id": f"{dataset_name}::{profile}::{external_name}::{group_id}",
            "sample_id": sample_id,
            "sequence_index": row_index * 1000 + normalized_index,
            "content_hash": content_hash,
            "ground_truth": target,
            "target_hash": hash_text(target),
            "prompt_token_count": prompt_token_count,
            "target_token_count": target_token_count,
            "source_dataset": external_name,
            "source_hf_dataset": hf_dataset,
            "source_row_index": row_index,
            "source_record_id": normalized.get("record_id"),
            "group_id": group_id,
            "external_longbench_source": True,
        }
        samples.append(
            MaterializedSample(
                sample_id=sample_id,
                prompt=prompt,
                target=target,
                expected_output_len=max(1, target_token_count),
                metadata=metadata,
            )
        )
    return samples


def prompt_fails_char_filter(prompt: str, *, filtering, counters) -> bool:
    prompt_chars = len(prompt)
    if filtering.min_prompt_chars is not None and prompt_chars < filtering.min_prompt_chars:
        counters.drops["prompt_char_too_short"] += 1
        return True
    if filtering.max_prompt_chars is not None and prompt_chars > filtering.max_prompt_chars:
        counters.drops["prompt_char_too_long"] += 1
        return True
    return False


def normalize_external_longbench_row(
    row: dict[str, Any],
    *,
    external_name: str,
    row_index: int,
) -> list[dict[str, Any]]:
    if external_name == "gov_report_original":
        document = first_external_text(row, ("document", "report", "text", "article"))
        target = first_external_text(row, ("summary", "target", "reference", "output"))
        record_id = first_external_scalar(row, ("id", "report_id", "doc_id"))
        record = external_summary_record(
                external_name=external_name,
                document=document,
                target=target,
                record_id=record_id,
                row_index=row_index,
                language="en",
            )
        return [record] if record is not None else []
    if external_name == "multi_news_original":
        document = first_external_text(row, ("document", "documents", "article", "articles", "text"))
        target = first_external_text(row, ("summary", "target", "reference", "output"))
        record_id = first_external_scalar(row, ("id", "article_cluster_id", "document_id"))
        record = external_summary_record(
                external_name=external_name,
                document=document,
                target=target,
                record_id=record_id,
                row_index=row_index,
                language="en",
            )
        return [record] if record is not None else []
    if external_name == "qmsum_original":
        prompt = first_external_text(row, ("input", "prompt"))
        target = first_external_text(row, ("output", "summary", "answer", "target", "reference"))
        if prompt is None:
            transcript = first_external_text(row, ("transcript", "meeting_transcript", "document", "context"))
            query = first_external_text(row, ("query", "question", "instruction"))
            if transcript is None or query is None:
                raise ValueError("qmsum_original rows require input/prompt or transcript plus query")
            prompt = EXTERNAL_QUERY_FOCUSED_SUMMARIZATION_PROMPT.format(query=query, transcript=transcript)
        record_id = first_external_scalar(row, ("id", "pid", "qid", "query_id"))
        group_id = first_external_scalar(row, ("meeting_id", "pid", "id")) or hash_text(prompt)
        return [
            {
                "prompt": prompt,
                "target": target,
                "record_id": record_id,
                "group_id": str(group_id),
                "prompt_template": "external_query_focused_summarization",
                "language": "en",
            }
        ]
    if external_name == "meetingbank":
        document = first_external_text(row, ("transcript", "document", "text"))
        target = first_external_text(row, ("summary", "target", "reference", "output"))
        record_id = first_external_scalar(row, ("uid", "id", "meeting_id"))
        record = external_summary_record(
                external_name=external_name,
                document=document,
                target=target,
                record_id=record_id,
                row_index=row_index,
                language="en",
            )
        return [record] if record is not None else []
    if external_name == "dureader_full":
        question = first_external_text(row, ("question", "query", "input"))
        raw_documents = row.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise ValueError(
                "dureader_full rows must come from Baidu DuReader 2.0 and include a non-empty documents list; "
                "single-context paragraph QA sources such as DuReader Robust are not accepted"
            )
        documents = text_value(raw_documents)
        if question is None or documents is None:
            raise ValueError("dureader_full rows require question/query and non-empty documents")
        target = external_reference_answer(row.get("answers")) or first_external_text(
            row,
            ("answer", "response", "target", "output"),
        )
        record_id = first_external_scalar(row, ("question_id", "id", "qid"))
        group_id = first_external_scalar(row, ("document_id", "doc_id")) or hash_text(documents)
        return [
            {
                "prompt": EXTERNAL_RAG_QA_PROMPT.format(question=question, documents=documents),
                "target": target,
                "record_id": record_id,
                "group_id": str(group_id),
                "prompt_template": "external_rag_qa",
                "language": "zh",
            }
        ]
    if external_name == "qasper_full":
        return normalize_qasper_external_rows(row, row_index=row_index)
    raise ValueError(f"unsupported LongBench external dataset: {external_name}")


def external_summary_record(
    *,
    external_name: str,
    document: str | None,
    target: str | None,
    record_id: Any,
    row_index: int,
    language: str,
) -> dict[str, Any] | None:
    if document is None:
        return None
    group_id = str(record_id) if record_id not in (None, "") else hash_text(document)
    return {
        "prompt": EXTERNAL_SUMMARIZATION_PROMPT.format(document=document),
        "target": target,
        "record_id": str(record_id) if record_id not in (None, "") else None,
        "group_id": group_id,
        "prompt_template": "external_summarization",
        "language": language,
        "sequence_index": row_index,
    }


def normalize_qasper_external_rows(row: dict[str, Any], *, row_index: int) -> list[dict[str, Any]]:
    paper_id = first_external_scalar(row, ("id", "paper_id", "doc_id")) or f"qasper-row-{row_index}"
    document = qasper_document_text(row)
    raw_qas = row.get("qas")
    if isinstance(raw_qas, list):
        normalized = []
        for question_index, qa in enumerate(raw_qas):
            if not isinstance(qa, dict):
                raise ValueError(f"qasper row.qas[{question_index}] must be a mapping")
            question = optional_string(qa.get("question"), f"qasper row.qas[{question_index}].question")
            if question is None:
                continue
            question_id = first_external_scalar(qa, ("question_id", "id")) or f"{paper_id}-q{question_index}"
            normalized.append(
                {
                    "prompt": EXTERNAL_DOCUMENT_QA_PROMPT.format(question=question, document=document),
                    "target": qasper_answer_text(qa.get("answers")),
                    "record_id": str(question_id),
                    "group_id": str(paper_id),
                    "prompt_template": "external_document_qa",
                    "language": "en",
                }
            )
        if not normalized:
            raise ValueError("qasper_full row produced no question records")
        return normalized
    qas = optional_mapping(raw_qas, "qasper row.qas")
    questions = qas.get("question")
    question_ids = qas.get("question_id")
    answers = qas.get("answers")
    if not isinstance(questions, list):
        question = first_external_text(row, ("question", "query"))
        if question is None:
            raise ValueError("qasper_full rows require qas.question or question")
        return [
            {
                "prompt": EXTERNAL_DOCUMENT_QA_PROMPT.format(question=question, document=document),
                "target": external_reference_answer(row.get("answers"))
                or first_external_text(row, ("answer", "target", "output", "reference")),
                "record_id": str(first_external_scalar(row, ("question_id", "id")) or f"{paper_id}-q0"),
                "group_id": str(paper_id),
                "prompt_template": "external_document_qa",
                "language": "en",
            }
        ]
    normalized: list[dict[str, Any]] = []
    for question_index, question in enumerate(questions):
        if not isinstance(question, str) or not question:
            continue
        question_id = list_value(question_ids, question_index) or f"{paper_id}-q{question_index}"
        target = qasper_answer_text(list_value(answers, question_index))
        normalized.append(
            {
                "prompt": EXTERNAL_DOCUMENT_QA_PROMPT.format(question=question, document=document),
                "target": target,
                "record_id": str(question_id),
                "group_id": str(paper_id),
                "prompt_template": "external_document_qa",
                "language": "en",
            }
        )
    if not normalized:
        raise ValueError("qasper_full row produced no question records")
    return normalized


def render_longbench_prompt(
    row: dict[str, Any],
    *,
    task_input: str | None = None,
    prompt_template: str = "longbench_context_task",
) -> str:
    context = expect_string(row.get("context"), "longbench row.context")
    resolved_task_input = task_input if task_input is not None else expect_string(
        row.get("input"),
        "longbench row.input",
    )
    if prompt_template == "longbench_official":
        task = expect_string(row.get("dataset"), "longbench row.dataset")
        return official_longbench_prompt(task).format(context=context, input=resolved_task_input)
    if prompt_template == "longbench_official_answer_style":
        task = expect_string(row.get("dataset"), "longbench row.dataset")
        return official_longbench_answer_style_prompt(task).format(context=context, input=resolved_task_input)
    if prompt_template != "longbench_context_task":
        raise ValueError(f"unsupported LongBench prompt template: {prompt_template}")
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


def official_longbench_prompt(task: str) -> str:
    base_task = metric_longbench_task(task)
    try:
        return LONGBENCH_OFFICIAL_PROMPTS[base_task]
    except KeyError as exc:
        raise ValueError(f"unsupported LongBench official prompt task: {task}") from exc


def official_longbench_answer_style_prompt(task: str) -> str:
    base_task = metric_longbench_task(task)
    try:
        return LONGBENCH_OFFICIAL_ANSWER_STYLE_PROMPTS[base_task]
    except KeyError as exc:
        raise ValueError(f"unsupported LongBench answer-style prompt task: {task}") from exc


def official_longbench_max_new_tokens(task: str) -> int:
    base_task = metric_longbench_task(task)
    try:
        return LONGBENCH_OFFICIAL_MAX_NEW_TOKENS[base_task]
    except KeyError as exc:
        raise ValueError(f"unsupported LongBench official max-new-tokens task: {task}") from exc


def metric_longbench_task(task: str) -> str:
    return task[:-2] if task.endswith("_e") else task


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


def select_longbench_external_samples(
    samples: list[MaterializedSample],
    *,
    source_name: str,
    seed: int,
    limit: int | None,
    max_group_reuse: int,
    group_counts: Counter[str],
) -> list[MaterializedSample]:
    ranked = sorted(
        samples,
        key=lambda sample: (
            hash_text(f"{seed}:{source_name}:{sample.sample_id}:{sample.metadata['content_hash']}"),
            sample.sample_id,
        ),
    )
    selected: list[MaterializedSample] = []
    for sample in ranked:
        group_id = str(sample.metadata["group_id"])
        if group_counts[group_id] >= max_group_reuse:
            continue
        selected.append(sample)
        group_counts[group_id] += 1
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(
            f"LongBench external dataset {source_name!r} has no rows after max_external_group_reuse="
            f"{max_group_reuse}"
        )
    return sorted(selected, key=sample_order_key)


def first_external_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key not in row:
            continue
        value = text_value(row[key])
        if value is not None:
            return value
    return None


def first_external_scalar(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, list):
        parts = [text_value(item) for item in value]
        joined = "\n\n".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        if "text" in value:
            text = text_value(value["text"])
            if text is not None:
                return text
        if "paragraphs" in value:
            text = text_value(value["paragraphs"])
            if text is not None:
                return text
        if "document" in value:
            text = text_value(value["document"])
            if text is not None:
                return text
        keys = [key for key in ("title", "abstract", "section_name", "context", "contents") if key in value]
        if keys:
            parts = [text_value(value[key]) for key in keys]
            joined = "\n\n".join(part for part in parts if part)
            return joined or None
    return None


def external_reference_answer(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        candidates = [external_reference_answer(item) for item in value]
        text_candidates = [candidate for candidate in candidates if candidate]
        if text_candidates:
            return max(text_candidates, key=len)
        return None
    if isinstance(value, dict):
        answer = value.get("answer")
        if answer is not None:
            resolved = external_reference_answer(answer)
            if resolved:
                return resolved
        text = value.get("text")
        if text is not None:
            resolved = external_reference_answer(text)
            if resolved:
                return resolved
        free_form = value.get("free_form_answer")
        if isinstance(free_form, str) and free_form:
            return free_form
        spans = value.get("extractive_spans")
        if isinstance(spans, list):
            text_spans = [str(item) for item in spans if str(item)]
            if text_spans:
                return "; ".join(text_spans)
        yes_no = value.get("yes_no")
        if isinstance(yes_no, bool):
            return "yes" if yes_no else "no"
        if value.get("unanswerable") is True:
            return "unanswerable"
    return None


def qasper_document_text(row: dict[str, Any]) -> str:
    title = first_external_text(row, ("title",))
    abstract = first_external_text(row, ("abstract",))
    full_text = row.get("full_text")
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if isinstance(full_text, dict):
        section_names = full_text.get("section_name")
        paragraphs = full_text.get("paragraphs")
        if isinstance(section_names, list) and isinstance(paragraphs, list):
            for section_index, section_paragraphs in enumerate(paragraphs):
                section_name = list_value(section_names, section_index)
                section_text = text_value(section_paragraphs)
                if not section_text:
                    continue
                if isinstance(section_name, str) and section_name:
                    parts.append(f"{section_name}\n{section_text}")
                else:
                    parts.append(section_text)
        else:
            full_text_value = text_value(full_text)
            if full_text_value:
                parts.append(full_text_value)
    else:
        full_text_value = text_value(full_text)
        if full_text_value:
            parts.append(full_text_value)
    if not parts:
        raise ValueError("qasper_full rows require title/abstract/full_text content")
    return "\n\n".join(parts)


def qasper_answer_text(value: Any) -> str | None:
    if isinstance(value, dict) and "answer" in value:
        return external_reference_answer(value.get("answer"))
    return external_reference_answer(value)


def list_value(value: Any, index: int) -> Any:
    if isinstance(value, list) and 0 <= index < len(value):
        return value[index]
    return None


def progress_iter(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
    enable: bool | None = None,
) -> Iterable[Any]:
    if enable is None:
        enable = total is not None and total >= 10_000
    if not enable:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, total=total, desc=desc, unit="rows", mininterval=5)


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
