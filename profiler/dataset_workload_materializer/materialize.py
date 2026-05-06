from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_mst_finder.workload import PromptTokenizer, resolve_tokenizer


SUPPORTED_DATASET_KINDS: dict[str, str] = {
    "crosscodeeval": "code_completion",
    "longbench": "long_context_nlp",
    "repobench": "code_completion",
}

DEFAULT_CROSSCODEEVAL_FIELD_ALIASES: dict[str, list[str]] = {
    "prompt": ["prompt", "input", "code_context", "current_file_prefix"],
    "target": ["target", "completion", "reference", "groundtruth"],
    "language": ["language", "lang"],
    "repo_id": ["repo", "repo_name", "repository"],
    "file_path": ["file_path", "path"],
    "cross_file_context": ["cross_file_context", "crossfile_context", "retrieved_context", "context"],
    "sequence_index": ["sequence_index", "cursor_index", "order"],
}


@dataclass(slots=True)
class MaterializedSample:
    sample_id: str
    prompt: str
    target: str
    expected_output_len: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class Counters:
    total_rows: int = 0
    materialized_rows: int = 0
    drops: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class LongBenchProfileSpec:
    tasks: tuple[str, ...]
    workload_type: str
    output_regime: str


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


def materialize_from_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("materialization config must be a mapping")
    return prepare(payload, config_source=path)


def prepare(config: dict[str, Any], *, config_source: Path | None = None) -> dict[str, Any]:
    name = _required_string(config, "name")
    dataset = _required_mapping(config, "dataset")
    dataset_name = _optional_string(dataset.get("name"), "dataset.name") or "crosscodeeval"
    dataset_kind = _dataset_kind(dataset_name)
    base_dir = config_source.parent if config_source is not None else Path.cwd()
    raw_path = _config_raw_path(dataset, base_dir=base_dir)
    split = _optional_string(dataset.get("split"), "dataset.split") or (
        "test" if dataset_name == "longbench" else "unspecified"
    )
    task = _optional_string(dataset.get("mode"), "dataset.mode") or (
        "cross_file_first" if dataset_name == "repobench" else "cross_file_materialized"
    )
    prompt_template = (
        "longbench_context_task"
        if dataset_name == "longbench"
        else _prompt_template(dataset.get("prompt_template", config.get("prompt_template", "plain_prefix")))
    )
    aliases = _field_aliases(dataset.get("field_aliases", {})) if dataset_name == "crosscodeeval" else None

    tokenization = _optional_mapping(config.get("tokenization"), "tokenization")
    tokenizer_name = _optional_string(tokenization.get("tokenizer"), "tokenization.tokenizer") or "whitespace"
    tokenizer = resolve_tokenizer(tokenizer_name)

    filtering = _optional_mapping(config.get("filtering"), "filtering")
    min_prompt_tokens = _int_setting(filtering, "min_prompt_tokens", 128)
    max_prompt_tokens = _int_setting(filtering, "max_prompt_tokens", 8192)
    min_target_tokens = _int_setting(filtering, "min_target_tokens", 1)
    max_target_tokens = _int_setting(filtering, "max_target_tokens", 128)
    language_filter = _language_filter(filtering.get("languages", {}))
    dedup_content_hash = _dedup_content_hash(filtering.get("dedup", {}))

    sampling = _optional_mapping(config.get("sampling"), "sampling")
    seed = _int_setting(sampling, "seed", 42)
    burst_size = _int_setting(sampling, "burst_size", 8)
    sampling_policy = _optional_string(sampling.get("policy"), "sampling.policy")
    samples_per_task = sampling.get("samples_per_task")
    if samples_per_task is not None:
        samples_per_task = _positive_int(samples_per_task, "sampling.samples_per_task")

    sharding = _required_mapping(config, "sharding")
    output_dir = _resolve_path(
        _required_string(sharding, "output_dir"),
        base_dir=config_source.parent if config_source is not None else Path.cwd(),
    )
    samples_per_shard = _int_setting(sharding, "samples_per_shard", 8000)
    requested_num_shards = sharding.get("num_shards")
    if requested_num_shards is not None:
        requested_num_shards = _positive_int(requested_num_shards, "sharding.num_shards")

    counters = Counters()
    profile: str | None = None
    selected_tasks: list[str] | None = None
    if dataset_name == "crosscodeeval":
        if raw_path is None:
            raise ValueError("crosscodeeval materialization requires dataset.raw_path")
        assert aliases is not None
        samples = _load_crosscodeeval_samples(
            raw_path,
            aliases=aliases,
            dataset_name=dataset_name,
            dataset_kind=dataset_kind,
            split=split,
            task=task,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            min_prompt_tokens=min_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_target_tokens=min_target_tokens,
            max_target_tokens=max_target_tokens,
            language_filter=language_filter,
            dedup_content_hash=dedup_content_hash,
            counters=counters,
        )
    elif dataset_name == "longbench":
        if raw_path is None:
            raise ValueError("longbench materialization requires dataset.raw_path")
        effective_policy = sampling_policy or "task_uniform"
        if effective_policy != "task_uniform":
            raise ValueError("longbench materialization only supports sampling.policy=task_uniform")
        if samples_per_task is None:
            raise ValueError("longbench materialization requires sampling.samples_per_task")
        profile, profile_spec, selected_tasks = _longbench_selection(dataset)
        task = profile
        samples = _load_longbench_samples(
            raw_path,
            dataset_name=dataset_name,
            dataset_kind=dataset_kind,
            split=split,
            profile=profile,
            profile_spec=profile_spec,
            selected_tasks=selected_tasks,
            tokenizer=tokenizer,
            min_prompt_tokens=min_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_target_tokens=min_target_tokens,
            max_target_tokens=max_target_tokens,
            language_filter=language_filter,
            dedup_content_hash=dedup_content_hash,
            counters=counters,
            seed=seed,
            samples_per_task=samples_per_task,
        )
    else:
        samples = _load_repobench_samples(
            raw_path,
            dataset_name=dataset_name,
            dataset_kind=dataset_kind,
            split=split,
            task=task,
            prompt_template=prompt_template,
            language=_required_string(dataset, "language") if task != "aggregate" else None,
            aggregate_sources=_repobench_aggregate_sources(dataset, base_dir=base_dir)
            if task == "aggregate"
            else None,
            tokenizer=tokenizer,
            min_prompt_tokens=min_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_target_tokens=min_target_tokens,
            max_target_tokens=max_target_tokens,
            language_filter=language_filter,
            dedup_content_hash=dedup_content_hash,
            counters=counters,
        )
    if not samples:
        raise ValueError("materialization produced no samples")
    ordered = _cache_realistic_order(samples, seed=seed, burst_size=burst_size)
    shards = _shard_samples(
        ordered,
        samples_per_shard=samples_per_shard,
        requested_num_shards=requested_num_shards,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)
    (output_dir / "workload_yamls").mkdir(exist_ok=True)
    normalized_config = dict(config)
    (output_dir / "materialization_config.yaml").write_text(
        yaml.safe_dump(normalized_config, sort_keys=False),
        encoding="utf-8",
    )
    shard_entries = _write_outputs(
        output_dir,
        name=name,
        shards=shards,
        request=_optional_mapping(config.get("request"), "request"),
        context_policy=_optional_mapping(
            _optional_mapping(config.get("workload_yaml"), "workload_yaml").get("context_policy"),
            "workload_yaml.context_policy",
        ),
    )
    manifest = _build_manifest(
        name=name,
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        task=task,
        profile=profile,
        prompt_template=prompt_template,
        shards=shards,
        shard_entries=shard_entries,
        samples_per_shard=samples_per_shard,
        selected_tasks=selected_tasks,
    )
    report = _build_report(
        name=name,
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        task=task,
        profile=profile,
        prompt_template=prompt_template,
        raw_path=raw_path,
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        samples=samples,
        counters=counters,
        selected_tasks=selected_tasks,
    )
    _write_json(output_dir / "shards_manifest.json", manifest)
    _write_json(output_dir / "materialization_report.json", report)
    return {
        "output_dir": str(output_dir),
        "materialization_report": str(output_dir / "materialization_report.json"),
        "shards_manifest": str(output_dir / "shards_manifest.json"),
        "num_shards": len(shards),
        "num_samples": sum(len(shard) for shard in shards),
    }


def _load_crosscodeeval_samples(
    raw_path: Path | None,
    *,
    aliases: dict[str, list[str]],
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    language_filter: dict[str, set[str]],
    dedup_content_hash: bool,
    counters: Counters,
) -> list[MaterializedSample]:
    files = _jsonl_files(raw_path)
    samples: list[MaterializedSample] = []
    seen_content_hashes: set[str] = set()
    global_index = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                counters.total_rows += 1
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{file_path}:{line_index + 1} must be a JSON object")
                sample = _crosscodeeval_row_to_sample(
                    row,
                    row_index=global_index,
                    input_path=file_path,
                    aliases=aliases,
                    dataset_name=dataset_name,
                    dataset_kind=dataset_kind,
                    split=split,
                    task=task,
                    prompt_template=prompt_template,
                    tokenizer=tokenizer,
                    min_prompt_tokens=min_prompt_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    min_target_tokens=min_target_tokens,
                    max_target_tokens=max_target_tokens,
                    language_filter=language_filter,
                    seen_content_hashes=seen_content_hashes,
                    dedup_content_hash=dedup_content_hash,
                    counters=counters,
                    source=f"{file_path}:{line_index + 1}",
                )
                global_index += 1
                if sample is not None:
                    samples.append(sample)
    counters.materialized_rows = len(samples)
    return samples


def _load_repobench_samples(
    raw_path: Path | None,
    *,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    language: str | None,
    aggregate_sources: list[tuple[str, Path, str]] | None = None,
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    language_filter: dict[str, set[str]],
    dedup_content_hash: bool,
    counters: Counters,
) -> list[MaterializedSample]:
    if task not in {"aggregate", "cross_file_first", "cross_file_random", "in_file"}:
        raise ValueError(
            "repobench dataset.mode must be one of: aggregate, cross_file_first, "
            "cross_file_random, in_file"
        )
    if task == "aggregate":
        if not aggregate_sources:
            raise ValueError("repobench aggregate mode requires dataset.raw_paths")
        sources = aggregate_sources
    else:
        if raw_path is None:
            raise ValueError("repobench non-aggregate mode requires dataset.raw_path")
        if language is None:
            raise ValueError("repobench non-aggregate mode requires dataset.language")
        sources = [(language, raw_path, task)]
    samples: list[MaterializedSample] = []
    seen_content_hashes: set[str] = set()
    global_index = 0
    for source_language, source_path, source_task in sources:
        if not _language_allowed(source_language, language_filter):
            raise ValueError(f"unsupported language for materialization: {source_language}")
        files = _repobench_parquet_files(source_path, mode=source_task)
        for file_path in files:
            rows = _read_parquet_rows(file_path)
            for row_index, row in enumerate(rows):
                counters.total_rows += 1
                sample = _repobench_row_to_sample(
                    row,
                    row_index=global_index,
                    dataset_name=dataset_name,
                    dataset_kind=dataset_kind,
                    split=split,
                    task=source_task,
                    prompt_template=prompt_template,
                    language=source_language,
                    tokenizer=tokenizer,
                    min_prompt_tokens=min_prompt_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    min_target_tokens=min_target_tokens,
                    max_target_tokens=max_target_tokens,
                    seen_content_hashes=seen_content_hashes,
                    dedup_content_hash=dedup_content_hash,
                    counters=counters,
                    source=f"{file_path}:{row_index}",
                )
                global_index += 1
                if sample is not None:
                    samples.append(sample)
    counters.materialized_rows = len(samples)
    return samples


def _load_longbench_samples(
    raw_path: Path | None,
    *,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    selected_tasks: list[str],
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    language_filter: dict[str, set[str]],
    dedup_content_hash: bool,
    counters: Counters,
    seed: int,
    samples_per_task: int,
) -> list[MaterializedSample]:
    if raw_path is None:
        raise ValueError("longbench materialization requires dataset.raw_path")
    samples: list[MaterializedSample] = []
    seen_content_hashes: set[str] = set()
    for task_name in selected_tasks:
        task_samples: list[MaterializedSample] = []
        for row_index, row, source in _iter_longbench_rows(raw_path, task_name):
            counters.total_rows += 1
            sample = _longbench_row_to_sample(
                row,
                row_index=row_index,
                dataset_name=dataset_name,
                dataset_kind=dataset_kind,
                split=split,
                task=task_name,
                profile=profile,
                profile_spec=profile_spec,
                tokenizer=tokenizer,
                min_prompt_tokens=min_prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
                min_target_tokens=min_target_tokens,
                max_target_tokens=max_target_tokens,
                language_filter=language_filter,
                seen_content_hashes=seen_content_hashes,
                dedup_content_hash=dedup_content_hash,
                counters=counters,
                source=source,
            )
            if sample is not None:
                task_samples.append(sample)
        if not task_samples:
            raise ValueError(f"LongBench task {task_name!r} produced no usable rows from {raw_path}")
        selected = _select_longbench_task_samples(
            task_samples,
            task_name=task_name,
            seed=seed,
            limit=samples_per_task,
        )
        counters.drops["not_selected_by_sampling"] += len(task_samples) - len(selected)
        samples.extend(selected)
    counters.materialized_rows = len(samples)
    return samples


def _iter_longbench_rows(raw_path: Path, task_name: str) -> list[tuple[int, dict[str, Any], str]]:
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
            return _iter_longbench_rows(zip_path, task_name)
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


def _longbench_row_to_sample(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    profile: str,
    profile_spec: LongBenchProfileSpec,
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    language_filter: dict[str, set[str]],
    seen_content_hashes: set[str],
    dedup_content_hash: bool,
    counters: Counters,
    source: str,
) -> MaterializedSample | None:
    prompt = _render_longbench_prompt(row)
    target = _longbench_reference_answer(row.get("answers"))
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(target))
    if prompt_token_count < min_prompt_tokens:
        counters.drops["prompt_too_short"] += 1
        return None
    if prompt_token_count > max_prompt_tokens:
        counters.drops["prompt_too_long"] += 1
        return None
    if target_token_count < min_target_tokens:
        counters.drops["missing_empty_target"] += 1
        return None
    if target_token_count > max_target_tokens:
        counters.drops["target_too_long"] += 1
        return None
    language = _optional_string(row.get("language"), f"{source}.language") or _default_longbench_language(task)
    if not _language_allowed(language, language_filter):
        counters.drops["unsupported_language"] += 1
        return None
    content_hash = _hash_text(prompt)
    if dedup_content_hash and content_hash in seen_content_hashes:
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
        "target_hash": _hash_text(target),
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "longbench_id": row.get("_id"),
        "longbench_row_index": row_index,
        "longbench_length": row.get("length"),
        "longbench_dataset": row.get("dataset"),
    }
    return MaterializedSample(
        sample_id=sample_id,
        prompt=prompt,
        target=target,
        expected_output_len=max(1, target_token_count),
        metadata=metadata,
    )


def _render_longbench_prompt(row: dict[str, Any]) -> str:
    context = _expect_string(row.get("context"), "longbench row.context")
    task_input = _expect_string(row.get("input"), "longbench row.input")
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
        f"{task_input}"
        f"{class_text}"
        f"{language_text}\n\n"
        "Answer:"
    )


def _longbench_reference_answer(answers: Any) -> str:
    if isinstance(answers, list):
        text_answers = [str(item) for item in answers if str(item)]
        if text_answers:
            return max(text_answers, key=len)
        return " "
    if answers is None:
        return " "
    return str(answers)


def _select_longbench_task_samples(
    samples: list[MaterializedSample],
    *,
    task_name: str,
    seed: int,
    limit: int,
) -> list[MaterializedSample]:
    if len(samples) <= limit:
        return sorted(samples, key=_sample_order_key)
    ranked = sorted(
        samples,
        key=lambda sample: (
            _hash_text(f"{seed}:{task_name}:{sample.sample_id}:{sample.metadata['content_hash']}"),
            sample.sample_id,
        ),
    )
    return sorted(ranked[:limit], key=_sample_order_key)


def _sample_order_key(sample: MaterializedSample) -> tuple[int, str, str]:
    return (
        int(sample.metadata["sequence_index"]),
        str(sample.metadata["file_path"]),
        str(sample.metadata["content_hash"]),
    )


def _repobench_row_to_sample(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    language: str,
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    seen_content_hashes: set[str],
    dedup_content_hash: bool,
    counters: Counters,
    source: str,
) -> MaterializedSample | None:
    repo_id = _expect_string(row.get("repo_name"), f"{source}.repo_name")
    file_path = _expect_string(row.get("file_path"), f"{source}.file_path")
    target = _expect_string(row.get("next_line"), f"{source}.next_line")
    cropped_code = _expect_string(row.get("cropped_code"), f"{source}.cropped_code")
    import_statement = _optional_string(row.get("import_statement"), f"{source}.import_statement") or ""
    context = row.get("context", [])
    if context is None:
        context = []
    if not isinstance(context, list):
        raise ValueError(f"{source}.context must be a list")
    prompt = _render_repobench_prompt(
        context=context,
        import_statement=import_statement,
        cropped_code=cropped_code,
        include_cross_file_context=task != "in_file",
        prompt_template=prompt_template,
    )
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(target))
    if prompt_token_count < min_prompt_tokens:
        counters.drops["prompt_too_short"] += 1
        return None
    if prompt_token_count > max_prompt_tokens:
        counters.drops["prompt_too_long"] += 1
        return None
    if target_token_count < min_target_tokens:
        counters.drops["missing_empty_target"] += 1
        return None
    if target_token_count > max_target_tokens:
        counters.drops["target_too_long"] += 1
        return None
    content_hash = _hash_text(prompt)
    if dedup_content_hash and content_hash in seen_content_hashes:
        counters.drops["duplicate_content_hash"] += 1
        return None
    seen_content_hashes.add(content_hash)
    directory = Path(file_path).parent.as_posix()
    if directory == ".":
        directory = ""
    sample_id = f"{dataset_name}-{language}-{row_index:06d}"
    metadata = {
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "split": split,
        "language": language,
        "repo_id": repo_id,
        "file_path": file_path,
        "session_id": f"{dataset_name}::{language}::{repo_id}::{directory}",
        "sample_id": sample_id,
        "sequence_index": row_index,
        "content_hash": content_hash,
        "ground_truth": target,
        "target_hash": _hash_text(target),
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "level": row.get("level"),
        "token_num": row.get("token_num"),
        "gold_snippet_index": row.get("gold_snippet_index"),
        "created_at": row.get("created_at"),
    }
    return MaterializedSample(
        sample_id=sample_id,
        prompt=prompt,
        target=target,
        expected_output_len=max(1, target_token_count),
        metadata=metadata,
    )


def _crosscodeeval_row_to_sample(
    row: dict[str, Any],
    *,
    row_index: int,
    input_path: Path,
    aliases: dict[str, list[str]],
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    tokenizer: PromptTokenizer,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_target_tokens: int,
    max_target_tokens: int,
    language_filter: dict[str, set[str]],
    seen_content_hashes: set[str],
    dedup_content_hash: bool,
    counters: Counters,
    source: str,
) -> MaterializedSample | None:
    current_file_prefix = _string_alias(row, aliases["prompt"], field_name="prompt", source=source)
    if current_file_prefix is None:
        counters.drops["missing_empty_prompt"] += 1
        return None
    target = _string_alias(row, aliases["target"], field_name="target", source=source)
    if target is None:
        counters.drops["missing_empty_target"] += 1
        return None
    cross_file_context = _string_alias(
        row,
        aliases["cross_file_context"],
        field_name="cross_file_context",
        source=source,
        required=False,
    )
    prompt = _render_crosscodeeval_prompt(
        current_file_prefix,
        cross_file_context,
        prompt_template=prompt_template,
    )
    prompt_token_count = len(tokenizer.encode(prompt))
    target_token_count = len(tokenizer.encode(target))
    if prompt_token_count < min_prompt_tokens:
        counters.drops["prompt_too_short"] += 1
        return None
    if prompt_token_count > max_prompt_tokens:
        counters.drops["prompt_too_long"] += 1
        return None
    if target_token_count < min_target_tokens:
        counters.drops["missing_empty_target"] += 1
        return None
    if target_token_count > max_target_tokens:
        counters.drops["target_too_long"] += 1
        return None

    language = _metadata_alias(row, aliases["language"], input_path.parent.name)
    if not _language_allowed(str(language), language_filter):
        counters.drops["unsupported_language"] += 1
        return None
    repo_id = str(_metadata_alias(row, aliases["repo_id"], "unknown"))
    source_file_path = str(_metadata_alias(row, aliases["file_path"], "unknown"))
    sequence_index = _sequence_index(row, aliases["sequence_index"], row_index, source=source)
    content_hash = _hash_text(prompt)
    if dedup_content_hash and content_hash in seen_content_hashes:
        counters.drops["duplicate_content_hash"] += 1
        return None
    seen_content_hashes.add(content_hash)
    directory = Path(source_file_path).parent.as_posix()
    if directory == ".":
        directory = ""
    sample_id = f"{dataset_name}-{language}-{row_index:06d}"
    metadata = {
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "split": split,
        "language": language,
        "repo_id": repo_id,
        "file_path": source_file_path,
        "session_id": f"{dataset_name}::{language}::{repo_id}::{directory}",
        "sample_id": sample_id,
        "sequence_index": sequence_index,
        "content_hash": content_hash,
        "ground_truth": target,
        "target_hash": _hash_text(target),
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
    }
    return MaterializedSample(
        sample_id=sample_id,
        prompt=prompt,
        target=target,
        expected_output_len=max(1, target_token_count),
        metadata=metadata,
    )


def _render_crosscodeeval_prompt(
    current_file_prefix: str,
    cross_file_context: str | None,
    *,
    prompt_template: str,
) -> str:
    if prompt_template == "plain_prefix":
        parts: list[str] = []
        if cross_file_context:
            parts.append("Relevant repository context:\n" + cross_file_context.strip())
        parts.append(current_file_prefix.rstrip())
        return "\n\n".join(parts)
    if prompt_template == "xml_tags" and cross_file_context:
        return (
            "Complete the code at the cursor. Return only the completion.\n\n"
            "<REPOSITORY_CONTEXT>\n"
            f"{cross_file_context}\n"
            "</REPOSITORY_CONTEXT>\n\n"
            "<CURRENT_FILE_PREFIX>\n"
            f"{current_file_prefix}\n"
            "</CURRENT_FILE_PREFIX>"
        )
    if prompt_template != "xml_tags":
        raise ValueError(f"unsupported prompt_template: {prompt_template}")
    return (
        "Complete the code at the cursor. Return only the completion.\n\n"
        "<CURRENT_FILE_PREFIX>\n"
        f"{current_file_prefix}\n"
        "</CURRENT_FILE_PREFIX>"
    )


def _render_repobench_prompt(
    *,
    context: list[Any],
    import_statement: str,
    cropped_code: str,
    include_cross_file_context: bool,
    prompt_template: str,
) -> str:
    context_blocks = _repobench_context_blocks(context) if include_cross_file_context and context else []
    if prompt_template == "plain_prefix":
        parts: list[str] = []
        if context_blocks:
            parts.append("Relevant repository context:\n" + "\n\n".join(context_blocks))
        if import_statement:
            parts.append(import_statement.strip())
        parts.append(cropped_code.rstrip())
        return "\n\n".join(parts)
    if prompt_template != "xml_tags":
        raise ValueError(f"unsupported prompt_template: {prompt_template}")
    parts = ["Complete the next line of code. Return only the next line."]
    if context_blocks:
        parts.append(
            "<REPOSITORY_CONTEXT>\n"
            + "\n\n".join(context_blocks)
            + "\n</REPOSITORY_CONTEXT>"
        )
    if import_statement:
        parts.append(f"<IMPORTS>\n{import_statement}\n</IMPORTS>")
    parts.append(f"<CURRENT_FILE_PREFIX>\n{cropped_code}\n</CURRENT_FILE_PREFIX>")
    return "\n\n".join(parts)


def _repobench_context_blocks(context: list[Any]) -> list[str]:
    context_blocks: list[str] = []
    for index, item in enumerate(context):
        if not isinstance(item, dict):
            raise ValueError(f"repobench context[{index}] must be a mapping")
        path = _optional_string(item.get("path"), f"repobench context[{index}].path") or "unknown"
        identifier = (
            _optional_string(item.get("identifier"), f"repobench context[{index}].identifier")
            or "unknown"
        )
        raw_snippet = item.get("snippet")
        if raw_snippet in (None, ""):
            continue
        snippet = _expect_string(raw_snippet, f"repobench context[{index}].snippet")
        context_blocks.append(f"# file: {path}\n# identifier: {identifier}\n{snippet}")
    return context_blocks


def _cache_realistic_order(
    samples: list[MaterializedSample],
    *,
    seed: int,
    burst_size: int,
) -> list[MaterializedSample]:
    by_session: dict[str, list[MaterializedSample]] = defaultdict(list)
    for sample in samples:
        by_session[str(sample.metadata["session_id"])].append(sample)
    session_ids = sorted(by_session)
    rng = random.Random(seed)
    rng.shuffle(session_ids)
    queues: deque[tuple[str, deque[MaterializedSample]]] = deque()
    for session_id in session_ids:
        ordered = sorted(
            by_session[session_id],
            key=lambda sample: (
                int(sample.metadata["sequence_index"]),
                str(sample.metadata["file_path"]),
                str(sample.metadata["content_hash"]),
            ),
        )
        queues.append((session_id, deque(ordered)))
    output: list[MaterializedSample] = []
    while queues:
        session_id, queue = queues.popleft()
        del session_id
        for _ in range(burst_size):
            if not queue:
                break
            output.append(queue.popleft())
        if queue:
            queues.append(("", queue))
    return output


def _shard_samples(
    samples: list[MaterializedSample],
    *,
    samples_per_shard: int,
    requested_num_shards: int | None,
) -> list[list[MaterializedSample]]:
    shards = [
        samples[index : index + samples_per_shard]
        for index in range(0, len(samples), samples_per_shard)
    ]
    if requested_num_shards is not None and len(shards) > requested_num_shards:
        raise ValueError(
            "materialized samples exceed sharding capacity: "
            f"need {len(shards)} shards with samples_per_shard={samples_per_shard}, "
            f"but sharding.num_shards={requested_num_shards}"
        )
    return [shard for shard in shards if shard]


def _write_outputs(
    output_dir: Path,
    *,
    name: str,
    shards: list[list[MaterializedSample]],
    request: dict[str, Any],
    context_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        shard_id = f"shard_{shard_index:03d}"
        shard_path = output_dir / "shards" / f"{shard_id}.runner.jsonl"
        workload_path = output_dir / "workload_yamls" / f"{shard_id}.yaml"
        with shard_path.open("w", encoding="utf-8") as handle:
            for sample in shard:
                metadata = dict(sample.metadata)
                metadata["shard_id"] = shard_id
                handle.write(
                    json.dumps(
                        {
                            "prompt": sample.prompt,
                            "expected_output_len": sample.expected_output_len,
                            "metadata": metadata,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        workload_payload = _workload_yaml_payload(
            name=name,
            shard_id=shard_id,
            shard_size=len(shard),
            request=request,
            context_policy=context_policy,
        )
        workload_path.write_text(
            yaml.safe_dump(workload_payload, sort_keys=False),
            encoding="utf-8",
        )
        entries.append(
            {
                "shard_id": shard_id,
                "num_samples": len(shard),
                "path": str(shard_path.relative_to(output_dir)),
                "workload_yaml_path": str(workload_path.relative_to(output_dir)),
            }
        )
    return entries


def _workload_yaml_payload(
    *,
    name: str,
    shard_id: str,
    shard_size: int,
    request: dict[str, Any],
    context_policy: dict[str, Any],
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "stream": request.get("stream", True),
        "temperature": request.get("temperature", 0.0),
        "ignore_eos": request.get("ignore_eos", False),
    }
    if "top_p" in request:
        request_payload["top_p"] = request["top_p"]
    extra_body = dict(_optional_mapping(request.get("extra_body"), "request.extra_body"))
    if "stop" in request:
        extra_body["stop"] = request["stop"]
    if extra_body:
        request_payload["extra_body"] = extra_body
    payload: dict[str, Any] = {
        "name": f"{name}-{shard_id}",
        "dataset": {
            "type": "jsonl",
            "path": f"../shards/{shard_id}.runner.jsonl",
        },
        "tokenizer": "whitespace",
        "sampling": {
            "seed": 42,
            "num_requests": shard_size,
            "entry_selection": "sequential",
            "prompt_len": {"mode": "from_dataset"},
            "output_len": {"mode": "from_dataset"},
        },
        "request": request_payload,
    }
    if context_policy:
        payload["context_policy"] = context_policy
    return payload


def _build_manifest(
    *,
    name: str,
    dataset_name: str,
    dataset_kind: str,
    task: str,
    profile: str | None,
    prompt_template: str,
    shards: list[list[MaterializedSample]],
    shard_entries: list[dict[str, Any]],
    samples_per_shard: int,
    selected_tasks: list[str] | None,
) -> dict[str, Any]:
    all_samples = [sample for shard in shards for sample in shard]
    manifest = {
        "workload_name": name,
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "num_shards": len(shards),
        "samples_per_shard": samples_per_shard,
        "shards": shard_entries,
        "selected_tasks": selected_tasks or sorted({str(sample.metadata["task"]) for sample in all_samples}),
        "language_counts": dict(Counter(str(sample.metadata["language"]) for sample in all_samples)),
        "prompt_tokens": _summary([int(sample.metadata["prompt_token_count"]) for sample in all_samples]),
        "target_tokens": _summary([int(sample.metadata["target_token_count"]) for sample in all_samples]),
        "profile_summaries": _group_summaries(all_samples, group_key="profile"),
        "task_summaries": _group_summaries(all_samples, group_key="task"),
        "unique_content_hashes": len({sample.metadata["content_hash"] for sample in all_samples}),
    }
    if profile is not None:
        manifest["profile"] = profile
    return manifest


def _build_report(
    *,
    name: str,
    dataset_name: str,
    dataset_kind: str,
    task: str,
    profile: str | None,
    prompt_template: str,
    raw_path: Path | None,
    output_dir: Path,
    tokenizer_name: str,
    samples: list[MaterializedSample],
    counters: Counters,
    selected_tasks: list[str] | None,
) -> dict[str, Any]:
    report = {
        "workload_name": name,
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "raw_path": str(raw_path) if raw_path is not None else None,
        "output_dir": str(output_dir),
        "selected_tasks": selected_tasks or sorted({str(sample.metadata["task"]) for sample in samples}),
        "tokenizer": {
            "name": tokenizer_name,
            "fallback_used": False,
        },
        "rows": {
            "total": counters.total_rows,
            "materialized": len(samples),
            "drops": dict(counters.drops),
        },
        "language_counts": dict(Counter(str(sample.metadata["language"]) for sample in samples)),
        "prompt_tokens": _summary([int(sample.metadata["prompt_token_count"]) for sample in samples]),
        "target_tokens": _summary([int(sample.metadata["target_token_count"]) for sample in samples]),
        "profile_summaries": _group_summaries(samples, group_key="profile"),
        "task_summaries": _group_summaries(samples, group_key="task"),
    }
    if profile is not None:
        report["profile"] = profile
    return report


def _group_summaries(
    samples: list[MaterializedSample],
    *,
    group_key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[MaterializedSample]] = defaultdict(list)
    for sample in samples:
        value = sample.metadata.get(group_key)
        if value in (None, ""):
            continue
        groups[str(value)].append(sample)
    summaries: dict[str, dict[str, Any]] = {}
    for group_name in sorted(groups):
        grouped_samples = groups[group_name]
        payload: dict[str, Any] = {
            "count": len(grouped_samples),
            "prompt_tokens": _summary([int(sample.metadata["prompt_token_count"]) for sample in grouped_samples]),
            "target_tokens": _summary([int(sample.metadata["target_token_count"]) for sample in grouped_samples]),
        }
        if group_key == "profile":
            payload["task_counts"] = dict(Counter(str(sample.metadata["task"]) for sample in grouped_samples))
        if group_key == "task":
            payload["language_counts"] = dict(Counter(str(sample.metadata["language"]) for sample in grouped_samples))
            profile_counts = Counter(
                str(sample.metadata["profile"])
                for sample in grouped_samples
                if sample.metadata.get("profile") not in (None, "")
            )
            if profile_counts:
                payload["profile_counts"] = dict(profile_counts)
            first_metadata = grouped_samples[0].metadata
            if "workload_type" in first_metadata:
                payload["workload_type"] = first_metadata["workload_type"]
            if "output_regime" in first_metadata:
                payload["output_regime"] = first_metadata["output_regime"]
        summaries[group_name] = payload
    return summaries


def _jsonl_files(raw_path: Path) -> list[Path]:
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


def _repobench_parquet_files(raw_path: Path, *, mode: str) -> list[Path]:
    if raw_path.is_file():
        if raw_path.suffix != ".parquet":
            raise ValueError(f"RepoBench raw_path file must be .parquet: {raw_path}")
        if not raw_path.name.startswith(f"{mode}-"):
            raise ValueError(f"RepoBench parquet file does not match mode {mode!r}: {raw_path}")
        return [raw_path]
    if raw_path.is_dir():
        files = sorted(raw_path.rglob(f"{mode}-*.parquet"))
        if files:
            return files
        raise FileNotFoundError(f"RepoBench raw_path directory has no {mode}-*.parquet files: {raw_path}")
    raise FileNotFoundError(f"RepoBench raw_path not found: {raw_path}")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pyarrow is required for RepoBench parquet materialization") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"RepoBench parquet file is empty: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"RepoBench parquet row {path}:{index} must be a mapping")
    return rows


def _field_aliases(payload: Any) -> dict[str, list[str]]:
    aliases = {key: list(value) for key, value in DEFAULT_CROSSCODEEVAL_FIELD_ALIASES.items()}
    overrides = _optional_mapping(payload, "dataset.field_aliases")
    for key, value in overrides.items():
        if key not in aliases:
            raise ValueError(f"dataset.field_aliases has unknown key: {key}")
        if not isinstance(value, list) or not value:
            raise ValueError(f"dataset.field_aliases.{key} must be a non-empty list")
        aliases[key] = [_expect_string(item, f"dataset.field_aliases.{key}[]") for item in value]
    return aliases


def _dataset_kind(dataset_name: str) -> str:
    try:
        return SUPPORTED_DATASET_KINDS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_DATASET_KINDS))
        raise ValueError(f"supported dataset.name values: {supported}") from exc


def _string_alias(
    row: dict[str, Any],
    aliases: list[str],
    *,
    field_name: str,
    source: str,
    required: bool = True,
) -> str | None:
    for alias in aliases:
        found, value = _lookup_alias(row, alias)
        if not found:
            continue
        if value is None or value == "":
            return None
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            value = value["text"]
        if not isinstance(value, str):
            raise ValueError(f"{source} field {alias!r} for {field_name} must be a string")
        return value
    if required:
        return None
    return None


def _metadata_alias(row: dict[str, Any], aliases: list[str], default: str) -> Any:
    for alias in aliases:
        found, value = _lookup_alias(row, alias)
        if found and value not in (None, ""):
            return value
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            found, value = _lookup_alias(metadata, alias)
            if found and value not in (None, ""):
                return value
    return default


def _lookup_alias(row: dict[str, Any], alias: str) -> tuple[bool, Any]:
    current: Any = row
    for part in alias.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _sequence_index(
    row: dict[str, Any],
    aliases: list[str],
    default: int,
    *,
    source: str,
) -> int:
    value = _metadata_alias(row, aliases, default)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"{source} sequence_index must be an integer")


def _language_filter(payload: Any) -> dict[str, set[str]]:
    languages = _optional_mapping(payload, "filtering.languages")
    return {
        "include": set(_string_list(languages.get("include", []), "filtering.languages.include")),
        "exclude": set(_string_list(languages.get("exclude", []), "filtering.languages.exclude")),
    }


def _language_allowed(language: str, language_filter: dict[str, set[str]]) -> bool:
    include = language_filter["include"]
    exclude = language_filter["exclude"]
    return (not include or language in include) and language not in exclude


def _dedup_content_hash(payload: Any) -> bool:
    dedup = _optional_mapping(payload, "filtering.dedup")
    value = dedup.get("content_hash", True)
    if not isinstance(value, bool):
        raise ValueError("filtering.dedup.content_hash must be a boolean")
    return value


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[int], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_raw_path(dataset: dict[str, Any], *, base_dir: Path) -> Path | None:
    raw_path = dataset.get("raw_path")
    if raw_path is None:
        return None
    return _resolve_path(_expect_string(raw_path, "dataset.raw_path"), base_dir=base_dir)


def _repobench_aggregate_sources(
    dataset: dict[str, Any],
    *,
    base_dir: Path,
) -> list[tuple[str, Path, str]]:
    raw_paths = _optional_mapping(dataset.get("raw_paths"), "dataset.raw_paths")
    if not raw_paths:
        raise ValueError("repobench aggregate mode requires dataset.raw_paths")
    tasks = _repobench_tasks(dataset.get("tasks"))
    sources: list[tuple[str, Path, str]] = []
    for language in sorted(raw_paths):
        raw_path = _resolve_path(
            _expect_string(raw_paths[language], f"dataset.raw_paths.{language}"),
            base_dir=base_dir,
        )
        for task in tasks:
            sources.append((language, raw_path, task))
    return sources


def _repobench_tasks(value: Any) -> list[str]:
    if value is None:
        return ["in_file", "cross_file_first", "cross_file_random"]
    tasks = _string_list(value, "dataset.tasks")
    allowed = {"in_file", "cross_file_first", "cross_file_random"}
    unknown = sorted(set(tasks) - allowed)
    if unknown:
        raise ValueError(f"dataset.tasks has unsupported RepoBench tasks: {unknown}")
    return tasks


def _longbench_selection(dataset: dict[str, Any]) -> tuple[str, LongBenchProfileSpec, list[str]]:
    profile = _required_string(dataset, "profile")
    try:
        profile_spec = LONGBENCH_PROFILE_SPECS[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(LONGBENCH_PROFILE_SPECS))
        raise ValueError(f"supported LongBench dataset.profile values: {supported}") from exc
    requested = (
        _string_list(dataset.get("configs"), "dataset.configs")
        if dataset.get("configs") is not None
        else list(profile_spec.tasks)
    )
    if not requested:
        raise ValueError("dataset.configs must not be empty when provided")
    excluded = sorted(task for task in requested if task in LONGBENCH_EXCLUDED_TASKS)
    if excluded:
        raise ValueError(
            "LongBench realistic-NL profiles exclude tasks: "
            + ", ".join(excluded)
        )
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


def _prompt_template(value: Any) -> str:
    template = _optional_string(value, "prompt_template") or "plain_prefix"
    if template not in {"plain_prefix", "xml_tags"}:
        raise ValueError("prompt_template must be one of: plain_prefix, xml_tags")
    return template


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_longbench_language(task_name: str) -> str:
    if task_name.endswith("_zh") or task_name == "dureader":
        return "zh"
    return "en"


def _resolve_path(path: str, *, base_dir: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return _optional_mapping(payload[key], key)


def _optional_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return _expect_string(payload[key], key)


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name)


def _expect_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [_expect_string(item, f"{field_name}[]") for item in value]


def _int_setting(payload: dict[str, Any], key: str, default: int) -> int:
    return _positive_int(payload.get(key, default), key)


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
