from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_mst_finder.workload import PromptTokenizer

from ..common import (
    expect_string,
    hash_text,
    jsonl_files,
    language_allowed,
    optional_mapping,
    optional_string,
    prompt_template,
    resolve_path,
    required_string,
    string_list,
)
from ..models import DatasetLoadResult, MaterializationContext, MaterializedSample


DEFAULT_CROSSCODEEVAL_FIELD_ALIASES: dict[str, list[str]] = {
    "prompt": ["prompt", "input", "code_context", "current_file_prefix"],
    "target": ["target", "completion", "reference", "groundtruth"],
    "language": ["language", "lang"],
    "repo_id": ["repo", "repo_name", "repository"],
    "file_path": ["file_path", "path"],
    "cross_file_context": ["cross_file_context", "crossfile_context", "retrieved_context", "context"],
    "sequence_index": ["sequence_index", "cursor_index", "order"],
}


def load_code_dataset(dataset: dict[str, Any], ctx: MaterializationContext) -> DatasetLoadResult:
    task = optional_string(dataset.get("mode"), "dataset.mode") or (
        "cross_file_first" if ctx.dataset_name == "repobench" else "cross_file_materialized"
    )
    rendered_prompt_template = prompt_template(
        dataset.get("prompt_template", "plain_prefix")
    )
    if ctx.dataset_name == "crosscodeeval":
        if ctx.raw_path is None:
            raise ValueError("crosscodeeval materialization requires dataset.raw_path")
        aliases = field_aliases(dataset.get("field_aliases", {}))
        samples = load_crosscodeeval_samples(
            ctx.raw_path,
            aliases=aliases,
            dataset_name=ctx.dataset_name,
            dataset_kind=ctx.dataset_kind,
            split=ctx.split,
            task=task,
            prompt_template=rendered_prompt_template,
            tokenizer=ctx.tokenizer,
            filtering=ctx.filtering,
            counters=ctx.counters,
        )
        return DatasetLoadResult(samples=samples, task=task, prompt_template=rendered_prompt_template)
    samples = load_repobench_samples(
        ctx.raw_path,
        dataset_name=ctx.dataset_name,
        dataset_kind=ctx.dataset_kind,
        split=ctx.split,
        task=task,
        prompt_template=rendered_prompt_template,
        language=required_string(dataset, "language") if task != "aggregate" else None,
        aggregate_sources=repobench_aggregate_sources(dataset, base_dir=ctx.base_dir)
        if task == "aggregate"
        else None,
        tokenizer=ctx.tokenizer,
        filtering=ctx.filtering,
        counters=ctx.counters,
    )
    return DatasetLoadResult(samples=samples, task=task, prompt_template=rendered_prompt_template)


def load_crosscodeeval_samples(
    raw_path: Path,
    *,
    aliases: dict[str, list[str]],
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    tokenizer: PromptTokenizer,
    filtering,
    counters,
) -> list[MaterializedSample]:
    files = jsonl_files(raw_path)
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
                sample = crosscodeeval_row_to_sample(
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
                    filtering=filtering,
                    seen_content_hashes=seen_content_hashes,
                    counters=counters,
                    source=f"{file_path}:{line_index + 1}",
                )
                global_index += 1
                if sample is not None:
                    samples.append(sample)
    counters.materialized_rows = len(samples)
    return samples


def load_repobench_samples(
    raw_path: Path | None,
    *,
    dataset_name: str,
    dataset_kind: str,
    split: str,
    task: str,
    prompt_template: str,
    language: str | None,
    aggregate_sources: list[tuple[str, Path, str]] | None,
    tokenizer: PromptTokenizer,
    filtering,
    counters,
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
        if not language_allowed(source_language, filtering.language_filter):
            raise ValueError(f"unsupported language for materialization: {source_language}")
        files = repobench_parquet_files(source_path, mode=source_task)
        for file_path in files:
            rows = read_parquet_rows(file_path)
            for row_index, row in enumerate(rows):
                counters.total_rows += 1
                sample = repobench_row_to_sample(
                    row,
                    row_index=global_index,
                    dataset_name=dataset_name,
                    dataset_kind=dataset_kind,
                    split=split,
                    task=source_task,
                    prompt_template=prompt_template,
                    language=source_language,
                    tokenizer=tokenizer,
                    filtering=filtering,
                    seen_content_hashes=seen_content_hashes,
                    counters=counters,
                    source=f"{file_path}:{row_index}",
                )
                global_index += 1
                if sample is not None:
                    samples.append(sample)
    counters.materialized_rows = len(samples)
    return samples


def repobench_row_to_sample(
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
    filtering,
    seen_content_hashes: set[str],
    counters,
    source: str,
) -> MaterializedSample | None:
    repo_id = expect_string(row.get("repo_name"), f"{source}.repo_name")
    file_path = expect_string(row.get("file_path"), f"{source}.file_path")
    target = expect_string(row.get("next_line"), f"{source}.next_line")
    cropped_code = expect_string(row.get("cropped_code"), f"{source}.cropped_code")
    import_statement = optional_string(row.get("import_statement"), f"{source}.import_statement") or ""
    context = row.get("context", [])
    if context is None:
        context = []
    if not isinstance(context, list):
        raise ValueError(f"{source}.context must be a list")
    prompt = render_repobench_prompt(
        context=context,
        import_statement=import_statement,
        cropped_code=cropped_code,
        include_cross_file_context=task != "in_file",
        prompt_template=prompt_template,
    )
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
    content_hash = hash_text(prompt)
    if filtering.dedup_content_hash and content_hash in seen_content_hashes:
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
        "target_hash": hash_text(target),
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


def crosscodeeval_row_to_sample(
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
    filtering,
    seen_content_hashes: set[str],
    counters,
    source: str,
) -> MaterializedSample | None:
    current_file_prefix = string_alias(row, aliases["prompt"], field_name="prompt", source=source)
    if current_file_prefix is None:
        counters.drops["missing_empty_prompt"] += 1
        return None
    target = string_alias(row, aliases["target"], field_name="target", source=source)
    if target is None:
        counters.drops["missing_empty_target"] += 1
        return None
    cross_file_context = string_alias(
        row,
        aliases["cross_file_context"],
        field_name="cross_file_context",
        source=source,
        required=False,
    )
    prompt = render_crosscodeeval_prompt(
        current_file_prefix,
        cross_file_context,
        prompt_template=prompt_template,
    )
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

    language = metadata_alias(row, aliases["language"], input_path.parent.name)
    if not language_allowed(str(language), filtering.language_filter):
        counters.drops["unsupported_language"] += 1
        return None
    repo_id = str(metadata_alias(row, aliases["repo_id"], "unknown"))
    source_file_path = str(metadata_alias(row, aliases["file_path"], "unknown"))
    sequence_index = parse_sequence_index(row, aliases["sequence_index"], row_index, source=source)
    content_hash = hash_text(prompt)
    if filtering.dedup_content_hash and content_hash in seen_content_hashes:
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
        "target_hash": hash_text(target),
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


def render_crosscodeeval_prompt(
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


def render_repobench_prompt(
    *,
    context: list[Any],
    import_statement: str,
    cropped_code: str,
    include_cross_file_context: bool,
    prompt_template: str,
) -> str:
    context_blocks = repobench_context_blocks(context) if include_cross_file_context and context else []
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


def repobench_context_blocks(context: list[Any]) -> list[str]:
    context_blocks: list[str] = []
    for index, item in enumerate(context):
        if not isinstance(item, dict):
            raise ValueError(f"repobench context[{index}] must be a mapping")
        path = optional_string(item.get("path"), f"repobench context[{index}].path") or "unknown"
        identifier = (
            optional_string(item.get("identifier"), f"repobench context[{index}].identifier")
            or "unknown"
        )
        raw_snippet = item.get("snippet")
        if raw_snippet in (None, ""):
            continue
        snippet = expect_string(raw_snippet, f"repobench context[{index}].snippet")
        context_blocks.append(f"# file: {path}\n# identifier: {identifier}\n{snippet}")
    return context_blocks


def repobench_parquet_files(raw_path: Path, *, mode: str) -> list[Path]:
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


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for RepoBench parquet materialization") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"RepoBench parquet file is empty: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"RepoBench parquet row {path}:{index} must be a mapping")
    return rows


def field_aliases(payload: Any) -> dict[str, list[str]]:
    aliases = {key: list(value) for key, value in DEFAULT_CROSSCODEEVAL_FIELD_ALIASES.items()}
    overrides = optional_mapping(payload, "dataset.field_aliases")
    for key, value in overrides.items():
        if key not in aliases:
            raise ValueError(f"dataset.field_aliases has unknown key: {key}")
        if not isinstance(value, list) or not value:
            raise ValueError(f"dataset.field_aliases.{key} must be a non-empty list")
        aliases[key] = [expect_string(item, f"dataset.field_aliases.{key}[]") for item in value]
    return aliases


def string_alias(
    row: dict[str, Any],
    aliases: list[str],
    *,
    field_name: str,
    source: str,
    required: bool = True,
) -> str | None:
    for alias in aliases:
        found, value = lookup_alias(row, alias)
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


def metadata_alias(row: dict[str, Any], aliases: list[str], default: str) -> Any:
    for alias in aliases:
        found, value = lookup_alias(row, alias)
        if found and value not in (None, ""):
            return value
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            found, value = lookup_alias(metadata, alias)
            if found and value not in (None, ""):
                return value
    return default


def lookup_alias(row: dict[str, Any], alias: str) -> tuple[bool, Any]:
    current: Any = row
    for part in alias.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def parse_sequence_index(
    row: dict[str, Any],
    aliases: list[str],
    default: int,
    *,
    source: str,
) -> int:
    value = metadata_alias(row, aliases, default)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"{source} sequence_index must be an integer")


def repobench_aggregate_sources(
    dataset: dict[str, Any],
    *,
    base_dir: Path,
) -> list[tuple[str, Path, str]]:
    raw_paths = optional_mapping(dataset.get("raw_paths"), "dataset.raw_paths")
    if not raw_paths:
        raise ValueError("repobench aggregate mode requires dataset.raw_paths")
    tasks = repobench_tasks(dataset.get("tasks"))
    sources: list[tuple[str, Path, str]] = []
    for language in sorted(raw_paths):
        raw_path = resolve_path(
            expect_string(raw_paths[language], f"dataset.raw_paths.{language}"),
            base_dir=base_dir,
        )
        for task in tasks:
            sources.append((language, raw_path, task))
    return sources


def repobench_tasks(value: Any) -> list[str]:
    if value is None:
        return ["in_file", "cross_file_first", "cross_file_random"]
    tasks = string_list(value, "dataset.tasks")
    allowed = {"in_file", "cross_file_first", "cross_file_random"}
    unknown = sorted(set(tasks) - allowed)
    if unknown:
        raise ValueError(f"dataset.tasks has unsupported RepoBench tasks: {unknown}")
    return tasks
