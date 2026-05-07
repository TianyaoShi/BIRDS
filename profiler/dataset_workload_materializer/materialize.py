from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_mst_finder.workload import resolve_tokenizer

from .common import (
    config_raw_path,
    dataset_kind,
    dedup_content_hash,
    int_setting,
    language_filter,
    optional_mapping,
    optional_string,
    positive_int,
    required_mapping,
    required_string,
    resolve_path,
)
from .datasets import load_dataset
from .models import Counters, FilteringConfig, MaterializationContext, SamplingConfig
from .outputs import (
    build_manifest,
    build_report,
    cache_realistic_order,
    epoch_shuffle_expand,
    shard_samples,
    write_json,
    write_outputs,
)


def materialize_from_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("materialization config must be a mapping")
    return prepare(payload, config_source=path)


def prepare(config: dict[str, Any], *, config_source: Path | None = None) -> dict[str, Any]:
    name = required_string(config, "name")
    dataset = required_mapping(config, "dataset")
    dataset_payload = dict(dataset)
    if "prompt_template" not in dataset_payload and "prompt_template" in config:
        dataset_payload["prompt_template"] = config["prompt_template"]
    dataset_name = optional_string(dataset.get("name"), "dataset.name") or "crosscodeeval"
    base_dir = config_source.parent if config_source is not None else Path.cwd()
    raw_path = config_raw_path(dataset, base_dir=base_dir)
    split = optional_string(dataset.get("split"), "dataset.split") or (
        "test" if dataset_name == "longbench" else "unspecified"
    )

    tokenization = optional_mapping(config.get("tokenization"), "tokenization")
    tokenizer_name = optional_string(tokenization.get("tokenizer"), "tokenization.tokenizer") or "whitespace"
    tokenizer_name = _resolve_tokenizer_spec(tokenizer_name, base_dir=base_dir)
    tokenizer = resolve_tokenizer(tokenizer_name)

    filtering_payload = optional_mapping(config.get("filtering"), "filtering")
    filtering = FilteringConfig(
        min_prompt_tokens=int_setting(filtering_payload, "min_prompt_tokens", 128),
        max_prompt_tokens=int_setting(filtering_payload, "max_prompt_tokens", 8192),
        min_target_tokens=int_setting(filtering_payload, "min_target_tokens", 1),
        max_target_tokens=int_setting(filtering_payload, "max_target_tokens", 128),
        language_filter=language_filter(filtering_payload.get("languages", {})),
        dedup_content_hash=dedup_content_hash(filtering_payload.get("dedup", {})),
    )

    sampling_payload = optional_mapping(config.get("sampling"), "sampling")
    samples_per_task = sampling_payload.get("samples_per_task")
    if samples_per_task is not None:
        samples_per_task = positive_int(samples_per_task, "sampling.samples_per_task")
    repeat_policy = optional_string(sampling_payload.get("repeat_policy"), "sampling.repeat_policy")
    if repeat_policy is not None and repeat_policy != "epoch_shuffle":
        raise ValueError("sampling.repeat_policy must be 'epoch_shuffle' when provided")
    target_samples = sampling_payload.get("target_samples")
    if target_samples is not None:
        target_samples = positive_int(target_samples, "sampling.target_samples")
    if repeat_policy is None and target_samples is not None:
        raise ValueError("sampling.target_samples requires sampling.repeat_policy")
    if repeat_policy is not None and target_samples is None:
        raise ValueError("sampling.repeat_policy requires sampling.target_samples")
    sampling = SamplingConfig(
        seed=int_setting(sampling_payload, "seed", 42),
        burst_size=int_setting(sampling_payload, "burst_size", 8),
        policy=optional_string(sampling_payload.get("policy"), "sampling.policy"),
        samples_per_task=samples_per_task,
        repeat_policy=repeat_policy,
        target_samples=target_samples,
    )

    sharding = required_mapping(config, "sharding")
    output_dir = resolve_path(
        required_string(sharding, "output_dir"),
        base_dir=base_dir,
    )
    samples_per_shard = int_setting(sharding, "samples_per_shard", 8000)
    requested_num_shards = sharding.get("num_shards")
    if requested_num_shards is not None:
        requested_num_shards = positive_int(requested_num_shards, "sharding.num_shards")

    counters = Counters()
    ctx = MaterializationContext(
        base_dir=base_dir,
        dataset_name=dataset_name,
        dataset_kind=dataset_kind(dataset_name),
        raw_path=raw_path,
        split=split,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        filtering=filtering,
        sampling=sampling,
        counters=counters,
    )
    loaded = load_dataset(dataset_payload, ctx)
    if not loaded.samples:
        raise ValueError("materialization produced no samples")

    ordered = cache_realistic_order(
        loaded.samples,
        seed=sampling.seed,
        burst_size=sampling.burst_size,
    )
    if sampling.repeat_policy == "epoch_shuffle":
        if sampling.target_samples is None:
            raise ValueError("sampling.target_samples is required for epoch_shuffle")
        ordered = epoch_shuffle_expand(
            ordered,
            seed=sampling.seed,
            target_samples=sampling.target_samples,
        )
    shards = shard_samples(
        ordered,
        samples_per_shard=samples_per_shard,
        requested_num_shards=requested_num_shards,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)
    (output_dir / "workload_yamls").mkdir(exist_ok=True)
    (output_dir / "materialization_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )

    request = optional_mapping(config.get("request"), "request")
    workload_yaml = optional_mapping(config.get("workload_yaml"), "workload_yaml")
    context_policy = optional_mapping(
        workload_yaml.get("context_policy"),
        "workload_yaml.context_policy",
    )
    output_len = optional_mapping(workload_yaml.get("output_len"), "workload_yaml.output_len")
    shard_entries = write_outputs(
        output_dir,
        name=name,
        shards=shards,
        request=request,
        context_policy=context_policy,
        output_len=output_len,
    )
    manifest = build_manifest(
        name=name,
        dataset_name=dataset_name,
        dataset_kind=ctx.dataset_kind,
        task=loaded.task,
        profile=loaded.profile,
        prompt_template=loaded.prompt_template,
        shards=shards,
        shard_entries=shard_entries,
        samples_per_shard=samples_per_shard,
        selected_tasks=loaded.selected_tasks,
    )
    report = build_report(
        name=name,
        dataset_name=dataset_name,
        dataset_kind=ctx.dataset_kind,
        task=loaded.task,
        profile=loaded.profile,
        prompt_template=loaded.prompt_template,
        raw_path=raw_path,
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        samples=ordered,
        counters=counters,
        selected_tasks=loaded.selected_tasks,
    )
    write_json(output_dir / "shards_manifest.json", manifest)
    write_json(output_dir / "materialization_report.json", report)
    return _result_payload(output_dir, shards)


def _result_payload(output_dir: Path, shards: list[list[Any]]) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "materialization_report": str(output_dir / "materialization_report.json"),
        "shards_manifest": str(output_dir / "shards_manifest.json"),
        "num_shards": len(shards),
        "num_samples": sum(len(shard) for shard in shards),
    }


def _resolve_tokenizer_spec(tokenizer_spec: str, *, base_dir: Path) -> str:
    if tokenizer_spec in {"whitespace", "character"}:
        return tokenizer_spec
    candidate = Path(tokenizer_spec).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    resolved = (base_dir / candidate).resolve()
    if candidate.parts and (tokenizer_spec.startswith(".") or resolved.exists()):
        return str(resolved)
    return tokenizer_spec
