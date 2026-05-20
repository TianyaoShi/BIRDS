from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from llm_mst_finder.workload import (
    DatasetConfig,
    DatasetEntry,
    LengthSpec,
    SamplingConfig,
    _load_hf_entry_sample_from_source,
    _load_sharegpt_entries_from_source,
    resolve_tokenizer,
)

from .models import (
    DEFAULT_BUCKET_POLICY,
    DEFAULT_DECODING_CONFIG,
    PromptLengthBucketName,
    PromptLengthBucketPolicy,
)


class QualityMaterializationConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QualitySourceConfig:
    name: str
    weight: float
    dataset: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QualityMaterializationConfig:
    path: Path
    output_dir: Path
    tokenizer: str
    seed: int
    total_requests: int
    shards: int
    sources: tuple[QualitySourceConfig, ...]
    bucket_policy: PromptLengthBucketPolicy
    minimum_per_source_bucket: int
    allow_replacement: bool


@dataclass(frozen=True, slots=True)
class QualityMaterializationResult:
    output_dir: Path
    request_manifest: Path
    materialization_report: Path
    shard_count: int
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "request_manifest": str(self.request_manifest),
            "materialization_report": str(self.materialization_report),
            "shard_count": self.shard_count,
            "request_count": self.request_count,
        }


def assign_prompt_length_bucket(
    prompt_tokens: int,
    policy: PromptLengthBucketPolicy = DEFAULT_BUCKET_POLICY,
) -> PromptLengthBucketName:
    return policy.bucket_for(prompt_tokens)


def load_materialization_config(path: str | Path) -> QualityMaterializationConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _expect_mapping(payload, "materialization config")
    _check_allowed_keys(root, "materialization config", {"output_dir", "tokenization", "sampling"})
    output_dir_raw = root.get("output_dir")
    if output_dir_raw is None:
        output_dir = config_path.with_suffix("")
    else:
        output_dir_value = Path(_expect_str(output_dir_raw, "output_dir"))
        output_dir = output_dir_value if output_dir_value.is_absolute() else (config_path.parent / output_dir_value).resolve()
    tokenization = _expect_mapping(root.get("tokenization"), "tokenization")
    _check_allowed_keys(tokenization, "tokenization", {"tokenizer"})
    tokenizer = _expect_str(tokenization.get("tokenizer"), "tokenization.tokenizer")
    if tokenizer == "whitespace":
        raise QualityMaterializationConfigError("tokenization.tokenizer must not be whitespace")
    sampling = _expect_mapping(root.get("sampling"), "sampling")
    _check_allowed_keys(
        sampling,
        "sampling",
        {
            "seed",
            "total_requests",
            "shards",
            "sources",
            "prompt_length_buckets",
            "allocation",
        },
    )
    total_requests = _expect_int(sampling.get("total_requests"), "sampling.total_requests", minimum=1)
    shards = _expect_int(sampling.get("shards"), "sampling.shards", minimum=1)
    if total_requests != 10000:
        raise QualityMaterializationConfigError("sampling.total_requests must be 10000 for V1")
    if shards != 10:
        raise QualityMaterializationConfigError("sampling.shards must be 10 for V1")
    if total_requests % shards != 0:
        raise QualityMaterializationConfigError("sampling.total_requests must divide evenly by shards")

    sources = _parse_sources(sampling.get("sources"))
    _validate_v1_sources(sources)
    bucket_policy = _parse_bucket_policy(sampling.get("prompt_length_buckets"))
    allocation = _expect_mapping(sampling.get("allocation"), "sampling.allocation")
    _check_allowed_keys(
        allocation,
        "sampling.allocation",
        {"source", "bucket", "minimum_per_source_bucket", "allow_replacement"},
    )
    if allocation.get("source") != "exact":
        raise QualityMaterializationConfigError("sampling.allocation.source must be exact")
    if allocation.get("bucket") != "proportional_with_minimum":
        raise QualityMaterializationConfigError(
            "sampling.allocation.bucket must be proportional_with_minimum"
        )
    minimum = _expect_int(
        allocation.get("minimum_per_source_bucket"),
        "sampling.allocation.minimum_per_source_bucket",
        minimum=1,
    )
    allow_replacement = _expect_bool(
        allocation.get("allow_replacement", False),
        "sampling.allocation.allow_replacement",
    )
    return QualityMaterializationConfig(
        path=config_path,
        output_dir=output_dir,
        tokenizer=tokenizer,
        seed=_expect_int(sampling.get("seed"), "sampling.seed"),
        total_requests=total_requests,
        shards=shards,
        sources=sources,
        bucket_policy=bucket_policy,
        minimum_per_source_bucket=minimum,
        allow_replacement=allow_replacement,
    )


def materialize_quality_requests(config_path: str | Path, *, force: bool = False) -> QualityMaterializationResult:
    config = load_materialization_config(config_path)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not force:
            raise QualityMaterializationConfigError(
                f"refusing to overwrite existing output directory: {config.output_dir}"
            )
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = config.output_dir / "shards"
    workloads_dir = config.output_dir / "workload_yamls"
    shards_dir.mkdir()
    workloads_dir.mkdir()
    shutil.copyfile(config.path, config.output_dir / "materialization_config.yaml")

    tokenizer = resolve_tokenizer(config.tokenizer)
    target_counts = source_request_counts(config)
    all_selected: list[dict[str, Any]] = []
    report_sources: dict[str, Any] = {}
    for source in config.sources:
        target_count = target_counts[source.name]
        dataset_entries, source_report = _load_source_entries(
            source,
            config=config,
            tokenizer=tokenizer,
            target_count=target_count,
            base_dir=config.path.parent,
        )
        selected, selection_report = _select_source_entries(
            dataset_entries,
            source_name=source.name,
            target_count=target_count,
            config=config,
        )
        all_selected.extend(selected)
        report_sources[source.name] = {
            **source_report,
            **selection_report,
        }

    shards = _balanced_shards(all_selected, shard_count=config.shards, seed=config.seed)
    shard_entries = []
    for shard_index, rows in enumerate(shards):
        shard_id = f"shard_{shard_index:03d}"
        shard_path = shards_dir / f"{shard_id}.runner.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for within_index, row in enumerate(rows):
                metadata = dict(row["metadata"])
                metadata["shard_id"] = shard_id
                metadata["within_shard_index"] = within_index
                payload = {
                    "prompt": row["prompt"],
                    "prompt_len": row["prompt_len"],
                    "expected_output_len": DEFAULT_DECODING_CONFIG.max_tokens,
                    "metadata": metadata,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        workload_path = workloads_dir / f"{shard_id}.yaml"
        workload_path.write_text(
            yaml.safe_dump(
                _workload_payload(
                    name=f"quality-sharegpt-wildchat-{shard_id}",
                    shard_id=shard_id,
                    shard_size=len(rows),
                    tokenizer=config.tokenizer,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        shard_entries.append(
            {
                "shard_id": shard_id,
                "num_requests": len(rows),
                "path": str(shard_path.relative_to(config.output_dir)),
                "workload_yaml_path": str(workload_path.relative_to(config.output_dir)),
                "source_counts": dict(Counter(row["metadata"]["source"] for row in rows)),
                "bucket_counts": dict(Counter(row["metadata"]["prompt_length_bucket"] for row in rows)),
            }
        )

    request_manifest = {
        "name": "sharegpt_wildchat_10k",
        "tokenizer": config.tokenizer,
        "total_requests": len(all_selected),
        "num_shards": len(shards),
        "source_counts": dict(Counter(row["metadata"]["source"] for row in all_selected)),
        "bucket_counts": dict(Counter(row["metadata"]["prompt_length_bucket"] for row in all_selected)),
        "stratum_counts": dict(Counter(row["metadata"]["stratum"] for row in all_selected)),
        "shards": shard_entries,
    }
    report = {
        "config_path": str(config.path),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "tokenizer": config.tokenizer,
        "bucket_policy": config.bucket_policy.to_dict(),
        "sources": report_sources,
        "manifest": request_manifest,
    }
    _write_json(config.output_dir / "request_manifest.json", request_manifest)
    _write_json(config.output_dir / "materialization_report.json", report)
    return QualityMaterializationResult(
        output_dir=config.output_dir,
        request_manifest=config.output_dir / "request_manifest.json",
        materialization_report=config.output_dir / "materialization_report.json",
        shard_count=len(shards),
        request_count=len(all_selected),
    )


def source_request_counts(config: QualityMaterializationConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    assigned = 0
    for source in config.sources[:-1]:
        count = int(config.total_requests * source.weight)
        counts[source.name] = count
        assigned += count
    counts[config.sources[-1].name] = config.total_requests - assigned
    return counts


def distribute_stratum_indices(count: int, shard_count: int) -> list[list[int]]:
    if count < 0:
        raise QualityMaterializationConfigError("count must be non-negative")
    if shard_count <= 0:
        raise QualityMaterializationConfigError("shard_count must be positive")
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    for index in range(count):
        shards[index % shard_count].append(index)
    return shards


def _load_source_entries(
    source: QualitySourceConfig,
    *,
    config: QualityMaterializationConfig,
    tokenizer: Any,
    target_count: int,
    base_dir: Path,
) -> tuple[list[DatasetEntry], dict[str, Any]]:
    sampling = SamplingConfig(
        seed=config.seed + (0 if source.name == "sharegpt" else 17),
        num_requests=max(target_count * 4, target_count + 1000),
        prompt_len=LengthSpec(mode="from_dataset"),
        output_len=LengthSpec(mode="natural_until_eos", value=DEFAULT_DECODING_CONFIG.max_tokens),
        entry_selection="sequential",
        conversation_mode="single_turn",
    )
    dataset = _dataset_config_from_source(source, base_dir=base_dir)
    if dataset.type == "sharegpt":
        if dataset.path is None:
            raise QualityMaterializationConfigError("sharegpt source requires dataset.path")
        entries = _load_sharegpt_entries_from_source(
            Path(dataset.path),
            tokenizer,
            sampling=sampling,
            include_prompt_len=True,
            include_output_len=False,
        )
        return entries, {
            "candidate_entries": len(entries),
            "sampling_method": "all_usable_sharegpt_rows",
            "dataset_path": dataset.path,
        }
    if dataset.type == "hf":
        sample = _load_hf_entry_sample_from_source(
            dataset,
            tokenizer,
            sampling=sampling,
            include_prompt_len=True,
            include_output_len=False,
            sample_size=sampling.num_requests,
            seed=sampling.seed,
            scan_limit=dataset.max_scan_rows,
        )
        return sample.entries, {
            "candidate_entries": len(sample.entries),
            "sampling_method": "reservoir_uniform",
            "dataset_path": dataset.path,
            "dataset_split": dataset.split,
            "scanned_rows": sample.scanned_rows,
            "usable_rows": sample.usable_rows,
            "scan_limit": sample.scan_limit,
            "sample_size": sample.sample_size,
        }
    raise QualityMaterializationConfigError(f"unsupported quality source dataset type: {dataset.type}")


def _dataset_config_from_source(source: QualitySourceConfig, *, base_dir: Path) -> DatasetConfig:
    dataset = source.dataset
    dataset_type = _expect_str(dataset.get("type"), f"source {source.name}.dataset.type")
    raw_path = dataset.get("path")
    path = _expect_str(raw_path, f"source {source.name}.dataset.path") if raw_path is not None else None
    if dataset_type != "hf" and path is not None:
        path_obj = Path(path).expanduser()
        if not path_obj.is_absolute():
            path_obj = (base_dir / path_obj).resolve()
        path = str(path_obj)
    max_scan_rows = dataset.get("max_scan_rows")
    if max_scan_rows is not None:
        max_scan_rows = _expect_int(max_scan_rows, f"source {source.name}.dataset.max_scan_rows", minimum=1)
    return DatasetConfig(
        type=dataset_type,
        path=path,
        split=dataset.get("split"),
        conversation_field=dataset.get("conversation_field"),
        max_scan_rows=max_scan_rows,
    )


def _select_source_entries(
    entries: list[DatasetEntry],
    *,
    source_name: str,
    target_count: int,
    config: QualityMaterializationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_bucket: dict[str, list[DatasetEntry]] = defaultdict(list)
    for entry in entries:
        if entry.prompt_len is None:
            raise QualityMaterializationConfigError("candidate entry missing prompt_len")
        by_bucket[config.bucket_policy.bucket_for(entry.prompt_len)].append(entry)
    raw_bucket_counts = {bucket: len(items) for bucket, items in sorted(by_bucket.items())}
    allocation = _bucket_allocation(
        raw_bucket_counts,
        target_count=target_count,
        minimum=config.minimum_per_source_bucket,
    )
    rng = random.Random(f"{config.seed}:{source_name}")
    selected: list[dict[str, Any]] = []
    for bucket_name in ("short", "medium", "long"):
        bucket_entries = list(by_bucket.get(bucket_name, []))
        rng.shuffle(bucket_entries)
        need = allocation[bucket_name]
        if len(bucket_entries) < need and not config.allow_replacement:
            raise QualityMaterializationConfigError(
                f"source {source_name} bucket {bucket_name} has {len(bucket_entries)} candidates, "
                f"but needs {need}; set allow_replacement only if this is intentional"
            )
        if len(bucket_entries) >= need:
            chosen = bucket_entries[:need]
        else:
            chosen = [rng.choice(bucket_entries) for _ in range(need)]
        for entry_index, entry in enumerate(chosen):
            selected.append(_entry_to_quality_row(entry, source_name=source_name, bucket_name=bucket_name, entry_index=entry_index))
    rng.shuffle(selected)
    return selected, {
        "raw_bucket_counts": raw_bucket_counts,
        "selected_bucket_counts": allocation,
        "selected_count": len(selected),
    }


def _bucket_allocation(raw_counts: dict[str, int], *, target_count: int, minimum: int) -> dict[str, int]:
    buckets = ("short", "medium", "long")
    total_raw = sum(raw_counts.get(bucket, 0) for bucket in buckets)
    if total_raw <= 0:
        raise QualityMaterializationConfigError("source has no bucketed candidates")
    allocation = {
        bucket: int(target_count * (raw_counts.get(bucket, 0) / total_raw))
        for bucket in buckets
    }
    for bucket in buckets:
        if raw_counts.get(bucket, 0) > 0:
            allocation[bucket] = max(allocation[bucket], min(minimum, target_count))
    while sum(allocation.values()) > target_count:
        candidates = [bucket for bucket in buckets if allocation[bucket] > (minimum if raw_counts.get(bucket, 0) > 0 else 0)]
        if not candidates:
            candidates = [bucket for bucket in buckets if allocation[bucket] > 0]
        bucket = max(candidates, key=lambda item: allocation[item])
        allocation[bucket] -= 1
    while sum(allocation.values()) < target_count:
        bucket = max(buckets, key=lambda item: raw_counts.get(item, 0) - allocation[item])
        allocation[bucket] += 1
    return allocation


def _entry_to_quality_row(
    entry: DatasetEntry,
    *,
    source_name: str,
    bucket_name: str,
    entry_index: int,
) -> dict[str, Any]:
    metadata = dict(entry.metadata)
    content_hash = metadata.get("content_hash") or f"{source_name}:{entry.source_index}:{entry_index}"
    request_id = f"{source_name}:{content_hash}"
    metadata.update(
        {
            "request_id": request_id,
            "source": source_name,
            "prompt_length_bucket": bucket_name,
            "stratum": f"{source_name}:{bucket_name}",
            "source_row_index": entry.source_index,
            "content_hash": content_hash,
        }
    )
    return {
        "prompt": entry.prompt,
        "prompt_len": int(entry.prompt_len or 0),
        "metadata": metadata,
    }


def _balanced_shards(rows: list[dict[str, Any]], *, shard_count: int, seed: int) -> list[list[dict[str, Any]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["metadata"]["stratum"])].append(row)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for stratum, stratum_rows in sorted(by_stratum.items()):
        rng = random.Random(f"{seed}:{stratum}:shards")
        shuffled = list(stratum_rows)
        rng.shuffle(shuffled)
        for index, row in enumerate(shuffled):
            shards[index % shard_count].append(row)
    for shard_index, shard in enumerate(shards):
        random.Random(f"{seed}:final-shard:{shard_index}").shuffle(shard)
    return shards


def _workload_payload(*, name: str, shard_id: str, shard_size: int, tokenizer: str) -> dict[str, Any]:
    return {
        "name": name,
        "tokenizer": tokenizer,
        "dataset": {
            "type": "jsonl",
            "path": f"../shards/{shard_id}.runner.jsonl",
        },
        "sampling": {
            "seed": 42,
            "num_requests": shard_size,
            "entry_selection": "sequential",
            "prompt_len": {"mode": "from_dataset"},
            "output_len": {
                "mode": "natural_until_eos",
                "max_tokens": DEFAULT_DECODING_CONFIG.max_tokens,
            },
        },
        "request": {
            "stream": True,
            "temperature": DEFAULT_DECODING_CONFIG.temperature,
            "top_p": DEFAULT_DECODING_CONFIG.top_p,
            "ignore_eos": False,
            "extra_body": {
                "top_k": DEFAULT_DECODING_CONFIG.top_k,
                "min_p": DEFAULT_DECODING_CONFIG.min_p,
                "n": DEFAULT_DECODING_CONFIG.n,
            },
        },
        "context_policy": {
            "max_model_len": DEFAULT_DECODING_CONFIG.max_tokens,
            "tokenizer_source": "vllm_model_config",
            "over_limit": "skip_sample",
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_sources(value: Any) -> tuple[QualitySourceConfig, ...]:
    if not isinstance(value, list) or not value:
        raise QualityMaterializationConfigError("sampling.sources must be a non-empty list")
    sources: list[QualitySourceConfig] = []
    for index, item in enumerate(value):
        payload = _expect_mapping(item, f"sampling.sources[{index}]")
        _check_allowed_keys(payload, f"sampling.sources[{index}]", {"name", "weight", "dataset"})
        sources.append(
            QualitySourceConfig(
                name=_expect_str(payload.get("name"), f"sampling.sources[{index}].name"),
                weight=_expect_float(payload.get("weight"), f"sampling.sources[{index}].weight"),
                dataset=_expect_mapping(payload.get("dataset"), f"sampling.sources[{index}].dataset"),
            )
        )
    return tuple(sources)


def _validate_v1_sources(sources: tuple[QualitySourceConfig, ...]) -> None:
    if len(sources) != 2:
        raise QualityMaterializationConfigError("V1 requires exactly two sources: sharegpt and wildchat")
    by_name = {source.name: source for source in sources}
    if set(by_name) != {"sharegpt", "wildchat"}:
        raise QualityMaterializationConfigError("V1 sources must be sharegpt and wildchat")
    for name, source in by_name.items():
        if source.weight != 0.5:
            raise QualityMaterializationConfigError(f"V1 source {name} must have weight 0.5")
    if by_name["sharegpt"].dataset.get("type") != "sharegpt":
        raise QualityMaterializationConfigError("sharegpt source dataset.type must be sharegpt")
    if by_name["wildchat"].dataset.get("type") != "hf":
        raise QualityMaterializationConfigError("wildchat source dataset.type must be hf")


def _parse_bucket_policy(value: Any) -> PromptLengthBucketPolicy:
    buckets = _expect_mapping(value, "sampling.prompt_length_buckets")
    _check_allowed_keys(buckets, "sampling.prompt_length_buckets", {"short", "medium", "long"})
    short = _expect_mapping(buckets.get("short"), "sampling.prompt_length_buckets.short")
    medium = _expect_mapping(buckets.get("medium"), "sampling.prompt_length_buckets.medium")
    long = _expect_mapping(buckets.get("long"), "sampling.prompt_length_buckets.long")
    _check_allowed_keys(short, "sampling.prompt_length_buckets.short", {"lt_tokens", "max_tokens"})
    _check_allowed_keys(medium, "sampling.prompt_length_buckets.medium", {"min_tokens", "max_tokens"})
    _check_allowed_keys(long, "sampling.prompt_length_buckets.long", {"gt_tokens", "min_tokens"})

    short_boundary = short.get("lt_tokens", short.get("max_tokens"))
    long_boundary = long.get("gt_tokens", long.get("min_tokens"))
    try:
        policy = PromptLengthBucketPolicy(
            short_max_tokens_exclusive=_expect_int(
                short_boundary,
                "sampling.prompt_length_buckets.short.lt_tokens",
                minimum=1,
            ),
            medium_min_tokens=_expect_int(
                medium.get("min_tokens"),
                "sampling.prompt_length_buckets.medium.min_tokens",
                minimum=1,
            ),
            medium_max_tokens=_expect_int(
                medium.get("max_tokens"),
                "sampling.prompt_length_buckets.medium.max_tokens",
                minimum=1,
            ),
            long_min_tokens_exclusive=_expect_int(
                long_boundary,
                "sampling.prompt_length_buckets.long.gt_tokens",
                minimum=1,
            ),
        )
    except ValueError as exc:
        raise QualityMaterializationConfigError(
            "V1 prompt buckets must be short <100, medium 100-512, long >512"
        ) from exc
    if policy != DEFAULT_BUCKET_POLICY:
        raise QualityMaterializationConfigError(
            "V1 prompt buckets must be short <100, medium 100-512, long >512"
        )
    return policy


def _expect_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityMaterializationConfigError(f"{field_name} must be a mapping")
    return value


def _expect_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityMaterializationConfigError(f"{field_name} must be a non-empty string")
    return value


def _expect_int(value: Any, field_name: str, minimum: int | None = None) -> int:
    if not isinstance(value, int):
        raise QualityMaterializationConfigError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise QualityMaterializationConfigError(f"{field_name} must be >= {minimum}")
    return value


def _expect_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise QualityMaterializationConfigError(f"{field_name} must be a number")
    return float(value)


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise QualityMaterializationConfigError(f"{field_name} must be a boolean")
    return value


def _check_allowed_keys(payload: Mapping[str, Any], field_name: str, allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise QualityMaterializationConfigError(f"{field_name} has unknown keys: {sorted(unknown)}")
