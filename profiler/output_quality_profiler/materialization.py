from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import DEFAULT_BUCKET_POLICY, PromptLengthBucketName, PromptLengthBucketPolicy


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
    seed: int
    total_requests: int
    shards: int
    sources: tuple[QualitySourceConfig, ...]
    bucket_policy: PromptLengthBucketPolicy
    minimum_per_source_bucket: int
    allow_replacement: bool


def assign_prompt_length_bucket(
    prompt_tokens: int,
    policy: PromptLengthBucketPolicy = DEFAULT_BUCKET_POLICY,
) -> PromptLengthBucketName:
    return policy.bucket_for(prompt_tokens)


def load_materialization_config(path: str | Path) -> QualityMaterializationConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _expect_mapping(payload, "materialization config")
    _check_allowed_keys(root, "materialization config", {"sampling"})
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
        seed=_expect_int(sampling.get("seed"), "sampling.seed"),
        total_requests=total_requests,
        shards=shards,
        sources=sources,
        bucket_policy=bucket_policy,
        minimum_per_source_bucket=minimum,
        allow_replacement=allow_replacement,
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
