from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from local_orchestrator.models import (
    HardwareConfig,
    LaunchConfig,
    ProbeConfig,
    RunConfig,
    SlurmConfig,
)


PromptLengthBucketName = Literal["short", "medium", "long"]
JudgeLabel = Literal["A_BETTER", "B_BETTER", "TIE", "INVALID"]


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_positive_float(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}")


def _require_probability(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class PromptLengthBucketPolicy:
    short_max_tokens_exclusive: int = 100
    medium_min_tokens: int = 100
    medium_max_tokens: int = 512
    long_min_tokens_exclusive: int = 512

    def __post_init__(self) -> None:
        _require_positive_int("short_max_tokens_exclusive", self.short_max_tokens_exclusive)
        _require_positive_int("medium_min_tokens", self.medium_min_tokens)
        _require_positive_int("medium_max_tokens", self.medium_max_tokens)
        _require_positive_int("long_min_tokens_exclusive", self.long_min_tokens_exclusive)
        if self.medium_min_tokens != self.short_max_tokens_exclusive:
            raise ValueError("medium_min_tokens must equal short_max_tokens_exclusive")
        if self.long_min_tokens_exclusive != self.medium_max_tokens:
            raise ValueError("long_min_tokens_exclusive must equal medium_max_tokens")
        if self.medium_min_tokens > self.medium_max_tokens:
            raise ValueError("medium_min_tokens must be <= medium_max_tokens")

    def bucket_for(self, prompt_tokens: int) -> PromptLengthBucketName:
        _require_non_negative_int("prompt_tokens", prompt_tokens)
        if prompt_tokens < self.short_max_tokens_exclusive:
            return "short"
        if prompt_tokens <= self.medium_max_tokens:
            return "medium"
        return "long"

    def to_dict(self) -> dict[str, Any]:
        return {
            "short": {"lt_tokens": self.short_max_tokens_exclusive},
            "medium": {
                "min_tokens": self.medium_min_tokens,
                "max_tokens": self.medium_max_tokens,
            },
            "long": {"gt_tokens": self.long_min_tokens_exclusive},
        }


DEFAULT_BUCKET_POLICY = PromptLengthBucketPolicy()


@dataclass(frozen=True, slots=True)
class QualityDecodingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    n: int = 1
    max_tokens: int = 32768
    max_tokens_policy: str = "model_context_minus_prompt_buffer"
    prompt_token_buffer: int = 128
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_probability("generation.decoding.temperature", self.temperature)
        _require_probability("generation.decoding.top_p", self.top_p)
        _require_positive_int("generation.decoding.top_k", self.top_k)
        _require_probability("generation.decoding.min_p", self.min_p)
        if self.n != 1:
            raise ValueError("generation.decoding.n must be 1 for V1 quality profiling")
        _require_positive_int("generation.decoding.max_tokens", self.max_tokens)
        if self.max_tokens != 32768:
            raise ValueError("generation.decoding.max_tokens must be 32768 for V1")
        if self.max_tokens_policy != "model_context_minus_prompt_buffer":
            raise ValueError(
                "generation.decoding.max_tokens_policy must be "
                "'model_context_minus_prompt_buffer'"
            )
        _require_non_negative_int("generation.decoding.prompt_token_buffer", self.prompt_token_buffer)
        if not isinstance(self.extra_body, dict):
            raise ValueError("generation.decoding.extra_body must be a mapping")

    def to_request_extra_body(self) -> dict[str, Any]:
        extra_body = dict(self.extra_body)
        extra_body.update(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "n": self.n,
            }
        )
        return extra_body

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "n": self.n,
            "max_tokens": self.max_tokens,
            "max_tokens_policy": self.max_tokens_policy,
            "prompt_token_buffer": self.prompt_token_buffer,
            "extra_body": dict(self.extra_body),
        }


DEFAULT_DECODING_CONFIG = QualityDecodingConfig()


@dataclass(frozen=True, slots=True)
class QualityGenerationConfig:
    request_timeout_s: float = 21600.0
    max_concurrency: int | None = None
    concurrency_source: str = "mst_fraction"
    concurrency_mst_fraction: float = 0.40
    preserve_request_order: bool = True
    response_text_max_chars: int = 65536
    include_prompt_text: bool = True
    decoding: QualityDecodingConfig = field(default_factory=QualityDecodingConfig)

    def __post_init__(self) -> None:
        _require_positive_float("generation.request_timeout_s", self.request_timeout_s)
        if self.max_concurrency is not None:
            _require_positive_int("generation.max_concurrency", self.max_concurrency)
        if self.concurrency_source not in {"mst_fraction", "explicit"}:
            raise ValueError("generation.concurrency_source must be mst_fraction or explicit")
        _require_probability("generation.concurrency_mst_fraction", self.concurrency_mst_fraction)
        if self.concurrency_source == "mst_fraction" and self.concurrency_mst_fraction != 0.40:
            raise ValueError("generation.concurrency_mst_fraction must be 0.40 for V1")
        if self.concurrency_source == "explicit" and self.max_concurrency is None:
            raise ValueError("generation.max_concurrency is required when concurrency_source=explicit")
        if not isinstance(self.preserve_request_order, bool):
            raise ValueError("generation.preserve_request_order must be a boolean")
        if not self.preserve_request_order:
            raise ValueError("generation.preserve_request_order must be true for V1")
        _require_positive_int("generation.response_text_max_chars", self.response_text_max_chars)
        if not isinstance(self.include_prompt_text, bool):
            raise ValueError("generation.include_prompt_text must be a boolean")
        if not self.include_prompt_text:
            raise ValueError("generation.include_prompt_text must be true for V1")


@dataclass(frozen=True, slots=True)
class QualityExperimentTemplate:
    source_index: int
    experiment_id: str | None
    models: tuple[str, ...]
    workloads: tuple[Path, ...]
    endpoint: str
    launch: LaunchConfig
    generation: QualityGenerationConfig
    hardware: HardwareConfig
    probe: ProbeConfig

    def __post_init__(self) -> None:
        _require_non_negative_int("source_index", self.source_index)
        if not self.models:
            raise ValueError("models must be non-empty")
        for model in self.models:
            _require_non_empty("model", model)
        if not self.workloads:
            raise ValueError("workloads must be non-empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")


@dataclass(frozen=True, slots=True)
class QualityRunManifest:
    manifest_path: Path
    run: RunConfig
    slurm: SlurmConfig
    hardware: HardwareConfig
    probe: ProbeConfig
    launch: LaunchConfig
    generation: QualityGenerationConfig
    experiments: tuple[QualityExperimentTemplate, ...]


@dataclass(frozen=True, slots=True)
class QualityExperimentJob:
    job_id: str
    source_index: int
    model: str
    workload: Path
    workloads: tuple[Path, ...]
    endpoint: str
    launch: LaunchConfig
    generation: QualityGenerationConfig
    hardware: HardwareConfig
    probe: ProbeConfig
    result_dir: Path
    model_slug: str
    shard_id: str
    server_signature_key: str

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("workloads must be non-empty")
