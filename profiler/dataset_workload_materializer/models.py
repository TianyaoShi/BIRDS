from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_mst_finder.workload import PromptTokenizer


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


@dataclass(frozen=True, slots=True)
class FilteringConfig:
    min_prompt_tokens: int
    max_prompt_tokens: int
    min_target_tokens: int
    max_target_tokens: int
    language_filter: dict[str, set[str]]
    dedup_content_hash: bool


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    seed: int
    burst_size: int
    policy: str | None
    samples_per_task: int | None
    repeat_policy: str | None
    target_samples: int | None


@dataclass(frozen=True, slots=True)
class MaterializationContext:
    base_dir: Path
    dataset_name: str
    dataset_kind: str
    raw_path: Path | None
    split: str
    tokenizer_name: str
    tokenizer: PromptTokenizer
    filtering: FilteringConfig
    sampling: SamplingConfig
    counters: Counters


@dataclass(frozen=True, slots=True)
class DatasetLoadResult:
    samples: list[MaterializedSample]
    task: str
    prompt_template: str
    profile: str | None = None
    selected_tasks: list[str] | None = None
