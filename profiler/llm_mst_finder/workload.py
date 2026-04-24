from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .model_context import (
    ContextPolicy,
    ContextValidationReport,
    parse_context_policy,
    resolve_model_tokenizer_for_policy,
    validate_samples_against_context_window,
)
from .records import SampleRequest


class PromptTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for token in text.split():
            token_id = self._token_to_id.get(token)
            if token_id is None:
                token_id = self._next_id
                self._next_id += 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


class CharacterTokenizer:
    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for character in text:
            token_id = self._char_to_id.get(character)
            if token_id is None:
                token_id = self._next_id
                self._next_id += 1
                self._char_to_id[character] = token_id
                self._id_to_char[token_id] = character
            token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._id_to_char[token_id] for token_id in token_ids)


class HuggingFaceTokenizer:
    def __init__(self, model_name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "transformers is required for non-built-in tokenizers"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "failed to load tokenizer from local cache: "
                f"{model_name_or_path!r}. Pre-download it locally."
            ) from exc

    def encode(self, text: str) -> list[int]:
        encoded = self._tokenizer(text).input_ids
        if not isinstance(encoded, list):
            raise TypeError("tokenizer output must be a list of token ids")
        return encoded


def resolve_tokenizer(tokenizer_spec: str | None) -> PromptTokenizer:
    if tokenizer_spec is None or tokenizer_spec == "whitespace":
        return WhitespaceTokenizer()
    if tokenizer_spec == "character":
        return CharacterTokenizer()
    return HuggingFaceTokenizer(tokenizer_spec)


def _expect_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _expect_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _expect_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _check_allowed_keys(payload: dict[str, Any], field_name: str, allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{field_name} has unknown keys: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class LengthBucket:
    value: int
    weight: float

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("bucket value must be positive")
        if self.weight <= 0:
            raise ValueError("bucket weight must be positive")


@dataclass(frozen=True, slots=True)
class LengthSpec:
    mode: str
    value: int | None = None
    buckets: tuple[LengthBucket, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "bucketed", "from_dataset"}:
            raise ValueError(f"unsupported length mode {self.mode!r}")
        if self.mode == "fixed":
            if self.value is None or self.value <= 0:
                raise ValueError("fixed mode requires positive value")
            if self.buckets:
                raise ValueError("fixed mode does not allow buckets")
        if self.mode == "bucketed":
            if not self.buckets:
                raise ValueError("bucketed mode requires buckets")
            if self.value is not None:
                raise ValueError("bucketed mode does not allow value")
        if self.mode == "from_dataset":
            if self.value is not None or self.buckets:
                raise ValueError("from_dataset mode does not allow value or buckets")

    def sample(self, rng: random.Random) -> int:
        if self.mode == "fixed":
            assert self.value is not None
            return self.value
        if self.mode == "bucketed":
            chosen = rng.choices(
                self.buckets,
                weights=[bucket.weight for bucket in self.buckets],
                k=1,
            )[0]
            return chosen.value
        raise RuntimeError("cannot sample from from_dataset mode")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    seed: int
    num_requests: int
    prompt_len: LengthSpec
    output_len: LengthSpec

    def __post_init__(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("sampling.num_requests must be positive")


@dataclass(frozen=True, slots=True)
class RequestConfig:
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    type: str
    path: str | None = None
    prompt: str | None = None
    prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"synthetic-fixed", "synthetic-distribution", "jsonl", "sharegpt"}
        if self.type not in allowed:
            raise ValueError(f"unsupported dataset type {self.type!r}")
        if self.type in {"jsonl", "sharegpt"} and not self.path:
            raise ValueError(f"dataset.path is required for dataset type {self.type!r}")
        if self.type == "synthetic-fixed":
            if self.path is not None:
                raise ValueError("synthetic-fixed must not define dataset.path")
            if self.prompts:
                raise ValueError("synthetic-fixed must not define dataset.prompts")
        if self.type == "synthetic-distribution":
            if self.path is not None:
                raise ValueError("synthetic-distribution must not define dataset.path")
            if not self.prompts:
                raise ValueError("synthetic-distribution requires dataset.prompts")


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    name: str
    dataset: DatasetConfig
    tokenizer: str | None
    sampling: SamplingConfig
    request: RequestConfig
    context_policy: ContextPolicy | None
    source_path: Path


@dataclass(frozen=True, slots=True)
class PreparedWorkload:
    config: WorkloadConfig
    samples: list[SampleRequest]
    metadata: dict[str, Any]
    context_validation_report: ContextValidationReport | None = None


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    prompt: str
    source_index: int
    expected_output_len: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_length_spec(payload: Any, field_name: str) -> LengthSpec:
    spec_payload = _expect_mapping(payload, field_name)
    mode = _expect_string(spec_payload.get("mode"), f"{field_name}.mode")
    if mode == "fixed_or_bucketed":
        if "value" in spec_payload:
            mode = "fixed"
        elif "buckets" in spec_payload:
            mode = "bucketed"
        else:
            raise ValueError(f"{field_name}.mode fixed_or_bucketed requires value or buckets")
    _check_allowed_keys(
        spec_payload,
        field_name,
        {"mode", "value", "buckets", "target_mean"},
    )
    if mode == "fixed":
        return LengthSpec(
            mode="fixed",
            value=_expect_int(spec_payload.get("value"), f"{field_name}.value", positive=True),
        )
    if mode == "bucketed":
        buckets_payload = spec_payload.get("buckets")
        if not isinstance(buckets_payload, list) or not buckets_payload:
            raise ValueError(f"{field_name}.buckets must be a non-empty list")
        buckets: list[LengthBucket] = []
        for index, raw_bucket in enumerate(buckets_payload):
            bucket_payload = _expect_mapping(raw_bucket, f"{field_name}.buckets[{index}]")
            _check_allowed_keys(bucket_payload, f"{field_name}.buckets[{index}]", {"value", "weight"})
            buckets.append(
                LengthBucket(
                    value=_expect_int(
                        bucket_payload.get("value"),
                        f"{field_name}.buckets[{index}].value",
                        positive=True,
                    ),
                    weight=float(bucket_payload.get("weight", 1.0)),
                )
            )
        return LengthSpec(mode="bucketed", buckets=tuple(buckets))
    if mode == "from_dataset":
        return LengthSpec(mode="from_dataset")
    raise ValueError(f"unsupported {field_name}.mode: {mode!r}")


def _parse_dataset(payload: Any, base_dir: Path) -> DatasetConfig:
    dataset_payload = _expect_mapping(payload, "dataset")
    _check_allowed_keys(dataset_payload, "dataset", {"type", "path", "prompt", "prompts"})
    dataset_type = _expect_string(dataset_payload.get("type"), "dataset.type")
    raw_path = dataset_payload.get("path")
    path_value: str | None = None
    if raw_path is not None:
        path_str = _expect_string(raw_path, "dataset.path")
        path_value = str((base_dir / path_str).resolve())
    prompt = dataset_payload.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or not prompt):
        raise ValueError("dataset.prompt must be a non-empty string when provided")
    prompts_payload = dataset_payload.get("prompts", [])
    prompts: tuple[str, ...] = ()
    if prompts_payload:
        if not isinstance(prompts_payload, list):
            raise ValueError("dataset.prompts must be a list when provided")
        prompts = tuple(_expect_string(item, "dataset.prompts[]") for item in prompts_payload)
    return DatasetConfig(
        type=dataset_type,
        path=path_value,
        prompt=prompt,
        prompts=prompts,
    )


def _parse_request(payload: Any) -> RequestConfig:
    if payload is None:
        return RequestConfig()
    request_payload = _expect_mapping(payload, "request")
    _check_allowed_keys(
        request_payload,
        "request",
        {"temperature", "top_p", "ignore_eos", "stream", "extra_body"},
    )
    extra_body = request_payload.get("extra_body")
    if extra_body is None:
        merged: dict[str, Any] = {}
    else:
        merged = dict(_expect_mapping(extra_body, "request.extra_body"))
    for key in ("temperature", "top_p", "ignore_eos", "stream"):
        if key in request_payload:
            merged[key] = request_payload[key]
    return RequestConfig(extra_body=merged)


def load_workload_config(path: str | Path) -> WorkloadConfig:
    workload_path = Path(path)
    if not workload_path.exists():
        raise FileNotFoundError(f"workload config not found: {workload_path}")
    with workload_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    root = _expect_mapping(payload, "workload")
    _check_allowed_keys(
        root,
        "workload",
        {"name", "dataset", "tokenizer", "sampling", "request", "context_policy"},
    )
    name = _expect_string(root.get("name"), "name")
    tokenizer = root.get("tokenizer")
    if tokenizer is not None and not isinstance(tokenizer, str):
        raise ValueError("tokenizer must be a string when provided")
    sampling_payload = _expect_mapping(root.get("sampling"), "sampling")
    _check_allowed_keys(sampling_payload, "sampling", {"seed", "num_requests", "prompt_len", "output_len"})
    sampling = SamplingConfig(
        seed=_expect_int(sampling_payload.get("seed"), "sampling.seed"),
        num_requests=_expect_int(sampling_payload.get("num_requests"), "sampling.num_requests", positive=True),
        prompt_len=_parse_length_spec(sampling_payload.get("prompt_len"), "sampling.prompt_len"),
        output_len=_parse_length_spec(sampling_payload.get("output_len"), "sampling.output_len"),
    )
    config = WorkloadConfig(
        name=name,
        dataset=_parse_dataset(root.get("dataset"), workload_path.parent.resolve()),
        tokenizer=tokenizer,
        sampling=sampling,
        request=_parse_request(root.get("request")),
        context_policy=parse_context_policy(root.get("context_policy")),
        source_path=workload_path.resolve(),
    )
    if config.dataset.type.startswith("synthetic") and config.sampling.prompt_len.mode == "from_dataset":
        raise ValueError("synthetic datasets do not support sampling.prompt_len.mode=from_dataset")
    if config.dataset.type.startswith("synthetic") and config.sampling.output_len.mode == "from_dataset":
        raise ValueError("synthetic datasets do not support sampling.output_len.mode=from_dataset")
    return config


def _load_jsonl_entries(path: Path) -> list[DatasetEntry]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl dataset not found: {path}")
    entries: list[DatasetEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl row {index} must be a mapping")
            prompt = _expect_string(row.get("prompt"), f"jsonl row {index}.prompt")
            expected_output_len = row.get("expected_output_len")
            if expected_output_len is not None:
                expected_output_len = _expect_int(
                    expected_output_len,
                    f"jsonl row {index}.expected_output_len",
                    positive=True,
                )
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"jsonl row {index}.metadata must be a mapping")
            entries.append(
                DatasetEntry(
                    prompt=prompt,
                    source_index=index,
                    expected_output_len=expected_output_len,
                    metadata=metadata,
                )
            )
    if not entries:
        raise ValueError(f"jsonl dataset is empty: {path}")
    return entries


def _find_sharegpt_text(
    conversations: list[Any],
    accepted_roles: set[str],
    field_name: str,
) -> str | None:
    for index, turn in enumerate(conversations):
        turn_payload = _expect_mapping(turn, f"{field_name}[{index}]")
        role = turn_payload.get("from", turn_payload.get("role"))
        text = turn_payload.get("value", turn_payload.get("content"))
        if isinstance(role, str) and role.lower() in accepted_roles and isinstance(text, str) and text:
            return text
    return None


def _load_sharegpt_entries(path: Path, tokenizer: PromptTokenizer) -> list[DatasetEntry]:
    if not path.exists():
        raise FileNotFoundError(f"sharegpt dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("sharegpt dataset must be a JSON list")
    entries: list[DatasetEntry] = []
    skipped_missing_prompt = 0
    skipped_missing_assistant = 0
    for index, row in enumerate(payload):
        row_payload = _expect_mapping(row, f"sharegpt row {index}")
        conversations = row_payload.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError(f"sharegpt row {index}.conversations must be a list")
        prompt = _find_sharegpt_text(conversations, {"human", "user"}, f"sharegpt row {index}.conversations")
        if prompt is None:
            skipped_missing_prompt += 1
            continue
        assistant = _find_sharegpt_text(
            conversations,
            {"gpt", "assistant"},
            f"sharegpt row {index}.conversations",
        )
        if assistant is None:
            skipped_missing_assistant += 1
            continue
        expected_output_len = len(tokenizer.encode(assistant)) if assistant is not None else None
        entries.append(
            DatasetEntry(
                prompt=prompt,
                source_index=index,
                expected_output_len=expected_output_len,
                metadata={"row_id": row_payload.get("id")},
            )
        )
    if not entries:
        raise ValueError(
            "sharegpt dataset has no usable rows with both prompt and assistant reply: "
            f"{path} (missing_prompt={skipped_missing_prompt}, "
            f"missing_assistant={skipped_missing_assistant})"
        )
    return entries


def _render_prompt(base_text: str, target_len: int, sample_index: int) -> str:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    base_tokens = base_text.split()
    if not base_tokens:
        base_tokens = ["synthetic"]
    expanded: list[str] = []
    token_index = 0
    while len(expanded) < target_len:
        token = base_tokens[token_index % len(base_tokens)]
        expanded.append(f"{token}_{sample_index}_{token_index}")
        token_index += 1
    return " ".join(expanded[:target_len])


def _sample_dataset_entries(
    config: WorkloadConfig,
    tokenizer: PromptTokenizer,
) -> list[DatasetEntry]:
    dataset = config.dataset
    if dataset.type == "synthetic-fixed":
        base_prompt = dataset.prompt or "synthetic fixed prompt"
        return [DatasetEntry(prompt=base_prompt, source_index=0, metadata={})]
    if dataset.type == "synthetic-distribution":
        return [
            DatasetEntry(prompt=prompt, source_index=index, metadata={})
            for index, prompt in enumerate(dataset.prompts)
        ]
    if dataset.type == "jsonl":
        assert dataset.path is not None
        return _load_jsonl_entries(Path(dataset.path))
    if dataset.type == "sharegpt":
        assert dataset.path is not None
        return _load_sharegpt_entries(Path(dataset.path), tokenizer)
    raise RuntimeError(f"unsupported dataset type {dataset.type!r}")


def _resolve_output_len(spec: LengthSpec, entry: DatasetEntry, rng: random.Random) -> int:
    if spec.mode == "from_dataset":
        if entry.expected_output_len is None:
            raise ValueError(
                "sampling.output_len.mode=from_dataset requires dataset entries with expected_output_len"
            )
        return entry.expected_output_len
    return spec.sample(rng)


def generate_sample_requests(
    config: WorkloadConfig,
    *,
    tokenizer: PromptTokenizer | None = None,
) -> list[SampleRequest]:
    resolved_tokenizer = tokenizer if tokenizer is not None else resolve_tokenizer(config.tokenizer)
    dataset_entries = _sample_dataset_entries(config, resolved_tokenizer)
    rng = random.Random(config.sampling.seed)
    samples: list[SampleRequest] = []

    for request_index in range(config.sampling.num_requests):
        if config.dataset.type == "synthetic-fixed":
            entry = dataset_entries[0]
        else:
            entry = rng.choice(dataset_entries)

        prompt = entry.prompt
        if config.sampling.prompt_len.mode != "from_dataset":
            prompt = _render_prompt(
                base_text=prompt,
                target_len=config.sampling.prompt_len.sample(rng),
                sample_index=request_index,
            )

        prompt_len = len(resolved_tokenizer.encode(prompt))
        expected_output_len = _resolve_output_len(config.sampling.output_len, entry, rng)
        samples.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=expected_output_len,
                extra_body=dict(config.request.extra_body) or None,
                metadata={
                    "workload_name": config.name,
                    "dataset_type": config.dataset.type,
                    "request_index": request_index,
                    "source_index": entry.source_index,
                    "seed": config.sampling.seed,
                    "sampling_prompt_len_mode": config.sampling.prompt_len.mode,
                    "sampling_output_len_mode": config.sampling.output_len.mode,
                    **entry.metadata,
                },
            )
        )
    return samples


def load_workload_samples(
    path: str | Path,
    *,
    tokenizer: PromptTokenizer | None = None,
) -> list[SampleRequest]:
    config = load_workload_config(path)
    return generate_sample_requests(config, tokenizer=tokenizer)


def _build_workload_metadata(
    config: WorkloadConfig,
    *,
    sample_count: int,
    context_validation_report: ContextValidationReport | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "workload": {
            "name": config.name,
            "source_path": str(config.source_path),
            "dataset_type": config.dataset.type,
            "num_requests": sample_count,
        }
    }
    if config.context_policy is not None:
        report = context_validation_report
        metadata["workload"]["context_policy"] = {
            "max_model_len": config.context_policy.max_model_len,
            "tokenizer_source": config.context_policy.tokenizer_source,
            "tokenizer": config.context_policy.tokenizer,
            "over_limit": config.context_policy.over_limit,
            "truncation_side": config.context_policy.truncation_side,
            "total_samples": report.total_samples if report is not None else sample_count,
            "kept_samples": report.kept_samples if report is not None else sample_count,
            "skipped_samples": report.skipped_samples if report is not None else 0,
            "truncated_samples": report.truncated_samples if report is not None else 0,
            "skipped_source_indexes": list(report.skipped_source_indexes) if report is not None else [],
            "truncated_source_indexes": list(report.truncated_source_indexes) if report is not None else [],
        }
    return metadata


def prepare_workload_for_trial(
    path: str | Path,
    *,
    model_name: str,
) -> PreparedWorkload:
    config = load_workload_config(path)
    workload_tokenizer = resolve_tokenizer(config.tokenizer)
    samples = generate_sample_requests(config, tokenizer=workload_tokenizer)
    requires_context_validation = config.dataset.type in {"jsonl", "sharegpt"}

    if config.context_policy is None:
        if requires_context_validation:
            raise ValueError(
                "real dataset workloads require context_policy for pre-trial context validation"
            )
        return PreparedWorkload(
            config=config,
            samples=samples,
            metadata=_build_workload_metadata(
                config,
                sample_count=len(samples),
                context_validation_report=None,
            ),
            context_validation_report=None,
        )

    model_tokenizer = resolve_model_tokenizer_for_policy(
        config.context_policy,
        workload_tokenizer=workload_tokenizer,
        model_name=model_name,
    )
    validation_result = validate_samples_against_context_window(
        samples,
        tokenizer=model_tokenizer,
        policy=config.context_policy,
    )
    if not validation_result.samples:
        raise ValueError(
            "context validation removed all workload samples; no requests remain for the trial"
        )
    return PreparedWorkload(
        config=config,
        samples=validation_result.samples,
        metadata=_build_workload_metadata(
            config,
            sample_count=len(validation_result.samples),
            context_validation_report=validation_result.report,
        ),
        context_validation_report=validation_result.report,
    )
