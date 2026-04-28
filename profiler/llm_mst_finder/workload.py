from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .model_context import (
    ContextPolicy,
    ContextValidationReport,
    ModelContextInfo,
    parse_context_policy,
    resolve_model_context_info,
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


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name)


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
    subset: str | None = None
    split: str | None = None
    conversation_field: str | None = None
    prompt_field: str | None = None
    completion_field: str | None = None
    prompt: str | None = None
    prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"synthetic-fixed", "synthetic-distribution", "jsonl", "sharegpt", "hf"}
        if self.type not in allowed:
            raise ValueError(f"unsupported dataset type {self.type!r}")
        if self.type in {"jsonl", "sharegpt", "hf"} and not self.path:
            raise ValueError(f"dataset.path is required for dataset type {self.type!r}")
        if self.type == "hf" and not self.split:
            raise ValueError("dataset.split is required for dataset type 'hf'")
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
    prompt_len: int | None = None
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
    _check_allowed_keys(
        dataset_payload,
        "dataset",
        {
            "type",
            "path",
            "subset",
            "split",
            "conversation_field",
            "prompt_field",
            "completion_field",
            "prompt",
            "prompts",
        },
    )
    dataset_type = _expect_string(dataset_payload.get("type"), "dataset.type")
    raw_path = dataset_payload.get("path")
    path_value: str | None = None
    if raw_path is not None:
        path_str = _expect_string(raw_path, "dataset.path")
        path_value = path_str if dataset_type == "hf" else str((base_dir / path_str).resolve())
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
        subset=_optional_string(dataset_payload.get("subset"), "dataset.subset"),
        split=_optional_string(dataset_payload.get("split"), "dataset.split"),
        conversation_field=_optional_string(
            dataset_payload.get("conversation_field"),
            "dataset.conversation_field",
        ),
        prompt_field=_optional_string(dataset_payload.get("prompt_field"), "dataset.prompt_field"),
        completion_field=_optional_string(
            dataset_payload.get("completion_field"),
            "dataset.completion_field",
        ),
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


def _normalized_tokenizer_spec(tokenizer_spec: str | None) -> str:
    return "whitespace" if tokenizer_spec is None else tokenizer_spec


def _tokenizer_cache_key(
    tokenizer_spec: str | None,
    *,
    tokenizer: PromptTokenizer | None = None,
) -> str:
    if tokenizer_spec is not None:
        return f"tokenizer:{_normalized_tokenizer_spec(tokenizer_spec)}"
    if tokenizer is None:
        return "tokenizer:whitespace"
    return f"tokenizer:{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"


def _context_tokenizer_cache_key(
    config: WorkloadConfig,
    *,
    model_name: str,
    workload_tokenizer_key: str,
) -> str:
    assert config.context_policy is not None
    policy = config.context_policy
    if policy.tokenizer_source == "workload_tokenizer":
        return workload_tokenizer_key
    if policy.tokenizer_source == "explicit":
        assert policy.tokenizer is not None
        return _tokenizer_cache_key(policy.tokenizer)
    if policy.tokenizer_source == "vllm_model_config":
        return _tokenizer_cache_key(policy.tokenizer or model_name)
    raise ValueError(f"unsupported context_policy.tokenizer_source {policy.tokenizer_source!r}")


def _manifest_cache_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".cache" / "llm_mst_finder" / "workload_manifests"


def _manifest_cache_path(
    config: WorkloadConfig,
    *,
    tokenizer_key: str,
) -> Path:
    key_payload = {
        "cache_version": 1,
        "dataset_path": config.dataset.path,
        "dataset_subset": config.dataset.subset,
        "dataset_split": config.dataset.split,
        "dataset_conversation_field": config.dataset.conversation_field,
        "dataset_prompt_field": config.dataset.prompt_field,
        "dataset_completion_field": config.dataset.completion_field,
        "dataset_type": config.dataset.type,
        "num_requests": config.sampling.num_requests,
        "output_len_mode": config.sampling.output_len.mode,
        "prompt_len_mode": config.sampling.prompt_len.mode,
        "tokenizer_key": tokenizer_key,
    }
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    filename = f"{config.dataset.type}_{digest}.json"
    return _manifest_cache_root() / filename


def _dataset_entry_to_payload(entry: DatasetEntry) -> dict[str, Any]:
    return {
        "expected_output_len": entry.expected_output_len,
        "metadata": entry.metadata,
        "prompt": entry.prompt,
        "prompt_len": entry.prompt_len,
        "source_index": entry.source_index,
    }


def _dataset_entry_from_payload(payload: Any, *, row_index: int) -> DatasetEntry:
    row = _expect_mapping(payload, f"manifest.entries[{row_index}]")
    _check_allowed_keys(
        row,
        f"manifest.entries[{row_index}]",
        {"expected_output_len", "metadata", "prompt", "prompt_len", "source_index"},
    )
    prompt_len = row.get("prompt_len")
    if prompt_len is not None:
        prompt_len = _expect_int(prompt_len, f"manifest.entries[{row_index}].prompt_len", positive=True)
    expected_output_len = row.get("expected_output_len")
    if expected_output_len is not None:
        expected_output_len = _expect_int(
            expected_output_len,
            f"manifest.entries[{row_index}].expected_output_len",
            positive=True,
        )
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"manifest.entries[{row_index}].metadata must be a mapping")
    return DatasetEntry(
        prompt=_expect_string(row.get("prompt"), f"manifest.entries[{row_index}].prompt"),
        source_index=_expect_int(row.get("source_index"), f"manifest.entries[{row_index}].source_index"),
        prompt_len=prompt_len,
        expected_output_len=expected_output_len,
        metadata=metadata,
    )


def _load_entries_from_manifest(path: Path) -> list[DatasetEntry]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = _expect_mapping(payload, "manifest")
    entries_payload = manifest.get("entries")
    if not isinstance(entries_payload, list) or not entries_payload:
        raise ValueError(f"manifest.entries must be a non-empty list: {path}")
    return [
        _dataset_entry_from_payload(item, row_index=index)
        for index, item in enumerate(entries_payload)
    ]


def _write_entries_to_manifest(
    path: Path,
    *,
    config: WorkloadConfig,
    tokenizer_key: str,
    entries: list[DatasetEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "cache_version": 1,
        "dataset_path": config.dataset.path,
        "dataset_subset": config.dataset.subset,
        "dataset_split": config.dataset.split,
        "dataset_conversation_field": config.dataset.conversation_field,
        "dataset_prompt_field": config.dataset.prompt_field,
        "dataset_completion_field": config.dataset.completion_field,
        "dataset_type": config.dataset.type,
        "num_requests": config.sampling.num_requests,
        "output_len_mode": config.sampling.output_len.mode,
        "prompt_len_mode": config.sampling.prompt_len.mode,
        "tokenizer_key": tokenizer_key,
        "entries": [_dataset_entry_to_payload(entry) for entry in entries],
    }
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    tmp_path.replace(path)


def _load_jsonl_entries_from_source(
    path: Path,
    *,
    tokenizer: PromptTokenizer | None = None,
    include_prompt_len: bool,
) -> list[DatasetEntry]:
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
            prompt_len = None
            if include_prompt_len:
                if tokenizer is None:
                    raise ValueError("tokenizer is required when include_prompt_len=True")
                prompt_len = len(tokenizer.encode(prompt))
            entries.append(
                DatasetEntry(
                    prompt=prompt,
                    source_index=index,
                    prompt_len=prompt_len,
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


def _load_sharegpt_entries_from_source(
    path: Path,
    tokenizer: PromptTokenizer,
    *,
    include_prompt_len: bool,
    include_output_len: bool,
) -> list[DatasetEntry]:
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
        prompt_len = len(tokenizer.encode(prompt)) if include_prompt_len else None
        expected_output_len = len(tokenizer.encode(assistant)) if include_output_len else None
        entries.append(
            DatasetEntry(
                prompt=prompt,
                source_index=index,
                prompt_len=prompt_len,
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


def _load_hf_entries_from_source(
    dataset: DatasetConfig,
    tokenizer: PromptTokenizer,
    *,
    include_prompt_len: bool,
    include_output_len: bool,
    max_entries: int,
) -> list[DatasetEntry]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("datasets is required for dataset.type=hf workloads") from exc
    assert dataset.path is not None
    assert dataset.split is not None
    rows = load_dataset(
        dataset.path,
        name=dataset.subset,
        split=dataset.split,
        streaming=True,
    )
    entries: list[DatasetEntry] = []
    skipped_missing_prompt = 0
    skipped_missing_completion = 0
    for index, row in enumerate(rows):
        if len(entries) >= max_entries:
            break
        if not isinstance(row, dict):
            raise ValueError(f"hf row {index} must be a mapping")
        prompt, completion = _extract_hf_prompt_completion(row, dataset, row_index=index)
        if prompt is None:
            skipped_missing_prompt += 1
            continue
        if completion is None or (include_output_len and not completion):
            skipped_missing_completion += 1
            continue
        prompt_len = len(tokenizer.encode(prompt)) if include_prompt_len else None
        expected_output_len = len(tokenizer.encode(completion)) if include_output_len else None
        entries.append(
            DatasetEntry(
                prompt=prompt,
                source_index=index,
                prompt_len=prompt_len,
                expected_output_len=expected_output_len,
                metadata={
                    "hf_dataset_path": dataset.path,
                    "hf_dataset_subset": dataset.subset,
                    "hf_dataset_split": dataset.split,
                },
            )
        )
    if not entries:
        raise ValueError(
            "hf dataset has no usable rows with prompt and completion: "
            f"{dataset.path} split={dataset.split} "
            f"(missing_prompt={skipped_missing_prompt}, "
            f"missing_completion={skipped_missing_completion})"
        )
    return entries


def _extract_hf_prompt_completion(
    row: dict[str, Any],
    dataset: DatasetConfig,
    *,
    row_index: int,
) -> tuple[str | None, str | None]:
    if dataset.prompt_field is not None:
        prompt = row.get(dataset.prompt_field)
        if not isinstance(prompt, str) or not prompt:
            return None, None
        if dataset.completion_field is None:
            return prompt, ""
        completion = row.get(dataset.completion_field)
        return prompt, completion if isinstance(completion, str) and completion else None

    conversation_field = dataset.conversation_field
    if conversation_field is None:
        for candidate in ("conversations", "conversation", "messages"):
            if candidate in row:
                conversation_field = candidate
                break
    if conversation_field is None:
        raise ValueError(
            f"hf row {row_index} does not contain a conversation field; "
            "set dataset.conversation_field or dataset.prompt_field"
        )
    conversations = row.get(conversation_field)
    if not isinstance(conversations, list):
        raise ValueError(f"hf row {row_index}.{conversation_field} must be a list")
    prompt = _find_hf_turn_text(conversations, {"human", "user"}, f"hf row {row_index}.{conversation_field}")
    if prompt is None:
        return None, None
    assistant = _find_hf_turn_text(
        conversations,
        {"gpt", "assistant"},
        f"hf row {row_index}.{conversation_field}",
    )
    return prompt, assistant


def _find_hf_turn_text(
    conversations: list[Any],
    accepted_roles: set[str],
    field_name: str,
) -> str | None:
    for index, turn in enumerate(conversations):
        turn_payload = _expect_mapping(turn, f"{field_name}[{index}]")
        role = turn_payload.get("from", turn_payload.get("role"))
        text = turn_payload.get("value", turn_payload.get("content", turn_payload.get("text")))
        if isinstance(role, str) and role.lower() in accepted_roles and isinstance(text, str) and text:
            return text
    return None


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
    *,
    tokenizer_key: str,
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
        manifest_path = _manifest_cache_path(config, tokenizer_key=tokenizer_key)
        if manifest_path.exists():
            return _load_entries_from_manifest(manifest_path)
        entries = _load_jsonl_entries_from_source(
            Path(dataset.path),
            tokenizer=tokenizer,
            include_prompt_len=config.sampling.prompt_len.mode == "from_dataset",
        )
        _write_entries_to_manifest(manifest_path, config=config, tokenizer_key=tokenizer_key, entries=entries)
        return entries
    if dataset.type == "sharegpt":
        assert dataset.path is not None
        manifest_path = _manifest_cache_path(config, tokenizer_key=tokenizer_key)
        if manifest_path.exists():
            return _load_entries_from_manifest(manifest_path)
        entries = _load_sharegpt_entries_from_source(
            Path(dataset.path),
            tokenizer,
            include_prompt_len=config.sampling.prompt_len.mode == "from_dataset",
            include_output_len=config.sampling.output_len.mode == "from_dataset",
        )
        _write_entries_to_manifest(manifest_path, config=config, tokenizer_key=tokenizer_key, entries=entries)
        return entries
    if dataset.type == "hf":
        manifest_path = _manifest_cache_path(config, tokenizer_key=tokenizer_key)
        if manifest_path.exists():
            return _load_entries_from_manifest(manifest_path)
        entries = _load_hf_entries_from_source(
            dataset,
            tokenizer,
            include_prompt_len=config.sampling.prompt_len.mode == "from_dataset",
            include_output_len=config.sampling.output_len.mode == "from_dataset",
            max_entries=min(max(config.sampling.num_requests * 4, config.sampling.num_requests), 4096),
        )
        _write_entries_to_manifest(manifest_path, config=config, tokenizer_key=tokenizer_key, entries=entries)
        return entries
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
    tokenizer_key: str | None = None,
) -> list[SampleRequest]:
    resolved_tokenizer = tokenizer if tokenizer is not None else resolve_tokenizer(config.tokenizer)
    workload_tokenizer_key = tokenizer_key or _tokenizer_cache_key(config.tokenizer, tokenizer=resolved_tokenizer)
    dataset_entries = _sample_dataset_entries(
        config,
        resolved_tokenizer,
        tokenizer_key=workload_tokenizer_key,
    )
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

        if config.sampling.prompt_len.mode == "from_dataset":
            if entry.prompt_len is None:
                raise ValueError(
                    "sampling.prompt_len.mode=from_dataset requires dataset entries with prompt_len"
                )
            prompt_len = entry.prompt_len
        else:
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
                    "prompt_tokenizer_key": workload_tokenizer_key,
                    **entry.metadata,
                },
            )
        )
    return samples


def load_workload_samples_for_sampling_only(
    path: str | Path,
    *,
    tokenizer: PromptTokenizer | None = None,
) -> list[SampleRequest]:
    """Sampling-only helper.

    This function intentionally does not apply model-context validation.
    Use prepare_workload_for_trial() for trial execution paths.
    """
    config = load_workload_config(path)
    return generate_sample_requests(config, tokenizer=tokenizer)


def load_workload_samples(
    path: str | Path,
    *,
    tokenizer: PromptTokenizer | None = None,
) -> list[SampleRequest]:
    """Backward-compatible alias for sampling-only behavior."""
    return load_workload_samples_for_sampling_only(path, tokenizer=tokenizer)


def _build_workload_metadata(
    config: WorkloadConfig,
    *,
    sample_count: int,
    context_validation_report: ContextValidationReport | None,
    effective_context_policy: ContextPolicy | None = None,
    model_context_info: ModelContextInfo | None = None,
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
        policy = effective_context_policy or config.context_policy
        report = context_validation_report
        metadata["workload"]["context_policy"] = {
            "max_model_len": policy.max_model_len,
            "tokenizer_source": policy.tokenizer_source,
            "tokenizer": policy.tokenizer,
            "over_limit": policy.over_limit,
            "truncation_side": policy.truncation_side,
            "unsafe_allow_workload_tokenizer_for_real_datasets": (
                policy.unsafe_allow_workload_tokenizer_for_real_datasets
            ),
            "total_samples": report.total_samples if report is not None else sample_count,
            "kept_samples": report.kept_samples if report is not None else sample_count,
            "skipped_samples": report.skipped_samples if report is not None else 0,
            "truncated_samples": report.truncated_samples if report is not None else 0,
            "skipped_source_indexes": list(report.skipped_source_indexes) if report is not None else [],
            "truncated_source_indexes": list(report.truncated_source_indexes) if report is not None else [],
        }
        if model_context_info is not None:
            metadata["workload"]["model_context"] = model_context_info.to_metadata()
    return metadata


def prepare_workload_for_trial(
    path: str | Path,
    *,
    model_name: str,
) -> PreparedWorkload:
    config = load_workload_config(path)
    fallback_tokenizer = resolve_tokenizer(config.tokenizer)
    fallback_tokenizer_key = _tokenizer_cache_key(config.tokenizer, tokenizer=fallback_tokenizer)
    fallback_tokenizer_name = _normalized_tokenizer_spec(config.tokenizer)
    requires_context_validation = config.dataset.type in {"jsonl", "sharegpt", "hf"}

    if config.context_policy is None:
        if requires_context_validation:
            raise ValueError(
                "real dataset workloads require context_policy for pre-trial context validation"
            )
        samples = generate_sample_requests(
            config,
            tokenizer=fallback_tokenizer,
            tokenizer_key=fallback_tokenizer_key,
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

    if (
        requires_context_validation
        and config.context_policy.tokenizer_source == "workload_tokenizer"
        and not config.context_policy.unsafe_allow_workload_tokenizer_for_real_datasets
    ):
        raise ValueError(
            "real dataset workloads must not use context_policy.tokenizer_source=workload_tokenizer. "
            "Set context_policy.unsafe_allow_workload_tokenizer_for_real_datasets=true only for "
            "explicitly unsafe test-only runs."
        )

    model_context_info = resolve_model_context_info(
        config.context_policy,
        workload_tokenizer=fallback_tokenizer,
        workload_tokenizer_key=fallback_tokenizer_key,
        model_name=model_name,
        fallback_tokenizer=fallback_tokenizer,
        fallback_tokenizer_key=fallback_tokenizer_key,
        fallback_tokenizer_name=fallback_tokenizer_name,
    )
    effective_context_policy = model_context_info.effective_policy(config.context_policy)
    samples = generate_sample_requests(
        config,
        tokenizer=model_context_info.tokenizer,
        tokenizer_key=model_context_info.tokenizer_key,
    )
    validation_result = validate_samples_against_context_window(
        samples,
        tokenizer=model_context_info.tokenizer,
        policy=effective_context_policy,
        tokenizer_key=model_context_info.tokenizer_key,
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
            effective_context_policy=effective_context_policy,
            model_context_info=model_context_info,
        ),
        context_validation_report=validation_result.report,
    )
