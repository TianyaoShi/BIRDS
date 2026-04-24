from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from .records import SampleRequest


class ModelTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


class DecodingModelTokenizer(ModelTokenizer, Protocol):
    def decode(self, token_ids: list[int]) -> str:
        ...


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    max_model_len: int
    tokenizer_source: str = "vllm_model_config"
    tokenizer: str | None = None
    over_limit: str = "fail"
    truncation_side: str = "left"
    unsafe_allow_workload_tokenizer_for_real_datasets: bool = False

    def __post_init__(self) -> None:
        if self.max_model_len <= 0:
            raise ValueError("context_policy.max_model_len must be positive")
        if self.tokenizer_source not in {"vllm_model_config", "explicit", "workload_tokenizer"}:
            raise ValueError(
                "context_policy.tokenizer_source must be one of "
                "{'vllm_model_config', 'explicit', 'workload_tokenizer'}"
            )
        if self.over_limit not in {"fail", "skip_sample", "truncate_prompt"}:
            raise ValueError(
                "context_policy.over_limit must be one of "
                "{'fail', 'skip_sample', 'truncate_prompt'}"
            )
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("context_policy.truncation_side must be one of {'left', 'right'}")
        if self.tokenizer_source == "explicit" and not self.tokenizer:
            raise ValueError("context_policy.tokenizer is required when tokenizer_source=explicit")
        if not isinstance(self.unsafe_allow_workload_tokenizer_for_real_datasets, bool):
            raise ValueError(
                "context_policy.unsafe_allow_workload_tokenizer_for_real_datasets must be a boolean"
            )


@dataclass(frozen=True, slots=True)
class ContextValidationReport:
    total_samples: int
    kept_samples: int
    skipped_samples: int
    truncated_samples: int
    skipped_source_indexes: tuple[int, ...]
    truncated_source_indexes: tuple[int, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "context_policy": {
                "total_samples": self.total_samples,
                "kept_samples": self.kept_samples,
                "skipped_samples": self.skipped_samples,
                "truncated_samples": self.truncated_samples,
                "skipped_source_indexes": list(self.skipped_source_indexes),
                "truncated_source_indexes": list(self.truncated_source_indexes),
            }
        }


@dataclass(frozen=True, slots=True)
class ContextValidationResult:
    samples: list[SampleRequest]
    report: ContextValidationReport


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


def parse_context_policy(payload: Any | None) -> ContextPolicy | None:
    if payload is None:
        return None
    policy_payload = _expect_mapping(payload, "context_policy")
    _check_allowed_keys(
        policy_payload,
        "context_policy",
        {
            "max_model_len",
            "tokenizer_source",
            "tokenizer",
            "over_limit",
            "truncation_side",
            "unsafe_allow_workload_tokenizer_for_real_datasets",
        },
    )
    if "max_model_len" not in policy_payload:
        raise ValueError("context_policy.max_model_len is required")
    max_model_len = _expect_int(policy_payload["max_model_len"], "context_policy.max_model_len", positive=True)
    tokenizer_source = policy_payload.get("tokenizer_source", "vllm_model_config")
    if not isinstance(tokenizer_source, str):
        raise ValueError("context_policy.tokenizer_source must be a string")
    tokenizer = policy_payload.get("tokenizer")
    if tokenizer is not None and not isinstance(tokenizer, str):
        raise ValueError("context_policy.tokenizer must be a string when provided")
    over_limit = policy_payload.get("over_limit", "fail")
    if not isinstance(over_limit, str):
        raise ValueError("context_policy.over_limit must be a string")
    truncation_side = policy_payload.get("truncation_side", "left")
    if not isinstance(truncation_side, str):
        raise ValueError("context_policy.truncation_side must be a string")
    unsafe_override = policy_payload.get("unsafe_allow_workload_tokenizer_for_real_datasets", False)
    if not isinstance(unsafe_override, bool):
        raise ValueError(
            "context_policy.unsafe_allow_workload_tokenizer_for_real_datasets must be a boolean"
        )
    return ContextPolicy(
        max_model_len=max_model_len,
        tokenizer_source=tokenizer_source,
        tokenizer=tokenizer,
        over_limit=over_limit,
        truncation_side=truncation_side,
        unsafe_allow_workload_tokenizer_for_real_datasets=unsafe_override,
    )


@contextmanager
def _force_hf_offline_mode():
    prior_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    prior_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prior_hf_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_hf_hub_offline
        if prior_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prior_transformers_offline


def _truncate_prompt_tokens(
    token_ids: list[int],
    *,
    keep_tokens: int,
    truncation_side: str,
) -> list[int]:
    if keep_tokens < 0:
        raise ValueError(f"keep_tokens must be non-negative, got {keep_tokens}")
    if truncation_side == "left":
        return token_ids[-keep_tokens:] if keep_tokens > 0 else []
    if truncation_side == "right":
        return token_ids[:keep_tokens]
    raise ValueError(f"unsupported truncation_side {truncation_side!r}")


def validate_samples_against_context_window(
    samples: list[SampleRequest],
    *,
    tokenizer: ModelTokenizer,
    policy: ContextPolicy,
) -> ContextValidationResult:
    kept: list[SampleRequest] = []
    skipped_source_indexes: list[int] = []
    truncated_source_indexes: list[int] = []
    skipped_samples = 0
    truncated_samples = 0

    for sample_index, sample in enumerate(samples):
        prompt_token_ids = tokenizer.encode(sample.prompt)
        prompt_token_count = len(prompt_token_ids)
        allowed_prompt_tokens = policy.max_model_len - sample.expected_output_len
        source_index = sample.metadata.get("source_index", sample_index)
        if not isinstance(source_index, int):
            raise TypeError(
                "sample.metadata['source_index'] must be an int when present, "
                f"got {type(source_index).__name__}"
            )

        if allowed_prompt_tokens < 0:
            raise ValueError(
                "sample expected_output_len exceeds context_policy.max_model_len "
                f"(source_index={source_index}, expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len})"
            )

        if prompt_token_count + sample.expected_output_len <= policy.max_model_len:
            kept.append(sample)
            continue

        if policy.over_limit == "fail":
            raise ValueError(
                "sample exceeds context window "
                f"(source_index={source_index}, prompt_tokens={prompt_token_count}, "
                f"expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len})"
            )

        if policy.over_limit == "skip_sample":
            skipped_samples += 1
            skipped_source_indexes.append(source_index)
            continue

        if policy.over_limit != "truncate_prompt":
            raise ValueError(f"unsupported over_limit policy {policy.over_limit!r}")

        if not hasattr(tokenizer, "decode"):
            raise TypeError(
                "tokenizer must implement decode(token_ids) when over_limit=truncate_prompt"
            )
        decoding_tokenizer = tokenizer  # typing helper
        truncated_token_ids = _truncate_prompt_tokens(
            prompt_token_ids,
            keep_tokens=allowed_prompt_tokens,
            truncation_side=policy.truncation_side,
        )
        truncated_prompt = getattr(decoding_tokenizer, "decode")(truncated_token_ids)
        if not isinstance(truncated_prompt, str):
            raise TypeError("tokenizer.decode must return a string")
        truncated_prompt_len = len(tokenizer.encode(truncated_prompt))
        if truncated_prompt_len + sample.expected_output_len > policy.max_model_len:
            raise RuntimeError(
                "truncate_prompt did not produce a context-fitting prompt "
                f"(source_index={source_index}, truncated_prompt_tokens={truncated_prompt_len}, "
                f"expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len})"
            )
        truncated_samples += 1
        truncated_source_indexes.append(source_index)
        updated_metadata = dict(sample.metadata)
        updated_metadata["context_truncated"] = True
        updated_metadata["context_original_prompt_len"] = sample.prompt_len
        kept.append(
            SampleRequest(
                prompt=truncated_prompt,
                prompt_len=truncated_prompt_len,
                expected_output_len=sample.expected_output_len,
                extra_body=sample.extra_body,
                metadata=updated_metadata,
            )
        )

    report = ContextValidationReport(
        total_samples=len(samples),
        kept_samples=len(kept),
        skipped_samples=skipped_samples,
        truncated_samples=truncated_samples,
        skipped_source_indexes=tuple(skipped_source_indexes),
        truncated_source_indexes=tuple(truncated_source_indexes),
    )
    return ContextValidationResult(samples=kept, report=report)


def resolve_model_tokenizer_for_policy(
    policy: ContextPolicy,
    *,
    workload_tokenizer: ModelTokenizer | None = None,
    model_name: str | None = None,
) -> ModelTokenizer:
    if policy.tokenizer_source == "workload_tokenizer":
        if workload_tokenizer is None:
            raise ValueError(
                "context_policy.tokenizer_source=workload_tokenizer requires workload_tokenizer"
            )
        return workload_tokenizer

    if policy.tokenizer_source == "explicit":
        from .workload import HuggingFaceTokenizer

        assert policy.tokenizer is not None
        return HuggingFaceTokenizer(policy.tokenizer)

    if policy.tokenizer_source != "vllm_model_config":
        raise ValueError(f"unsupported context_policy.tokenizer_source {policy.tokenizer_source!r}")

    resolved_name = policy.tokenizer or model_name
    if not resolved_name:
        raise ValueError(
            "context_policy.tokenizer_source=vllm_model_config requires "
            "context_policy.tokenizer or model_name"
        )
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        try:
            from vllm.transformers_utils.tokenizer import get_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "vLLM tokenizer utilities are unavailable; cannot resolve "
                "context_policy.tokenizer_source=vllm_model_config"
            ) from exc
    with _force_hf_offline_mode():
        tokenizer = get_tokenizer(
            tokenizer_name=resolved_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
        )
    if not hasattr(tokenizer, "encode"):
        raise TypeError("resolved tokenizer does not implement encode(text)")
    return tokenizer
