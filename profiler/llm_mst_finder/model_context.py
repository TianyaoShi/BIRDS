from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
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
    max_model_len: int | None = None
    tokenizer_source: str = "vllm_model_config"
    tokenizer: str | None = None
    over_limit: str = "fail"
    truncation_side: str = "left"
    reserve_tokens: int = 0
    unsafe_allow_workload_tokenizer_for_real_datasets: bool = False

    def __post_init__(self) -> None:
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("context_policy.max_model_len must be positive")
        if self.reserve_tokens < 0:
            raise ValueError("context_policy.reserve_tokens must be non-negative")
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


@dataclass(frozen=True, slots=True)
class ModelContextInfo:
    tokenizer: ModelTokenizer
    tokenizer_key: str
    max_model_len: int
    model_max_model_len: int | None
    workload_max_model_len: int | None
    tokenizer_source: str
    tokenizer_name: str | None
    fallback_used: bool = False
    fallback_reason: str | None = None

    def effective_policy(self, policy: ContextPolicy) -> ContextPolicy:
        return replace(policy, max_model_len=self.max_model_len)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fallback_reason": self.fallback_reason,
            "fallback_used": self.fallback_used,
            "max_model_len": self.max_model_len,
            "model_max_model_len": self.model_max_model_len,
            "tokenizer": self.tokenizer_name,
            "tokenizer_key": self.tokenizer_key,
            "tokenizer_source": self.tokenizer_source,
            "workload_max_model_len": self.workload_max_model_len,
        }


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
            "reserve_tokens",
            "unsafe_allow_workload_tokenizer_for_real_datasets",
        },
    )
    max_model_len = None
    if "max_model_len" in policy_payload:
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
    reserve_tokens = policy_payload.get("reserve_tokens", 0)
    reserve_tokens = _expect_int(reserve_tokens, "context_policy.reserve_tokens")
    if reserve_tokens < 0:
        raise ValueError("context_policy.reserve_tokens must be non-negative")
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
        reserve_tokens=reserve_tokens,
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


def _is_chat_endpoint(endpoint: str | None) -> bool:
    return bool(endpoint and endpoint.rstrip("/").endswith("/chat/completions"))


def _request_prompt_token_count(
    prompt: str,
    *,
    tokenizer: ModelTokenizer,
    endpoint: str | None,
) -> int:
    if _is_chat_endpoint(endpoint) and hasattr(tokenizer, "apply_chat_template"):
        token_ids = getattr(tokenizer, "apply_chat_template")(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(token_ids, list):
            raise TypeError("tokenizer.apply_chat_template(..., tokenize=True) must return a list")
        return len(token_ids)
    return len(tokenizer.encode(prompt))


def _truncate_prompt_to_fit(
    prompt_token_ids: list[int],
    *,
    tokenizer: DecodingModelTokenizer,
    allowed_total_prompt_tokens: int,
    endpoint: str | None,
    truncation_side: str,
) -> tuple[str, int]:
    low = 0
    high = len(prompt_token_ids)
    best_prompt = ""
    best_len = _request_prompt_token_count("", tokenizer=tokenizer, endpoint=endpoint)
    while low <= high:
        keep_tokens = (low + high) // 2
        candidate_ids = _truncate_prompt_tokens(
            prompt_token_ids,
            keep_tokens=keep_tokens,
            truncation_side=truncation_side,
        )
        candidate_prompt = tokenizer.decode(candidate_ids)
        if not isinstance(candidate_prompt, str):
            raise TypeError("tokenizer.decode must return a string")
        candidate_len = _request_prompt_token_count(candidate_prompt, tokenizer=tokenizer, endpoint=endpoint)
        if candidate_len <= allowed_total_prompt_tokens:
            best_prompt = candidate_prompt
            best_len = candidate_len
            low = keep_tokens + 1
        else:
            high = keep_tokens - 1
    return best_prompt, best_len


def validate_samples_against_context_window(
    samples: list[SampleRequest],
    *,
    tokenizer: ModelTokenizer,
    policy: ContextPolicy,
    tokenizer_key: str | None = None,
    endpoint: str | None = None,
) -> ContextValidationResult:
    kept: list[SampleRequest] = []
    skipped_source_indexes: list[int] = []
    truncated_source_indexes: list[int] = []
    skipped_samples = 0
    truncated_samples = 0

    for sample_index, sample in enumerate(samples):
        prompt_token_ids: list[int] | None = None
        cached_prompt_tokenizer_key = sample.metadata.get("prompt_tokenizer_key")
        if cached_prompt_tokenizer_key is not None and not isinstance(cached_prompt_tokenizer_key, str):
            raise TypeError(
                "sample.metadata['prompt_tokenizer_key'] must be a string when present, "
                f"got {type(cached_prompt_tokenizer_key).__name__}"
            )
        if (
            not _is_chat_endpoint(endpoint)
            and tokenizer_key is not None
            and cached_prompt_tokenizer_key == tokenizer_key
        ):
            prompt_token_count = sample.prompt_len
        else:
            prompt_token_ids = tokenizer.encode(sample.prompt)
            prompt_token_count = _request_prompt_token_count(
                sample.prompt,
                tokenizer=tokenizer,
                endpoint=endpoint,
            )
        allowed_prompt_tokens = policy.max_model_len - sample.expected_output_len - policy.reserve_tokens
        source_index = sample.metadata.get("source_index", sample_index)
        if not isinstance(source_index, int):
            raise TypeError(
                "sample.metadata['source_index'] must be an int when present, "
                f"got {type(source_index).__name__}"
            )

        if allowed_prompt_tokens < 0:
            if policy.over_limit == "skip_sample":
                skipped_samples += 1
                skipped_source_indexes.append(source_index)
                continue
            raise ValueError(
                "sample expected_output_len exceeds context_policy.max_model_len "
                f"(source_index={source_index}, expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len}, reserve_tokens={policy.reserve_tokens})"
            )

        if prompt_token_count + sample.expected_output_len + policy.reserve_tokens <= policy.max_model_len:
            kept.append(sample)
            continue

        if policy.over_limit == "fail":
            raise ValueError(
                "sample exceeds context window "
                f"(source_index={source_index}, prompt_tokens={prompt_token_count}, "
                f"expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len}, reserve_tokens={policy.reserve_tokens})"
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
        if prompt_token_ids is None:
            prompt_token_ids = tokenizer.encode(sample.prompt)
        truncated_prompt, truncated_prompt_len = _truncate_prompt_to_fit(
            prompt_token_ids,
            tokenizer=decoding_tokenizer,
            allowed_total_prompt_tokens=allowed_prompt_tokens,
            endpoint=endpoint,
            truncation_side=policy.truncation_side,
        )
        if truncated_prompt_len + sample.expected_output_len + policy.reserve_tokens > policy.max_model_len:
            raise RuntimeError(
                "truncate_prompt did not produce a context-fitting prompt "
                f"(source_index={source_index}, truncated_prompt_tokens={truncated_prompt_len}, "
                f"expected_output_len={sample.expected_output_len}, "
                f"max_model_len={policy.max_model_len}, reserve_tokens={policy.reserve_tokens})"
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
    return _resolve_vllm_tokenizer(resolved_name)


def resolve_model_context_info(
    policy: ContextPolicy,
    *,
    workload_tokenizer: ModelTokenizer | None = None,
    workload_tokenizer_key: str | None = None,
    model_name: str | None,
    fallback_tokenizer: ModelTokenizer,
    fallback_tokenizer_key: str,
    fallback_tokenizer_name: str,
    default_max_model_len: int = 4096,
) -> ModelContextInfo:
    if default_max_model_len <= 0:
        raise ValueError("default_max_model_len must be positive")

    if policy.tokenizer_source == "workload_tokenizer":
        if workload_tokenizer is None or workload_tokenizer_key is None:
            raise ValueError(
                "context_policy.tokenizer_source=workload_tokenizer requires workload_tokenizer"
            )
        max_model_len = _effective_max_model_len(
            model_max_model_len=None,
            workload_max_model_len=policy.max_model_len,
            default_max_model_len=default_max_model_len,
        )
        return ModelContextInfo(
            tokenizer=workload_tokenizer,
            tokenizer_key=workload_tokenizer_key,
            max_model_len=max_model_len,
            model_max_model_len=None,
            workload_max_model_len=policy.max_model_len,
            tokenizer_source="workload_tokenizer",
            tokenizer_name=fallback_tokenizer_name,
        )

    if policy.tokenizer_source == "explicit":
        tokenizer_name = policy.tokenizer
        if tokenizer_name is None:
            raise ValueError("context_policy.tokenizer is required when tokenizer_source=explicit")
        from .workload import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_name)
        model_max_model_len = infer_model_max_model_len(tokenizer_name, tokenizer=tokenizer)
        max_model_len = _effective_max_model_len(
            model_max_model_len=model_max_model_len,
            workload_max_model_len=policy.max_model_len,
            default_max_model_len=default_max_model_len,
        )
        return ModelContextInfo(
            tokenizer=tokenizer,
            tokenizer_key=f"tokenizer:{tokenizer_name}",
            max_model_len=max_model_len,
            model_max_model_len=model_max_model_len,
            workload_max_model_len=policy.max_model_len,
            tokenizer_source="explicit",
            tokenizer_name=tokenizer_name,
        )

    if policy.tokenizer_source != "vllm_model_config":
        raise ValueError(f"unsupported context_policy.tokenizer_source {policy.tokenizer_source!r}")

    tokenizer_name = policy.tokenizer or model_name
    if tokenizer_name:
        try:
            tokenizer = _resolve_vllm_tokenizer(tokenizer_name)
            model_max_model_len = infer_model_max_model_len(tokenizer_name, tokenizer=tokenizer)
            max_model_len = _effective_max_model_len(
                model_max_model_len=model_max_model_len,
                workload_max_model_len=policy.max_model_len,
                default_max_model_len=default_max_model_len,
            )
            return ModelContextInfo(
                tokenizer=tokenizer,
                tokenizer_key=f"tokenizer:{tokenizer_name}",
                max_model_len=max_model_len,
                model_max_model_len=model_max_model_len,
                workload_max_model_len=policy.max_model_len,
                tokenizer_source="vllm_model_config",
                tokenizer_name=tokenizer_name,
            )
        except Exception as exc:
            fallback_reason = str(exc)
    else:
        fallback_reason = (
            "context_policy.tokenizer_source=vllm_model_config requires "
            "context_policy.tokenizer or model_name"
        )

    max_model_len = _effective_max_model_len(
        model_max_model_len=None,
        workload_max_model_len=policy.max_model_len,
        default_max_model_len=default_max_model_len,
    )
    return ModelContextInfo(
        tokenizer=fallback_tokenizer,
        tokenizer_key=fallback_tokenizer_key,
        max_model_len=max_model_len,
        model_max_model_len=None,
        workload_max_model_len=policy.max_model_len,
        tokenizer_source="fallback",
        tokenizer_name=fallback_tokenizer_name,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )


def infer_model_max_model_len(
    model_name: str | None,
    *,
    tokenizer: ModelTokenizer | None = None,
) -> int | None:
    config = _load_cached_hf_config(model_name)
    if config is not None:
        for key in (
            "max_model_len",
            "max_position_embeddings",
            "model_max_length",
            "n_positions",
            "seq_length",
            "max_seq_len",
            "max_sequence_length",
        ):
            value = config.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and 0 < value < 1_000_000_000:
                return value
            if isinstance(value, float) and value.is_integer() and 0 < value < 1_000_000_000:
                return int(value)
    if tokenizer is not None:
        value = getattr(tokenizer, "model_max_length", None)
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            return value
        if isinstance(value, float) and value.is_integer() and 0 < value < 1_000_000_000:
            return int(value)
    return None


def _effective_max_model_len(
    *,
    model_max_model_len: int | None,
    workload_max_model_len: int | None,
    default_max_model_len: int,
) -> int:
    candidates = [
        value
        for value in (model_max_model_len, workload_max_model_len)
        if value is not None
    ]
    if candidates:
        return min(candidates)
    return default_max_model_len


def _resolve_vllm_tokenizer(tokenizer_name: str) -> ModelTokenizer:
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
            tokenizer_name=tokenizer_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
        )
    if not hasattr(tokenizer, "encode"):
        raise TypeError("resolved tokenizer does not implement encode(text)")
    return tokenizer


def _load_cached_hf_config(model_name: str | None) -> dict[str, Any] | None:
    if model_name is None or "/" not in model_name:
        return None
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidates: list[Path] = []
    ref_path = model_cache / "refs" / "main"
    try:
        ref = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        ref = ""
    if ref:
        candidates.append(snapshots_dir / ref / "config.json")
    candidates.extend(sorted(snapshots_dir.glob("*/config.json")))

    for config_path in candidates:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None
