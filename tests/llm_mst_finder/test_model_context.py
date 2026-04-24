from __future__ import annotations

import os
import sys
import types

import pytest

from llm_mst_finder.model_context import (
    ContextPolicy,
    parse_context_policy,
    resolve_model_tokenizer_for_policy,
    validate_samples_against_context_window,
)
from llm_mst_finder.records import SampleRequest


class WordTokenizer:
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


def _sample(
    prompt: str,
    *,
    prompt_len: int,
    expected_output_len: int,
    source_index: int,
) -> SampleRequest:
    return SampleRequest(
        prompt=prompt,
        prompt_len=prompt_len,
        expected_output_len=expected_output_len,
        metadata={"source_index": source_index},
    )


def test_context_validation_fits_context_returns_unchanged_samples() -> None:
    tokenizer = WordTokenizer()
    samples = [
        _sample("a b c", prompt_len=3, expected_output_len=2, source_index=0),
        _sample("d e", prompt_len=2, expected_output_len=1, source_index=1),
    ]
    policy = ContextPolicy(max_model_len=8, tokenizer_source="workload_tokenizer", over_limit="fail")

    result = validate_samples_against_context_window(samples, tokenizer=tokenizer, policy=policy)

    assert result.samples == samples
    assert result.report.total_samples == 2
    assert result.report.kept_samples == 2
    assert result.report.skipped_samples == 0
    assert result.report.truncated_samples == 0


def test_context_validation_over_limit_fail_raises() -> None:
    tokenizer = WordTokenizer()
    samples = [_sample("a b c d e", prompt_len=5, expected_output_len=2, source_index=7)]
    policy = ContextPolicy(max_model_len=6, tokenizer_source="workload_tokenizer", over_limit="fail")

    with pytest.raises(ValueError, match="sample exceeds context window"):
        validate_samples_against_context_window(samples, tokenizer=tokenizer, policy=policy)


def test_context_validation_skip_sample_records_counts_and_indexes() -> None:
    tokenizer = WordTokenizer()
    samples = [
        _sample("a b c", prompt_len=3, expected_output_len=2, source_index=10),
        _sample("d e f g h", prompt_len=5, expected_output_len=2, source_index=11),
        _sample("i j k l", prompt_len=4, expected_output_len=3, source_index=12),
    ]
    policy = ContextPolicy(max_model_len=6, tokenizer_source="workload_tokenizer", over_limit="skip_sample")

    result = validate_samples_against_context_window(samples, tokenizer=tokenizer, policy=policy)

    assert [sample.metadata["source_index"] for sample in result.samples] == [10]
    assert result.report.total_samples == 3
    assert result.report.kept_samples == 1
    assert result.report.skipped_samples == 2
    assert result.report.truncated_samples == 0
    assert result.report.skipped_source_indexes == (11, 12)


def test_context_validation_truncate_prompt_shortens_and_records_indexes() -> None:
    tokenizer = WordTokenizer()
    samples = [_sample("a b c d e", prompt_len=5, expected_output_len=2, source_index=33)]
    policy = ContextPolicy(
        max_model_len=6,
        tokenizer_source="workload_tokenizer",
        over_limit="truncate_prompt",
        truncation_side="left",
    )

    result = validate_samples_against_context_window(samples, tokenizer=tokenizer, policy=policy)

    assert len(result.samples) == 1
    truncated = result.samples[0]
    assert truncated.metadata["context_truncated"] is True
    assert truncated.metadata["context_original_prompt_len"] == 5
    assert truncated.metadata["source_index"] == 33
    assert truncated.prompt == "b c d e"
    assert truncated.prompt_len == 4
    assert result.report.total_samples == 1
    assert result.report.kept_samples == 1
    assert result.report.skipped_samples == 0
    assert result.report.truncated_samples == 1
    assert result.report.truncated_source_indexes == (33,)


def test_parse_context_policy_missing_max_model_len_raises() -> None:
    with pytest.raises(ValueError, match="context_policy.max_model_len is required"):
        parse_context_policy({"over_limit": "fail"})


def test_resolve_model_tokenizer_missing_workload_tokenizer_raises() -> None:
    policy = ContextPolicy(max_model_len=4096, tokenizer_source="workload_tokenizer")
    with pytest.raises(
        ValueError,
        match="tokenizer_source=workload_tokenizer requires workload_tokenizer",
    ):
        resolve_model_tokenizer_for_policy(policy)


def test_parse_context_policy_parses_unsafe_override_flag() -> None:
    policy = parse_context_policy(
        {
            "max_model_len": 8192,
            "tokenizer_source": "workload_tokenizer",
            "unsafe_allow_workload_tokenizer_for_real_datasets": True,
        }
    )
    assert policy is not None
    assert policy.unsafe_allow_workload_tokenizer_for_real_datasets is True


def test_resolve_model_tokenizer_vllm_forces_offline_env(monkeypatch) -> None:
    observed: dict[str, str | None] = {}

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    def fake_get_tokenizer(*, tokenizer_name: str, tokenizer_mode: str, trust_remote_code: bool):
        del tokenizer_name, tokenizer_mode, trust_remote_code
        observed["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        observed["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return FakeTokenizer()

    fake_vllm_module = types.ModuleType("vllm")
    fake_transformers_utils = types.ModuleType("vllm.transformers_utils")
    fake_tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizer")
    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer
    fake_transformers_utils.tokenizer = fake_tokenizer_module
    fake_vllm_module.transformers_utils = fake_transformers_utils

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils", fake_transformers_utils)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer", fake_tokenizer_module)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    policy = ContextPolicy(max_model_len=4096, tokenizer_source="vllm_model_config")
    tokenizer = resolve_model_tokenizer_for_policy(policy, model_name="fake-model")

    assert tokenizer.encode("abc") == [3]
    assert observed["HF_HUB_OFFLINE"] == "1"
    assert observed["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None
