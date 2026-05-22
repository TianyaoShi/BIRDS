from __future__ import annotations

from llm_mst_finder.records import SampleRequest

from output_quality_profiler.generation import _sample_with_quality_decoding
from output_quality_profiler.models import QualityDecodingConfig


def test_workload_expected_output_len_policy_uses_sample_cap() -> None:
    sample = SampleRequest(prompt="Question?", prompt_len=10, expected_output_len=64)
    decoded = _sample_with_quality_decoding(
        sample,
        decoding=QualityDecodingConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            max_tokens_policy="workload_expected_output_len",
        ),
        serving_max_model_len=32768,
    )

    assert decoded.expected_output_len == 64


def test_workload_expected_output_len_policy_still_respects_context_room() -> None:
    sample = SampleRequest(prompt="Question?", prompt_len=90, expected_output_len=64)
    decoded = _sample_with_quality_decoding(
        sample,
        decoding=QualityDecodingConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            max_tokens_policy="workload_expected_output_len",
            prompt_token_buffer=8,
        ),
        serving_max_model_len=120,
    )

    assert decoded.expected_output_len == 22
