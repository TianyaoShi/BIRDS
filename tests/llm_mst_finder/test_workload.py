from __future__ import annotations

from pathlib import Path

import pytest

from llm_mst_finder.workload import (
    generate_sample_requests,
    load_workload_config,
    load_workload_samples,
)


FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def test_synthetic_distribution_is_deterministic_and_records_token_lengths() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_distribution.yaml"
    first = load_workload_samples(workload_path)
    second = load_workload_samples(workload_path)

    assert [sample.to_dict() for sample in first] == [sample.to_dict() for sample in second]
    assert len(first) == 6
    assert all(sample.prompt_len == len(sample.prompt.split()) for sample in first)
    assert all(sample.expected_output_len == 4 for sample in first)
    assert all(sample.extra_body is not None for sample in first)
    assert all(sample.extra_body["ignore_eos"] is True for sample in first if sample.extra_body is not None)


def test_jsonl_from_dataset_uses_dataset_lengths() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "jsonl_from_dataset.yaml"
    samples = load_workload_samples(workload_path)

    assert len(samples) == 5
    assert all(sample.expected_output_len in {4, 5, 6} for sample in samples)
    assert all(sample.prompt_len == len(sample.prompt.split()) for sample in samples)
    assert all(sample.metadata["dataset_type"] == "jsonl" for sample in samples)


def test_sharegpt_from_dataset_uses_assistant_output_length() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_from_dataset.yaml"
    samples = load_workload_samples(workload_path)

    assert len(samples) == 4
    assert all(sample.expected_output_len in {2, 5} for sample in samples)
    assert all(sample.metadata["dataset_type"] == "sharegpt" for sample in samples)


def test_sharegpt_skips_rows_missing_prompt_or_assistant() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_with_invalid_rows.yaml"
    samples = load_workload_samples(workload_path)

    assert len(samples) == 3
    assert all(sample.prompt == "valid prompt text" for sample in samples)
    assert all(sample.expected_output_len == 4 for sample in samples)


def test_missing_dataset_file_raises(tmp_path: Path) -> None:
    workload_path = tmp_path / "missing_dataset.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: missing-dataset",
                "dataset:",
                "  type: jsonl",
                "  path: not-there.jsonl",
                "tokenizer: whitespace",
                "sampling:",
                "  seed: 1",
                "  num_requests: 3",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    config = load_workload_config(workload_path)
    with pytest.raises(FileNotFoundError):
        generate_sample_requests(config)


def test_invalid_fixed_or_bucketed_raises(tmp_path: Path) -> None:
    workload_path = tmp_path / "invalid.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: invalid-fixed-or-bucketed",
                "dataset:",
                "  type: synthetic-fixed",
                "sampling:",
                "  seed: 1",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: fixed_or_bucketed",
                "  output_len:",
                "    mode: fixed",
                "    value: 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed_or_bucketed requires value or buckets"):
        load_workload_config(workload_path)
