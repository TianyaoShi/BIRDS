from __future__ import annotations

from pathlib import Path

import pytest

from llm_mst_finder.workload import (
    generate_sample_requests,
    load_workload_config,
    load_workload_samples,
    load_workload_samples_for_sampling_only,
    prepare_workload_for_trial,
)


FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def test_synthetic_distribution_is_deterministic_and_records_token_lengths() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_distribution.yaml"
    first = load_workload_samples_for_sampling_only(workload_path)
    second = load_workload_samples_for_sampling_only(workload_path)

    assert [sample.to_dict() for sample in first] == [sample.to_dict() for sample in second]
    assert len(first) == 6
    assert all(sample.prompt_len == len(sample.prompt.split()) for sample in first)
    assert all(sample.expected_output_len == 4 for sample in first)
    assert all(sample.extra_body is not None for sample in first)
    assert all(sample.extra_body["ignore_eos"] is True for sample in first if sample.extra_body is not None)


def test_load_workload_samples_alias_matches_sampling_only_helper() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_distribution.yaml"
    via_alias = load_workload_samples(workload_path)
    via_explicit_helper = load_workload_samples_for_sampling_only(workload_path)
    assert [sample.to_dict() for sample in via_alias] == [sample.to_dict() for sample in via_explicit_helper]


def test_jsonl_from_dataset_uses_dataset_lengths() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "jsonl_from_dataset.yaml"
    samples = load_workload_samples_for_sampling_only(workload_path)

    assert len(samples) == 5
    assert all(sample.expected_output_len in {4, 5, 6} for sample in samples)
    assert all(sample.prompt_len == len(sample.prompt.split()) for sample in samples)
    assert all(sample.metadata["dataset_type"] == "jsonl" for sample in samples)


def test_sharegpt_from_dataset_uses_assistant_output_length() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_from_dataset.yaml"
    samples = load_workload_samples_for_sampling_only(workload_path)

    assert len(samples) == 4
    assert all(sample.expected_output_len in {2, 5} for sample in samples)
    assert all(sample.metadata["dataset_type"] == "sharegpt" for sample in samples)


def test_sharegpt_skips_rows_missing_prompt_or_assistant() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_with_invalid_rows.yaml"
    samples = load_workload_samples_for_sampling_only(workload_path)

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


def test_context_policy_is_loaded_from_yaml(tmp_path: Path) -> None:
    workload_path = tmp_path / "with_context_policy.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: with-context-policy",
                "dataset:",
                "  type: synthetic-fixed",
                "sampling:",
                "  seed: 1",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 3",
                "  output_len:",
                "    mode: fixed",
                "    value: 4",
                "context_policy:",
                "  max_model_len: 16",
                "  tokenizer_source: workload_tokenizer",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    config = load_workload_config(workload_path)
    assert config.context_policy is not None
    assert config.context_policy.max_model_len == 16
    assert config.context_policy.tokenizer_source == "workload_tokenizer"


def test_context_policy_missing_max_model_len_raises_from_yaml(tmp_path: Path) -> None:
    workload_path = tmp_path / "bad_context_policy.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: bad-context-policy",
                "dataset:",
                "  type: synthetic-fixed",
                "sampling:",
                "  seed: 1",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 3",
                "  output_len:",
                "    mode: fixed",
                "    value: 4",
                "context_policy:",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_policy.max_model_len is required"):
        load_workload_config(workload_path)


def test_prepare_workload_for_trial_requires_context_policy_for_real_dataset() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "jsonl_from_dataset.yaml"

    with pytest.raises(
        ValueError,
        match="real dataset workloads require context_policy for pre-trial context validation",
    ):
        prepare_workload_for_trial(workload_path, model_name="fake-model")


def test_prepare_workload_for_trial_rejects_workload_tokenizer_for_real_dataset_by_default(
    tmp_path: Path,
) -> None:
    workload_path = tmp_path / "jsonl_context_unsafe_default.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-real-default-reject",
                "dataset:",
                "  type: jsonl",
                f"  path: {FIXTURES_ROOT / 'data' / 'requests.jsonl'}",
                "tokenizer: whitespace",
                "sampling:",
                "  seed: 1",
                "  num_requests: 3",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 8192",
                "  tokenizer_source: workload_tokenizer",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must not use context_policy.tokenizer_source=workload_tokenizer",
    ):
        prepare_workload_for_trial(workload_path, model_name="fake-model")


def test_prepare_workload_for_trial_allows_explicit_unsafe_override_and_records_it(
    tmp_path: Path,
) -> None:
    workload_path = tmp_path / "jsonl_context_unsafe_override.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-real-unsafe-override",
                "dataset:",
                "  type: jsonl",
                f"  path: {FIXTURES_ROOT / 'data' / 'requests.jsonl'}",
                "tokenizer: whitespace",
                "sampling:",
                "  seed: 1",
                "  num_requests: 3",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 8192",
                "  tokenizer_source: workload_tokenizer",
                "  over_limit: fail",
                "  unsafe_allow_workload_tokenizer_for_real_datasets: true",
            ]
        ),
        encoding="utf-8",
    )

    prepared = prepare_workload_for_trial(workload_path, model_name="fake-model")
    context_policy_metadata = prepared.metadata["workload"]["context_policy"]
    assert context_policy_metadata["unsafe_allow_workload_tokenizer_for_real_datasets"] is True


def test_jsonl_manifest_cache_is_reused_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        (FIXTURES_ROOT / "data" / "requests.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_cached.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-cached",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
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

    first = load_workload_samples_for_sampling_only(workload_path)
    assert len(first) == 3

    def fail_if_source_is_used(*args, **kwargs):
        del args, kwargs
        raise AssertionError("expected workload sampling to use the manifest cache")

    monkeypatch.setattr("llm_mst_finder.workload._load_jsonl_entries_from_source", fail_if_source_is_used)

    second = load_workload_samples_for_sampling_only(workload_path)
    assert [sample.to_dict() for sample in second] == [sample.to_dict() for sample in first]
