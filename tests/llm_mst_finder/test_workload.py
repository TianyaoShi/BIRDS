from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from llm_mst_finder.workload import (
    generate_sample_requests,
    inspect_workload_dataset,
    load_workload_config,
    load_workload_samples,
    load_workload_samples_for_sampling_only,
    prepare_workload_for_trial,
)
from llm_mst_finder.vllm_compat import build_openai_payload


FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def test_synthetic_distribution_is_deterministic_and_records_token_lengths() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_distribution.yaml"
    first = load_workload_samples_for_sampling_only(workload_path)
    second = load_workload_samples_for_sampling_only(workload_path)

    assert [sample.to_dict() for sample in first] == [sample.to_dict() for sample in second]
    assert len(first) == 6
    assert all(sample.prompt_len == len(sample.prompt) for sample in first)
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
    assert all(sample.prompt_len == len(sample.prompt) for sample in samples)
    assert all(sample.metadata["dataset_type"] == "jsonl" for sample in samples)


def test_jsonl_sequential_entry_selection_replays_rows_in_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "row zero", "expected_output_len": 1}),
                json.dumps({"prompt": "row one", "expected_output_len": 2}),
                json.dumps({"prompt": "row two", "expected_output_len": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_sequential.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-sequential",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 99",
                "  num_requests: 5",
                "  entry_selection: sequential",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    samples = load_workload_samples_for_sampling_only(workload_path)

    assert [sample.prompt for sample in samples] == [
        "row zero",
        "row one",
        "row two",
        "row zero",
        "row one",
    ]
    assert [sample.expected_output_len for sample in samples] == [1, 2, 3, 1, 2]
    assert all(sample.metadata["sampling_entry_selection"] == "sequential" for sample in samples)


def test_jsonl_from_dataset_honors_materialized_prompt_len(tmp_path: Path) -> None:
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "one two", "prompt_len": 1234, "expected_output_len": 5}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_prompt_len.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-prompt-len",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 99",
                "  num_requests: 1",
                "  entry_selection: sequential",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    sample = load_workload_samples_for_sampling_only(workload_path)[0]

    assert sample.prompt_len == 1234
    assert sample.expected_output_len == 5


def test_jsonl_natural_until_eos_uses_cap_without_dataset_output_length(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reasoning_questions.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Question: What is 2+2? Think carefully.",
                "metadata": {"ground_truth": "4"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "reasoning_natural.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: reasoning-natural",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 99",
                "  num_requests: 1",
                "  entry_selection: sequential",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: natural_until_eos",
                "    max_tokens: 2048",
                "request:",
                "  temperature: 0.0",
                "  ignore_eos: false",
            ]
        ),
        encoding="utf-8",
    )

    sample = load_workload_samples_for_sampling_only(workload_path)[0]
    payload = build_openai_payload("/v1/completions", "fake-model", sample)

    assert sample.expected_output_len == 2048
    assert sample.metadata["ground_truth"] == "4"
    assert sample.metadata["sampling_output_len_mode"] == "natural_until_eos"
    assert sample.metadata["sampling_output_len_is_cap"] is True
    assert sample.metadata["max_output_tokens"] == 2048
    assert sample.extra_body == {"temperature": 0.0, "ignore_eos": False}
    assert payload["max_tokens"] == 2048
    assert payload["ignore_eos"] is False


def test_natural_until_eos_rejects_ignore_eos_true(tmp_path: Path) -> None:
    workload_path = tmp_path / "invalid_reasoning_natural.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: invalid-reasoning-natural",
                "dataset:",
                "  type: synthetic-fixed",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 8",
                "  output_len:",
                "    mode: natural_until_eos",
                "    max_tokens: 128",
                "request:",
                "  ignore_eos: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires request.ignore_eos=false"):
        load_workload_config(workload_path)


def test_sharegpt_from_dataset_uses_assistant_output_length() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_from_dataset.yaml"
    samples = load_workload_samples_for_sampling_only(workload_path)
    dataset = json.loads(
        (FIXTURES_ROOT / "data" / "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(samples) == 4
    for sample in samples:
        row = dataset[sample.metadata["source_index"]]
        assistant_turn = next(
            turn for turn in row["conversations"] if turn.get("from", "").lower() == "gpt"
        )
        assert sample.expected_output_len == len(assistant_turn["value"])
    assert all(sample.metadata["dataset_type"] == "sharegpt" for sample in samples)


def test_sharegpt_skips_rows_missing_prompt_or_assistant() -> None:
    workload_path = FIXTURES_ROOT / "workloads" / "sharegpt_with_invalid_rows.yaml"
    samples = load_workload_samples_for_sampling_only(workload_path)

    assert len(samples) == 3
    assert all(sample.prompt == "valid prompt text" for sample in samples)
    assert all(sample.expected_output_len == len("valid assistant text here") for sample in samples)


def test_hf_dataset_uses_conversation_rows(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "conversation": [
                {"role": "user", "content": "How do I roast carrots?"},
                {"role": "assistant", "content": "Use oil, salt, and a hot oven."},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "What is TTFT?"},
                {"role": "assistant", "content": "Time to first token."},
            ]
        },
    ]

    def load_dataset(path, *, name, split, streaming):
        assert path == "allenai/WildChat"
        assert name is None
        assert split == "train"
        assert streaming is True
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr("llm_mst_finder.workload._manifest_cache_root", lambda: tmp_path / "cache")
    workload_path = tmp_path / "hf_wildchat.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: hf-wildchat",
                "dataset:",
                "  type: hf",
                "  path: allenai/WildChat",
                "  split: train",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "request:",
                "  stream: true",
                "context_policy:",
                "  max_model_len: 4096",
                "  tokenizer_source: workload_tokenizer",
                "  unsafe_allow_workload_tokenizer_for_real_datasets: true",
            ]
        ),
        encoding="utf-8",
    )

    samples = prepare_workload_for_trial(workload_path, model_name="fake-model").samples

    assert len(samples) == 2
    assert {sample.prompt for sample in samples} <= {"How do I roast carrots?", "What is TTFT?"}
    assert all(sample.metadata["dataset_type"] == "hf" for sample in samples)
    assert all(sample.metadata["hf_dataset_path"] == "allenai/WildChat" for sample in samples)
    assert all(sample.expected_output_len > 0 for sample in samples)


def test_hf_dataset_uses_reservoir_uniform_sample_not_first_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "conversation": [
                {"role": "user", "content": f"prompt {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        }
        for index in range(30)
    ]

    def load_dataset(path, *, name, split, streaming):
        assert path == "allenai/WildChat"
        assert name is None
        assert split == "train"
        assert streaming is True
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr("llm_mst_finder.workload._manifest_cache_root", lambda: tmp_path / "cache")
    workload_path = tmp_path / "hf_wildchat_uniform.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: hf-wildchat-uniform",
                "dataset:",
                "  type: hf",
                "  path: allenai/WildChat",
                "  split: train",
                "tokenizer: character",
                "sampling:",
                "  seed: 7",
                "  num_requests: 5",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 4096",
                "  tokenizer_source: workload_tokenizer",
                "  unsafe_allow_workload_tokenizer_for_real_datasets: true",
            ]
        ),
        encoding="utf-8",
    )

    samples = prepare_workload_for_trial(workload_path, model_name="fake-model").samples

    source_indexes = [sample.metadata["source_index"] for sample in samples]
    assert len(source_indexes) == 5
    assert source_indexes != [0, 1, 2, 3, 4]
    assert all(sample.metadata["hf_sampling_method"] == "reservoir_uniform" for sample in samples)


def test_inspect_workload_dataset_reports_hf_length_distribution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "conversation": [
                {"role": "user", "content": "short prompt"},
                {"role": "assistant", "content": "short answer"},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "longer prompt with more words"},
                {"role": "assistant", "content": "longer answer with several output words"},
            ]
        },
    ]

    def load_dataset(path, *, name, split, streaming):
        del name
        assert path == "allenai/WildChat"
        assert split == "train"
        assert streaming is True
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    workload_path = tmp_path / "hf_wildchat_inspect.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: hf-wildchat-inspect",
                "dataset:",
                "  type: hf",
                "  path: allenai/WildChat",
                "  split: train",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workload_dataset(
        workload_path,
        model_name="fake-model",
        sample_size=2,
        max_scan_rows=2,
    )

    assert report["source_summary"]["sampling_method"] == "reservoir_uniform"
    assert report["source_summary"]["scanned_rows"] == 2
    assert report["lengths"]["prompt_tokens"]["count"] == 2
    assert report["lengths"]["output_tokens"]["max"] == len("longer answer with several output words")
    assert "trial_min_duration_s" in report["suggested_search_overrides"]


def test_inspect_workload_dataset_accepts_tokenizer_override_without_workload_tokenizer(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "abc", "expected_output_len": 2}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_inspect_no_tokenizer.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-inspect-no-tokenizer",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workload_dataset(
        workload_path,
        model_name="fake-model",
        tokenizer_name="character",
    )

    assert report["inspection_tokenizer"] == "character"
    assert report["tokenizer_key"] == "tokenizer:character"
    assert report["lengths"]["prompt_tokens"]["p50"] == 3.0


def test_longbench_dataset_reads_zip_and_formats_prompts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "longbench.zip"
    rows = [
        {
            "_id": "row-0",
            "dataset": "gov_report",
            "input": "Summarize the report.",
            "context": "alpha beta gamma delta",
            "answers": ["short summary", "longer reference summary"],
            "length": 6,
            "language": "en",
            "all_classes": [],
        },
        {
            "_id": "row-1",
            "dataset": "trec",
            "input": "Classify the question.",
            "context": "question text here",
            "answers": ["DESC"],
            "length": 5,
            "language": "en",
            "all_classes": ["DESC", "ENTY"],
        },
    ]
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "data/gov_report.jsonl",
            "\n".join(json.dumps(row) for row in rows[:1]) + "\n",
        )
        archive.writestr(
            "data/trec.jsonl",
            "\n".join(json.dumps(row) for row in rows[1:]) + "\n",
        )

    monkeypatch.setattr("llm_mst_finder.workload._manifest_cache_root", lambda: tmp_path / "cache")
    workload_path = tmp_path / "longbench.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: longbench-smoke",
                "dataset:",
                "  type: longbench",
                f"  path: {zip_path}",
                "  configs:",
                "    - gov_report",
                "    - trec",
                "tokenizer: character",
                "sampling:",
                "  seed: 3",
                "  num_requests: 2",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 256",
                "  tokenizer_source: workload_tokenizer",
                "  unsafe_allow_workload_tokenizer_for_real_datasets: true",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    samples = prepare_workload_for_trial(workload_path, model_name="fake-model").samples

    assert len(samples) == 2
    assert {sample.metadata["longbench_config"] for sample in samples} == {"gov_report", "trec"}
    assert all(sample.prompt.startswith("Context:\n") for sample in samples)
    assert all("\n\nTask:\n" in sample.prompt for sample in samples)
    assert all(sample.prompt.endswith("Answer:") for sample in samples)
    assert any("Candidate labels/classes: DESC, ENTY" in sample.prompt for sample in samples)
    gov_report_sample = next(sample for sample in samples if sample.metadata["longbench_config"] == "gov_report")
    assert gov_report_sample.expected_output_len == len("longer reference summary")
    assert gov_report_sample.metadata["longbench_language"] == "en"


def test_longbench_inspection_reports_selected_configs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "data/multi_news.jsonl",
            json.dumps(
                {
                    "_id": "row-0",
                    "dataset": "multi_news",
                    "input": "Summarize.",
                    "context": "one two three four five",
                    "answers": ["summary text"],
                    "length": 7,
                    "language": "en",
                    "all_classes": [],
                }
            )
            + "\n",
        )

    monkeypatch.setattr("llm_mst_finder.workload._manifest_cache_root", lambda: tmp_path / "cache")
    workload_path = tmp_path / "longbench_inspect.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: longbench-inspect",
                "dataset:",
                "  type: longbench",
                f"  path: {zip_path}",
                "  configs:",
                "    - multi_news",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workload_dataset(
        workload_path,
        model_name="fake-model",
        sample_size=1,
    )

    assert report["workload"]["dataset_type"] == "longbench"
    assert report["workload"]["dataset_configs"] == ["multi_news"]
    assert report["source_summary"]["sampling_method"] == "reservoir_uniform"
    assert report["source_summary"]["configs"] == ["multi_news"]
    assert report["source_summary"]["scanned_rows"] == 1
    assert report["lengths"]["output_tokens"]["count"] == 1


def test_missing_dataset_file_raises(tmp_path: Path) -> None:
    workload_path = tmp_path / "missing_dataset.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: missing-dataset",
                "dataset:",
                "  type: jsonl",
                "  path: not-there.jsonl",
                "tokenizer: character",
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
                "tokenizer: character",
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


def test_workload_config_rejects_whitespace_tokenizer(tmp_path: Path) -> None:
    workload_path = tmp_path / "whitespace_tokenizer.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: whitespace-tokenizer",
                "dataset:",
                "  type: synthetic-fixed",
                "tokenizer: whitespace",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 8",
                "  output_len:",
                "    mode: fixed",
                "    value: 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tokenizer must not be 'whitespace'"):
        load_workload_config(workload_path)


def test_context_policy_is_loaded_from_yaml(tmp_path: Path) -> None:
    workload_path = tmp_path / "with_context_policy.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: with-context-policy",
                "dataset:",
                "  type: synthetic-fixed",
                "tokenizer: character",
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


def test_context_policy_can_omit_model_specific_max_model_len(tmp_path: Path) -> None:
    workload_path = tmp_path / "model_resolved_context_policy.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: model-resolved-context-policy",
                "dataset:",
                "  type: synthetic-fixed",
                "tokenizer: character",
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

    config = load_workload_config(workload_path)
    assert config.context_policy is not None
    assert config.context_policy.max_model_len is None
    assert config.context_policy.tokenizer_source == "vllm_model_config"


def test_prepare_workload_for_trial_uses_model_tokenizer_for_dataset_lengths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class CharacterModelTokenizer:
        model_max_length = 128

        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

    monkeypatch.setattr(
        "llm_mst_finder.model_context._resolve_vllm_tokenizer",
        lambda tokenizer_name: CharacterModelTokenizer(),
    )
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "alpha beta", "expected_output_len": 2}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_model_tokenizer.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-model-tokenizer",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    prepared = prepare_workload_for_trial(workload_path, model_name="fake/model")

    assert prepared.samples[0].prompt_len == len("alpha beta")
    assert prepared.samples[0].metadata["prompt_tokenizer_key"] == "tokenizer:fake/model"
    model_context = prepared.metadata["workload"]["model_context"]
    assert model_context["max_model_len"] == 128
    assert model_context["model_max_model_len"] == 128


def test_prepare_workload_for_trial_truncates_to_serving_max_model_len(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class CharacterModelTokenizer:
        model_max_length = 32768

        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

        def decode(self, token_ids: list[int]) -> str:
            return "x" * len(token_ids)

    monkeypatch.setattr(
        "llm_mst_finder.model_context._resolve_vllm_tokenizer",
        lambda tokenizer_name: CharacterModelTokenizer(),
    )
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "a" * 8191, "expected_output_len": 2}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_serving_context.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-serving-context",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 32768",
                "  over_limit: truncate_prompt",
            ]
        ),
        encoding="utf-8",
    )

    prepared = prepare_workload_for_trial(
        workload_path,
        model_name="fake/model",
        serving_max_model_len=8192,
    )

    assert prepared.samples[0].prompt_len == 8190
    assert prepared.samples[0].expected_output_len == 2
    assert prepared.samples[0].metadata["context_truncated"] is True
    context_policy = prepared.metadata["workload"]["context_policy"]
    assert context_policy["max_model_len"] == 8192
    assert context_policy["truncated_samples"] == 1
    model_context = prepared.metadata["workload"]["model_context"]
    assert model_context["model_max_model_len"] == 32768
    assert model_context["workload_max_model_len"] == 32768
    assert model_context["serving_max_model_len"] == 8192


def test_prepare_workload_for_trial_falls_back_to_workload_tokenizer_when_model_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "llm_mst_finder.model_context._resolve_vllm_tokenizer",
        lambda tokenizer_name: (_ for _ in ()).throw(RuntimeError("missing model tokenizer")),
    )
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "alpha beta", "expected_output_len": 2}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_fallback_tokenizer.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-fallback-tokenizer",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "context_policy:",
                "  max_model_len: 16",
                "  over_limit: fail",
            ]
        ),
        encoding="utf-8",
    )

    prepared = prepare_workload_for_trial(workload_path, model_name="missing/model")

    assert prepared.samples[0].prompt_len == len("alpha beta")
    model_context = prepared.metadata["workload"]["model_context"]
    assert model_context["fallback_used"] is True
    assert model_context["tokenizer_source"] == "fallback"
    assert "missing model tokenizer" in model_context["fallback_reason"]


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
                "tokenizer: character",
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
                "tokenizer: character",
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
                "tokenizer: character",
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


def test_jsonl_manifest_cache_invalidates_when_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("llm_mst_finder.workload._manifest_cache_root", lambda: tmp_path / "cache")
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "first", "expected_output_len": 1, "metadata": {"version": 1}}) + "\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "jsonl_cached.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: jsonl-cached-invalidates",
                "dataset:",
                "  type: jsonl",
                f"  path: {dataset_path}",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
            ]
        ),
        encoding="utf-8",
    )

    first = load_workload_samples_for_sampling_only(workload_path)
    dataset_path.write_text(
        json.dumps({"prompt": "second", "expected_output_len": 1, "metadata": {"version": 2}}) + "\n",
        encoding="utf-8",
    )
    second = load_workload_samples_for_sampling_only(workload_path)

    assert first[0].prompt == "first"
    assert second[0].prompt == "second"
    assert second[0].metadata["version"] == 2
