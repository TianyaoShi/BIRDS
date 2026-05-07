from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from dataset_workload_materializer.materialize import materialize_from_config
from llm_mst_finder.workload import prepare_workload_for_trial


def test_crosscodeeval_like_materialization_from_local_jsonl(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    rows = [
        {
            "current_file_prefix": "def alpha():\n    return",
            "completion": " 1",
            "language": "python",
            "repo_name": "owner/repo",
            "path": "src/a.py",
            "retrieved_context": "def helper():\n    return 1",
            "cursor_index": 1,
        },
        {
            "current_file_prefix": "def beta():\n    return",
            "completion": " 2",
            "lang": "python",
            "repository": "owner/repo",
            "file_path": "src/b.py",
            "sequence_index": 2,
        },
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, raw_path=raw_path, output_dir=output_dir)

    result = materialize_from_config(config_path)

    assert result["num_samples"] == 2
    assert (output_dir / "materialization_config.yaml").is_file()
    report = json.loads((output_dir / "materialization_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "shards_manifest.json").read_text(encoding="utf-8"))
    shard_rows = [
        json.loads(line)
        for line in (output_dir / "shards" / "shard_000.runner.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["rows"]["materialized"] == 2
    assert report["dataset_kind"] == "code_completion"
    assert manifest["num_shards"] == 1
    assert manifest["dataset_kind"] == "code_completion"
    assert manifest["language_counts"] == {"python": 2}
    assert len(shard_rows) == 2
    assert shard_rows[0]["metadata"]["dataset"] == "crosscodeeval"
    assert shard_rows[0]["metadata"]["dataset_kind"] == "code_completion"
    assert shard_rows[0]["metadata"]["task"] == "cross_file_materialized"
    assert shard_rows[0]["metadata"]["prompt_template"] == "plain_prefix"
    assert shard_rows[0]["metadata"]["content_hash"]
    assert "<CURRENT_FILE_PREFIX>" not in shard_rows[0]["prompt"]
    assert "Relevant repository context:" in shard_rows[0]["prompt"]
    assert shard_rows[0]["prompt"].rstrip().endswith("return")


def test_crosscodeeval_like_materialization_from_jsonl_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "part_b.jsonl").write_text(
        json.dumps({"prompt": "b prefix", "target": " b", "language": "python"}) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "part_a.jsonl").write_text(
        json.dumps({"prompt": "a prefix", "target": " a", "language": "python"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, raw_path=raw_dir, output_dir=output_dir)

    materialize_from_config(config_path)

    manifest = json.loads((output_dir / "shards_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_shards"] == 1
    assert manifest["shards"][0]["num_samples"] == 2


def test_generated_workload_yaml_loads_through_llm_mst_finder(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "prompt": "def alpha():\n    return",
                "target": " 1",
                "language": "python",
                "repo": "owner/repo",
                "path": "src/a.py",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, raw_path=raw_path, output_dir=output_dir)
    materialize_from_config(config_path)

    prepared = prepare_workload_for_trial(
        output_dir / "workload_yamls" / "shard_000.yaml",
        model_name="fake-model",
    )

    assert len(prepared.samples) == 1
    assert prepared.samples[0].metadata["dataset"] == "crosscodeeval"
    assert prepared.samples[0].metadata["sampling_entry_selection"] == "sequential"
    assert prepared.samples[0].metadata["shard_id"] == "shard_000"


def test_longbench_materialization_rejects_excluded_tasks(tmp_path: Path) -> None:
    raw_path = _write_longbench_zip(
        tmp_path,
        {
            "multi_news": [_longbench_row("multi-news-0", "multi_news", "en")],
            "trec": [_longbench_row("trec-0", "trec", "en")],
        },
    )
    config_path = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        profile="medium_output_summarization",
        configs=["multi_news", "trec"],
        samples_per_task=1,
    )

    with pytest.raises(ValueError, match="exclude tasks: trec"):
        materialize_from_config(config_path)


def test_longbench_materialization_is_deterministic_and_reports_profile_metadata(tmp_path: Path) -> None:
    raw_path = _write_longbench_zip(
        tmp_path,
        {
            "multi_news": [
                _longbench_row("multi-news-0", "multi_news", "en", suffix="alpha"),
                _longbench_row("multi-news-1", "multi_news", "en", suffix="beta"),
                _longbench_row("multi-news-2", "multi_news", "en", suffix="gamma"),
            ],
            "qmsum": [
                _longbench_row("qmsum-0", "qmsum", "en", suffix="delta"),
                _longbench_row("qmsum-1", "qmsum", "en", suffix="epsilon"),
                _longbench_row("qmsum-2", "qmsum", "en", suffix="zeta"),
            ],
        },
    )
    config_a = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out_a",
        profile="medium_output_summarization",
        configs=["multi_news", "qmsum"],
        samples_per_task=2,
        config_name="longbench_a.yaml",
    )
    config_b = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out_b",
        profile="medium_output_summarization",
        configs=["multi_news", "qmsum"],
        samples_per_task=2,
        config_name="longbench_b.yaml",
    )

    result_a = materialize_from_config(config_a)
    result_b = materialize_from_config(config_b)

    assert result_a["num_samples"] == 4
    assert result_b["num_samples"] == 4

    shard_rows_a = [
        json.loads(line)
        for line in (tmp_path / "out_a" / "shards" / "shard_000.runner.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    shard_rows_b = [
        json.loads(line)
        for line in (tmp_path / "out_b" / "shards" / "shard_000.runner.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert shard_rows_a == shard_rows_b

    report = json.loads((tmp_path / "out_a" / "materialization_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out_a" / "shards_manifest.json").read_text(encoding="utf-8"))

    assert report["dataset_kind"] == "long_context_nlp"
    assert report["profile"] == "medium_output_summarization"
    assert report["selected_tasks"] == ["multi_news", "qmsum"]
    assert report["rows"]["drops"]["not_selected_by_sampling"] == 2
    assert report["profile_summaries"]["medium_output_summarization"]["count"] == 4
    assert report["profile_summaries"]["medium_output_summarization"]["task_counts"] == {
        "multi_news": 2,
        "qmsum": 2,
    }
    assert report["task_summaries"]["multi_news"]["count"] == 2
    assert report["task_summaries"]["multi_news"]["workload_type"] == "summarization"
    assert report["task_summaries"]["multi_news"]["output_regime"] == "medium"
    assert manifest["profile_summaries"]["medium_output_summarization"]["count"] == 4
    assert manifest["task_summaries"]["qmsum"]["count"] == 2

    first_metadata = shard_rows_a[0]["metadata"]
    assert first_metadata["dataset"] == "longbench"
    assert first_metadata["dataset_kind"] == "long_context_nlp"
    assert first_metadata["profile"] == "medium_output_summarization"
    assert first_metadata["task"] in {"multi_news", "qmsum"}
    assert first_metadata["workload_type"] == "summarization"
    assert first_metadata["output_regime"] == "medium"
    assert first_metadata["language"] == "en"
    assert first_metadata["prompt_token_count"] >= 4
    assert first_metadata["target_token_count"] >= 1


def test_longbench_generated_workload_yaml_loads_through_llm_mst_finder(tmp_path: Path) -> None:
    raw_path = _write_longbench_zip(
        tmp_path,
        {
            "qasper": [_longbench_row("qasper-0", "qasper", "en", suffix="paper")],
        },
    )
    config_path = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        profile="short_answer_document_qa",
        configs=["qasper"],
        samples_per_task=1,
    )
    materialize_from_config(config_path)

    prepared = prepare_workload_for_trial(
        tmp_path / "out" / "workload_yamls" / "shard_000.yaml",
        model_name="fake-model",
    )

    assert len(prepared.samples) == 1
    assert prepared.samples[0].metadata["dataset"] == "longbench"
    assert prepared.samples[0].metadata["profile"] == "short_answer_document_qa"
    assert prepared.samples[0].metadata["task"] == "qasper"
    assert prepared.samples[0].metadata["sampling_entry_selection"] == "sequential"
    assert prepared.samples[0].metadata["shard_id"] == "shard_000"


def test_longbench_summarization_with_empty_input_synthesizes_task_instruction(tmp_path: Path) -> None:
    raw_path = _write_longbench_zip(
        tmp_path,
        {
            "multi_news": [
                _longbench_row("multi-news-0", "multi_news", "en", suffix="brief", empty_input=True),
            ],
        },
    )
    config_path = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        profile="medium_output_summarization",
        configs=["multi_news"],
        samples_per_task=1,
    )

    materialize_from_config(config_path)

    shard_row = json.loads(
        (tmp_path / "out" / "shards" / "shard_000.runner.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "Summarize the document in about" in shard_row["prompt"]
    assert shard_row["metadata"]["longbench_task_input_source"] == "synthesized_summarization_instruction"


def test_longbench_epoch_shuffle_expansion_preserves_unique_corpus_metadata(tmp_path: Path) -> None:
    raw_path = _write_longbench_zip(
        tmp_path,
        {
            "qasper": [
                _longbench_row("qasper-0", "qasper", "en", suffix="paper-0"),
                _longbench_row("qasper-1", "qasper", "en", suffix="paper-1"),
                _longbench_row("qasper-2", "qasper", "en", suffix="paper-2"),
            ],
        },
    )
    config_path = _write_longbench_config(
        tmp_path,
        raw_path=raw_path,
        output_dir=tmp_path / "out",
        profile="short_answer_document_qa",
        configs=["qasper"],
        samples_per_task=3,
        repeat_policy="epoch_shuffle",
        target_samples=7,
    )

    result = materialize_from_config(config_path)

    assert result["num_samples"] == 7
    shard_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "shards" / "shard_000.runner.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    original_ids = [row["metadata"]["original_sample_id"] for row in shard_rows]
    assert len(set(original_ids[:3])) == 3
    assert len(set(original_ids[3:6])) == 3
    assert shard_rows[0]["metadata"]["repeat_policy"] == "epoch_shuffle"
    assert shard_rows[0]["metadata"]["unique_sample_count"] == 3
    assert shard_rows[0]["metadata"]["expanded_sample_count"] == 7
    assert shard_rows[0]["metadata"]["epoch_shuffle_seed"] == 11
    assert {row["metadata"]["epoch_index"] for row in shard_rows} == {0, 1, 2}

    report = json.loads((tmp_path / "out" / "materialization_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "shards_manifest.json").read_text(encoding="utf-8"))
    assert report["rows"]["materialized"] == 7
    assert report["sampling"]["unique_sample_ids"] == 3
    assert report["sampling"]["repeat_policy"] == "epoch_shuffle"
    assert manifest["unique_sample_ids"] == 3
    assert manifest["repeat_policy"] == "epoch_shuffle"


def _write_config(tmp_path: Path, *, raw_path: Path, output_dir: Path) -> Path:
    config = {
        "name": "crosscodeeval_tiny",
        "dataset": {
            "name": "crosscodeeval",
            "raw_path": str(raw_path),
            "split": "test",
            "mode": "cross_file_materialized",
            "prompt_template": "plain_prefix",
        },
        "tokenization": {"tokenizer": "whitespace"},
        "filtering": {
            "min_prompt_tokens": 1,
            "max_prompt_tokens": 128,
            "min_target_tokens": 1,
            "max_target_tokens": 8,
        },
        "sampling": {"seed": 7, "burst_size": 2},
        "request": {"temperature": 0.0, "stream": True, "ignore_eos": False},
        "sharding": {"output_dir": str(output_dir), "samples_per_shard": 8},
        "workload_yaml": {
            "context_policy": {
                "max_model_len": 512,
                "tokenizer_source": "workload_tokenizer",
                "unsafe_allow_workload_tokenizer_for_real_datasets": True,
                "over_limit": "fail",
            }
        },
    }
    config_path = tmp_path / "materialize.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_longbench_config(
    tmp_path: Path,
    *,
    raw_path: Path,
    output_dir: Path,
    profile: str,
    configs: list[str],
    samples_per_task: int,
    repeat_policy: str | None = None,
    target_samples: int | None = None,
    config_name: str = "longbench_materialize.yaml",
) -> Path:
    config = {
        "name": f"longbench_{profile}",
        "dataset": {
            "name": "longbench",
            "raw_path": str(raw_path),
            "split": "test",
            "profile": profile,
            "configs": configs,
        },
        "tokenization": {"tokenizer": "whitespace"},
        "filtering": {
            "min_prompt_tokens": 4,
            "max_prompt_tokens": 512,
            "min_target_tokens": 1,
            "max_target_tokens": 64,
        },
        "sampling": {
            "seed": 11,
            "policy": "task_uniform",
            "samples_per_task": samples_per_task,
            "burst_size": 1,
        },
        "request": {"temperature": 0.0, "stream": True, "ignore_eos": False},
        "sharding": {"output_dir": str(output_dir), "samples_per_shard": 8},
        "workload_yaml": {
            "context_policy": {
                "max_model_len": 1024,
                "tokenizer_source": "workload_tokenizer",
                "unsafe_allow_workload_tokenizer_for_real_datasets": True,
                "over_limit": "fail",
            }
        },
    }
    if repeat_policy is not None:
        config["sampling"]["repeat_policy"] = repeat_policy
    if target_samples is not None:
        config["sampling"]["target_samples"] = target_samples
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_longbench_zip(tmp_path: Path, rows_by_task: dict[str, list[dict[str, object]]]) -> Path:
    zip_path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for task_name, rows in rows_by_task.items():
            archive.writestr(
                f"data/{task_name}.jsonl",
                "\n".join(json.dumps(row) for row in rows) + "\n",
            )
    return zip_path


def _longbench_row(
    row_id: str,
    task_name: str,
    language: str,
    *,
    suffix: str = "sample",
    empty_input: bool = False,
) -> dict[str, object]:
    return {
        "_id": row_id,
        "dataset": task_name,
        "input": "" if empty_input else f"Summarize the material for {suffix}.",
        "context": f"context tokens for {task_name} {suffix} include several informative words",
        "answers": [f"reference answer {suffix}", f"longer reference answer {suffix}"],
        "length": 7,
        "language": language,
        "all_classes": [],
    }
