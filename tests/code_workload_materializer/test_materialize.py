from __future__ import annotations

import json
from pathlib import Path

import yaml

from code_workload_materializer.materialize import materialize_from_config
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
    assert manifest["num_shards"] == 1
    assert manifest["language_counts"] == {"python": 2}
    assert len(shard_rows) == 2
    assert shard_rows[0]["metadata"]["dataset"] == "crosscodeeval"
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
