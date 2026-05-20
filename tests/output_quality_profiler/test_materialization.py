from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from output_quality_profiler.materialization import (
    QualityMaterializationConfigError,
    assign_prompt_length_bucket,
    distribute_stratum_indices,
    load_materialization_config,
    source_request_counts,
)


def _base_config() -> dict:
    return {
        "sampling": {
            "seed": 20260520,
            "total_requests": 10000,
            "shards": 10,
            "sources": [
                {
                    "name": "sharegpt",
                    "weight": 0.5,
                    "dataset": {"type": "sharegpt", "path": "sharegpt.json"},
                },
                {
                    "name": "wildchat",
                    "weight": 0.5,
                    "dataset": {
                        "type": "hf",
                        "path": "allenai/WildChat",
                        "split": "train",
                        "conversation_field": "conversation",
                    },
                },
            ],
            "prompt_length_buckets": {
                "short": {"lt_tokens": 100},
                "medium": {"min_tokens": 100, "max_tokens": 512},
                "long": {"gt_tokens": 512},
            },
            "allocation": {
                "source": "exact",
                "bucket": "proportional_with_minimum",
                "minimum_per_source_bucket": 100,
                "allow_replacement": False,
            },
        }
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_prompt_length_bucket_boundaries_are_v1_contract() -> None:
    assert assign_prompt_length_bucket(0) == "short"
    assert assign_prompt_length_bucket(99) == "short"
    assert assign_prompt_length_bucket(100) == "medium"
    assert assign_prompt_length_bucket(512) == "medium"
    assert assign_prompt_length_bucket(513) == "long"


def test_load_materialization_config_enforces_10k_50_50_contract(tmp_path: Path) -> None:
    config = load_materialization_config(_write_config(tmp_path, _base_config()))

    assert config.total_requests == 10000
    assert config.shards == 10
    assert source_request_counts(config) == {"sharegpt": 5000, "wildchat": 5000}
    assert config.bucket_policy.bucket_for(100) == "medium"


def test_load_materialization_config_rejects_old_bucket_thresholds(tmp_path: Path) -> None:
    payload = _base_config()
    payload["sampling"]["prompt_length_buckets"] = {
        "short": {"max_tokens": 256},
        "medium": {"min_tokens": 257, "max_tokens": 1024},
        "long": {"min_tokens": 1025},
    }

    with pytest.raises(QualityMaterializationConfigError, match="short <100"):
        load_materialization_config(_write_config(tmp_path, payload))


def test_load_materialization_config_rejects_non_v1_request_count(tmp_path: Path) -> None:
    payload = _base_config()
    payload["sampling"]["total_requests"] = 100

    with pytest.raises(QualityMaterializationConfigError, match="10000"):
        load_materialization_config(_write_config(tmp_path, payload))


def test_distribute_stratum_indices_spreads_evenly_across_shards() -> None:
    shards = distribute_stratum_indices(11, 10)

    assert [len(shard) for shard in shards] == [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert shards[0] == [0, 10]
    assert shards[1] == [1]

