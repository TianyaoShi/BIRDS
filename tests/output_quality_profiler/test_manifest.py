from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from output_quality_profiler.manifest import QualityManifestValidationError, load_quality_manifest


def _write_workload(tmp_path: Path, name: str = "shard_000.yaml") -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "name": "quality-shard",
                "dataset": {"type": "jsonl", "path": "../shards/shard_000.runner.jsonl"},
                "sampling": {
                    "seed": 42,
                    "num_requests": 1,
                    "entry_selection": "sequential",
                    "prompt_len": {"mode": "from_dataset"},
                    "output_len": {"mode": "natural_until_eos", "max_tokens": 32768},
                },
                "request": {"stream": True, "temperature": 0.6, "ignore_eos": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_manifest(workload: Path) -> dict:
    return {
        "run": {
            "run_id": "quality-run",
            "output_root": "results/quality",
            "default_endpoint": "/v1/chat/completions",
            "allowed_gpu_ids": [0, 1],
            "max_active_gpus": 1,
            "keep_one_gpu_spare": True,
        },
        "launch": {"gpu_count": 1, "tensor_parallel_size": 1, "max_model_len": 32768},
        "generation": {
            "concurrency_source": "mst_fraction",
            "concurrency_mst_fraction": 0.40,
            "preserve_request_order": True,
            "include_prompt_text": True,
            "decoding": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "n": 1,
                "max_tokens": 32768,
                "max_tokens_policy": "model_context_minus_prompt_buffer",
                "prompt_token_buffer": 128,
                "extra_body": {},
            },
        },
        "experiments": [
            {
                "id": "quality-exp",
                "model": "Qwen/Qwen3-8B",
                "workload": str(workload),
            }
        ],
    }


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_quality_manifest_accepts_v1_defaults(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path)
    manifest = load_quality_manifest(_write_manifest(tmp_path, _base_manifest(workload)))

    assert manifest.run.run_id == "quality-run"
    assert manifest.generation.decoding.temperature == pytest.approx(0.6)
    assert manifest.generation.decoding.top_p == pytest.approx(0.95)
    assert manifest.generation.decoding.top_k == 20
    assert manifest.generation.decoding.min_p == pytest.approx(0.0)
    assert manifest.generation.decoding.n == 1
    assert manifest.generation.decoding.max_tokens == 32768
    assert manifest.generation.concurrency_source == "mst_fraction"
    assert manifest.generation.concurrency_mst_fraction == pytest.approx(0.40)
    assert manifest.generation.include_prompt_text is True
    assert manifest.experiments[0].workloads == (workload.resolve(),)


def test_load_quality_manifest_rejects_search_section(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path)
    payload = _base_manifest(workload)
    payload["search"] = {"search_mode": "open-loop"}

    with pytest.raises(QualityManifestValidationError, match="unknown keys"):
        load_quality_manifest(_write_manifest(tmp_path, payload))


def test_load_quality_manifest_rejects_multiple_responses_per_prompt(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path)
    payload = _base_manifest(workload)
    payload["generation"]["decoding"]["n"] = 2

    with pytest.raises(QualityManifestValidationError, match="must be 1"):
        load_quality_manifest(_write_manifest(tmp_path, payload))


def test_load_quality_manifest_rejects_prompt_omission(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path)
    payload = _base_manifest(workload)
    payload["generation"]["include_prompt_text"] = False

    with pytest.raises(QualityManifestValidationError, match="include_prompt_text"):
        load_quality_manifest(_write_manifest(tmp_path, payload))


def test_load_quality_manifest_rejects_non_40_percent_mst_fraction(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path)
    payload = _base_manifest(workload)
    payload["generation"]["concurrency_mst_fraction"] = 0.5

    with pytest.raises(QualityManifestValidationError, match="0.40"):
        load_quality_manifest(_write_manifest(tmp_path, payload))

