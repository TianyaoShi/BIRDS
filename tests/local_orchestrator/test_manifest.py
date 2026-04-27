from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from local_orchestrator.manifest import ManifestValidationError, load_manifest


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_workload(tmp_path: Path, name: str) -> Path:
    workload_path = tmp_path / name
    workload_path.write_text("name: stub\n", encoding="utf-8")
    return workload_path


def test_load_manifest_accepts_valid_structured_config(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "workload.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {
                "allowed_gpu_ids": [0, 1, 2, 3],
                "max_active_gpus": 3,
                "default_endpoint": "/v1/chat/completions",
            },
            "launch": {
                "executable": "vllm",
                "tensor_parallel_size": 1,
                "gpu_count": 1,
                "max_num_seqs": 128,
            },
            "search": {
                "search_mode": "open-loop",
                "ttft_slo_ms": 2000,
                "tpot_slo_ms": 80,
                "ttft_slo_field": "ttft_p90_ms",
                "tpot_slo_field": "tpot_p90_ms",
            },
            "experiments": [
                {
                    "id": "exp-chat",
                    "model": "google/gemma-4-E4B-it",
                    "workload": str(workload),
                }
            ],
        },
    )

    manifest = load_manifest(manifest_path)
    assert manifest.manifest_path == manifest_path.resolve()
    assert manifest.run.max_active_gpus == 3
    assert len(manifest.experiments) == 1
    assert manifest.experiments[0].experiment_id == "exp-chat"


def test_load_manifest_rejects_missing_workload_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "experiments": [
                {
                    "model": "google/gemma-4-E4B-it",
                    "workload": str(missing),
                }
            ]
        },
    )

    with pytest.raises(ManifestValidationError, match="missing workload path"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_duplicate_explicit_ids(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "workload.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "experiments": [
                {"id": "dup", "model": "m1", "workload": str(workload)},
                {"id": "dup", "model": "m2", "workload": str(workload)},
            ]
        },
    )

    with pytest.raises(ManifestValidationError, match="duplicate experiment id"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_invalid_slo_field(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "workload.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "search": {"ttft_slo_field": "ttft_p95_ms"},
            "experiments": [
                {
                    "model": "google/gemma-4-E4B-it",
                    "workload": str(workload),
                }
            ],
        },
    )

    with pytest.raises(ManifestValidationError, match="unsupported ttft_slo_field"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_mixed_template_and_structured_launch(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "workload.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "experiments": [
                {
                    "model": "google/gemma-4-E4B-it",
                    "workload": str(workload),
                    "launch": {
                        "template": "vllm serve {model} --port {port}",
                        "dtype": "float16",
                    },
                }
            ]
        },
    )

    with pytest.raises(ManifestValidationError, match="choose one launch style"):
        load_manifest(manifest_path)
