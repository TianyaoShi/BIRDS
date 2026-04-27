from __future__ import annotations

from pathlib import Path

import yaml

from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_workload(tmp_path: Path, name: str) -> Path:
    workload_path = tmp_path / name
    workload_path.write_text("name: stub\n", encoding="utf-8")
    return workload_path


def test_expand_manifest_is_deterministic_and_uses_expected_layout(tmp_path: Path) -> None:
    workload_a = _write_workload(tmp_path, "sharegpt.yaml")
    workload_b = _write_workload(tmp_path, "wildchat.yaml")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {
                "allowed_gpu_ids": [0, 1, 2, 3],
                "max_active_gpus": 3,
            },
            "launch": {
                "executable": "vllm",
                "gpu_count": 1,
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 2048,
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
                    "id": "matrix",
                    "models": ["model-a", "model-b"],
                    "workloads": [str(workload_a), str(workload_b)],
                    "endpoint": "/v1/chat/completions",
                }
            ],
        },
    )

    manifest = load_manifest(manifest_path)
    first = expand_manifest(manifest)
    second = expand_manifest(manifest)

    assert len(first) == 4
    assert [job.experiment_id for job in first] == [job.experiment_id for job in second]
    assert [str(job.result_dir) for job in first] == [str(job.result_dir) for job in second]
    assert [job.server_signature_key for job in first] == [job.server_signature_key for job in second]

    for job in first:
        parts = job.result_dir.parts
        assert parts[0] == "results"
        assert parts[1] == "mst"
        assert job.server_config_slug.startswith("server-")
