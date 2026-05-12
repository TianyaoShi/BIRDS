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


def test_expand_manifest_can_namespace_mst_results_by_run(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "sharegpt.yaml")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "experiments": [
                {
                    "id": "matrix",
                    "model": "model-a",
                    "workload": str(workload),
                    "endpoint": "/v1/chat/completions",
                }
            ],
        },
    )

    manifest = load_manifest(manifest_path)
    jobs = expand_manifest(manifest, mst_output_root=tmp_path / "results" / "mst" / "run-1")

    assert len(jobs) == 1
    assert jobs[0].result_dir.parts[-6:-4] == ("results", "mst")
    assert jobs[0].result_dir.parts[-4] == "run-1"


def test_expand_manifest_disambiguates_generated_shard_workloads(tmp_path: Path) -> None:
    code_a = tmp_path / "code_workloads" / "crosscodeeval_rg1" / "workload_yamls" / "shard_000.yaml"
    code_b = tmp_path / "code_workloads" / "repobench_8k" / "workload_yamls" / "shard_000.yaml"
    code_a.parent.mkdir(parents=True)
    code_b.parent.mkdir(parents=True)
    code_a.write_text("name: crosscodeeval\n", encoding="utf-8")
    code_b.write_text("name: repobench\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {"mst_output_root": str(tmp_path / "results" / "mst")},
            "experiments": [
                {
                    "id": "matrix",
                    "model": "google/gemma-4-26b-a4b-it",
                    "workloads": [str(code_a), str(code_b)],
                    "endpoint": "/v1/completions",
                }
            ],
        },
    )

    jobs = expand_manifest(load_manifest(manifest_path))

    assert len(jobs) == 2
    assert {job.dataset_slug for job in jobs} == {
        "crosscodeeval-rg1-shard-000",
        "repobench-8k-shard-000",
    }
    assert len({job.result_dir for job in jobs}) == 2


def test_expand_manifest_applies_selector_overrides_and_probe_auto_gpu_count(tmp_path: Path) -> None:
    workload = _write_workload(tmp_path, "synthetic_512_128.yaml")
    workload.write_text(
        yaml.safe_dump(
            {
                "name": "synthetic_512_128",
                "dataset": {"type": "synthetic-fixed"},
                "tokenizer": "character",
                "sampling": {
                    "seed": 1,
                    "num_requests": 100,
                    "prompt_len": {"mode": "fixed", "value": 512},
                    "output_len": {"mode": "fixed", "value": 128},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path,
        {
            "run": {
                "allowed_gpu_ids": [0, 1, 2, 3],
                "max_active_gpus": 4,
                "keep_one_gpu_spare": False,
            },
            "hardware": {
                "name": "a100-test",
                "gpu_memory_gb": 10,
                "gpu_memory_utilization": 0.9,
            },
            "probe": {
                "auto_gpu_count": True,
                "activation_memory_gb": 2,
                "memory_safety_factor": 1.2,
            },
            "search": {"search_mode": "open-loop", "max_request_rate": 2},
            "overrides": [
                {
                    "match": {"model": "*8B*"},
                    "search": {"max_request_rate": 8, "max_binary_steps": 7},
                },
                {
                    "match": {"workload": "*synthetic*"},
                    "search": {"ttft_slo_ms": 1500, "tpot_slo_ms": 100},
                },
            ],
            "experiments": [
                {
                    "id": "matrix",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "workload": str(workload),
                }
            ],
        },
    )

    job = expand_manifest(load_manifest(manifest_path))[0]

    assert job.search.max_request_rate == 8
    assert job.search.max_binary_steps == 7
    assert job.search.ttft_slo_ms == 1500
    assert job.search.tpot_slo_ms == 100
    assert job.probe is not None
    assert job.probe.model_params_b == 8
    assert job.probe.required_gpu_count is not None
    assert job.launch.gpu_count == job.probe.required_gpu_count
    assert job.launch.tensor_parallel_size == job.probe.required_gpu_count
