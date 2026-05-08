from __future__ import annotations

import json
from pathlib import Path

from local_orchestrator.models import HardwareConfig, LaunchConfig, ProbeConfig
from local_orchestrator.planning import estimate_resource_probe


def _write_workload(tmp_path: Path) -> Path:
    workload = tmp_path / "workload.yaml"
    workload.write_text(
        """
name: synthetic
dataset:
  type: synthetic-fixed
tokenizer: character
sampling:
  seed: 1
  num_requests: 10
  prompt_len:
    mode: fixed
    value: 512
  output_len:
    mode: fixed
    value: 128
""".lstrip(),
        encoding="utf-8",
    )
    return workload


def test_probe_infers_million_suffix_models(tmp_path: Path) -> None:
    result = estimate_resource_probe(
        model="facebook/opt-125m",
        workload=_write_workload(tmp_path),
        launch=LaunchConfig(dtype="float16"),
        hardware=HardwareConfig(name="l40", gpu_memory_gb=48),
        probe=ProbeConfig(),
    )

    assert result.model_params_b == 0.125
    assert result.required_gpu_count == 1
    assert result.warnings == ()


def test_probe_uses_local_hf_quantization_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    snapshot = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--gpt-oss-20b"
        / "snapshots"
        / "abc"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "mxfp4"}}),
        encoding="utf-8",
    )

    quantized = estimate_resource_probe(
        model="openai/gpt-oss-20b",
        workload=_write_workload(tmp_path),
        launch=LaunchConfig(),
        hardware=HardwareConfig(name="l40", gpu_memory_gb=48),
        probe=ProbeConfig(),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    fp16 = estimate_resource_probe(
        model="openai/gpt-oss-20b",
        workload=_write_workload(tmp_path),
        launch=LaunchConfig(dtype="float16"),
        hardware=HardwareConfig(name="l40", gpu_memory_gb=48),
        probe=ProbeConfig(),
    )

    assert quantized.estimated_required_gb is not None
    assert fp16.estimated_required_gb is not None
    assert quantized.estimated_required_gb < fp16.estimated_required_gb
    assert quantized.required_gpu_count == 1
