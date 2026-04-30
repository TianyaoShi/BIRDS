from __future__ import annotations

import math
import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping

from llm_mst_finder.workload import LengthSpec, load_workload_config

from .models import HardwareConfig, LaunchConfig, ProbeConfig, ResourceProbeResult


_SIZE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:E)?(\d+(?:\.\d+)?)\s*([BM])(?![A-Za-z0-9])", re.IGNORECASE)


def estimate_resource_probe(
    *,
    model: str,
    workload: Path,
    launch: LaunchConfig,
    hardware: HardwareConfig,
    probe: ProbeConfig,
) -> ResourceProbeResult:
    warnings: list[str] = []
    model_params_b = infer_model_size_billions(model, model_size_overrides_b=probe.model_size_overrides_b)
    if model_params_b is None:
        warnings.append(f"could not infer model parameter count from {model!r}")

    context_tokens = _workload_context_tokens(workload, probe=probe, warnings=warnings)
    model_config = load_cached_hf_config(model)
    dtype_bytes = _dtype_bytes(launch=launch, model_config=model_config)
    weight_gb = None if model_params_b is None else model_params_b * dtype_bytes * 0.9313225746
    kv_cache_gb = None
    if model_params_b is not None:
        kv_cache_gb = (
            context_tokens
            * model_params_b
            * 0.000065
            * (dtype_bytes / 2.0)
            * probe.kv_cache_request_count
        )

    required_gb = None
    if weight_gb is not None and kv_cache_gb is not None:
        required_gb = (
            weight_gb + probe.activation_memory_gb + kv_cache_gb
        ) * probe.memory_safety_factor

    gpu_utilization = launch.gpu_memory_utilization or hardware.gpu_memory_utilization
    usable_per_gpu = None
    if hardware.gpu_memory_gb is not None:
        usable_per_gpu = hardware.gpu_memory_gb * gpu_utilization

    required_gpu_count = None
    if required_gb is not None and usable_per_gpu is not None:
        required_gpu_count = max(1, math.ceil(required_gb / usable_per_gpu))
    elif hardware.gpu_memory_gb is None:
        warnings.append("hardware.gpu_memory_gb is unset; cannot estimate required GPU count")

    return ResourceProbeResult(
        hardware_name=hardware.name,
        gpu_memory_gb=hardware.gpu_memory_gb,
        model_params_b=model_params_b,
        estimated_weight_gb=weight_gb,
        estimated_activation_gb=probe.activation_memory_gb,
        estimated_kv_cache_gb=kv_cache_gb,
        estimated_required_gb=required_gb,
        usable_memory_per_gpu_gb=usable_per_gpu,
        required_gpu_count=required_gpu_count,
        context_tokens=context_tokens,
        warnings=tuple(warnings),
    )


def infer_model_size_billions(
    model: str,
    *,
    model_size_overrides_b: Mapping[str, float] | None = None,
) -> float | None:
    for pattern, size_b in (model_size_overrides_b or {}).items():
        if fnmatchcase(model.lower(), pattern.lower()):
            return float(size_b)
    matches = _SIZE_PATTERN.findall(model.replace("-", " ").replace("_", " "))
    if not matches:
        matches = _SIZE_PATTERN.findall(model)
    if not matches:
        return None
    value, unit = matches[0]
    scale = 1.0 if unit.lower() == "b" else 0.001
    return float(value) * scale


def _workload_context_tokens(workload: Path, *, probe: ProbeConfig, warnings: list[str]) -> int:
    try:
        config = load_workload_config(workload)
    except Exception as exc:
        warnings.append(f"failed to inspect workload context: {exc}")
        return probe.default_context_tokens

    if config.context_policy is not None and config.context_policy.max_model_len is not None:
        return config.context_policy.max_model_len

    prompt_tokens = _max_length_spec(config.sampling.prompt_len)
    output_tokens = _max_length_spec(config.sampling.output_len)
    if prompt_tokens is not None and output_tokens is not None:
        return prompt_tokens + output_tokens

    warnings.append(
        "workload uses from_dataset lengths without context_policy.max_model_len; "
        f"using default_context_tokens={probe.default_context_tokens}"
    )
    return probe.default_context_tokens


def _max_length_spec(spec: LengthSpec) -> int | None:
    if spec.mode == "fixed":
        return spec.value
    if spec.mode == "bucketed":
        return max(bucket.value for bucket in spec.buckets)
    return None


def _dtype_bytes(*, launch: LaunchConfig, model_config: dict[str, Any] | None = None) -> float:
    quantization = (launch.quantization or "").lower()
    if not quantization and model_config is not None:
        quant_config = model_config.get("quantization_config")
        if isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")
            if isinstance(quant_method, str):
                quantization = quant_method.lower()
    if any(token in quantization for token in ("int4", "4bit", "awq", "gptq")):
        return 0.5
    if "mxfp4" in quantization:
        return 0.5
    if any(token in quantization for token in ("int8", "8bit")):
        return 1.0

    dtype = (launch.dtype or "").lower()
    if not dtype and model_config is not None:
        config_dtype = model_config.get("torch_dtype")
        if isinstance(config_dtype, str):
            dtype = config_dtype.lower()
    if dtype in {"float32", "fp32"}:
        return 4.0
    if dtype in {"float8", "fp8"}:
        return 1.0
    return 2.0


def load_cached_hf_config(model: str) -> dict[str, Any] | None:
    if "/" not in model:
        return None
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_root / f"models--{model.replace('/', '--')}"
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_path = model_cache / "refs" / "main"
    candidates: list[Path] = []
    try:
        ref = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        ref = ""
    if ref:
        candidates.append(snapshots_dir / ref / "config.json")
    candidates.extend(sorted(snapshots_dir.glob("*/config.json")))

    for config_path in candidates:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _model_size_billions(model: str, *, probe: ProbeConfig) -> float | None:
    return infer_model_size_billions(model, model_size_overrides_b=probe.model_size_overrides_b)


def _load_cached_hf_config(model: str) -> dict[str, Any] | None:
    return load_cached_hf_config(model)
