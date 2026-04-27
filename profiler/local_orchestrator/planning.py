from __future__ import annotations

import math
import re
from fnmatch import fnmatchcase
from pathlib import Path

from llm_mst_finder.workload import LengthSpec, load_workload_config

from .models import HardwareConfig, LaunchConfig, ProbeConfig, ResourceProbeResult


_SIZE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:E)?(\d+(?:\.\d+)?)\s*B(?![A-Za-z0-9])", re.IGNORECASE)


def estimate_resource_probe(
    *,
    model: str,
    workload: Path,
    launch: LaunchConfig,
    hardware: HardwareConfig,
    probe: ProbeConfig,
) -> ResourceProbeResult:
    warnings: list[str] = []
    model_params_b = _model_size_billions(model, probe=probe)
    if model_params_b is None:
        warnings.append(f"could not infer model parameter count from {model!r}")

    context_tokens = _workload_context_tokens(workload, probe=probe, warnings=warnings)
    dtype_bytes = _dtype_bytes(launch=launch)
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


def _model_size_billions(model: str, *, probe: ProbeConfig) -> float | None:
    for pattern, size_b in probe.model_size_overrides_b.items():
        if fnmatchcase(model.lower(), pattern.lower()):
            return float(size_b)
    matches = _SIZE_PATTERN.findall(model.replace("-", " ").replace("_", " "))
    if not matches:
        matches = _SIZE_PATTERN.findall(model)
    if not matches:
        return None
    return float(matches[-1])


def _workload_context_tokens(workload: Path, *, probe: ProbeConfig, warnings: list[str]) -> int:
    try:
        config = load_workload_config(workload)
    except Exception as exc:
        warnings.append(f"failed to inspect workload context: {exc}")
        return probe.default_context_tokens

    if config.context_policy is not None:
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


def _dtype_bytes(*, launch: LaunchConfig) -> float:
    quantization = (launch.quantization or "").lower()
    if any(token in quantization for token in ("int4", "4bit", "awq", "gptq")):
        return 0.5
    if any(token in quantization for token in ("int8", "8bit")):
        return 1.0

    dtype = (launch.dtype or "").lower()
    if dtype in {"float32", "fp32"}:
        return 4.0
    if dtype in {"float8", "fp8"}:
        return 1.0
    return 2.0
