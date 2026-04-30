from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path

from .manifest import _merge_launch_config, _merge_search_config
from .models import (
    ExpandedExperimentJob,
    ExperimentOverride,
    HardwareConfig,
    LaunchConfig,
    OrchestratorManifest,
    ProbeConfig,
    ResourceProbeResult,
    SearchConfig,
)
from .planning import estimate_resource_probe
from .utils import slugify, stable_hash


def expand_manifest(
    manifest: OrchestratorManifest,
    *,
    mst_output_root: Path | None = None,
) -> list[ExpandedExperimentJob]:
    jobs: list[ExpandedExperimentJob] = []
    seen_ids: set[str] = set()
    for template in manifest.experiments:
        base_id = template.experiment_id or f"exp-{template.source_index + 1:03d}"
        combinations = [(model, workload) for model in template.models for workload in template.workloads]
        for model, workload in combinations:
            launch, search = _apply_overrides(
                model=model,
                workload=workload,
                hardware=template.hardware,
                launch=template.launch,
                search=template.search,
                overrides=manifest.overrides + template.overrides,
            )
            probe = _build_probe(
                model=model,
                workload=workload,
                launch=launch,
                hardware=template.hardware,
                probe_config=template.probe,
            )
            if (
                template.probe.enabled
                and template.probe.auto_gpu_count
                and probe is not None
                and probe.required_gpu_count is not None
                and probe.required_gpu_count > launch.gpu_count
            ):
                old_gpu_count = launch.gpu_count
                launch = replace(
                    launch,
                    gpu_count=probe.required_gpu_count,
                    tensor_parallel_size=(
                        probe.required_gpu_count
                        if launch.tensor_parallel_size == old_gpu_count
                        else launch.tensor_parallel_size
                    ),
                )

            if len(combinations) == 1 and template.experiment_id is not None:
                experiment_id = template.experiment_id
            else:
                combo_hash = stable_hash(
                    {
                        "base_id": base_id,
                        "model": model,
                        "workload": str(workload),
                    },
                    length=10,
                )
                experiment_id = f"{base_id}-{combo_hash}"
            if experiment_id in seen_ids:
                raise ValueError(f"expanded experiment id collision detected: {experiment_id!r}")
            seen_ids.add(experiment_id)

            model_slug = slugify(model, max_length=56)
            dataset_slug = slugify(workload.stem, max_length=56)
            server_signature_key = _server_signature_key(
                model=model,
                endpoint=template.endpoint,
                launch=launch,
            )
            server_config_slug = _server_config_slug(
                model=model,
                endpoint=template.endpoint,
                launch=launch,
                search=search,
            )
            result_root = Path("results") / "mst" if mst_output_root is None else mst_output_root
            result_dir = result_root / model_slug / dataset_slug / server_config_slug
            jobs.append(
                ExpandedExperimentJob(
                    experiment_id=experiment_id,
                    source_index=template.source_index,
                    model=model,
                    workload=workload,
                    endpoint=template.endpoint,
                    launch=launch,
                    search=search,
                    hardware=template.hardware,
                    probe=probe,
                    result_dir=result_dir,
                    model_slug=model_slug,
                    dataset_slug=dataset_slug,
                    server_config_slug=server_config_slug,
                    server_signature_key=server_signature_key,
                    server_metadata_file=template.server_metadata_file,
                )
            )
    return jobs


def _apply_overrides(
    *,
    model: str,
    workload: Path,
    hardware: HardwareConfig,
    launch: LaunchConfig,
    search: SearchConfig,
    overrides: tuple[ExperimentOverride, ...],
) -> tuple[LaunchConfig, SearchConfig]:
    resolved_launch = launch
    resolved_search = search
    for override in overrides:
        if not _override_matches(override, model=model, workload=workload, hardware=hardware):
            continue
        if override.launch is not None:
            resolved_launch = _merge_launch_config(
                resolved_launch,
                override.launch,
                field_name=f"override[{override.source_index}].launch",
            )
        if override.search is not None:
            resolved_search = _merge_search_config(
                resolved_search,
                override.search,
                field_name=f"override[{override.source_index}].search",
            )
    return resolved_launch, resolved_search


def _override_matches(
    override: ExperimentOverride,
    *,
    model: str,
    workload: Path,
    hardware: HardwareConfig,
) -> bool:
    if override.model_patterns and not _matches_any(model, override.model_patterns):
        return False
    if override.hardware_patterns and not _matches_any(hardware.name, override.hardware_patterns):
        return False
    if override.workload_patterns:
        workload_candidates = (str(workload), workload.name, workload.stem)
        if not any(_matches_any(candidate, override.workload_patterns) for candidate in workload_candidates):
            return False
    return True


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def _build_probe(
    *,
    model: str,
    workload: Path,
    launch: LaunchConfig,
    hardware: HardwareConfig,
    probe_config: ProbeConfig,
) -> ResourceProbeResult | None:
    if not probe_config.enabled:
        return None
    return estimate_resource_probe(
        model=model,
        workload=workload,
        launch=launch,
        hardware=hardware,
        probe=probe_config,
    )


def _server_signature_key(*, model: str, endpoint: str, launch: LaunchConfig) -> str:
    payload: dict[str, object] = {
        "model": model,
        "endpoint": endpoint,
        "gpu_count": launch.gpu_count,
    }
    if launch.template is not None:
        payload["launch_template"] = list(launch.template)
    else:
        payload.update(
            {
                "executable": launch.executable,
                "extra_args": list(launch.extra_args),
                "tensor_parallel_size": launch.tensor_parallel_size,
                "dtype": launch.dtype,
                "quantization": launch.quantization,
                "tokenizer_mode": launch.tokenizer_mode,
                "gpu_memory_utilization": launch.gpu_memory_utilization,
                "max_model_len": launch.max_model_len,
                "max_num_seqs": launch.max_num_seqs,
                "max_num_batched_tokens": launch.max_num_batched_tokens,
            }
        )
    return stable_hash(payload)


def _server_config_slug(*, model: str, endpoint: str, launch: LaunchConfig, search: SearchConfig) -> str:
    payload = {
        "signature": _server_signature_key(model=model, endpoint=endpoint, launch=launch),
        "slo": {
            "ttft_slo_ms": search.ttft_slo_ms,
            "tpot_slo_ms": search.tpot_slo_ms,
            "ttft_slo_field": search.ttft_slo_field,
            "tpot_slo_field": search.tpot_slo_field,
        },
        "search_mode": search.search_mode,
    }
    return f"server-{stable_hash(payload, length=12)}"
