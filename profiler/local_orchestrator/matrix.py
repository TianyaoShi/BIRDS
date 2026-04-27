from __future__ import annotations

from pathlib import Path

from .models import ExpandedExperimentJob, LaunchConfig, OrchestratorManifest, SearchConfig
from .utils import slugify, stable_hash


def expand_manifest(manifest: OrchestratorManifest) -> list[ExpandedExperimentJob]:
    jobs: list[ExpandedExperimentJob] = []
    seen_ids: set[str] = set()
    for template in manifest.experiments:
        base_id = template.experiment_id or f"exp-{template.source_index + 1:03d}"
        combinations = [(model, workload) for model in template.models for workload in template.workloads]
        for model, workload in combinations:
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
                launch=template.launch,
            )
            server_config_slug = _server_config_slug(
                model=model,
                endpoint=template.endpoint,
                launch=template.launch,
                search=template.search,
            )
            result_dir = Path("results") / "mst" / model_slug / dataset_slug / server_config_slug
            jobs.append(
                ExpandedExperimentJob(
                    experiment_id=experiment_id,
                    source_index=template.source_index,
                    model=model,
                    workload=workload,
                    endpoint=template.endpoint,
                    launch=template.launch,
                    search=template.search,
                    result_dir=result_dir,
                    model_slug=model_slug,
                    dataset_slug=dataset_slug,
                    server_config_slug=server_config_slug,
                    server_signature_key=server_signature_key,
                    server_metadata_file=template.server_metadata_file,
                )
            )
    return jobs


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
