from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from local_orchestrator.matrix import _build_probe, _dataset_slug, _server_signature_key
from local_orchestrator.utils import slugify, stable_hash

from .models import QualityExperimentJob, QualityRunManifest


def expand_quality_manifest(
    manifest: QualityRunManifest,
    *,
    run_root: Path | None = None,
) -> list[QualityExperimentJob]:
    jobs: list[QualityExperimentJob] = []
    seen_ids: set[str] = set()
    result_root = run_root / "responses" if run_root is not None else Path("results") / "quality" / "responses"
    for template in manifest.experiments:
        base_id = template.experiment_id or f"quality-exp-{template.source_index + 1:03d}"
        combinations = [(model, workload) for model in template.models for workload in template.workloads]
        for model, workload in combinations:
            launch = template.launch
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
                job_id = template.experiment_id
            else:
                combo_hash = stable_hash(
                    {
                        "base_id": base_id,
                        "model": model,
                        "workload": str(workload),
                    },
                    length=10,
                )
                job_id = f"{base_id}-{combo_hash}"
            if job_id in seen_ids:
                raise ValueError(f"expanded quality job id collision detected: {job_id!r}")
            seen_ids.add(job_id)

            model_slug = slugify(model, max_length=56)
            shard_id = workload.stem
            server_signature_key = _server_signature_key(
                model=model,
                endpoint=template.endpoint,
                launch=launch,
            )
            jobs.append(
                QualityExperimentJob(
                    job_id=job_id,
                    source_index=template.source_index,
                    model=model,
                    workload=workload,
                    endpoint=template.endpoint,
                    launch=launch,
                    generation=template.generation,
                    hardware=template.hardware,
                    probe=template.probe,
                    result_dir=result_root / model_slug / _dataset_slug(workload),
                    model_slug=model_slug,
                    shard_id=shard_id,
                    server_signature_key=server_signature_key,
                )
            )
    return jobs

