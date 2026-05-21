from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from local_orchestrator.manifest import (
    _merge_hardware_config,
    _merge_launch_config,
    _merge_probe_config,
    _parse_run_config,
    _parse_slurm_config,
)
from local_orchestrator.models import HardwareConfig, LaunchConfig, ProbeConfig

from .models import (
    QualityDecodingConfig,
    QualityExperimentTemplate,
    QualityGenerationConfig,
    QualityRunManifest,
)


class QualityManifestValidationError(ValueError):
    pass


_TOP_LEVEL_KEYS = {"run", "slurm", "hardware", "probe", "launch", "generation", "experiments"}
_GENERATION_KEYS = {
    "request_timeout_s",
    "max_concurrency",
    "request_rate",
    "load_mode",
    "concurrency_source",
    "concurrency_mst_fraction",
    "preserve_request_order",
    "response_text_max_chars",
    "include_prompt_text",
    "decoding",
}
_DECODING_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "n",
    "max_tokens",
    "max_tokens_policy",
    "prompt_token_buffer",
    "extra_body",
}
_EXPERIMENT_KEYS = {
    "id",
    "model",
    "models",
    "workload",
    "workloads",
    "endpoint",
    "hardware",
    "probe",
    "launch",
    "generation",
}


def load_quality_manifest(path: str | Path) -> QualityRunManifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise QualityManifestValidationError(f"manifest path does not exist: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = _expect_mapping(payload, "manifest")
    _check_allowed_keys(root, "manifest", _TOP_LEVEL_KEYS)
    if "search" in root:
        raise QualityManifestValidationError("quality manifests must use generation, not search")

    try:
        run = _parse_run_config(root.get("run"), manifest_path=manifest_path)
        slurm = _parse_slurm_config(root.get("slurm"))
        hardware = _merge_hardware_config(HardwareConfig(), root.get("hardware"), field_name="hardware")
        probe = _merge_probe_config(ProbeConfig(), root.get("probe"), field_name="probe")
        launch = _merge_launch_config(LaunchConfig(), root.get("launch"), field_name="launch")
    except ValueError as exc:
        raise QualityManifestValidationError(str(exc)) from exc
    generation = _merge_generation_config(
        QualityGenerationConfig(),
        root.get("generation"),
        field_name="generation",
    )

    experiments_raw = root.get("experiments")
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise QualityManifestValidationError("manifest.experiments must be a non-empty list")
    explicit_ids: set[str] = set()
    experiments: list[QualityExperimentTemplate] = []
    for index, raw_experiment in enumerate(experiments_raw):
        field_name = f"experiments[{index}]"
        item = _expect_mapping(raw_experiment, field_name)
        _check_allowed_keys(item, field_name, _EXPERIMENT_KEYS)
        experiment_id = _optional_str(item.get("id"), f"{field_name}.id")
        if experiment_id is not None:
            if experiment_id in explicit_ids:
                raise QualityManifestValidationError(f"duplicate experiment id found: {experiment_id!r}")
            explicit_ids.add(experiment_id)
        experiment_launch = _merge_launch_config(
            launch,
            item.get("launch"),
            field_name=f"{field_name}.launch",
        )
        experiment_hardware = _merge_hardware_config(
            hardware,
            item.get("hardware"),
            field_name=f"{field_name}.hardware",
        )
        experiment_probe = _merge_probe_config(
            probe,
            item.get("probe"),
            field_name=f"{field_name}.probe",
        )
        experiments.append(
            QualityExperimentTemplate(
                source_index=index,
                experiment_id=experiment_id,
                models=_parse_models(item, field_name),
                workloads=_parse_workloads(item, field_name, manifest_dir=manifest_path.parent),
                endpoint=_optional_str(item.get("endpoint"), f"{field_name}.endpoint")
                or run.default_endpoint,
                launch=experiment_launch,
                generation=_merge_generation_config(
                    generation,
                    item.get("generation"),
                    field_name=f"{field_name}.generation",
                ),
                hardware=experiment_hardware,
                probe=experiment_probe,
            )
        )

    return QualityRunManifest(
        manifest_path=manifest_path,
        run=run,
        slurm=slurm,
        hardware=hardware,
        probe=probe,
        launch=launch,
        generation=generation,
        experiments=tuple(experiments),
    )


def _merge_generation_config(
    base: QualityGenerationConfig,
    payload: Any,
    *,
    field_name: str,
) -> QualityGenerationConfig:
    if payload is None:
        return base
    item = _expect_mapping(payload, field_name)
    _check_allowed_keys(item, field_name, _GENERATION_KEYS)
    decoding_payload = item.get("decoding")
    decoding = base.decoding
    if decoding_payload is not None:
        decoding = _merge_decoding_config(decoding, decoding_payload, field_name=f"{field_name}.decoding")
    try:
        return replace(
            base,
            request_timeout_s=float(item.get("request_timeout_s", base.request_timeout_s)),
            max_concurrency=(
                base.max_concurrency
                if "max_concurrency" not in item
                else _optional_int(item.get("max_concurrency"), f"{field_name}.max_concurrency")
            ),
            request_rate=(
                base.request_rate
                if "request_rate" not in item
                else _optional_float(item.get("request_rate"), f"{field_name}.request_rate")
            ),
            load_mode=str(item.get("load_mode", base.load_mode)),
            concurrency_source=str(item.get("concurrency_source", base.concurrency_source)),
            concurrency_mst_fraction=float(
                item.get("concurrency_mst_fraction", base.concurrency_mst_fraction)
            ),
            preserve_request_order=_expect_bool(
                item.get("preserve_request_order", base.preserve_request_order),
                f"{field_name}.preserve_request_order",
            ),
            response_text_max_chars=_expect_int(
                item.get("response_text_max_chars", base.response_text_max_chars),
                f"{field_name}.response_text_max_chars",
                minimum=1,
            ),
            include_prompt_text=_expect_bool(
                item.get("include_prompt_text", base.include_prompt_text),
                f"{field_name}.include_prompt_text",
            ),
            decoding=decoding,
        )
    except ValueError as exc:
        raise QualityManifestValidationError(str(exc)) from exc


def _merge_decoding_config(
    base: QualityDecodingConfig,
    payload: Any,
    *,
    field_name: str,
) -> QualityDecodingConfig:
    item = _expect_mapping(payload, field_name)
    _check_allowed_keys(item, field_name, _DECODING_KEYS)
    try:
        return replace(
            base,
            temperature=float(item.get("temperature", base.temperature)),
            top_p=float(item.get("top_p", base.top_p)),
            top_k=_expect_int(item.get("top_k", base.top_k), f"{field_name}.top_k", minimum=1),
            min_p=float(item.get("min_p", base.min_p)),
            n=_expect_int(item.get("n", base.n), f"{field_name}.n", minimum=1),
            max_tokens=_expect_int(
                item.get("max_tokens", base.max_tokens),
                f"{field_name}.max_tokens",
                minimum=1,
            ),
            max_tokens_policy=str(item.get("max_tokens_policy", base.max_tokens_policy)),
            prompt_token_buffer=_expect_int(
                item.get("prompt_token_buffer", base.prompt_token_buffer),
                f"{field_name}.prompt_token_buffer",
                minimum=0,
            ),
            extra_body=dict(_expect_mapping(item.get("extra_body", base.extra_body), f"{field_name}.extra_body")),
        )
    except ValueError as exc:
        raise QualityManifestValidationError(str(exc)) from exc


def _parse_models(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    has_model = "model" in payload
    has_models = "models" in payload
    if has_model == has_models:
        raise QualityManifestValidationError(f"{field_name} must define exactly one of model or models")
    if has_model:
        return (_expect_str(payload.get("model"), f"{field_name}.model"),)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise QualityManifestValidationError(f"{field_name}.models must be a non-empty list")
    return tuple(_expect_str(item, f"{field_name}.models[]") for item in raw_models)


def _parse_workloads(payload: Mapping[str, Any], field_name: str, *, manifest_dir: Path) -> tuple[Path, ...]:
    has_workload = "workload" in payload
    has_workloads = "workloads" in payload
    if has_workload == has_workloads:
        raise QualityManifestValidationError(f"{field_name} must define exactly one of workload or workloads")
    raw_values = [payload["workload"]] if has_workload else payload["workloads"]
    if not isinstance(raw_values, list) or not raw_values:
        raise QualityManifestValidationError(f"{field_name}.workloads must be a non-empty list")
    paths: list[Path] = []
    for index, raw_path in enumerate(raw_values):
        path = Path(_expect_str(raw_path, f"{field_name}.workloads[{index}]"))
        resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
        if not resolved.is_file():
            raise QualityManifestValidationError(f"missing workload path: {resolved}")
        paths.append(resolved)
    return tuple(paths)


def _expect_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityManifestValidationError(f"{field_name} must be a mapping")
    return value


def _expect_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityManifestValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, field_name)


def _expect_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int):
        raise QualityManifestValidationError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise QualityManifestValidationError(f"{field_name} must be >= {minimum}")
    return value


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise QualityManifestValidationError(f"{field_name} must be a boolean")
    return value


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name, minimum=1)


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise QualityManifestValidationError(f"{field_name} must be a number")
    return float(value)


def _check_allowed_keys(payload: Mapping[str, Any], field_name: str, allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise QualityManifestValidationError(f"{field_name} has unknown keys: {sorted(unknown)}")
