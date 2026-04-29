from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import (
    ExperimentTemplate,
    ExperimentOverride,
    HardwareConfig,
    LaunchConfig,
    OrchestratorManifest,
    ProbeConfig,
    RetryPolicy,
    RunConfig,
    SearchConfig,
    SlurmConfig,
)


_TOP_LEVEL_KEYS = {"run", "slurm", "hardware", "probe", "launch", "search", "overrides", "experiments"}
_RUN_KEYS = {
    "run_id",
    "output_root",
    "allowed_gpu_ids",
    "max_active_gpus",
    "keep_one_gpu_spare",
    "default_endpoint",
    "startup_attempts",
    "search_attempts",
    "base_port_start",
    "base_port_end",
    "metrics_port_offset",
    "python_executable",
}
_SLURM_KEYS = {
    "partition",
    "account",
    "qos",
    "time",
    "mem",
    "modules",
    "setup_commands",
    "python_executable",
    "sbatch_extra_args",
    "array_concurrency_limit",
    "base_port",
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
    "search",
    "overrides",
    "server_metadata_file",
}
_HARDWARE_KEYS = {
    "name",
    "gpu_memory_gb",
    "gpu_memory_utilization",
}
_PROBE_KEYS = {
    "enabled",
    "auto_gpu_count",
    "activation_memory_gb",
    "memory_safety_factor",
    "kv_cache_request_count",
    "default_context_tokens",
    "model_size_overrides_b",
}
_LAUNCH_KEYS = {
    "template",
    "executable",
    "extra_args",
    "env",
    "tensor_parallel_size",
    "gpu_count",
    "dtype",
    "quantization",
    "tokenizer_mode",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "host",
    "readiness_path",
    "readiness_timeout_s",
    "readiness_interval_s",
}
_SEARCH_KEYS = {
    "search_mode",
    "trial_min_duration_s",
    "trial_max_duration_s",
    "final_confirmation_duration_s",
    "rate_precision",
    "initial_request_rate",
    "max_request_rate",
    "max_binary_steps",
    "max_bracket_trials",
    "closed_loop_initial_concurrency",
    "closed_loop_min_trials",
    "max_closed_loop_concurrency",
    "closed_loop_plateau_relative_gain",
    "metrics_interval_s",
    "window_s",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "ttft_slo_field",
    "tpot_slo_field",
    "max_num_seqs",
    "max_num_batched_tokens",
}
_OVERRIDE_KEYS = {"match", "launch", "search", "reason"}
_OVERRIDE_MATCH_KEYS = {"model", "models", "workload", "workloads", "hardware", "hardwares"}
_STRUCTURED_LAUNCH_KEYS = {
    "executable",
    "extra_args",
    "tensor_parallel_size",
    "gpu_count",
    "dtype",
    "quantization",
    "tokenizer_mode",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "host",
}


class ManifestValidationError(ValueError):
    pass


def load_manifest(manifest_path: str | Path) -> OrchestratorManifest:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ManifestValidationError(f"manifest path does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _expect_mapping(raw, "manifest")
    _check_allowed_keys(payload, "manifest", _TOP_LEVEL_KEYS)

    run = _parse_run_config(payload.get("run"), manifest_path=path)
    slurm = _parse_slurm_config(payload.get("slurm"))
    default_hardware = _merge_hardware_config(HardwareConfig(), payload.get("hardware"), field_name="hardware")
    default_probe = _merge_probe_config(ProbeConfig(), payload.get("probe"), field_name="probe")
    default_launch = _merge_launch_config(LaunchConfig(), payload.get("launch"), field_name="launch")
    default_search = _merge_search_config(SearchConfig(), payload.get("search"), field_name="search")
    default_overrides = _parse_override_rules(payload.get("overrides"), field_name="overrides")

    experiments_raw = payload.get("experiments")
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise ManifestValidationError("manifest.experiments must be a non-empty list")

    explicit_ids: set[str] = set()
    experiments: list[ExperimentTemplate] = []
    for index, raw_experiment in enumerate(experiments_raw):
        field_name = f"experiments[{index}]"
        experiment_payload = _expect_mapping(raw_experiment, field_name)
        _check_allowed_keys(experiment_payload, field_name, _EXPERIMENT_KEYS)

        experiment_id = _optional_non_empty_string(experiment_payload.get("id"), f"{field_name}.id")
        if experiment_id is not None:
            if experiment_id in explicit_ids:
                raise ManifestValidationError(f"duplicate experiment id found: {experiment_id!r}")
            explicit_ids.add(experiment_id)

        models = _parse_models(experiment_payload, field_name)
        workloads = _parse_workloads(experiment_payload, field_name, manifest_dir=path.parent)

        endpoint = _optional_non_empty_string(experiment_payload.get("endpoint"), f"{field_name}.endpoint")
        resolved_endpoint = endpoint or run.default_endpoint
        if not resolved_endpoint.startswith("/"):
            raise ManifestValidationError(f"{field_name}.endpoint must start with '/'")

        launch = _merge_launch_config(
            default_launch,
            experiment_payload.get("launch"),
            field_name=f"{field_name}.launch",
        )
        search = _merge_search_config(
            default_search,
            experiment_payload.get("search"),
            field_name=f"{field_name}.search",
        )
        hardware = _merge_hardware_config(
            default_hardware,
            experiment_payload.get("hardware"),
            field_name=f"{field_name}.hardware",
        )
        probe = _merge_probe_config(
            default_probe,
            experiment_payload.get("probe"),
            field_name=f"{field_name}.probe",
        )
        overrides = _parse_override_rules(
            experiment_payload.get("overrides"),
            field_name=f"{field_name}.overrides",
        )

        metadata_file_raw = experiment_payload.get("server_metadata_file")
        server_metadata_file: Path | None = None
        if metadata_file_raw is not None:
            metadata_path = _resolve_path(
                _expect_non_empty_string(metadata_file_raw, f"{field_name}.server_metadata_file"),
                base_dir=path.parent,
            )
            if not metadata_path.is_file():
                raise ManifestValidationError(
                    f"{field_name}.server_metadata_file does not exist: {metadata_path}"
                )
            server_metadata_file = metadata_path

        experiments.append(
            ExperimentTemplate(
                source_index=index,
                experiment_id=experiment_id,
                models=models,
                workloads=workloads,
                endpoint=resolved_endpoint,
                launch=launch,
                search=search,
                hardware=hardware,
                probe=probe,
                overrides=overrides,
                server_metadata_file=server_metadata_file,
            )
        )

    return OrchestratorManifest(
        manifest_path=path,
        run=run,
        slurm=slurm,
        hardware=default_hardware,
        probe=default_probe,
        overrides=default_overrides,
        experiments=tuple(experiments),
    )


def _parse_run_config(raw: Any, *, manifest_path: Path) -> RunConfig:
    if raw is None:
        return RunConfig(output_root=(manifest_path.parent / "results/orchestrator").resolve())
    payload = _expect_mapping(raw, "run")
    _check_allowed_keys(payload, "run", _RUN_KEYS)

    run_id = _optional_non_empty_string(payload.get("run_id"), "run.run_id")
    output_root_raw = payload.get("output_root")
    if output_root_raw is None:
        output_root = (manifest_path.parent / "results/orchestrator").resolve()
    else:
        output_root = _resolve_path(
            _expect_non_empty_string(output_root_raw, "run.output_root"),
            base_dir=manifest_path.parent,
        )
    keep_one_gpu_spare = _expect_bool(payload.get("keep_one_gpu_spare", True), "run.keep_one_gpu_spare")

    allowed_gpu_ids_raw = payload.get("allowed_gpu_ids", [0, 1, 2, 3])
    if not isinstance(allowed_gpu_ids_raw, list) or not allowed_gpu_ids_raw:
        raise ManifestValidationError("run.allowed_gpu_ids must be a non-empty list")
    allowed_gpu_ids = tuple(_expect_int(item, "run.allowed_gpu_ids[]", minimum=0) for item in allowed_gpu_ids_raw)

    default_max_active = min(3, len(allowed_gpu_ids) - 1 if keep_one_gpu_spare else len(allowed_gpu_ids))
    if default_max_active <= 0:
        default_max_active = 1
    max_active_gpus = _expect_int(
        payload.get("max_active_gpus", default_max_active),
        "run.max_active_gpus",
        minimum=1,
    )

    default_endpoint = _expect_non_empty_string(
        payload.get("default_endpoint", "/v1/chat/completions"),
        "run.default_endpoint",
    )
    if not default_endpoint.startswith("/"):
        raise ManifestValidationError("run.default_endpoint must start with '/'")

    startup_attempts = _expect_int(payload.get("startup_attempts", 2), "run.startup_attempts", minimum=1)
    search_attempts = _expect_int(payload.get("search_attempts", 2), "run.search_attempts", minimum=1)
    base_port_start = _expect_int(payload.get("base_port_start", 8000), "run.base_port_start", minimum=1)
    base_port_end = _expect_int(payload.get("base_port_end", 8099), "run.base_port_end", minimum=1)
    metrics_port_offset = _expect_int(
        payload.get("metrics_port_offset", 1000),
        "run.metrics_port_offset",
        minimum=1,
    )
    python_executable = _optional_non_empty_string(payload.get("python_executable"), "run.python_executable")

    try:
        return RunConfig(
            run_id=run_id,
            output_root=output_root,
            allowed_gpu_ids=allowed_gpu_ids,
            max_active_gpus=max_active_gpus,
            keep_one_gpu_spare=keep_one_gpu_spare,
            default_endpoint=default_endpoint,
            retry=RetryPolicy(startup_attempts=startup_attempts, search_attempts=search_attempts),
            base_port_start=base_port_start,
            base_port_end=base_port_end,
            metrics_port_offset=metrics_port_offset,
            python_executable=python_executable,
        )
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _parse_slurm_config(raw: Any) -> SlurmConfig:
    if raw is None:
        return SlurmConfig()
    payload = _expect_mapping(raw, "slurm")
    _check_allowed_keys(payload, "slurm", _SLURM_KEYS)

    updated: dict[str, Any] = {
        "partition": None,
        "account": None,
        "qos": None,
        "time": None,
        "mem": None,
        "modules": (),
        "setup_commands": (),
        "python_executable": None,
        "sbatch_extra_args": (),
        "array_concurrency_limit": None,
        "base_port": 8000,
    }
    for key, value in payload.items():
        if key in {"partition", "account", "qos", "time", "mem", "python_executable"}:
            updated[key] = _expect_non_empty_string(value, f"slurm.{key}")
            continue
        if key in {"modules", "setup_commands", "sbatch_extra_args"}:
            updated[key] = _parse_string_list(value, field_name=f"slurm.{key}")
            continue
        if key == "base_port":
            updated[key] = _expect_int(value, f"slurm.{key}", minimum=1)
            continue
        if key == "array_concurrency_limit":
            updated[key] = _expect_optional_positive_int(value, f"slurm.{key}")
            continue
        raise ManifestValidationError(f"unsupported slurm field {key!r}")
    try:
        return SlurmConfig(**updated)
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _merge_hardware_config(base: HardwareConfig, raw: Any, *, field_name: str) -> HardwareConfig:
    if raw is None:
        return base
    payload = _expect_mapping(raw, field_name)
    _check_allowed_keys(payload, field_name, _HARDWARE_KEYS)
    updated = {
        "name": base.name,
        "gpu_memory_gb": base.gpu_memory_gb,
        "gpu_memory_utilization": base.gpu_memory_utilization,
    }
    for key, value in payload.items():
        if key == "name":
            updated[key] = _expect_non_empty_string(value, f"{field_name}.{key}")
            continue
        if key in {"gpu_memory_gb", "gpu_memory_utilization"}:
            updated[key] = _expect_optional_positive_float(value, f"{field_name}.{key}")
            continue
        raise ManifestValidationError(f"unsupported hardware field {key!r}")
    try:
        return HardwareConfig(**updated)
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _merge_probe_config(base: ProbeConfig, raw: Any, *, field_name: str) -> ProbeConfig:
    if raw is None:
        return base
    payload = _expect_mapping(raw, field_name)
    _check_allowed_keys(payload, field_name, _PROBE_KEYS)
    updated = {
        "enabled": base.enabled,
        "auto_gpu_count": base.auto_gpu_count,
        "activation_memory_gb": base.activation_memory_gb,
        "memory_safety_factor": base.memory_safety_factor,
        "kv_cache_request_count": base.kv_cache_request_count,
        "default_context_tokens": base.default_context_tokens,
        "model_size_overrides_b": dict(base.model_size_overrides_b),
    }
    for key, value in payload.items():
        if key in {"enabled", "auto_gpu_count"}:
            updated[key] = _expect_bool(value, f"{field_name}.{key}")
            continue
        if key in {"activation_memory_gb", "memory_safety_factor"}:
            updated[key] = _expect_positive_float(value, f"{field_name}.{key}")
            continue
        if key in {"kv_cache_request_count", "default_context_tokens"}:
            updated[key] = _expect_int(value, f"{field_name}.{key}", minimum=1)
            continue
        if key == "model_size_overrides_b":
            overrides = _expect_mapping(value, f"{field_name}.{key}")
            updated[key] = {
                _expect_non_empty_string(pattern, f"{field_name}.{key}.pattern"): _expect_positive_float(
                    size_b,
                    f"{field_name}.{key}.{pattern}",
                )
                for pattern, size_b in overrides.items()
            }
            continue
        raise ManifestValidationError(f"unsupported probe field {key!r}")
    try:
        return ProbeConfig(**updated)
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _merge_launch_config(base: LaunchConfig, raw: Any, *, field_name: str) -> LaunchConfig:
    if raw is None:
        return base
    payload = _expect_mapping(raw, field_name)
    _check_allowed_keys(payload, field_name, _LAUNCH_KEYS)

    structured_keys = set(payload) & _STRUCTURED_LAUNCH_KEYS
    if "template" in payload and payload["template"] is not None:
        for key in _STRUCTURED_LAUNCH_KEYS:
            if key in payload:
                raise ManifestValidationError(
                    f"{field_name} uses raw template and structured field {key!r}; choose one launch style"
                )
    clears_template = "template" in payload and payload["template"] is None
    if base.template is not None and structured_keys and not clears_template:
        raise ManifestValidationError(
            f"{field_name} cannot combine structured launch overrides with an existing template; "
            "set template: null before structured fields"
        )

    updated = {
        "template": base.template,
        "executable": base.executable,
        "extra_args": base.extra_args,
        "env": dict(base.env),
        "tensor_parallel_size": base.tensor_parallel_size,
        "gpu_count": base.gpu_count,
        "dtype": base.dtype,
        "quantization": base.quantization,
        "tokenizer_mode": base.tokenizer_mode,
        "gpu_memory_utilization": base.gpu_memory_utilization,
        "max_model_len": base.max_model_len,
        "max_num_seqs": base.max_num_seqs,
        "max_num_batched_tokens": base.max_num_batched_tokens,
        "host": base.host,
        "readiness_path": base.readiness_path,
        "readiness_timeout_s": base.readiness_timeout_s,
        "readiness_interval_s": base.readiness_interval_s,
    }

    for key, value in payload.items():
        if key == "template":
            updated["template"] = _parse_template(value, field_name=f"{field_name}.template")
            continue
        if key == "extra_args":
            updated["extra_args"] = _parse_string_list(value, field_name=f"{field_name}.extra_args")
            continue
        if key == "env":
            updated["env"] = _parse_env(value, field_name=f"{field_name}.env")
            continue
        if key in {"executable", "dtype", "quantization", "tokenizer_mode", "host", "readiness_path"}:
            updated[key] = _expect_non_empty_string(value, f"{field_name}.{key}")
            continue
        if key == "tensor_parallel_size":
            updated[key] = _expect_int(value, f"{field_name}.tensor_parallel_size", minimum=1)
            continue
        if key == "gpu_count":
            updated[key] = _expect_int(value, f"{field_name}.gpu_count", minimum=1)
            continue
        if key == "max_model_len":
            updated[key] = _expect_int(value, f"{field_name}.max_model_len", minimum=1)
            continue
        if key in {
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
            "readiness_timeout_s",
            "readiness_interval_s",
        }:
            updated[key] = _expect_positive_float(value, f"{field_name}.{key}")
            continue
        raise ManifestValidationError(f"unsupported launch field {key!r}")

    try:
        return LaunchConfig(**updated)
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _merge_search_config(base: SearchConfig, raw: Any, *, field_name: str) -> SearchConfig:
    if raw is None:
        return base
    payload = _expect_mapping(raw, field_name)
    _check_allowed_keys(payload, field_name, _SEARCH_KEYS)

    for key in payload:
        if "e2e" in key.lower():
            raise ManifestValidationError(
                f"{field_name}.{key} is not supported; MST orchestrator only accepts TTFT/TPOT SLO fields"
            )

    updated = {
        "search_mode": base.search_mode,
        "trial_min_duration_s": base.trial_min_duration_s,
        "trial_max_duration_s": base.trial_max_duration_s,
        "final_confirmation_duration_s": base.final_confirmation_duration_s,
        "rate_precision": base.rate_precision,
        "initial_request_rate": base.initial_request_rate,
        "max_request_rate": base.max_request_rate,
        "max_binary_steps": base.max_binary_steps,
        "max_bracket_trials": base.max_bracket_trials,
        "closed_loop_initial_concurrency": base.closed_loop_initial_concurrency,
        "closed_loop_min_trials": base.closed_loop_min_trials,
        "max_closed_loop_concurrency": base.max_closed_loop_concurrency,
        "closed_loop_plateau_relative_gain": base.closed_loop_plateau_relative_gain,
        "metrics_interval_s": base.metrics_interval_s,
        "window_s": base.window_s,
        "ttft_slo_ms": base.ttft_slo_ms,
        "tpot_slo_ms": base.tpot_slo_ms,
        "ttft_slo_field": base.ttft_slo_field,
        "tpot_slo_field": base.tpot_slo_field,
    }

    for key, value in payload.items():
        if key == "search_mode":
            updated[key] = _expect_non_empty_string(value, f"{field_name}.search_mode")
            continue
        if key in {
            "trial_min_duration_s",
            "rate_precision",
            "initial_request_rate",
            "metrics_interval_s",
            "window_s",
        }:
            updated[key] = _expect_positive_float(value, f"{field_name}.{key}")
            continue
        if key in {"trial_max_duration_s", "final_confirmation_duration_s", "max_request_rate"}:
            updated[key] = _expect_optional_positive_float(value, f"{field_name}.{key}")
            continue
        if key in {"max_binary_steps", "max_bracket_trials", "closed_loop_initial_concurrency", "closed_loop_min_trials", "max_closed_loop_concurrency"}:
            updated[key] = _expect_int(value, f"{field_name}.{key}", minimum=1)
            continue
        if key == "closed_loop_plateau_relative_gain":
            updated[key] = _expect_positive_float(value, f"{field_name}.{key}")
            continue
        if key in {"ttft_slo_ms", "tpot_slo_ms"}:
            updated[key] = _expect_optional_positive_float(value, f"{field_name}.{key}")
            continue
        if key in {"ttft_slo_field", "tpot_slo_field"}:
            updated[key] = _expect_non_empty_string(value, f"{field_name}.{key}")
            continue
        if key in {"max_num_seqs", "max_num_batched_tokens"}:
            updated[key] = _expect_optional_positive_int(value, f"{field_name}.{key}")
            continue
        raise ManifestValidationError(f"unsupported search field {key!r}")

    try:
        return replace(base, **updated)
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc


def _parse_override_rules(raw: Any, *, field_name: str) -> tuple[ExperimentOverride, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestValidationError(f"{field_name} must be a list")
    rules: list[ExperimentOverride] = []
    for index, item in enumerate(raw):
        rule_field = f"{field_name}[{index}]"
        payload = _expect_mapping(item, rule_field)
        _check_allowed_keys(payload, rule_field, _OVERRIDE_KEYS)
        match_payload = _expect_mapping(payload.get("match"), f"{rule_field}.match")
        _check_allowed_keys(match_payload, f"{rule_field}.match", _OVERRIDE_MATCH_KEYS)

        launch_updates = payload.get("launch")
        if launch_updates is not None:
            launch_updates = dict(_expect_mapping(launch_updates, f"{rule_field}.launch"))
            _check_allowed_keys(launch_updates, f"{rule_field}.launch", _LAUNCH_KEYS)
        search_updates = payload.get("search")
        if search_updates is not None:
            search_updates = dict(_expect_mapping(search_updates, f"{rule_field}.search"))
            _check_allowed_keys(search_updates, f"{rule_field}.search", _SEARCH_KEYS)

        try:
            rules.append(
                ExperimentOverride(
                    source_index=index,
                    model_patterns=_parse_match_patterns(
                        match_payload,
                        singular="model",
                        plural="models",
                        field_name=f"{rule_field}.match",
                    ),
                    workload_patterns=_parse_match_patterns(
                        match_payload,
                        singular="workload",
                        plural="workloads",
                        field_name=f"{rule_field}.match",
                    ),
                    hardware_patterns=_parse_match_patterns(
                        match_payload,
                        singular="hardware",
                        plural="hardwares",
                        field_name=f"{rule_field}.match",
                    ),
                    launch=launch_updates,
                    search=search_updates,
                    reason=_optional_non_empty_string(payload.get("reason"), f"{rule_field}.reason"),
                )
            )
        except ValueError as exc:
            raise ManifestValidationError(str(exc)) from exc
    return tuple(rules)


def _parse_match_patterns(
    payload: Mapping[str, Any],
    *,
    singular: str,
    plural: str,
    field_name: str,
) -> tuple[str, ...]:
    has_singular = singular in payload
    has_plural = plural in payload
    if has_singular and has_plural:
        raise ManifestValidationError(f"{field_name} cannot define both {singular} and {plural}")
    if has_singular:
        return (_expect_non_empty_string(payload[singular], f"{field_name}.{singular}"),)
    if has_plural:
        return _parse_string_list(payload[plural], field_name=f"{field_name}.{plural}")
    return ()


def _parse_models(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    has_model = "model" in payload
    has_models = "models" in payload
    if has_model and has_models:
        raise ManifestValidationError(f"{field_name} cannot define both model and models")
    if not has_model and not has_models:
        raise ManifestValidationError(f"{field_name} must define model or models")
    if has_model:
        return (_expect_non_empty_string(payload["model"], f"{field_name}.model"),)
    return _parse_string_list(payload["models"], field_name=f"{field_name}.models")


def _parse_workloads(payload: Mapping[str, Any], field_name: str, *, manifest_dir: Path) -> tuple[Path, ...]:
    has_workload = "workload" in payload
    has_workloads = "workloads" in payload
    if has_workload and has_workloads:
        raise ManifestValidationError(f"{field_name} cannot define both workload and workloads")
    if not has_workload and not has_workloads:
        raise ManifestValidationError(f"{field_name} must define workload or workloads")
    raw_values: tuple[str, ...]
    if has_workload:
        raw_values = (_expect_non_empty_string(payload["workload"], f"{field_name}.workload"),)
    else:
        raw_values = _parse_string_list(payload["workloads"], field_name=f"{field_name}.workloads")
    resolved_paths: list[Path] = []
    for raw_path in raw_values:
        workload_path = _resolve_path(raw_path, base_dir=manifest_dir)
        if not workload_path.is_file():
            raise ManifestValidationError(f"{field_name} references missing workload path: {workload_path}")
        resolved_paths.append(workload_path)
    return tuple(resolved_paths)


def _parse_template(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = tuple(shlex.split(value))
        if not tokens:
            raise ManifestValidationError(f"{field_name} must not be empty")
        return tokens
    if isinstance(value, list):
        tokens = _parse_string_list(value, field_name=field_name)
        if not tokens:
            raise ManifestValidationError(f"{field_name} must not be empty")
        return tokens
    raise ManifestValidationError(f"{field_name} must be a string, list of strings, or null")


def _parse_env(value: Any, *, field_name: str) -> dict[str, str]:
    payload = _expect_mapping(value, field_name)
    env: dict[str, str] = {}
    for key, item in payload.items():
        env_key = _expect_non_empty_string(key, f"{field_name}.key")
        env[env_key] = _expect_non_empty_string(item, f"{field_name}.{env_key}")
    return env


def _parse_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(f"{field_name} must be a non-empty list")
    items = tuple(_expect_non_empty_string(item, f"{field_name}[]") for item in value)
    return items


def _resolve_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _expect_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field_name} must be a mapping")
    return value


def _expect_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_non_empty_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_non_empty_string(value, field_name)


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestValidationError(f"{field_name} must be a boolean")
    return value


def _expect_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{field_name} must be an integer")
    if value < minimum:
        raise ManifestValidationError(f"{field_name} must be >= {minimum}")
    return value

def _expect_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "off", "disabled"}:
        return None
    return _expect_int(value, field_name, minimum=1)


def _expect_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{field_name} must be a positive number")
    numeric = float(value)
    if numeric <= 0.0:
        raise ManifestValidationError(f"{field_name} must be a positive number")
    return numeric


def _expect_optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "off", "disabled"}:
        return None
    return _expect_positive_float(value, field_name)


def _check_allowed_keys(payload: Mapping[str, Any], field_name: str, allowed_keys: set[str]) -> None:
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise ManifestValidationError(f"{field_name} has unknown keys: {unknown}")
