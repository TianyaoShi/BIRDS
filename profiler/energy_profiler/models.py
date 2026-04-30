from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

from local_orchestrator.models import LaunchConfig


EnergyPlanMode = Literal["mst-rounded", "sweep", "explicit"]
EnergyRateSource = Literal["max_slo", "max_no_drift"]
EnergyJobStatus = Literal["planned", "running", "succeeded", "failed", "skipped"]


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _require_positive_float(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}")


def _require_non_negative_float(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite float, got {value!r}")


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_optional_rate_list(name: str, values: tuple[float, ...]) -> None:
    for value in values:
        _require_positive_float(name, value)


@dataclass(frozen=True, slots=True)
class EnergyPlanHeader:
    plan_id: str
    source_orchestrator_run_root: Path
    output_root: Path = Path("results/energy")
    python_executable: str | None = None
    mode: EnergyPlanMode = "mst-rounded"

    def __post_init__(self) -> None:
        _require_non_empty("plan_id", self.plan_id)
        if self.mode not in {"mst-rounded", "sweep", "explicit"}:
            raise ValueError(f"unsupported plan mode {self.mode!r}")
        if self.python_executable is not None:
            _require_non_empty("python_executable", self.python_executable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source_orchestrator_run_root": str(self.source_orchestrator_run_root),
            "output_root": str(self.output_root),
            "python_executable": self.python_executable,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanHeader":
        return cls(
            plan_id=_expect_str(payload.get("plan_id"), "plan.plan_id"),
            source_orchestrator_run_root=Path(
                _expect_str(
                    payload.get("source_orchestrator_run_root"),
                    "plan.source_orchestrator_run_root",
                )
            ),
            output_root=Path(_expect_str(payload.get("output_root"), "plan.output_root")),
            python_executable=_optional_str(payload.get("python_executable"), "plan.python_executable"),
            mode=_expect_mode(payload.get("mode"), "plan.mode"),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlanSelectionSweep:
    enabled: bool = False
    models: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    max_steps: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("selection.sweep.enabled must be a boolean")
        _require_positive_int("selection.sweep.max_steps", self.max_steps)
        if self.max_steps > 20:
            raise ValueError("selection.sweep.max_steps must be <= 20")
        for value in self.models:
            _require_non_empty("selection.sweep.models[]", value)
        for value in self.experiment_ids:
            _require_non_empty("selection.sweep.experiment_ids[]", value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "models": list(self.models),
            "experiment_ids": list(self.experiment_ids),
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanSelectionSweep":
        return cls(
            enabled=_expect_bool(payload.get("enabled", False), "selection.sweep.enabled"),
            models=_expect_string_tuple(payload.get("models", []), "selection.sweep.models"),
            experiment_ids=_expect_string_tuple(
                payload.get("experiment_ids", []),
                "selection.sweep.experiment_ids",
            ),
            max_steps=_expect_int(payload.get("max_steps", 20), "selection.sweep.max_steps", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlanSelection:
    models: tuple[str, ...] = ()
    workloads: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    explicit_request_rates: tuple[float, ...] = ()
    sweep: EnergyPlanSelectionSweep = field(default_factory=EnergyPlanSelectionSweep)

    def __post_init__(self) -> None:
        for value in self.models:
            _require_non_empty("selection.models[]", value)
        for value in self.workloads:
            _require_non_empty("selection.workloads[]", value)
        for value in self.experiment_ids:
            _require_non_empty("selection.experiment_ids[]", value)
        _require_optional_rate_list("selection.explicit_request_rates[]", self.explicit_request_rates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "workloads": list(self.workloads),
            "experiment_ids": list(self.experiment_ids),
            "explicit_request_rates": list(self.explicit_request_rates),
            "sweep": self.sweep.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanSelection":
        return cls(
            models=_expect_string_tuple(payload.get("models", []), "selection.models"),
            workloads=_expect_string_tuple(payload.get("workloads", []), "selection.workloads"),
            experiment_ids=_expect_string_tuple(payload.get("experiment_ids", []), "selection.experiment_ids"),
            explicit_request_rates=_expect_float_tuple(
                payload.get("explicit_request_rates", []),
                "selection.explicit_request_rates",
            ),
            sweep=EnergyPlanSelectionSweep.from_dict(
                _expect_mapping(payload.get("sweep", {}), "selection.sweep")
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlanDefaults:
    duration_s: float = 180.0
    warmup_s: float = 30.0
    cooldown_s: float = 15.0
    metrics_interval_s: float = 1.0
    window_s: float = 10.0
    gpu_monitor_interval_s: float = 0.025
    gpu_monitor_truncate_s: float = 5.0
    monitor_clock: bool = False
    request_timeout_s: float = 6 * 60 * 60
    safety_max_outstanding: int | None = None

    def __post_init__(self) -> None:
        _require_positive_float("defaults.duration_s", self.duration_s)
        _require_non_negative_float("defaults.warmup_s", self.warmup_s)
        _require_non_negative_float("defaults.cooldown_s", self.cooldown_s)
        _require_positive_float("defaults.metrics_interval_s", self.metrics_interval_s)
        _require_positive_float("defaults.window_s", self.window_s)
        _require_positive_float("defaults.gpu_monitor_interval_s", self.gpu_monitor_interval_s)
        _require_non_negative_float("defaults.gpu_monitor_truncate_s", self.gpu_monitor_truncate_s)
        _require_positive_float("defaults.request_timeout_s", self.request_timeout_s)
        if not isinstance(self.monitor_clock, bool):
            raise ValueError("defaults.monitor_clock must be a boolean")
        if self.safety_max_outstanding is not None:
            _require_positive_int("defaults.safety_max_outstanding", self.safety_max_outstanding)

    @property
    def idle_monitor_duration_s(self) -> float:
        return self.warmup_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "warmup_s": self.warmup_s,
            "cooldown_s": self.cooldown_s,
            "metrics_interval_s": self.metrics_interval_s,
            "window_s": self.window_s,
            "gpu_monitor_interval_s": self.gpu_monitor_interval_s,
            "gpu_monitor_truncate_s": self.gpu_monitor_truncate_s,
            "monitor_clock": self.monitor_clock,
            "request_timeout_s": self.request_timeout_s,
            "safety_max_outstanding": self.safety_max_outstanding,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanDefaults":
        return cls(
            duration_s=_expect_float(payload.get("duration_s", 180.0), "defaults.duration_s", minimum=0.0, strict_gt=True),
            warmup_s=_expect_float(payload.get("warmup_s", 30.0), "defaults.warmup_s", minimum=0.0),
            cooldown_s=_expect_float(payload.get("cooldown_s", 15.0), "defaults.cooldown_s", minimum=0.0),
            metrics_interval_s=_expect_float(payload.get("metrics_interval_s", 1.0), "defaults.metrics_interval_s", minimum=0.0, strict_gt=True),
            window_s=_expect_float(payload.get("window_s", 10.0), "defaults.window_s", minimum=0.0, strict_gt=True),
            gpu_monitor_interval_s=_expect_float(
                payload.get("gpu_monitor_interval_s", 0.025),
                "defaults.gpu_monitor_interval_s",
                minimum=0.0,
                strict_gt=True,
            ),
            gpu_monitor_truncate_s=_expect_float(
                payload.get("gpu_monitor_truncate_s", 5.0),
                "defaults.gpu_monitor_truncate_s",
                minimum=0.0,
            ),
            monitor_clock=_expect_bool(payload.get("monitor_clock", False), "defaults.monitor_clock"),
            request_timeout_s=_expect_float(
                payload.get("request_timeout_s", 6 * 60 * 60),
                "defaults.request_timeout_s",
                minimum=0.0,
                strict_gt=True,
            ),
            safety_max_outstanding=_expect_optional_int(
                payload.get("safety_max_outstanding"),
                "defaults.safety_max_outstanding",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlanRounding:
    mode: str = "floor_preferred"
    preferred_steps: tuple[float, ...] = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0)
    minimum_rate: float = 0.1

    def __post_init__(self) -> None:
        _require_non_empty("rounding.mode", self.mode)
        if not self.preferred_steps:
            raise ValueError("rounding.preferred_steps must be non-empty")
        previous = 0.0
        for step in self.preferred_steps:
            _require_positive_float("rounding.preferred_steps[]", step)
            if step < previous:
                raise ValueError("rounding.preferred_steps must be sorted ascending")
            previous = step
        _require_positive_float("rounding.minimum_rate", self.minimum_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "preferred_steps": list(self.preferred_steps),
            "minimum_rate": self.minimum_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanRounding":
        return cls(
            mode=_expect_str(payload.get("mode", "floor_preferred"), "rounding.mode"),
            preferred_steps=_expect_float_tuple(payload.get("preferred_steps", [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]), "rounding.preferred_steps"),
            minimum_rate=_expect_float(payload.get("minimum_rate", 0.1), "rounding.minimum_rate", minimum=0.0, strict_gt=True),
        )


@dataclass(frozen=True, slots=True)
class EnergyLaunchConfig:
    template: tuple[str, ...] | None = None
    executable: str = "vllm"
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    tensor_parallel_size: int = 1
    gpu_count: int = 1
    dtype: str | None = None
    quantization: str | None = None
    tokenizer_mode: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: float | None = None
    max_num_batched_tokens: float | None = None
    host: str = "127.0.0.1"
    readiness_path: str = "/v1/models"
    readiness_timeout_s: float = 300.0
    readiness_interval_s: float = 2.0

    def __post_init__(self) -> None:
        if self.template is not None and len(self.template) == 0:
            raise ValueError("launch.template cannot be empty")
        _require_non_empty("launch.executable", self.executable)
        _require_positive_int("launch.tensor_parallel_size", self.tensor_parallel_size)
        _require_positive_int("launch.gpu_count", self.gpu_count)
        if self.tensor_parallel_size > self.gpu_count:
            raise ValueError("launch.tensor_parallel_size cannot exceed launch.gpu_count")
        if self.gpu_memory_utilization is not None:
            _require_positive_float("launch.gpu_memory_utilization", self.gpu_memory_utilization)
        if self.max_model_len is not None:
            _require_positive_int("launch.max_model_len", self.max_model_len)
        if self.max_num_seqs is not None:
            _require_positive_float("launch.max_num_seqs", self.max_num_seqs)
        if self.max_num_batched_tokens is not None:
            _require_positive_float("launch.max_num_batched_tokens", self.max_num_batched_tokens)
        _require_non_empty("launch.host", self.host)
        if not self.readiness_path.startswith("/"):
            raise ValueError("launch.readiness_path must start with '/'")
        _require_positive_float("launch.readiness_timeout_s", self.readiness_timeout_s)
        _require_positive_float("launch.readiness_interval_s", self.readiness_interval_s)
        for key, value in self.env.items():
            _require_non_empty("launch.env key", key)
            _require_non_empty("launch.env value", value)

    @classmethod
    def from_launch_config(cls, launch: LaunchConfig) -> "EnergyLaunchConfig":
        return cls(
            template=launch.template,
            executable=launch.executable,
            extra_args=launch.extra_args,
            env=dict(launch.env),
            tensor_parallel_size=launch.tensor_parallel_size,
            gpu_count=launch.gpu_count,
            dtype=launch.dtype,
            quantization=launch.quantization,
            tokenizer_mode=launch.tokenizer_mode,
            gpu_memory_utilization=launch.gpu_memory_utilization,
            max_model_len=launch.max_model_len,
            max_num_seqs=launch.max_num_seqs,
            max_num_batched_tokens=launch.max_num_batched_tokens,
            host=launch.host,
            readiness_path=launch.readiness_path,
            readiness_timeout_s=launch.readiness_timeout_s,
            readiness_interval_s=launch.readiness_interval_s,
        )

    def to_launch_config(self) -> LaunchConfig:
        return LaunchConfig(
            template=self.template,
            executable=self.executable,
            extra_args=self.extra_args,
            env=dict(self.env),
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_count=self.gpu_count,
            dtype=self.dtype,
            quantization=self.quantization,
            tokenizer_mode=self.tokenizer_mode,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            host=self.host,
            readiness_path=self.readiness_path,
            readiness_timeout_s=self.readiness_timeout_s,
            readiness_interval_s=self.readiness_interval_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": None if self.template is None else list(self.template),
            "executable": self.executable,
            "extra_args": list(self.extra_args),
            "env": dict(self.env),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_count": self.gpu_count,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "tokenizer_mode": self.tokenizer_mode,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "host": self.host,
            "readiness_path": self.readiness_path,
            "readiness_timeout_s": self.readiness_timeout_s,
            "readiness_interval_s": self.readiness_interval_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyLaunchConfig":
        raw_template = payload.get("template")
        template = None
        if raw_template is not None:
            template = _expect_string_tuple(raw_template, "jobs[].launch.template")
        return cls(
            template=template,
            executable=_expect_str(payload.get("executable", "vllm"), "jobs[].launch.executable"),
            extra_args=_expect_string_tuple(payload.get("extra_args", []), "jobs[].launch.extra_args"),
            env=_expect_string_mapping(payload.get("env", {}), "jobs[].launch.env"),
            tensor_parallel_size=_expect_int(
                payload.get("tensor_parallel_size", 1),
                "jobs[].launch.tensor_parallel_size",
                minimum=1,
            ),
            gpu_count=_expect_int(payload.get("gpu_count", 1), "jobs[].launch.gpu_count", minimum=1),
            dtype=_optional_str(payload.get("dtype"), "jobs[].launch.dtype"),
            quantization=_optional_str(payload.get("quantization"), "jobs[].launch.quantization"),
            tokenizer_mode=_optional_str(payload.get("tokenizer_mode"), "jobs[].launch.tokenizer_mode"),
            gpu_memory_utilization=_expect_optional_float(
                payload.get("gpu_memory_utilization"),
                "jobs[].launch.gpu_memory_utilization",
                minimum=0.0,
                strict_gt=True,
            ),
            max_model_len=_expect_optional_int(payload.get("max_model_len"), "jobs[].launch.max_model_len", minimum=1),
            max_num_seqs=_expect_optional_float(
                payload.get("max_num_seqs"),
                "jobs[].launch.max_num_seqs",
                minimum=0.0,
                strict_gt=True,
            ),
            max_num_batched_tokens=_expect_optional_float(
                payload.get("max_num_batched_tokens"),
                "jobs[].launch.max_num_batched_tokens",
                minimum=0.0,
                strict_gt=True,
            ),
            host=_expect_str(payload.get("host", "127.0.0.1"), "jobs[].launch.host"),
            readiness_path=_expect_str(
                payload.get("readiness_path", "/v1/models"),
                "jobs[].launch.readiness_path",
            ),
            readiness_timeout_s=_expect_float(
                payload.get("readiness_timeout_s", 300.0),
                "jobs[].launch.readiness_timeout_s",
                minimum=0.0,
                strict_gt=True,
            ),
            readiness_interval_s=_expect_float(
                payload.get("readiness_interval_s", 2.0),
                "jobs[].launch.readiness_interval_s",
                minimum=0.0,
                strict_gt=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlanJob:
    id: str
    source_experiment_id: str
    source_result_dir: Path
    model: str
    workload: Path
    endpoint: str
    request_rate: float
    mst_rate: float | None
    mst_rate_source: str | None
    launch: EnergyLaunchConfig
    server_signature_key: str
    server_config_slug: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("jobs[].id", self.id)
        _require_non_empty("jobs[].source_experiment_id", self.source_experiment_id)
        _require_non_empty("jobs[].model", self.model)
        if not self.endpoint.startswith("/"):
            raise ValueError("jobs[].endpoint must start with '/'")
        _require_positive_float("jobs[].request_rate", self.request_rate)
        if self.mst_rate is not None:
            _require_positive_float("jobs[].mst_rate", self.mst_rate)
        if self.mst_rate_source is not None:
            _require_non_empty("jobs[].mst_rate_source", self.mst_rate_source)
        _require_non_empty("jobs[].server_signature_key", self.server_signature_key)
        if self.server_config_slug is not None:
            _require_non_empty("jobs[].server_config_slug", self.server_config_slug)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_experiment_id": self.source_experiment_id,
            "source_result_dir": str(self.source_result_dir),
            "model": self.model,
            "workload": str(self.workload),
            "endpoint": self.endpoint,
            "request_rate": self.request_rate,
            "mst_rate": self.mst_rate,
            "mst_rate_source": self.mst_rate_source,
            "launch": self.launch.to_dict(),
            "server_signature_key": self.server_signature_key,
            "server_config_slug": self.server_config_slug,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlanJob":
        return cls(
            id=_expect_str(payload.get("id"), "jobs[].id"),
            source_experiment_id=_expect_str(payload.get("source_experiment_id"), "jobs[].source_experiment_id"),
            source_result_dir=Path(_expect_str(payload.get("source_result_dir"), "jobs[].source_result_dir")),
            model=_expect_str(payload.get("model"), "jobs[].model"),
            workload=Path(_expect_str(payload.get("workload"), "jobs[].workload")),
            endpoint=_expect_str(payload.get("endpoint"), "jobs[].endpoint"),
            request_rate=_expect_float(payload.get("request_rate"), "jobs[].request_rate", minimum=0.0, strict_gt=True),
            mst_rate=_expect_optional_float(payload.get("mst_rate"), "jobs[].mst_rate", minimum=0.0, strict_gt=True),
            mst_rate_source=_optional_str(payload.get("mst_rate_source"), "jobs[].mst_rate_source"),
            launch=EnergyLaunchConfig.from_dict(_expect_mapping(payload.get("launch"), "jobs[].launch")),
            server_signature_key=_expect_str(payload.get("server_signature_key"), "jobs[].server_signature_key"),
            server_config_slug=_optional_str(payload.get("server_config_slug"), "jobs[].server_config_slug"),
            metadata=dict(_expect_mapping(payload.get("metadata", {}), "jobs[].metadata")),
        )


@dataclass(frozen=True, slots=True)
class EnergyPlan:
    plan: EnergyPlanHeader
    selection: EnergyPlanSelection = field(default_factory=EnergyPlanSelection)
    defaults: EnergyPlanDefaults = field(default_factory=EnergyPlanDefaults)
    rounding: EnergyPlanRounding = field(default_factory=EnergyPlanRounding)
    jobs: tuple[EnergyPlanJob, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "selection": self.selection.to_dict(),
            "defaults": self.defaults.to_dict(),
            "rounding": self.rounding.to_dict(),
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyPlan":
        return cls(
            plan=EnergyPlanHeader.from_dict(_expect_mapping(payload.get("plan"), "plan")),
            selection=EnergyPlanSelection.from_dict(_expect_mapping(payload.get("selection", {}), "selection")),
            defaults=EnergyPlanDefaults.from_dict(_expect_mapping(payload.get("defaults", {}), "defaults")),
            rounding=EnergyPlanRounding.from_dict(_expect_mapping(payload.get("rounding", {}), "rounding")),
            jobs=tuple(
                EnergyPlanJob.from_dict(_expect_mapping(item, "jobs[]"))
                for item in _expect_list(payload.get("jobs", []), "jobs")
            ),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorJobRecord:
    source_run_id: str
    source_run_root: Path
    experiment_id: str
    model: str
    workload: Path
    endpoint: str
    result_dir: Path
    status: str
    max_no_drift_request_rate: float | None
    max_slo_satisfying_request_rate: float | None
    search_id: str | None
    search_mode: str | None
    confirmation_trial_id: str | None
    launch: EnergyLaunchConfig
    server_signature_key: str
    server_config_slug: str


def _expect_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _expect_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _expect_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, field_name)


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _expect_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _expect_optional_int(value: Any, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name, minimum=minimum)


def _expect_float(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    strict_gt: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if strict_gt:
        if numeric <= minimum:
            raise ValueError(f"{field_name} must be > {minimum}")
    elif numeric < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return numeric


def _expect_optional_float(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    strict_gt: bool = False,
) -> float | None:
    if value is None:
        return None
    return _expect_float(value, field_name, minimum=minimum, strict_gt=strict_gt)


def _expect_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    items = _expect_list(value, field_name)
    return tuple(_expect_str(item, f"{field_name}[]") for item in items)


def _expect_float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    items = _expect_list(value, field_name)
    return tuple(
        _expect_float(item, f"{field_name}[]", minimum=0.0, strict_gt=True)
        for item in items
    )


def _expect_string_mapping(value: Any, field_name: str) -> dict[str, str]:
    payload = _expect_mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        result[key] = _expect_str(item, f"{field_name}.{key}")
    return result


def _expect_mode(value: Any, field_name: str) -> EnergyPlanMode:
    mode = _expect_str(value, field_name)
    if mode not in {"mst-rounded", "sweep", "explicit"}:
        raise ValueError(f"{field_name} must be one of mst-rounded, sweep, explicit")
    return mode
