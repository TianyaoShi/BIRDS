from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


SearchMode = Literal["closed-loop", "open-loop", "hybrid"]
JobStatus = Literal["planned", "running", "succeeded", "failed", "skipped"]


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    startup_attempts: int = 2
    search_attempts: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("startup_attempts", self.startup_attempts)
        _require_positive_int("search_attempts", self.search_attempts)


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str | None = None
    output_root: Path = Path("results/orchestrator")
    mst_output_root: Path | None = None
    allowed_gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    max_active_gpus: int = 3
    keep_one_gpu_spare: bool = True
    default_endpoint: str = "/v1/chat/completions"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    base_port_start: int = 8000
    base_port_end: int = 8099
    metrics_port_offset: int = 1000
    python_executable: str | None = None

    def __post_init__(self) -> None:
        if not self.allowed_gpu_ids:
            raise ValueError("allowed_gpu_ids must be non-empty")
        seen: set[int] = set()
        for gpu_id in self.allowed_gpu_ids:
            _require_non_negative_int("gpu_id", gpu_id)
            if gpu_id in seen:
                raise ValueError(f"duplicate GPU id found: {gpu_id!r}")
            seen.add(gpu_id)
        _require_positive_int("max_active_gpus", self.max_active_gpus)
        if self.max_active_gpus > len(self.allowed_gpu_ids):
            raise ValueError("max_active_gpus cannot exceed number of allowed_gpu_ids")
        if self.keep_one_gpu_spare and self.max_active_gpus >= len(self.allowed_gpu_ids):
            raise ValueError(
                "keep_one_gpu_spare=true requires at least one unused GPU id beyond max_active_gpus"
            )
        _require_positive_int("base_port_start", self.base_port_start)
        _require_positive_int("base_port_end", self.base_port_end)
        if self.base_port_end < self.base_port_start:
            raise ValueError("base_port_end must be >= base_port_start")
        _require_positive_int("metrics_port_offset", self.metrics_port_offset)
        if not self.default_endpoint.startswith("/"):
            raise ValueError("default_endpoint must start with '/'")
        if self.python_executable is not None and not self.python_executable:
            raise ValueError("python_executable must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time: str | None = None
    mem: str | None = None
    cpus_per_task: int | None = None
    cpus_per_gpu: int = 14
    modules: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    python_executable: str | None = None
    sbatch_extra_args: tuple[str, ...] = ()
    array_concurrency_limit: int | None = None
    base_port: int = 8000

    def __post_init__(self) -> None:
        for field_name in ("partition", "account", "qos", "time", "mem"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must be non-empty when provided")
        if self.cpus_per_task is not None:
            _require_positive_int("slurm cpus_per_task", self.cpus_per_task)
        _require_positive_int("slurm cpus_per_gpu", self.cpus_per_gpu)
        if self.python_executable is not None and not self.python_executable:
            raise ValueError("slurm python_executable must be non-empty when provided")
        if self.array_concurrency_limit is not None:
            _require_positive_int("array_concurrency_limit", self.array_concurrency_limit)
        _require_positive_int("base_port", self.base_port)
        for field_name in ("modules", "setup_commands", "sbatch_extra_args"):
            values = getattr(self, field_name)
            for value in values:
                if not value:
                    raise ValueError(f"slurm {field_name} entries must be non-empty")


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    name: str = "local"
    gpu_memory_gb: float | None = None
    gpu_memory_utilization: float = 0.90

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hardware name must be non-empty")
        if self.gpu_memory_gb is not None:
            _require_positive_float("gpu_memory_gb", self.gpu_memory_gb)
        _require_positive_float("gpu_memory_utilization", self.gpu_memory_utilization)
        if self.gpu_memory_utilization > 1.0:
            raise ValueError("gpu_memory_utilization must be <= 1.0")


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    enabled: bool = True
    auto_gpu_count: bool = False
    activation_memory_gb: float = 2.0
    memory_safety_factor: float = 1.20
    kv_cache_request_count: int = 1
    default_context_tokens: int = 4096
    model_size_overrides_b: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("probe.enabled must be a boolean")
        if not isinstance(self.auto_gpu_count, bool):
            raise ValueError("probe.auto_gpu_count must be a boolean")
        _require_positive_float("activation_memory_gb", self.activation_memory_gb)
        _require_positive_float("memory_safety_factor", self.memory_safety_factor)
        if self.memory_safety_factor < 1.0:
            raise ValueError("memory_safety_factor must be >= 1.0")
        _require_positive_int("kv_cache_request_count", self.kv_cache_request_count)
        _require_positive_int("default_context_tokens", self.default_context_tokens)
        for pattern, size_b in self.model_size_overrides_b.items():
            if not pattern:
                raise ValueError("model_size_overrides_b patterns must be non-empty")
            _require_positive_float("model_size_overrides_b value", size_b)


@dataclass(frozen=True, slots=True)
class LaunchConfig:
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
            raise ValueError("launch template cannot be empty")
        if not self.executable:
            raise ValueError("launch executable must be non-empty")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if self.tensor_parallel_size > self.gpu_count:
            raise ValueError("tensor_parallel_size cannot exceed gpu_count")
        if self.gpu_memory_utilization is not None:
            _require_positive_float("gpu_memory_utilization", self.gpu_memory_utilization)
            if self.gpu_memory_utilization > 1.0:
                raise ValueError("gpu_memory_utilization must be <= 1.0")
        if self.max_model_len is not None:
            _require_positive_int("max_model_len", self.max_model_len)
        if self.max_num_seqs is not None:
            _require_positive_float("max_num_seqs", self.max_num_seqs)
        if self.max_num_batched_tokens is not None:
            _require_positive_float("max_num_batched_tokens", self.max_num_batched_tokens)
        if not self.host:
            raise ValueError("host must be non-empty")
        if not self.readiness_path.startswith("/"):
            raise ValueError("readiness_path must start with '/'")
        _require_positive_float("readiness_timeout_s", self.readiness_timeout_s)
        _require_positive_float("readiness_interval_s", self.readiness_interval_s)
        for key, value in self.env.items():
            if not key:
                raise ValueError("launch env keys must be non-empty")
            if not isinstance(value, str):
                raise ValueError("launch env values must be strings")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    search_mode: SearchMode = "open-loop"
    trial_min_duration_s: float = 90.0
    trial_max_duration_s: float | None = 180.0
    final_confirmation_duration_s: float | None = 180.0
    rate_precision: float = 0.05
    initial_request_rate: float = 1.0
    max_request_rate: float | None = None
    max_binary_steps: int = 24
    max_bracket_trials: int = 16
    open_loop_bracket_growth_factor: float = 2.0
    client_limited_retry_attempts: int = 1
    client_limited_retry_cooldown_s: float = 30.0
    closed_loop_initial_concurrency: int = 1
    closed_loop_min_trials: int = 2
    max_closed_loop_concurrency: int = 128
    closed_loop_plateau_relative_gain: float = 0.05
    metrics_interval_s: float = 1.0
    window_s: float = 10.0
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    ttft_slo_field: str = "ttft_p90_ms"
    tpot_slo_field: str = "tpot_p90_ms"
    ttft_slo_mode: str = "static"
    longbench_ttft_static_preset: str | None = None
    request_reuse_policy: str = "no-repeat-across-search"
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.search_mode not in {"closed-loop", "open-loop", "hybrid"}:
            raise ValueError(f"unsupported search_mode {self.search_mode!r}")
        _require_positive_float("trial_min_duration_s", self.trial_min_duration_s)
        if self.trial_max_duration_s is not None:
            _require_positive_float("trial_max_duration_s", self.trial_max_duration_s)
            if self.trial_max_duration_s < self.trial_min_duration_s:
                raise ValueError("trial_max_duration_s must be >= trial_min_duration_s")
        if self.final_confirmation_duration_s is not None:
            _require_positive_float("final_confirmation_duration_s", self.final_confirmation_duration_s)
        _require_positive_float("rate_precision", self.rate_precision)
        if self.rate_precision >= 1.0:
            raise ValueError("rate_precision must be < 1.0")
        _require_positive_float("initial_request_rate", self.initial_request_rate)
        if self.max_request_rate is not None:
            _require_positive_float("max_request_rate", self.max_request_rate)
            if self.max_request_rate < self.initial_request_rate:
                raise ValueError("max_request_rate must be >= initial_request_rate")
        _require_positive_int("max_binary_steps", self.max_binary_steps)
        _require_positive_int("max_bracket_trials", self.max_bracket_trials)
        _require_positive_float("open_loop_bracket_growth_factor", self.open_loop_bracket_growth_factor)
        if self.open_loop_bracket_growth_factor <= 1.0:
            raise ValueError("open_loop_bracket_growth_factor must be > 1.0")
        _require_non_negative_int("client_limited_retry_attempts", self.client_limited_retry_attempts)
        _require_non_negative_float(
            "client_limited_retry_cooldown_s",
            self.client_limited_retry_cooldown_s,
        )
        _require_positive_int("closed_loop_initial_concurrency", self.closed_loop_initial_concurrency)
        _require_positive_int("closed_loop_min_trials", self.closed_loop_min_trials)
        _require_positive_int("max_closed_loop_concurrency", self.max_closed_loop_concurrency)
        if self.closed_loop_initial_concurrency > self.max_closed_loop_concurrency:
            raise ValueError("closed_loop_initial_concurrency must be <= max_closed_loop_concurrency")
        _require_positive_float("closed_loop_plateau_relative_gain", self.closed_loop_plateau_relative_gain)
        _require_positive_float("metrics_interval_s", self.metrics_interval_s)
        _require_positive_float("window_s", self.window_s)
        if self.ttft_slo_ms is not None:
            _require_positive_float("ttft_slo_ms", self.ttft_slo_ms)
        if self.tpot_slo_ms is not None:
            _require_positive_float("tpot_slo_ms", self.tpot_slo_ms)
        if self.ttft_slo_field not in {"ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms"}:
            raise ValueError(f"unsupported ttft_slo_field {self.ttft_slo_field!r}")
        if self.tpot_slo_field not in {"tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms"}:
            raise ValueError(f"unsupported tpot_slo_field {self.tpot_slo_field!r}")
        if self.ttft_slo_mode not in {"static", "length_scaled"}:
            raise ValueError(f"unsupported ttft_slo_mode {self.ttft_slo_mode!r}")
        if self.longbench_ttft_static_preset is not None:
            if self.longbench_ttft_static_preset not in {"default", "tight", "relaxed"}:
                raise ValueError(
                    "longbench_ttft_static_preset must be one of: default, tight, relaxed"
                )
            if self.ttft_slo_mode != "static":
                raise ValueError("longbench_ttft_static_preset requires ttft_slo_mode='static'")
        if self.request_reuse_policy not in {
            "cycle",
            "no-repeat-per-trial",
            "no-repeat-across-search",
            "unique-then-cycle",
        }:
            raise ValueError(f"unsupported request_reuse_policy {self.request_reuse_policy!r}")
        if self.max_num_seqs is not None:
            _require_positive_int("max_num_seqs", self.max_num_seqs)
        if self.max_num_batched_tokens is not None:
            _require_positive_int("max_num_batched_tokens", self.max_num_batched_tokens)


@dataclass(frozen=True, slots=True)
class ExperimentOverride:
    source_index: int
    model_patterns: tuple[str, ...] = ()
    workload_patterns: tuple[str, ...] = ()
    hardware_patterns: tuple[str, ...] = ()
    launch: Mapping[str, Any] | None = None
    search: Mapping[str, Any] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("source_index", self.source_index)
        if not self.model_patterns and not self.workload_patterns and not self.hardware_patterns:
            raise ValueError("override match must include model, workload, or hardware")
        if self.launch is None and self.search is None:
            raise ValueError("override must include launch or search updates")


@dataclass(frozen=True, slots=True)
class ResourceProbeResult:
    hardware_name: str
    gpu_memory_gb: float | None
    model_params_b: float | None
    estimated_weight_gb: float | None
    estimated_activation_gb: float
    estimated_kv_cache_gb: float | None
    estimated_required_gb: float | None
    usable_memory_per_gpu_gb: float | None
    required_gpu_count: int | None
    context_tokens: int
    warnings: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "hardware_name": self.hardware_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "model_params_b": self.model_params_b,
            "estimated_weight_gb": self.estimated_weight_gb,
            "estimated_activation_gb": self.estimated_activation_gb,
            "estimated_kv_cache_gb": self.estimated_kv_cache_gb,
            "estimated_required_gb": self.estimated_required_gb,
            "usable_memory_per_gpu_gb": self.usable_memory_per_gpu_gb,
            "required_gpu_count": self.required_gpu_count,
            "context_tokens": self.context_tokens,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ExperimentTemplate:
    source_index: int
    experiment_id: str | None
    models: tuple[str, ...]
    workloads: tuple[Path, ...]
    endpoint: str
    launch: LaunchConfig
    search: SearchConfig
    hardware: HardwareConfig
    probe: ProbeConfig
    overrides: tuple[ExperimentOverride, ...] = ()
    server_metadata_file: Path | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("source_index", self.source_index)
        if not self.models:
            raise ValueError("models must be non-empty")
        if not self.workloads:
            raise ValueError("workloads must be non-empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")


@dataclass(frozen=True, slots=True)
class OrchestratorManifest:
    manifest_path: Path
    run: RunConfig
    slurm: SlurmConfig
    hardware: HardwareConfig
    probe: ProbeConfig
    overrides: tuple[ExperimentOverride, ...]
    experiments: tuple[ExperimentTemplate, ...]


@dataclass(frozen=True, slots=True)
class ExpandedExperimentJob:
    experiment_id: str
    source_index: int
    model: str
    workload: Path
    endpoint: str
    launch: LaunchConfig
    search: SearchConfig
    hardware: HardwareConfig
    probe: ResourceProbeResult | None
    result_dir: Path
    model_slug: str
    dataset_slug: str
    server_config_slug: str
    server_signature_key: str
    server_metadata_file: Path | None = None


@dataclass(frozen=True, slots=True)
class GPULease:
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        for gpu_id in self.gpu_ids:
            _require_non_negative_int("gpu_id", gpu_id)

    @property
    def gpu_id(self) -> int:
        return self.gpu_ids[0]


@dataclass(frozen=True, slots=True)
class PortReservation:
    base_port: int
    metrics_port: int


@dataclass(slots=True)
class ActiveServer:
    reuse_key: str
    runtime_signature: str
    model: str
    endpoint: str
    gpu_id: int
    gpu_ids: tuple[int, ...]
    base_port: int
    metrics_port: int
    command: tuple[str, ...]
    base_url: str
    stdout_log: Path
    stderr_log: Path
    process: object
    stdout_handle: object
    stderr_handle: object


@dataclass(frozen=True, slots=True)
class SearchExecutionResult:
    success: bool
    return_code: int
    commands: tuple[tuple[str, ...], ...]
    stdout_log: Path
    stderr_log: Path
    search_trace_path: Path | None
    final_report_json_path: Path | None
    final_report_md_path: Path | None
    error: str | None = None
