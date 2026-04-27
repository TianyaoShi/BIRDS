from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


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
        if self.max_active_gpus > 3:
            raise ValueError("V1 supports at most 3 active GPUs")
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
    max_num_seqs: float | None = None
    max_num_batched_tokens: float | None = None
    host: str = "127.0.0.1"
    readiness_path: str = "/v1/models"
    readiness_timeout_s: float = 180.0
    readiness_interval_s: float = 2.0

    def __post_init__(self) -> None:
        if self.template is not None and len(self.template) == 0:
            raise ValueError("launch template cannot be empty")
        if not self.executable:
            raise ValueError("launch executable must be non-empty")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.gpu_count != 1:
            raise ValueError("V1 supports only single-GPU jobs (gpu_count must be 1)")
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
    metrics_interval_s: float = 1.0
    window_s: float = 10.0
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    ttft_slo_field: str = "ttft_p90_ms"
    tpot_slo_field: str = "tpot_p90_ms"

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


@dataclass(frozen=True, slots=True)
class ExperimentTemplate:
    source_index: int
    experiment_id: str | None
    models: tuple[str, ...]
    workloads: tuple[Path, ...]
    endpoint: str
    launch: LaunchConfig
    search: SearchConfig
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
    result_dir: Path
    model_slug: str
    dataset_slug: str
    server_config_slug: str
    server_signature_key: str
    server_metadata_file: Path | None = None


@dataclass(frozen=True, slots=True)
class GPULease:
    gpu_id: int


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
