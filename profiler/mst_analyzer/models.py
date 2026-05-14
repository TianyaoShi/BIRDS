from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Confidence = Literal["high", "medium", "low"]
ComparabilityLabel = Literal["direct", "contextual"]
SeverityLabel = Literal["high", "medium", "low"]
AnomalyFamily = Literal[
    "within_size_outlier",
    "larger_model_inversion",
    "same_family_non_monotonicity",
    "search_rate_cap_reached",
    "missing_confirmed_mst_rate",
    "trace_instability_suspect",
    "slo_driven_disagreement",
]


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TrialArtifactRef:
    trial_id: str
    trial_dir: Path | None
    summary_json: Path | None
    analysis_json: Path | None
    trial_validity: str | None
    stability_status: str | None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class TraceInstabilityEvidence:
    conflicting_rate_labels: tuple[str, ...] = ()
    majority_confirmation_used: bool = False
    uncertain_retry_count: int = 0
    low_confidence: bool = False
    suspect_termination_reason: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class MSTRow:
    experiment_id: str
    model: str
    model_family: str
    model_variant: str | None
    model_size_b: float | None
    size_bucket: str | None
    hardware: str
    workload_name: str
    workload_path: Path
    endpoint: str
    endpoint_type: str
    mst_rps: float | None
    termination_reason: str | None
    bottleneck_class: str | None
    confidence: Confidence | None
    server_signature_key: str | None
    max_num_seqs: float | None
    max_num_batched_tokens: float | None
    max_model_len: int | None
    tensor_parallel_size: int | None
    gpu_count: int | None
    dtype: str | None
    quantization: str | None
    is_quantized: bool
    is_moe: bool
    ttft_slo_ms: float | None
    tpot_slo_ms: float | None
    ttft_slo_field: str | None
    tpot_slo_field: str | None
    confirmation_trial: TrialArtifactRef | None
    confirmation_successful_completion_rate: float | None
    confirmation_total_token_throughput: float | None
    confirmation_generation_token_throughput: float | None
    confirmation_prompt_len_mean: float | None
    confirmation_output_len_mean: float | None
    high_bound_rate: float | None
    high_bound_trial: TrialArtifactRef | None
    result_dir: Path
    search_trace_path: Path
    final_report_json_path: Path | None
    trace_instability: TraceInstabilityEvidence = field(default_factory=TraceInstabilityEvidence)
    has_slo_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class ComparatorEvidence:
    relation: str
    comparison_label: ComparabilityLabel
    model: str
    experiment_id: str
    serving_config_label: str
    tensor_parallel_size: int | None
    gpu_count: int | None
    mst_rps: float
    model_size_b: float | None
    rate_ratio_vs_comparator: float
    absolute_delta_rps: float
    same_slo: bool
    variant_mismatch: bool
    confirmation_trial_id: str | None
    high_bound_trial_id: str | None
    reason: str
    search_trace_path: Path
    confirmation_summary_json: Path | None
    high_bound_summary_json: Path | None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class AnomalyCandidate:
    experiment_id: str
    model: str
    serving_config_label: str
    tensor_parallel_size: int | None
    gpu_count: int | None
    mst_rps: float
    confidence: Confidence | None
    severity_score: int
    severity: SeverityLabel
    families: tuple[AnomalyFamily, ...]
    summary: str
    reasons: tuple[str, ...]
    family_reasons: dict[str, tuple[str, ...]]
    comparators: tuple[ComparatorEvidence, ...]
    confirmation_trial_id: str | None
    high_bound_trial_id: str | None
    suggested_action: str
    search_trace_path: Path
    final_report_json_path: Path | None
    evidence_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class TraceDiagnostic:
    experiment_id: str
    model: str
    serving_config_label: str
    tensor_parallel_size: int | None
    gpu_count: int | None
    mst_rps: float | None
    confidence: Confidence | None
    reasons: tuple[str, ...]
    confirmation_trial_id: str | None
    high_bound_trial_id: str | None
    search_trace_path: Path
    evidence_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class BucketSummary:
    bucket_name: str
    comparable_group: str
    model_count: int
    median_mst_rps: float
    median_total_token_throughput: float | None
    models: tuple[str, ...]
    member_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class SuggestedRerunPlan:
    manifest_path: Path
    selected_models: tuple[str, ...]
    selected_experiment_ids: tuple[str, ...]
    selected_workloads: tuple[Path, ...]
    workload_copies: tuple[Path, ...]
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalysisArtifacts:
    rows_json_path: Path
    report_json_path: Path
    report_md_path: Path
    rerun_manifest_path: Path | None
    row_count: int
    anomaly_count: int
    trace_diagnostic_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))
