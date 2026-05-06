from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bottleneck import classify_bottleneck
from .records import RequestRecord, ServerMetricSample, TrialAnalysisResult, TrialSummary
from .stability import StabilityConfig, classify_stability, load_window_summaries_csv
from .windowing import FixedWindowAggregator

_OPEN_LOOP_SEND_RATE_TOLERANCE = 0.05
_SCHEDULING_DELAY_WARNING_S = 0.25


def analyze_trial_dir(
    trial_dir: str | Path,
    *,
    stability_config: StabilityConfig | None = None,
) -> TrialAnalysisResult:
    directory = Path(trial_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"trial directory not found: {directory}")

    summary_payload = _load_json_file(directory / "summary.json")
    config_payload = _require_mapping(summary_payload.get("config"), "summary.json.config")
    summary = _load_trial_summary(summary_payload)
    request_records = _load_request_records(directory / "request_records.jsonl")
    windows = load_window_summaries_csv(directory / "windows.csv")
    _validate_trial_id_consistency(summary, windows, request_records)
    server_metadata = _load_server_metadata(directory, config_payload, summary_payload)

    validity_reasons = _invalid_workload_reasons(request_records)
    if validity_reasons:
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="invalid_workload",
            validity_reasons=validity_reasons,
            stability=None,
            bottleneck=None,
        )

    if _is_client_limited(summary):
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="client_limited",
            validity_reasons=_client_limited_reasons(summary),
            stability=None,
            bottleneck=None,
        )

    server_metrics_result = _load_server_metrics(directory, summary_payload)
    if isinstance(server_metrics_result, TrialAnalysisResult):
        return server_metrics_result
    server_metrics: list[ServerMetricSample] = []
    if server_metrics_result:
        server_metrics = server_metrics_result
        windows = FixedWindowAggregator(window_s=_analysis_window_s(config_payload)).summarize(
            trial_id=summary.trial_id,
            request_records=request_records,
            server_metrics=server_metrics,
        )

    stability = classify_stability(
        windows,
        config=stability_config or _stability_config_from_metadata(config_payload),
        aborted_safety=summary.status == "aborted_safety",
        request_records=request_records,
        trial_start_ts=_trial_start_ts(request_records, server_metrics),
    )
    bottleneck = classify_bottleneck(
        windows,
        stability_result=stability,
        trial_summary=summary,
        request_records=request_records,
        server_metadata=server_metadata,
    )

    if bottleneck.bottleneck_class == "client_limited":
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="client_limited",
            validity_reasons=list(bottleneck.evidence),
            stability=None,
            bottleneck=None,
        )

    return TrialAnalysisResult(
        trial_id=summary.trial_id,
        trial_validity="valid",
        validity_reasons=["artifacts were well-formed and workload validity checks passed"],
        stability=stability,
        bottleneck=bottleneck,
    )


def write_analysis_artifact(trial_dir: str | Path, result: TrialAnalysisResult) -> Path:
    output_path = Path(trial_dir) / "analysis.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _load_trial_summary(summary_payload: Mapping[str, Any]) -> TrialSummary:
    summary_mapping = _require_mapping(summary_payload.get("summary"), "summary.json.summary")
    return TrialSummary(**summary_mapping)


def _load_request_records(path: Path) -> list[RequestRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"request records artifact not found: {path}")
    records: list[RequestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"{path}:{line_idx}: request record line must not be blank")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_idx}: request record must decode to an object")
            records.append(RequestRecord(**payload))
    if not records:
        raise ValueError(f"request records artifact is empty: {path}")
    return records


def _load_server_metrics(
    trial_dir: Path,
    summary_payload: Mapping[str, Any],
) -> list[ServerMetricSample] | TrialAnalysisResult | None:
    summary = _load_trial_summary(summary_payload)
    config_payload = _require_mapping(summary_payload.get("config"), "summary.json.config")
    metrics_url = config_payload.get("metrics_url")
    path = trial_dir / "server_metrics.jsonl"
    if not path.exists():
        if metrics_url is not None:
            return TrialAnalysisResult(
                trial_id=summary.trial_id,
                trial_validity="metrics_invalid",
                validity_reasons=[
                    f"metrics_url was configured but server_metrics.jsonl is missing: {path}",
                    "server-side evidence is required for confident search decisions",
                ],
                stability=None,
                bottleneck=None,
            )
        return None

    samples: list[ServerMetricSample] = []
    poll_errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"{path}:{line_idx}: server metric line must not be blank")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_idx}: server metric must decode to an object")
            raw = payload.get("raw")
            if isinstance(raw, Mapping) and raw.get("poll_error") is not None:
                poll_errors.append(str(raw["poll_error"]))
                continue
            samples.append(ServerMetricSample(**payload))

    if poll_errors:
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="metrics_invalid",
            validity_reasons=[
                "saved server metrics contain Prometheus poll failures: "
                f"{len(poll_errors)} sample(s), first={poll_errors[0]!r}",
                "server-side evidence is unavailable; search bounds must not be updated",
            ],
            stability=None,
            bottleneck=None,
        )
    if not samples:
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="metrics_invalid",
            validity_reasons=[
                f"server metrics artifact is empty: {path}",
                "server-side evidence is required for confident search decisions",
            ],
            stability=None,
            bottleneck=None,
        )
    if summary.metrics_sample_count != len(samples):
        return TrialAnalysisResult(
            trial_id=summary.trial_id,
            trial_validity="metrics_invalid",
            validity_reasons=[
                "summary metrics_sample_count does not match server_metrics.jsonl: "
                f"{summary.metrics_sample_count} != {len(samples)}",
                "server-side evidence is inconsistent; search bounds must not be updated",
            ],
            stability=None,
            bottleneck=None,
        )
    return samples


def _validate_trial_id_consistency(
    summary: TrialSummary,
    windows: Sequence[Any],
    request_records: Sequence[RequestRecord],
) -> None:
    for window in windows:
        if window.trial_id != summary.trial_id:
            raise ValueError(
                "trial_id mismatch between summary.json and windows.csv: "
                f"{summary.trial_id!r} != {window.trial_id!r}"
            )
    for record in request_records:
        if record.trial_id != summary.trial_id:
            raise ValueError(
                "trial_id mismatch between summary.json and request_records.jsonl: "
                f"{summary.trial_id!r} != {record.trial_id!r}"
            )


def _analysis_window_s(config_payload: Mapping[str, Any]) -> float:
    raw_window_s = config_payload.get("window_s", 10.0)
    if not isinstance(raw_window_s, (int, float)):
        raise ValueError("summary.json.config.window_s must be numeric when provided")
    return float(raw_window_s)


def _trial_start_ts(
    request_records: Sequence[RequestRecord],
    server_metrics: Sequence[ServerMetricSample],
) -> float:
    request_start = min(
        record.actual_send_ts
        for record in request_records
        if record.actual_send_ts is not None
    )
    metric_start = min((sample.ts for sample in server_metrics), default=float("inf"))
    return min(request_start, metric_start)


def _load_server_metadata(
    trial_dir: Path,
    config_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
) -> Mapping[str, object] | None:
    candidates: list[Mapping[str, object]] = []

    config_metadata = config_payload.get("metadata")
    if isinstance(config_metadata, Mapping):
        for key in ("server_metadata", "server_config", "vllm_config"):
            value = config_metadata.get(key)
            if value is not None:
                candidates.append(_require_mapping(value, f"summary.json.config.metadata.{key}"))

    for key in ("server_metadata", "server_config", "vllm_config"):
        value = summary_payload.get(key)
        if value is not None:
            candidates.append(_require_mapping(value, f"summary.json.{key}"))

    for filename in ("server_metadata.json", "server_config.json", "vllm_config.json"):
        path = trial_dir / filename
        if path.exists():
            candidates.append(_load_json_mapping_file(path))

    if not candidates:
        return None

    merged: dict[str, object] = {}
    for candidate in candidates:
        merged = _merge_mappings(merged, candidate)
    return merged


def _stability_config_from_metadata(config_payload: Mapping[str, Any]) -> StabilityConfig:
    metadata = config_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return StabilityConfig()
    payload = metadata.get("stability_policy")
    if payload is None:
        return StabilityConfig()
    policy = _require_mapping(payload, "summary.json.config.metadata.stability_policy")
    allowed = {
        "warmup_windows",
        "min_eval_windows",
        "completion_arrival_tolerance",
        "max_positive_backlog_slope",
        "min_backlog_growth_for_hard_pressure",
        "min_backlog_relative_increase",
        "backlog_trend_alpha",
        "min_waiting_queue_mean_for_pressure",
        "min_waiting_queue_active_fraction",
        "token_throughput_plateau_relative_growth",
        "max_error_rate",
        "ttft_slo_ms",
        "tpot_slo_ms",
        "ttft_slo_field",
        "tpot_slo_field",
        "ttft_slo_mode",
        "longbench_ttft_static_preset",
    }
    unknown = set(policy) - allowed
    if unknown:
        raise ValueError(f"stability_policy has unknown keys: {sorted(unknown)}")
    return StabilityConfig(**dict(policy))


def _merge_mappings(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    path: str = "server_metadata",
) -> dict[str, object]:
    merged = dict(left)
    for key, right_value in right.items():
        current_path = f"{path}.{key}"
        if key not in merged:
            merged[key] = right_value
            continue
        left_value = merged[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            merged[key] = _merge_mappings(
                _require_mapping(left_value, current_path),
                _require_mapping(right_value, current_path),
                path=current_path,
            )
            continue
        if left_value != right_value:
            raise ValueError(
                f"conflicting recorded server metadata for {current_path}: "
                f"{left_value!r} != {right_value!r}"
            )
    return merged


def _client_limited_reasons(summary: TrialSummary) -> list[str]:
    reasons: list[str] = []
    if summary.status == "aborted_safety":
        reasons.append(
            "trial was aborted by the client safety cap"
            + (f": {summary.abort_reason}" if summary.abort_reason else "")
        )
    if (
        summary.mode == "open-loop"
        and summary.requested_request_rate is not None
        and summary.actual_send_rate
        < summary.requested_request_rate * (1.0 - _OPEN_LOOP_SEND_RATE_TOLERANCE)
    ):
        reasons.append(
            "actual open-loop send rate lagged the configured rate: "
            f"actual={summary.actual_send_rate:.3f} req/s, "
            f"configured={summary.requested_request_rate:.3f} req/s"
        )
    if (
        summary.max_scheduling_delay_s is not None
        and summary.max_scheduling_delay_s > _SCHEDULING_DELAY_WARNING_S
    ):
        reasons.append(
            "client scheduling delay was elevated: "
            f"max_scheduling_delay_s={summary.max_scheduling_delay_s:.3f}"
        )
    if not reasons:
        raise ValueError("client-limited analysis requires explicit evidence")
    return reasons


def _is_client_limited(summary: TrialSummary) -> bool:
    if summary.status == "aborted_safety":
        return True
    if summary.mode != "open-loop" or summary.requested_request_rate is None:
        return False
    if summary.actual_send_rate >= summary.requested_request_rate * (1.0 - _OPEN_LOOP_SEND_RATE_TOLERANCE):
        return False
    return (
        summary.max_scheduling_delay_s is not None
        and summary.max_scheduling_delay_s > _SCHEDULING_DELAY_WARNING_S
    )


def _invalid_workload_reasons(request_records: Sequence[RequestRecord]) -> list[str]:
    failing_request_ids = [
        record.request_id
        for record in request_records
        if record.error is not None
        and any(marker in record.error.lower() for marker in _CONTEXT_ERROR_MARKERS)
    ]
    if not failing_request_ids:
        return []
    return [
        "saved request records contain model/request validation failures that invalidate the workload: "
        f"{len(failing_request_ids)} request(s), first={failing_request_ids[0]!r}",
        "context-length or model-validation failures are excluded from stability and bottleneck inference",
    ]


def _load_json_file(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, str(path))


def _load_json_mapping_file(path: Path) -> Mapping[str, object]:
    payload = _load_json_file(path)
    return _require_mapping(payload, str(path))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


_CONTEXT_ERROR_MARKERS = (
    "context length",
    "context window",
    "max_model_len",
    "maximum context",
    "maximum sequence length",
    "model validation",
    "validation error",
    "prompt is too long",
    "prompt too long",
    "tokens exceeds",
    "token count exceeds",
)


__all__ = ["analyze_trial_dir", "write_analysis_artifact"]
