from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_KNOWN_FAMILIES = {
    "within_size_outlier",
    "larger_model_inversion",
    "same_family_non_monotonicity",
    "trace_instability_suspect",
    "slo_driven_disagreement",
}


@dataclass(frozen=True, slots=True)
class OutlierBandConfig:
    min_rate: float
    max_rate: float | None
    ratio_threshold: float
    absolute_delta_rps: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutlierBandConfig":
        min_rate = _expect_float(payload.get("min_rate"), "outlier_band.min_rate")
        max_rate_raw = payload.get("max_rate")
        max_rate = None if max_rate_raw is None else _expect_float(max_rate_raw, "outlier_band.max_rate")
        ratio_threshold = _expect_float(payload.get("ratio_threshold"), "outlier_band.ratio_threshold")
        absolute_delta_rps = _expect_float(
            payload.get("absolute_delta_rps"),
            "outlier_band.absolute_delta_rps",
        )
        return cls(
            min_rate=min_rate,
            max_rate=max_rate,
            ratio_threshold=ratio_threshold,
            absolute_delta_rps=absolute_delta_rps,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyzerSuppressions:
    disable_families: tuple[str, ...] = ()
    suppress_trace_instability_below_rps: float | None = None
    suppress_contextual_only_findings: bool = False
    suppress_quantized_bucket_verdicts: bool = True
    suppress_moe_bucket_verdicts: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyzerSuppressions":
        raw_disable = payload.get("disable_families", ())
        if not isinstance(raw_disable, list):
            raise ValueError("suppressions.disable_families must be a list when provided")
        disable_families = []
        for family in raw_disable:
            if not isinstance(family, str):
                raise ValueError("suppressions.disable_families entries must be strings")
            if family not in _KNOWN_FAMILIES:
                raise ValueError(f"unknown analyzer family in suppressions.disable_families: {family}")
            disable_families.append(family)
        raw_threshold = payload.get("suppress_trace_instability_below_rps")
        suppress_trace_instability_below_rps = (
            None
            if raw_threshold is None
            else _expect_float(raw_threshold, "suppressions.suppress_trace_instability_below_rps")
        )
        return cls(
            disable_families=tuple(disable_families),
            suppress_trace_instability_below_rps=suppress_trace_instability_below_rps,
            suppress_contextual_only_findings=_expect_bool(
                payload.get("suppress_contextual_only_findings", False),
                "suppressions.suppress_contextual_only_findings",
            ),
            suppress_quantized_bucket_verdicts=_expect_bool(
                payload.get("suppress_quantized_bucket_verdicts", True),
                "suppressions.suppress_quantized_bucket_verdicts",
            ),
            suppress_moe_bucket_verdicts=_expect_bool(
                payload.get("suppress_moe_bucket_verdicts", True),
                "suppressions.suppress_moe_bucket_verdicts",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyzerSettings:
    outlier_bands: tuple[OutlierBandConfig, ...] = (
        OutlierBandConfig(min_rate=10.0, max_rate=None, ratio_threshold=1.5, absolute_delta_rps=5.0),
        OutlierBandConfig(min_rate=2.0, max_rate=10.0, ratio_threshold=1.5, absolute_delta_rps=1.0),
        OutlierBandConfig(min_rate=0.0, max_rate=2.0, ratio_threshold=2.5, absolute_delta_rps=1.0),
    )
    larger_model_min_size_ratio: float = 1.5
    larger_model_max_relative_rate: float = 1.15
    larger_model_min_rate_for_relative_compare: float = 2.0
    larger_model_min_absolute_delta_rps: float = 1.0
    same_family_max_relative_rate: float = 0.8
    same_family_min_rate: float = 2.0
    trace_instability_min_uncertain_retries: int = 2
    trace_instability_require_low_confidence_uncertain_retries: int = 1
    severity_weight_within_size_outlier: int = 30
    severity_weight_larger_model_inversion: int = 25
    severity_weight_same_family_non_monotonicity: int = 20
    severity_weight_trace_instability_suspect: int = 15
    severity_weight_low_confidence_result: int = 10
    severity_penalty_all_below_one_rps: int = 20
    severity_penalty_slo_mismatch: int = 10
    severity_penalty_variant_mismatch: int = 10
    suppressions: AnalyzerSuppressions = field(default_factory=AnalyzerSuppressions)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyzerSettings":
        bands_payload = payload.get("outlier_bands")
        if bands_payload is None:
            outlier_bands = cls().outlier_bands
        else:
            if not isinstance(bands_payload, list) or not bands_payload:
                raise ValueError("outlier_bands must be a non-empty list when provided")
            outlier_bands = tuple(
                OutlierBandConfig.from_dict(_expect_mapping(item, "outlier_bands[]"))
                for item in bands_payload
            )
        raw_suppressions = payload.get("suppressions", {})
        suppressions = AnalyzerSuppressions.from_dict(
            _expect_mapping(raw_suppressions, "suppressions")
        )
        return cls(
            outlier_bands=outlier_bands,
            larger_model_min_size_ratio=_expect_float(
                payload.get("larger_model_min_size_ratio", cls().larger_model_min_size_ratio),
                "larger_model_min_size_ratio",
            ),
            larger_model_max_relative_rate=_expect_float(
                payload.get("larger_model_max_relative_rate", cls().larger_model_max_relative_rate),
                "larger_model_max_relative_rate",
            ),
            larger_model_min_rate_for_relative_compare=_expect_float(
                payload.get(
                    "larger_model_min_rate_for_relative_compare",
                    cls().larger_model_min_rate_for_relative_compare,
                ),
                "larger_model_min_rate_for_relative_compare",
            ),
            larger_model_min_absolute_delta_rps=_expect_float(
                payload.get(
                    "larger_model_min_absolute_delta_rps",
                    cls().larger_model_min_absolute_delta_rps,
                ),
                "larger_model_min_absolute_delta_rps",
            ),
            same_family_max_relative_rate=_expect_float(
                payload.get("same_family_max_relative_rate", cls().same_family_max_relative_rate),
                "same_family_max_relative_rate",
            ),
            same_family_min_rate=_expect_float(
                payload.get("same_family_min_rate", cls().same_family_min_rate),
                "same_family_min_rate",
            ),
            trace_instability_min_uncertain_retries=_expect_int(
                payload.get(
                    "trace_instability_min_uncertain_retries",
                    cls().trace_instability_min_uncertain_retries,
                ),
                "trace_instability_min_uncertain_retries",
            ),
            trace_instability_require_low_confidence_uncertain_retries=_expect_int(
                payload.get(
                    "trace_instability_require_low_confidence_uncertain_retries",
                    cls().trace_instability_require_low_confidence_uncertain_retries,
                ),
                "trace_instability_require_low_confidence_uncertain_retries",
            ),
            severity_weight_within_size_outlier=_expect_int(
                payload.get(
                    "severity_weight_within_size_outlier",
                    cls().severity_weight_within_size_outlier,
                ),
                "severity_weight_within_size_outlier",
            ),
            severity_weight_larger_model_inversion=_expect_int(
                payload.get(
                    "severity_weight_larger_model_inversion",
                    cls().severity_weight_larger_model_inversion,
                ),
                "severity_weight_larger_model_inversion",
            ),
            severity_weight_same_family_non_monotonicity=_expect_int(
                payload.get(
                    "severity_weight_same_family_non_monotonicity",
                    cls().severity_weight_same_family_non_monotonicity,
                ),
                "severity_weight_same_family_non_monotonicity",
            ),
            severity_weight_trace_instability_suspect=_expect_int(
                payload.get(
                    "severity_weight_trace_instability_suspect",
                    cls().severity_weight_trace_instability_suspect,
                ),
                "severity_weight_trace_instability_suspect",
            ),
            severity_weight_low_confidence_result=_expect_int(
                payload.get(
                    "severity_weight_low_confidence_result",
                    cls().severity_weight_low_confidence_result,
                ),
                "severity_weight_low_confidence_result",
            ),
            severity_penalty_all_below_one_rps=_expect_int(
                payload.get(
                    "severity_penalty_all_below_one_rps",
                    cls().severity_penalty_all_below_one_rps,
                ),
                "severity_penalty_all_below_one_rps",
            ),
            severity_penalty_slo_mismatch=_expect_int(
                payload.get(
                    "severity_penalty_slo_mismatch",
                    cls().severity_penalty_slo_mismatch,
                ),
                "severity_penalty_slo_mismatch",
            ),
            severity_penalty_variant_mismatch=_expect_int(
                payload.get(
                    "severity_penalty_variant_mismatch",
                    cls().severity_penalty_variant_mismatch,
                ),
                "severity_penalty_variant_mismatch",
            ),
            suppressions=suppressions,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outlier_bands"] = [band.to_dict() for band in self.outlier_bands]
        payload["suppressions"] = self.suppressions.to_dict()
        return payload


def load_settings(path: str | Path) -> AnalyzerSettings:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return AnalyzerSettings()
    return AnalyzerSettings.from_dict(_expect_mapping(payload, "settings_yaml"))


def _expect_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _expect_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _expect_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value
