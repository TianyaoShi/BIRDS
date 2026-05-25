"""Reusable biodiversity-impact calculations for energy-summary rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from bi_modeling.code import modeling

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
DEFAULT_LIFETIME_YEARS = 5
DEFAULT_OPERATIONAL_YEAR = 2024
DEFAULT_LOCATION = "US"
DEFAULT_DATACENTER_WUE_L_PER_KWH = 0.36
DEFAULT_DATACENTER_PUE = 1.16


@dataclass(frozen=True)
class EnergyBiConfig:
    lifetime_years: int = DEFAULT_LIFETIME_YEARS
    operational_year: int = DEFAULT_OPERATIONAL_YEAR
    location: str = DEFAULT_LOCATION
    datacenter_wue_l_per_kwh: float = DEFAULT_DATACENTER_WUE_L_PER_KWH
    datacenter_pue: float = DEFAULT_DATACENTER_PUE
    use_incremental_energy: bool = True
    calculate_upstream_materials: bool = False
    include_logic_fab_scope3: bool | None = None


def discover_summary_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/**/summary_compact.csv"))


def accelerator_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "a100" in parts:
        return "A100"
    if "h100" in parts:
        return "H100"
    raise ValueError(f"Could not infer accelerator type from {path}.")


def accelerator_specs(accelerator: str) -> dict:
    key = accelerator.upper()
    if key == "A100":
        return dict(modeling.a100_40g_specs)
    if key == "H100":
        return dict(modeling.h100_specs)
    raise ValueError(f"Unsupported accelerator {accelerator!r}.")


def biodiversity_total(endpoint_impacts: dict[str, float]) -> float:
    return float(sum(endpoint_impacts.values()))


def _normalize_workload_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    normalized = normalized.replace("mmlu_pro_reasoning", "mmlu-pro-reasoning")
    normalized = normalized.replace("supergpqa_hard_reasoning", "supergpqa-hard-reasoning")
    normalized = normalized.replace(
        "longbench-medium-output-summarization",
        "longbench_medium_output_summarization",
    )
    normalized = normalized.replace(
        "longbench-medium-answer-rag-qa",
        "longbench_medium_answer_rag_qa",
    )
    normalized = normalized.replace(
        "longbench-short-answer-document-qa",
        "longbench_short_answer_document_qa",
    )
    return normalized


def _resolve_workload_from_job_summary(experiment_dir: Path, job_id: str) -> str | None:
    summary_path = experiment_dir / "jobs" / job_id / "summary.json"
    if not summary_path.exists():
        return None

    try:
        with summary_path.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    workload_meta = (
        payload.get("config", {})
        .get("metadata", {})
        .get("workload", {})
    )
    raw_name = workload_meta.get("name")
    if raw_name:
        return _normalize_workload_name(str(raw_name))

    source_path = workload_meta.get("source_path")
    if not source_path:
        return None

    source = Path(str(source_path))
    if source.stem and source.stem != "shard_000":
        return _normalize_workload_name(source.stem)
    if len(source.parents) >= 2:
        return _normalize_workload_name(source.parents[1].name)
    return None


def _resolve_generic_workloads(frame: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    if "workload" not in frame.columns or "job_id" not in frame.columns:
        return frame

    generic_mask = frame["workload"].astype(str).str.fullmatch(r"shard_\d+(?:_mst_anomaly_rerun)?")
    if not generic_mask.any():
        return frame

    resolved = {}
    experiment_dir = summary_path.parent
    for job_id in frame.loc[generic_mask, "job_id"].dropna().unique():
        workload = _resolve_workload_from_job_summary(experiment_dir, str(job_id))
        if workload:
            resolved[str(job_id)] = workload

    if not resolved:
        return frame

    updated = frame.copy()
    updated.loc[generic_mask, "workload"] = (
        updated.loc[generic_mask, "job_id"].astype(str).map(resolved).fillna(updated.loc[generic_mask, "workload"])
    )
    return updated


def operational_bi_for_energy_joules(
    energy_joules: float,
    *,
    config: EnergyBiConfig,
) -> float:
    midpoint = modeling.calculate_operational_impacts_given_energy(
        energy_joules,
        config.operational_year,
        location=config.location,
        energy_unit="J",
        datacenter_wue=config.datacenter_wue_l_per_kwh,
        pue=config.datacenter_pue,
    )
    endpoint = modeling.midpoint_to_endpoint(midpoint)
    return biodiversity_total(endpoint)


def embodied_bi_per_accelerator_second(accelerator: str, *, config: EnergyBiConfig) -> float:
    specs = accelerator_specs(accelerator)
    manufacturing = modeling.calculate_total_impact(
        specs,
        manufacturing_only=True,
        calculate_upstream_materials=config.calculate_upstream_materials,
        include_logic_fab_scope3=config.include_logic_fab_scope3,
    )
    embodied_bi = biodiversity_total(manufacturing["manufacturing"]["endpoint"])
    return embodied_bi / (config.lifetime_years * SECONDS_PER_YEAR)


def load_energy_summary_rows(summary_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in summary_files:
        frame = pd.read_csv(path)
        if "workload" in frame.columns:
            frame["workload"] = frame["workload"].astype(str).map(_normalize_workload_name)
        frame = _resolve_generic_workloads(frame, path)
        frame["summary_path"] = str(path)
        frame["accelerator"] = accelerator_from_path(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    if "status" in rows.columns:
        rows = rows[rows["status"] == "succeeded"].copy()
    return rows


def add_biodiversity_metrics(rows: pd.DataFrame, *, config: EnergyBiConfig) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    result = rows.copy()
    energy_request_column = (
        "incremental_energy_per_total_request_j"
        if config.use_incremental_energy
        else "energy_per_total_request_j"
    )
    energy_token_column = (
        "incremental_energy_per_total_token_j"
        if config.use_incremental_energy
        else "energy_per_total_token_j"
    )

    embodied_per_second = {
        accelerator: embodied_bi_per_accelerator_second(accelerator, config=config)
        for accelerator in sorted(result["accelerator"].dropna().unique())
    }

    result["tokens_per_request"] = result["total_tokens"] / result["started_requests"]
    result["operational_bi_per_request"] = result[energy_request_column].apply(
        lambda joules: operational_bi_for_energy_joules(joules, config=config)
    )
    result["operational_bi_per_token"] = result[energy_token_column].apply(
        lambda joules: operational_bi_for_energy_joules(joules, config=config)
    )
    result["embodied_bi_per_second_per_gpu"] = result["accelerator"].map(embodied_per_second)
    result["embodied_bi_per_request"] = (
        result["gpu_count"] * result["embodied_bi_per_second_per_gpu"] / result["request_rate"]
    )
    result["embodied_bi_per_token"] = result["embodied_bi_per_request"] / result["tokens_per_request"]
    result["bi_per_request"] = result["operational_bi_per_request"] + result["embodied_bi_per_request"]
    result["bi_per_token"] = result["operational_bi_per_token"] + result["embodied_bi_per_token"]
    result["bi_energy_basis"] = "incremental" if config.use_incremental_energy else "total"
    result["bi_operational_year"] = config.operational_year
    result["bi_location"] = config.location
    result["bi_datacenter_wue_l_per_kwh"] = config.datacenter_wue_l_per_kwh
    result["bi_datacenter_pue"] = config.datacenter_pue
    result["bi_lifetime_years"] = config.lifetime_years
    result["bi_calculate_upstream_materials"] = config.calculate_upstream_materials
    result["bi_include_logic_fab_scope3"] = config.include_logic_fab_scope3
    return result


def build_energy_bi_dataset(results_dir: Path, *, config: EnergyBiConfig) -> pd.DataFrame:
    return add_biodiversity_metrics(
        load_energy_summary_rows(discover_summary_files(results_dir)),
        config=config,
    )
