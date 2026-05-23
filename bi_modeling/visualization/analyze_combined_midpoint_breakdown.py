"""Analyze combined H100+A100 midpoint-attributed BI ratio breakdowns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.code import modeling  # noqa: E402
from bi_modeling.visualization.analyze_h100_lifecycle_stage_breakdown import (  # noqa: E402
    ACCELERATOR_SPECS,
    select_most_energy_efficient_config_per_model,
)
from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    SECONDS_PER_YEAR,
    EnergyBiConfig,
    build_energy_bi_dataset,
)

MIDPOINT_BUCKETS = ("GWP", "WC", "AP", "POFP", "Others")
TRACKED_MIDPOINTS = {"GWP", "WC", "AP", "POFP"}
MIDPOINT_SCOPES = tuple(modeling.MIDPOINT_SCOPES)


def _midpoint_stage_impacts_per_accelerator(accelerator: str, config: EnergyBiConfig) -> dict[str, dict[str, float]]:
    impacts = modeling.calculate_total_impact(
        ACCELERATOR_SPECS[accelerator],
        manufacturing_only=False,
        calculate_upstream_materials=config.calculate_upstream_materials,
        include_logic_fab_scope3=config.include_logic_fab_scope3,
    )
    return {
        stage: impacts[stage]["midpoint"]
        for stage in ("manufacturing", "transportation", "recycling")
    }


def _bucket_midpoint_values(midpoint_values: dict[str, float]) -> dict[str, float]:
    result = {bucket: 0.0 for bucket in MIDPOINT_BUCKETS}
    for midpoint, value in midpoint_values.items():
        if midpoint in TRACKED_MIDPOINTS:
            result[midpoint] += float(value)
        else:
            result["Others"] += float(value)
    return result


def _endpoint_bi_by_midpoint(midpoint_values: dict[str, float]) -> dict[str, float]:
    endpoint_bi = {}
    for midpoint in MIDPOINT_SCOPES:
        value = float(midpoint_values.get(midpoint, 0.0))
        isolated_midpoint = {scope: 0.0 for scope in MIDPOINT_SCOPES}
        isolated_midpoint[midpoint] = value
        endpoint_bi[midpoint] = float(sum(modeling.midpoint_to_endpoint(isolated_midpoint).values()))
    return endpoint_bi


def build_combined_midpoint_breakdown(results_dir: Path, *, config: EnergyBiConfig) -> pd.DataFrame:
    rows = build_energy_bi_dataset(results_dir, config=config)
    rows = rows[rows["accelerator"].isin(("H100", "A100"))].copy()

    per_accelerator_stage_midpoints = {
        accelerator: _midpoint_stage_impacts_per_accelerator(accelerator, config)
        for accelerator in ("H100", "A100")
    }
    lifetime_seconds = config.lifetime_years * SECONDS_PER_YEAR

    per_accelerator_stage_endpoint_midpoints = {
        accelerator: {
            stage: _endpoint_bi_by_midpoint(stage_midpoint)
            for stage, stage_midpoint in per_accelerator_stage_midpoints[accelerator].items()
        }
        for accelerator in ("H100", "A100")
    }

    midpoint_columns = []
    for bucket in MIDPOINT_BUCKETS:
        column = f"{bucket.lower()}_bi_per_request"
        midpoint_columns.append(column)
        rows[column] = 0.0

    for index, row in rows.iterrows():
        operational_midpoint = modeling.calculate_operational_impacts_given_energy(
            row["incremental_energy_per_total_request_j"],
            config.operational_year,
            location=config.location,
            energy_unit="J",
            datacenter_wue=config.datacenter_wue_l_per_kwh,
            pue=config.datacenter_pue,
        )
        combined = _bucket_midpoint_values(_endpoint_bi_by_midpoint(operational_midpoint))

        for stage in ("manufacturing", "transportation", "recycling"):
            stage_midpoint = per_accelerator_stage_endpoint_midpoints[row["accelerator"]][stage]
            stage_scale = row["gpu_count"] / lifetime_seconds / row["request_rate"]
            for bucket, value in _bucket_midpoint_values(stage_midpoint).items():
                combined[bucket] += stage_scale * value

        for bucket in MIDPOINT_BUCKETS:
            rows.at[index, f"{bucket.lower()}_bi_per_request"] = combined[bucket]

    rows["midpoint_bi_total_per_request"] = rows[midpoint_columns].sum(axis=1)
    for bucket in MIDPOINT_BUCKETS:
        rows[f"{bucket.lower()}_ratio"] = (
            rows[f"{bucket.lower()}_bi_per_request"] / rows["midpoint_bi_total_per_request"]
        )

    return rows


def summarize_midpoint_ratios(rows: pd.DataFrame) -> pd.DataFrame:
    percentiles = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    records = []
    for bucket in MIDPOINT_BUCKETS:
        values = rows[f"{bucket.lower()}_ratio"]
        record = {
            "bucket": bucket,
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
        }
        for percentile in percentiles:
            record[f"p{int(percentile * 100):02d}"] = values.quantile(percentile)
        records.append(record)
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "derived",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EnergyBiConfig()
    rows = build_combined_midpoint_breakdown(args.results_dir, config=config)
    best_rows = select_most_energy_efficient_config_per_model(rows)
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("", rows),
        ("_best_energy_per_model", best_rows),
    ]
    for suffix, selected_rows in outputs:
        row_path = args.derived_dir / f"combined_h100_a100_midpoint_bi_per_request{suffix}.csv"
        summary_path = args.derived_dir / f"combined_h100_a100_midpoint_ratio_summary{suffix}.csv"
        selected_rows.to_csv(row_path, index=False)
        summarize_midpoint_ratios(selected_rows).to_csv(summary_path, index=False)
        print(f"Wrote {row_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
