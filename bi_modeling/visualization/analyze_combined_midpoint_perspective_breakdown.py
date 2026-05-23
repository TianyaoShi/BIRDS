"""Analyze combined H100+A100 midpoint contribution ratios across ReCiPe perspectives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.code import modeling  # noqa: E402
from bi_modeling.visualization.analyze_combined_midpoint_breakdown import (  # noqa: E402
    MIDPOINT_BUCKETS,
    MIDPOINT_SCOPES,
    TRACKED_MIDPOINTS,
    _bucket_midpoint_values,
)
from bi_modeling.visualization.analyze_h100_lifecycle_stage_breakdown import (  # noqa: E402
    ACCELERATOR_SPECS,
    select_most_energy_efficient_config_per_model,
)
from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    SECONDS_PER_YEAR,
    EnergyBiConfig,
    build_energy_bi_dataset,
)

PERSPECTIVE_TO_HORIZON = {
    "individualistic": 20,
    "hierarchist": 100,
    "egalitarian": 1000,
}


def _endpoint_bi_by_midpoint(midpoint_values: dict[str, float], *, perspective: str) -> dict[str, float]:
    endpoint_bi = {}
    for midpoint in MIDPOINT_SCOPES:
        isolated_midpoint = {scope: 0.0 for scope in MIDPOINT_SCOPES}
        isolated_midpoint[midpoint] = float(midpoint_values.get(midpoint, 0.0))
        endpoint_bi[midpoint] = float(
            sum(modeling.midpoint_to_endpoint(isolated_midpoint, perspective=perspective).values())
        )
    return endpoint_bi


def _midpoint_stage_impacts_per_accelerator(
    accelerator: str,
    config: EnergyBiConfig,
    *,
    perspective: str,
) -> dict[str, dict[str, float]]:
    impacts = modeling.calculate_total_impact(
        ACCELERATOR_SPECS[accelerator],
        manufacturing_only=False,
        calculate_upstream_materials=config.calculate_upstream_materials,
        include_logic_fab_scope3=config.include_logic_fab_scope3,
        perspective=perspective,
    )
    return {
        stage: impacts[stage]["midpoint"]
        for stage in ("manufacturing", "transportation", "recycling")
    }


def build_combined_midpoint_breakdown(results_dir: Path, *, config: EnergyBiConfig, perspective: str) -> pd.DataFrame:
    rows = build_energy_bi_dataset(results_dir, config=config)
    rows = rows[rows["accelerator"].isin(("H100", "A100"))].copy()

    per_accelerator_stage_midpoints = {
        accelerator: _midpoint_stage_impacts_per_accelerator(accelerator, config, perspective=perspective)
        for accelerator in ("H100", "A100")
    }
    lifetime_seconds = config.lifetime_years * SECONDS_PER_YEAR

    per_accelerator_stage_endpoint_midpoints = {
        accelerator: {
            stage: _endpoint_bi_by_midpoint(stage_midpoint, perspective=perspective)
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
        combined = _bucket_midpoint_values(
            _endpoint_bi_by_midpoint(operational_midpoint, perspective=perspective)
        )

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
    rows["recipe_perspective"] = perspective
    rows["time_horizon_years"] = PERSPECTIVE_TO_HORIZON[perspective]
    return rows


def summarize_midpoint_means(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    perspective = rows["recipe_perspective"].iloc[0]
    horizon = int(rows["time_horizon_years"].iloc[0])
    for bucket in MIDPOINT_BUCKETS:
        values = rows[f"{bucket.lower()}_ratio"]
        records.append(
            {
                "recipe_perspective": perspective,
                "time_horizon_years": horizon,
                "bucket": bucket,
                "mean": values.mean(),
            }
        )
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
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_best_rows = []
    all_summaries = []
    all_best_summaries = []
    for perspective in PERSPECTIVE_TO_HORIZON:
        rows = build_combined_midpoint_breakdown(args.results_dir, config=config, perspective=perspective)
        best_rows = select_most_energy_efficient_config_per_model(rows)
        all_rows.append(rows)
        all_best_rows.append(best_rows)
        all_summaries.append(summarize_midpoint_means(rows))
        all_best_summaries.append(summarize_midpoint_means(best_rows))

    outputs = [
        (
            "",
            pd.concat(all_rows, ignore_index=True),
            pd.concat(all_summaries, ignore_index=True),
        ),
        (
            "_best_energy_per_model",
            pd.concat(all_best_rows, ignore_index=True),
            pd.concat(all_best_summaries, ignore_index=True),
        ),
    ]
    for suffix, rows, summary in outputs:
        row_path = args.derived_dir / f"combined_h100_a100_midpoint_perspective_bi_per_request{suffix}.csv"
        summary_path = args.derived_dir / f"combined_h100_a100_midpoint_perspective_mean_ratio_summary{suffix}.csv"
        rows.to_csv(row_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {row_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
