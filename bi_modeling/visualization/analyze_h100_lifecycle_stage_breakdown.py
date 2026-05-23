"""Analyze GPU BI-per-request lifecycle-stage breakdowns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.code import modeling  # noqa: E402
from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    SECONDS_PER_YEAR,
    EnergyBiConfig,
    biodiversity_total,
    build_energy_bi_dataset,
    operational_bi_for_energy_joules,
)

STAGES = ("operational", "manufacturing", "transportation", "recycling")
PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
ACCELERATOR_SPECS = {
    "H100": modeling.h100_specs,
    "A100": modeling.a100_40g_specs,
}


def _stage_bi_per_accelerator(accelerator: str, config: EnergyBiConfig) -> dict[str, float]:
    impacts = modeling.calculate_total_impact(
        ACCELERATOR_SPECS[accelerator],
        manufacturing_only=False,
        calculate_upstream_materials=config.calculate_upstream_materials,
        include_logic_fab_scope3=config.include_logic_fab_scope3,
    )
    return {
        stage: biodiversity_total(impacts[stage]["endpoint"])
        for stage in ("manufacturing", "transportation", "recycling")
    }


def build_lifecycle_breakdown(
    results_dir: Path,
    *,
    config: EnergyBiConfig,
    accelerators: tuple[str, ...],
) -> pd.DataFrame:
    rows = build_energy_bi_dataset(results_dir, config=config)
    rows = rows[rows["accelerator"].isin(accelerators)].copy()
    stage_bi = {
        accelerator: _stage_bi_per_accelerator(accelerator, config)
        for accelerator in accelerators
    }
    lifetime_seconds = config.lifetime_years * SECONDS_PER_YEAR

    rows["operational_bi_per_request"] = rows["incremental_energy_per_total_request_j"].apply(
        lambda joules: operational_bi_for_energy_joules(joules, config=config)
    )
    for stage in ("manufacturing", "transportation", "recycling"):
        rows[f"{stage}_bi_per_request"] = (
            rows.apply(
                lambda row: row["gpu_count"]
                * stage_bi[row["accelerator"]][stage]
                / lifetime_seconds
                / row["request_rate"],
                axis=1,
            )
        )

    stage_columns = [f"{stage}_bi_per_request" for stage in STAGES]
    rows["total_lifecycle_bi_per_request"] = rows[stage_columns].sum(axis=1)
    for stage in STAGES:
        rows[f"{stage}_ratio"] = (
            rows[f"{stage}_bi_per_request"] / rows["total_lifecycle_bi_per_request"]
        )

    return rows


def summarize_stage_ratios(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for stage in STAGES:
        values = rows[f"{stage}_ratio"]
        record = {
            "stage": stage,
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
        }
        for percentile in PERCENTILES:
            record[f"p{int(percentile * 100):02d}"] = values.quantile(percentile)
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_stage_bi_per_request(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for stage in STAGES:
        values = rows[f"{stage}_bi_per_request"]
        record = {
            "stage": stage,
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
        }
        for percentile in PERCENTILES:
            record[f"p{int(percentile * 100):02d}"] = values.quantile(percentile)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _format_scientific(value: float) -> str:
    return f"{value:.6e}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.6f}%"


def _markdown_table(df: pd.DataFrame, *, percent: bool) -> str:
    columns = ["stage", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    formatter = _format_percent if percent else _format_scientific
    for _, row in df[columns].iterrows():
        values = [str(row["stage"])] + [formatter(float(row[column])) for column in columns[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    rows: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    bi_summary: pd.DataFrame,
    report_path: Path,
    *,
    config: EnergyBiConfig,
    selection_note: str,
    title: str,
    accelerator_note: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    operational_ratio = ratio_summary.loc[ratio_summary["stage"] == "operational"].iloc[0]
    text = f"""# {title}

Rows analyzed: {len(rows)}

Scope and assumptions:
- Accelerators: {accelerator_note}.
- Selection: {selection_note}
- Operational BI uses incremental energy per total request.
- Operational year/location: {config.operational_year}, {config.location}.
- Datacenter WUE/PUE: {config.datacenter_wue_l_per_kwh} L/kWh, {config.datacenter_pue}.
- Embodied lifecycle stages use accelerator-specific total-impact endpoint BI allocated uniformly across a {config.lifetime_years}-year lifetime, then scaled by `gpu_count / request_rate`.
- Upstream material shortcut: `calculate_upstream_materials={config.calculate_upstream_materials}`, `include_logic_fab_scope3={config.include_logic_fab_scope3}`.

Operational stage contribution range:
- min: {_format_percent(float(operational_ratio["min"]))}
- median: {_format_percent(float(operational_ratio["p50"]))}
- max: {_format_percent(float(operational_ratio["max"]))}

## Stage Contribution Ratio To Total BI

{_markdown_table(ratio_summary, percent=True)}

## BI Per Request By Stage

Units: species·yr/request.

{_markdown_table(bi_summary, percent=False)}
"""
    report_path.write_text(text, encoding="utf-8")


def select_most_energy_efficient_config_per_model(rows: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "model",
        "incremental_energy_per_total_request_j",
        "energy_per_total_request_j",
        "gpu_count",
        "tensor_parallel_size",
        "request_rate",
    ]
    ascending = [True, True, True, True, True, False]
    return (
        rows.sort_values(sort_columns, ascending=ascending)
        .groupby("model", as_index=False, sort=False)
        .head(1)
        .sort_values(["model"])
        .reset_index(drop=True)
    )


def write_low_operational_ratio_audit(
    rows: pd.DataFrame,
    output_csv: Path,
    output_md: Path,
    *,
    threshold: float = 0.95,
    title: str = "H100 Configs With Operational BI Ratio Below 95%",
) -> None:
    audit = rows[rows["operational_ratio"] < threshold].copy()
    audit = audit.sort_values(["operational_ratio", "model", "workload", "request_rate"])
    columns = [
        "job_id",
        "source_experiment_id",
        "model",
        "workload",
        "gpu_count",
        "tensor_parallel_size",
        "request_rate",
        "mst_rate",
        "tokens_per_request",
        "started_requests",
        "successful_requests",
        "incremental_energy_per_total_request_j",
        "energy_per_total_request_j",
        "incremental_energy_per_total_token_j",
        "energy_per_total_token_j",
        "operational_bi_per_request",
        "manufacturing_bi_per_request",
        "transportation_bi_per_request",
        "recycling_bi_per_request",
        "total_lifecycle_bi_per_request",
        "operational_ratio",
        "manufacturing_ratio",
        "transportation_ratio",
        "recycling_ratio",
        "summary_path",
    ]
    audit = audit[columns]
    audit.to_csv(output_csv, index=False)

    def fmt_num(value: float) -> str:
        return f"{value:.6g}"

    table_columns = [
        "model",
        "workload",
        "gpu_count",
        "request_rate",
        "tokens_per_request",
        "incremental_energy_per_total_request_j",
        "energy_per_total_request_j",
        "operational_bi_per_request",
        "manufacturing_bi_per_request",
        "transportation_bi_per_request",
        "recycling_bi_per_request",
        "operational_ratio",
        "manufacturing_ratio",
        "transportation_ratio",
        "recycling_ratio",
    ]
    headers = [
        "model",
        "workload",
        "gpus",
        "req/s",
        "tok/req",
        "incr J/req",
        "total J/req",
        "op BI/req",
        "mfg BI/req",
        "transport BI/req",
        "recycle BI/req",
        "op %",
        "mfg %",
        "transport %",
        "recycle %",
    ]
    lines = [
        f"# {title}",
        "",
        f"Rows: {len(audit)}",
        "",
        f"Filter: `operational_ratio < {threshold}`.",
        "",
        "Units: energy/request is J/request; BI values are species·yr/request.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in audit.iterrows():
        values = []
        for column in table_columns:
            value = row[column]
            if column.endswith("_bi_per_request"):
                values.append(_format_scientific(float(value)))
            elif column.endswith("_ratio"):
                values.append(_format_percent(float(value)))
            elif column in {
                "request_rate",
                "tokens_per_request",
                "incremental_energy_per_total_request_j",
                "energy_per_total_request_j",
            }:
                values.append(fmt_num(float(value)))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    analysis_sets = [
        (
            "h100",
            ("H100",),
            "H100 Lifecycle-Stage BI Per Request Breakdown",
            "H100 records only from `results/energy/**/summary_compact.csv`",
            "H100 Configs With Operational BI Ratio Below 95%",
            "all H100 succeeded energy-summary rows",
        ),
        (
            "combined_h100_a100",
            ("H100", "A100"),
            "Combined H100+A100 Lifecycle-Stage BI Per Request Breakdown",
            "H100 and A100 records from `results/energy/**/summary_compact.csv`",
            "Combined H100+A100 Configs With Operational BI Ratio Below 95%",
            "all succeeded H100 and A100 energy-summary rows",
        ),
    ]
    for prefix, accelerators, title, accelerator_note, audit_title, all_rows_note in analysis_sets:
        rows = build_lifecycle_breakdown(args.results_dir, config=config, accelerators=accelerators)
        output_sets = [
            (
                "",
                rows,
                all_rows_note,
            ),
            (
                "_best_energy_per_model",
                select_most_energy_efficient_config_per_model(rows),
                "one row per model, selected by minimum incremental energy per total request",
            ),
        ]

        for suffix, selected_rows, selection_note in output_sets:
            ratio_summary = summarize_stage_ratios(selected_rows)
            bi_summary = summarize_stage_bi_per_request(selected_rows)

            row_path = args.derived_dir / f"{prefix}_lifecycle_stage_bi_per_request{suffix}.csv"
            ratio_path = args.derived_dir / f"{prefix}_lifecycle_stage_ratio_summary{suffix}.csv"
            bi_path = args.derived_dir / f"{prefix}_lifecycle_stage_bi_per_request_summary{suffix}.csv"
            report_path = args.derived_dir / f"{prefix}_lifecycle_stage_breakdown_report{suffix}.md"
            low_ratio_csv = args.derived_dir / f"{prefix}_lifecycle_stage_low_operational_ratio_audit{suffix}.csv"
            low_ratio_md = args.derived_dir / f"{prefix}_lifecycle_stage_low_operational_ratio_audit{suffix}.md"

            selected_rows.to_csv(row_path, index=False)
            ratio_summary.to_csv(ratio_path, index=False)
            bi_summary.to_csv(bi_path, index=False)
            write_report(
                selected_rows,
                ratio_summary,
                bi_summary,
                report_path,
                config=config,
                selection_note=selection_note,
                title=title,
                accelerator_note=accelerator_note,
            )
            write_low_operational_ratio_audit(
                selected_rows,
                low_ratio_csv,
                low_ratio_md,
                title=audit_title,
            )

            print(f"Wrote {row_path}")
            print(f"Wrote {ratio_path}")
            print(f"Wrote {bi_path}")
            print(f"Wrote {report_path}")
            print(f"Wrote {low_ratio_csv}")
            print(f"Wrote {low_ratio_md}")


if __name__ == "__main__":
    main()
