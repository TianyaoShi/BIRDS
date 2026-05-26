"""Plot traffic-load sensitivity for selected H100 ShareGPT exhaustive sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    EnergyBiConfig,
    add_biodiversity_metrics,
    load_energy_summary_rows,
)
from bi_modeling.visualization.plot_style import (  # noqa: E402
    FONTSIZE,
    LEGEND_FONTSIZE,
    LINEWIDTH,
    MARKERSIZE,
    TICK_FONTSIZE,
    apply_academic_style,
)

DEFAULT_SWEEP_SUMMARY = (
    REPO_ROOT
    / "results"
    / "energy"
    / "exhaustive_sweep"
    / "h100-chat-selected-sharegpt-exhaustive-sweep"
    / "slurm-energy-20260525T050016Z"
    / "summary_compact.csv"
)
QUALITY_SCORES_PATH = REPO_ROOT / "results" / "quality_scores" / "quality_scores_compiled.csv"

MODEL_SELECTIONS = {
    "qwen": [
        ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3 30B-A3B"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen3 235B-A22B"),
    ],
    "gemma": [
        ("google/gemma-4-E4B-it", "Gemma4 4B"),
        ("google/gemma-4-26B-A4B-it", "Gemma4 26B-A4B"),
        ("google/gemma-4-31B-it", "Gemma4 31B"),
    ],
}
MODEL_COLORS = {
    "Qwen/Qwen3-4B-Instruct-2507": "#4C78A8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "#59A14F",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "#E07A5F",
    "google/gemma-4-E4B-it": "#4C78A8",
    "google/gemma-4-26B-A4B-it": "#59A14F",
    "google/gemma-4-31B-it": "#E07A5F",
}
MODEL_MARKERS = {
    "Qwen/Qwen3-4B-Instruct-2507": "o",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "s",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "D",
    "google/gemma-4-E4B-it": "o",
    "google/gemma-4-26B-A4B-it": "s",
    "google/gemma-4-31B-it": "D",
}


def _normalize_quality_score(value: float) -> float:
    numeric = float(value)
    if numeric > 1.0:
        return numeric / 100.0
    return numeric


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _plain_tick(value: float, _: int) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


def _latency_log_tick(value: float, _: int) -> str:
    if np.isclose(value, 10.0):
        return "10"
    if np.isclose(value, 100.0):
        return r"$10^2$"
    if np.isclose(value, 1000.0):
        return r"$10^3$"
    return ""


def _markevery_for_series(series_len: int) -> int:
    return max(1, int(np.ceil(series_len / 10)))


def build_case_study_rows(summary_path: Path, *, config: EnergyBiConfig) -> pd.DataFrame:
    rows = add_biodiversity_metrics(load_energy_summary_rows([summary_path]), config=config)
    selected_models = {
        model
        for selection in MODEL_SELECTIONS.values()
        for model, _ in selection
    }
    rows = rows[rows["model"].isin(selected_models)].copy()

    quality = pd.read_csv(QUALITY_SCORES_PATH)[["model", "chat_q"]].rename(
        columns={"chat_q": "quality_score"}
    )
    rows = rows.merge(quality, on="model", how="left")
    rows["normalized_quality_score"] = rows["quality_score"].apply(_normalize_quality_score)
    rows["qnbi_per_request"] = rows["bi_per_request"] / rows["normalized_quality_score"]

    model_to_label = {
        model: label
        for selection in MODEL_SELECTIONS.values()
        for model, label in selection
    }
    model_to_selection = {
        model: selection_name
        for selection_name, selection in MODEL_SELECTIONS.items()
        for model, _ in selection
    }
    model_to_order = {
        model: idx
        for selection in MODEL_SELECTIONS.values()
        for idx, (model, _) in enumerate(selection)
    }
    rows["selection"] = rows["model"].map(model_to_selection)
    rows["model_label"] = rows["model"].map(model_to_label)
    rows["model_order"] = rows["model"].map(model_to_order)
    rows["legend_label"] = rows.apply(
        lambda row: f"{row['model_label']} (TP{int(row['tensor_parallel_size'])})",
        axis=1,
    )
    rows = rows.sort_values(["selection", "model_order", "request_rate"]).reset_index(drop=True)
    return rows


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _plot_metric_lines(
    ax: plt.Axes,
    rows: pd.DataFrame,
    selection_key: str,
    *,
    y_column: str,
    ylabel: str,
    log_y: bool,
    latency_ticks: bool = False,
) -> None:
    for model, label in MODEL_SELECTIONS[selection_key]:
        subset = rows[rows["model"] == model].sort_values("request_rate")
        if subset.empty:
            continue
        ax.plot(
            subset["request_rate"],
            subset[y_column],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markevery=_markevery_for_series(len(subset)),
            markersize=MARKERSIZE*0.75,
            markerfacecolor="white",
            markeredgewidth=LINEWIDTH * 0.28,
            linewidth=LINEWIDTH,
            label=str(subset["legend_label"].iloc[0]),
        )

    if log_y:
        ax.set_yscale("log")
        values = rows.loc[rows["selection"] == selection_key, y_column].dropna().to_numpy(dtype=float)
        values = values[values > 0]
        ax.set_ylim(values.min() * 0.75, values.max() * 1.45)
        if latency_ticks:
            ax.set_yticks([10, 100, 1000])
            ax.yaxis.set_major_formatter(FuncFormatter(_latency_log_tick))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(_scientific_tick))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_tick))
    ax.set_xlabel("Request rate (req/s)", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    _style_axis(ax)


def plot_selection(
    rows: pd.DataFrame,
    selection_key: str,
    output_path: Path,
    *,
    latency_percentile: str,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(25, 7), constrained_layout=True)
    selection_rows = rows[rows["selection"] == selection_key].copy()
    latency_label = latency_percentile.upper()

    _plot_metric_lines(
        axes[0],
        selection_rows,
        selection_key,
        y_column="qnbi_per_request",
        ylabel=r"QNBI (species$\cdot$yr)",
        log_y=True,
    )
    _plot_metric_lines(
        axes[1],
        selection_rows,
        selection_key,
        y_column=f"ttft_{latency_percentile}_ms",
        ylabel=f"{latency_label} TTFT (ms)",
        log_y=True,
        latency_ticks=True,
    )
    _plot_metric_lines(
        axes[2],
        selection_rows,
        selection_key,
        y_column=f"tpot_{latency_percentile}_ms",
        ylabel=f"{latency_label} TPOT (ms)",
        log_y=True,
        latency_ticks=True,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE + 2,
        bbox_to_anchor=(0.5, 1.15),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SWEEP_SUMMARY)
    parser.add_argument("--latency-percentile", choices=("p90", "p95", "p99"), default="p90")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "figures",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "derived",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_case_study_rows(args.summary_path, config=EnergyBiConfig())
    args.derived_dir.mkdir(parents=True, exist_ok=True)
    derived_path = args.derived_dir / "sharegpt_h100_traffic_load_case_study_rows.csv"
    rows.to_csv(derived_path, index=False)
    print(f"Wrote {derived_path}")

    for selection_key in MODEL_SELECTIONS:
        suffix = "" if args.latency_percentile == "p90" else f"_{args.latency_percentile}"
        figure_path = args.figures_dir / f"sharegpt_h100_{selection_key}_traffic_load_case_study_1x3{suffix}.pdf"
        plot_selection(
            rows,
            selection_key,
            figure_path,
            latency_percentile=args.latency_percentile,
        )
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
