"""Plot lifecycle-stage ratios and perspective midpoint ratios in one 1x2 figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.plot_style import (  # noqa: E402
    FONTSIZE,
    LEGEND_FONTSIZE,
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)

STAGES = ("operational", "manufacturing", "transportation", "recycling")
STAGE_LABELS = ("Operation", "Manufacturing", "Transportation", "End of Life")
STAGE_COLORS = ("#1F77B4", "#5B8C5A", "#C44E52", "#7B5EA7")

MIDPOINT_BUCKETS = ("GWP", "WC", "AP", "POFP")
MIDPOINT_LABELS = {
    "GWP": "GW",
    "WC": "WC",
    "AP": "TA",
    "POFP": "POF",
}
MIDPOINT_COLORS = {
    "GWP": "#1F77B4",
    "WC": "#17A589",
    "AP": "#C44E52",
    "POFP": "#7B5EA7",
}
HORIZONS = (20, 100, 1000)


def _stage_ratio_values(rows: pd.DataFrame) -> list[pd.Series]:
    return [rows[f"{stage}_ratio"] * 100 for stage in STAGES]


def _format_log_percent(value: float, _: int) -> str:
    if value < 0.1:
        return f"{value:.2f}"
    if value < 1:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _plot_lifecycle_boxplot(ax: plt.Axes, rows: pd.DataFrame) -> None:
    box = ax.boxplot(
        _stage_ratio_values(rows),
        tick_labels=STAGE_LABELS,
        patch_artist=True,
        widths=0.55,
        vert=False,
        showfliers=True,
        medianprops={"color": "black", "linewidth": LINEWIDTH * 0.55},
        boxprops={"linewidth": LINEWIDTH * 0.45, "color": "black"},
        whiskerprops={"linewidth": LINEWIDTH * 0.45, "color": "black"},
        capprops={"linewidth": LINEWIDTH * 0.45, "color": "black"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 7,
            "linestyle": "none",
        },
    )
    for patch, color in zip(box["boxes"], STAGE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xlabel("Contribution Ratio (%)", fontsize=FONTSIZE)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 110)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_log_percent))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE - 4, width=0, length=0)


def _plot_perspective_stackedbars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    summary = summary[summary["bucket"].isin(MIDPOINT_BUCKETS)].copy()
    pivot = (
        summary.pivot(index="time_horizon_years", columns="bucket", values="mean")
        .reindex(HORIZONS)
        .fillna(0.0)
    )
    y_positions = list(range(len(HORIZONS)))
    left = [0.0] * len(HORIZONS)

    for bucket in MIDPOINT_BUCKETS:
        widths = (pivot[bucket] * 100.0).tolist()
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=0.58,
            color=MIDPOINT_COLORS[bucket],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.35,
            label=MIDPOINT_LABELS[bucket],
        )
        left = [curr + width for curr, width in zip(left, widths)]

    ax.set_yticks(y_positions, [str(horizon) for horizon in HORIZONS])
    ax.set_xlabel("Contribution Ratio (%)", fontsize=FONTSIZE)
    ax.set_ylabel("Time Horizon (years)", fontsize=FONTSIZE)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.2,
        columnspacing=1.1,
    )


def plot_combined_figure(lifecycle_rows: pd.DataFrame, midpoint_summary: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 8.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    _plot_lifecycle_boxplot(axes[0], lifecycle_rows)
    _plot_perspective_stackedbars(axes[1], midpoint_summary)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(LINEWIDTH * 0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    derived_dir = REPO_ROOT / "bi_modeling" / "visualization" / "derived"
    figures_dir = REPO_ROOT / "bi_modeling" / "visualization" / "figures"
    parser.add_argument(
        "--lifecycle-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_lifecycle_stage_bi_per_request.csv",
    )
    parser.add_argument(
        "--midpoint-summary-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_midpoint_perspective_mean_ratio_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=figures_dir / "combined_h100_a100_lifecycle_midpoint_1x2.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_combined_figure(
        pd.read_csv(args.lifecycle_csv),
        pd.read_csv(args.midpoint_summary_csv),
        args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
