"""Plot vertical lifecycle-stage ratios and perspective midpoint ratios in one 1x2 figure."""

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
ANNOTATION_FONTSIZE = TICK_FONTSIZE - 3


def _stage_ratio_values(rows: pd.DataFrame) -> list[pd.Series]:
    return [rows[f"{stage}_ratio"] * 100 for stage in STAGES]


def _format_log_percent(value: float, _: int) -> str:
    if value < 0.1:
        return f"{value:.2f}"
    if value < 1:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _format_ratio(value: float) -> str:
    if value < 0.1:
        return f"{value:.2f}%"
    if value < 100:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def _plot_lifecycle_boxplot(ax: plt.Axes, rows: pd.DataFrame) -> None:
    box = ax.boxplot(
        _stage_ratio_values(rows),
        tick_labels=STAGE_LABELS,
        patch_artist=True,
        widths=0.55,
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

    ax.set_ylabel("Contribution Ratio (%)", fontsize=FONTSIZE+2)
    ax.set_yscale("log")
    ax.set_ylim(0.0015, 110)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_percent))
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE+4, width=LINEWIDTH * 0.45, length=12)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE+4, width=0, length=0)
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_ha("center")


def _plot_perspective_stackedbars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    summary = summary[summary["bucket"].isin(MIDPOINT_BUCKETS)].copy()
    pivot = (
        summary.pivot(index="time_horizon_years", columns="bucket", values="mean")
        .reindex(HORIZONS)
        .fillna(0.0)
    )
    x_positions = list(range(len(HORIZONS)))
    bottom = [0.0] * len(HORIZONS)

    for bucket in MIDPOINT_BUCKETS:
        heights = (pivot[bucket] * 100.0).tolist()
        ax.bar(
            x_positions,
            heights,
            bottom=bottom,
            width=0.55,
            color=MIDPOINT_COLORS[bucket],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.35,
            label=MIDPOINT_LABELS[bucket],
        )
        bottom = [curr + height for curr, height in zip(bottom, heights)]

    carbon_heights = (pivot["GWP"] * 100.0).tolist()
    for x_pos, carbon_height in zip(x_positions, carbon_heights):
        ax.text(
            x_pos,
            carbon_height * 0.5,
            f"{_format_ratio(carbon_height)}",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONTSIZE,
            color="white",
        )

    horizon_to_x = {horizon: idx for idx, horizon in enumerate(HORIZONS)}
    starts = pivot[list(MIDPOINT_BUCKETS)].cumsum(axis=1).shift(axis=1, fill_value=0.0) * 100.0
    heights = pivot[list(MIDPOINT_BUCKETS)] * 100.0

    for bucket in ("AP", "POFP"):
        x_pos = horizon_to_x[20]
        center = starts.loc[20, bucket] + heights.loc[20, bucket] * 0.5
        ax.text(
            x_pos,
            center,
            f"{_format_ratio(heights.loc[20, bucket])}",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONTSIZE,
            color="white",
        )

    arrow_props = {
        "arrowstyle": "->",
        "linewidth": LINEWIDTH * 0.25,
        "color": "black",
        "shrinkA": 0,
        "shrinkB": 4,
    }
    for bucket, text_xy in {
        "WC": (1.51, 60),
        "AP": (1.51, 75),
        "POFP": (1.51, 90),
    }.items():
        x_pos = horizon_to_x[100]
        target = starts.loc[100, bucket] + heights.loc[100, bucket] * 0.5
        ax.annotate(
            f"{_format_ratio(heights.loc[100, bucket])}",
            xy=(x_pos, target),
            xytext=text_xy,
            fontsize=ANNOTATION_FONTSIZE,
            ha="center",
            va="center",
            arrowprops=arrow_props,
        )

    x_pos = horizon_to_x[1000]
    aggregate_buckets = ("WC", "AP", "POFP")
    aggregate_height = sum(heights.loc[1000, bucket] for bucket in aggregate_buckets)
    aggregate_start = starts.loc[1000, "WC"]
    # ax.annotate(
    #     f"WC+TA+POF {_format_ratio(aggregate_height)}",
    #     xy=(x_pos, aggregate_start + aggregate_height * 0.5),
    #     xytext=(1.65, 55),
    #     fontsize=ANNOTATION_FONTSIZE,
    #     ha="center",
    #     va="center",
    #     arrowprops=arrow_props,
    # )

    ax.set_xticks(x_positions, [str(horizon) for horizon in HORIZONS])
    ax.set_xlabel("Time Horizon (years)", fontsize=FONTSIZE+4)
    # ax.set_ylabel("Contribution Ratio (%)", fontsize=FONTSIZE+2)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE+4, width=LINEWIDTH * 0.45, length=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.35, 1.2),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE+4,
        handlelength=1.2,
        columnspacing=1.1,
    )


def plot_combined_figure(lifecycle_rows: pd.DataFrame, midpoint_summary: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
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
        default=figures_dir / "combined_h100_a100_lifecycle_midpoint_1x2_vertical.pdf",
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
