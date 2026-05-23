"""Plot combined H100+A100 midpoint mean ratios as perspective-stacked bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

BUCKETS = ("GWP", "WC", "AP", "POFP")
BUCKET_LABELS = {
    "GWP": "Carbon",
    "WC": "Water",
    "AP": "Acidification",
    "POFP": "POF",
}
BUCKET_COLORS = {
    "GWP": "#1F77B4",
    "WC": "#17A589",
    "AP": "#C44E52",
    "POFP": "#7B5EA7",
}
HORIZONS = (20, 100, 1000)


def plot_midpoint_perspective_stackedbars(summary: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    summary = summary[summary["bucket"].isin(BUCKETS)].copy()
    pivot = (
        summary.pivot(index="time_horizon_years", columns="bucket", values="mean")
        .reindex(HORIZONS)
        .fillna(0.0)
    )
    x = range(len(HORIZONS))
    bottom = [0.0] * len(HORIZONS)

    for bucket in BUCKETS:
        heights = (pivot[bucket] * 100.0).tolist()
        ax.bar(
            x,
            heights,
            bottom=bottom,
            width=0.62,
            color=BUCKET_COLORS[bucket],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.35,
            label=BUCKET_LABELS[bucket],
        )
        bottom = [curr + height for curr, height in zip(bottom, heights)]

    ax.set_xticks(list(x), [str(horizon) for horizon in HORIZONS])
    ax.set_xlabel("Time Horizon (years)", fontsize=FONTSIZE)
    ax.set_ylabel("Contribution Ratio (%)", fontsize=FONTSIZE)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.2,
        columnspacing=1.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    derived_dir = REPO_ROOT / "bi_modeling" / "visualization" / "derived"
    figures_dir = REPO_ROOT / "bi_modeling" / "visualization" / "figures"
    parser.add_argument(
        "--all-summary-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_midpoint_perspective_mean_ratio_summary.csv",
    )
    parser.add_argument(
        "--best-summary-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_midpoint_perspective_mean_ratio_summary_best_energy_per_model.csv",
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        default=figures_dir / "combined_h100_a100_midpoint_perspective_stackedbar_all.pdf",
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        default=figures_dir / "combined_h100_a100_midpoint_perspective_stackedbar_best_energy_per_model.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_midpoint_perspective_stackedbars(pd.read_csv(args.all_summary_csv), args.all_output)
    plot_midpoint_perspective_stackedbars(pd.read_csv(args.best_summary_csv), args.best_output)
    print(f"Wrote {args.all_output}")
    print(f"Wrote {args.best_output}")


if __name__ == "__main__":
    main()
