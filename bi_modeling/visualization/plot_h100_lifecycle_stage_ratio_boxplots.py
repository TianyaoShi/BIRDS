"""Plot GPU lifecycle-stage contribution ratio box plots."""

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
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)

STAGES = ("operational", "manufacturing", "transportation", "recycling")
STAGE_LABELS = ("Operation", "Manufacturing", "Transportation", "End of Life")
STAGE_COLORS = ("#1F77B4", "#5B8C5A", "#C44E52", "#7B5EA7")


def _stage_ratio_values(rows: pd.DataFrame) -> list[pd.Series]:
    return [rows[f"{stage}_ratio"] * 100 for stage in STAGES]


def plot_stage_ratio_boxplot(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
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

    ax.set_ylabel("Contribution Ratio (%)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 110)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    def _format_func(y, _: int) -> str:
        if y < 0.1:
            return f"{y:.2f}"
        elif y < 1:
            return f"{y:.1f}"
        else:
            return f"{y:.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(_format_func))
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE - 2, width=0, length=0)
    for label in ax.get_xticklabels():
        label.set_rotation(10)
        label.set_ha("center")
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    derived_dir = REPO_ROOT / "bi_modeling" / "visualization" / "derived"
    figures_dir = REPO_ROOT / "bi_modeling" / "visualization" / "figures"
    parser.add_argument("--all-rows-csv", type=Path, default=derived_dir / "h100_lifecycle_stage_bi_per_request.csv")
    parser.add_argument(
        "--best-per-model-csv",
        type=Path,
        default=derived_dir / "h100_lifecycle_stage_bi_per_request_best_energy_per_model.csv",
    )
    parser.add_argument("--all-output", type=Path, default=figures_dir / "h100_lifecycle_stage_ratio_boxplot_all.pdf")
    parser.add_argument(
        "--best-output",
        type=Path,
        default=figures_dir / "h100_lifecycle_stage_ratio_boxplot_best_energy_per_model.pdf",
    )
    parser.add_argument(
        "--combined-all-rows-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_lifecycle_stage_bi_per_request.csv",
    )
    parser.add_argument(
        "--combined-best-per-model-csv",
        type=Path,
        default=derived_dir / "combined_h100_a100_lifecycle_stage_bi_per_request_best_energy_per_model.csv",
    )
    parser.add_argument(
        "--combined-all-output",
        type=Path,
        default=figures_dir / "combined_h100_a100_lifecycle_stage_ratio_boxplot_all.pdf",
    )
    parser.add_argument(
        "--combined-best-output",
        type=Path,
        default=figures_dir / "combined_h100_a100_lifecycle_stage_ratio_boxplot_best_energy_per_model.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_stage_ratio_boxplot(pd.read_csv(args.all_rows_csv), args.all_output)
    plot_stage_ratio_boxplot(pd.read_csv(args.best_per_model_csv), args.best_output)
    plot_stage_ratio_boxplot(pd.read_csv(args.combined_all_rows_csv), args.combined_all_output)
    plot_stage_ratio_boxplot(pd.read_csv(args.combined_best_per_model_csv), args.combined_best_output)
    print(f"Wrote {args.all_output}")
    print(f"Wrote {args.best_output}")
    print(f"Wrote {args.combined_all_output}")
    print(f"Wrote {args.combined_best_output}")


if __name__ == "__main__":
    main()
