"""Plot H100 BI-per-request violins with workload length box plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    EnergyBiConfig,
    build_energy_bi_dataset,
)
from bi_modeling.visualization.plot_style import (  # noqa: E402
    FONTSIZE,
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)
from bi_modeling.visualization.plot_workload_bi_per_request_violin import (  # noqa: E402
    GROUP_ORDER,
    INPUT_COLOR,
    LENGTH_OFFSET,
    OUTPUT_COLOR,
    WORKLOAD_SPECS,
    _build_layout_positions,
    _build_length_distributions,
    _draw_group_labels,
    _draw_group_separators,
    _style_axes,
    _style_violin_parts,
    build_workload_distribution_rows,
)

VIOLIN_WIDTH = 0.78
BOX_WIDTH = 0.28


def _style_boxplot(boxplot, *, facecolor: str, alpha: float) -> None:
    for patch in boxplot["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("black")
        patch.set_alpha(alpha)
        patch.set_linewidth(LINEWIDTH * 0.45)
    for key in ("medians", "whiskers", "caps"):
        for artist in boxplot[key]:
            artist.set_color("black")
            artist.set_linewidth(LINEWIDTH * 0.45)


def plot_violin_with_length_boxes(rows, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(28, 15.2),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    bi_ax, length_ax = axes

    positions, tick_labels, datasets, group_centers, separator_positions = _build_layout_positions(rows)
    bi_data = [rows.loc[rows["dataset"] == dataset, "bi_per_request"].dropna().to_numpy() for dataset in datasets]
    bi_colors = [
        next(spec["color"] for spec in WORKLOAD_SPECS if spec["dataset"] == dataset)
        for dataset in datasets
    ]
    bi_parts = bi_ax.violinplot(
        bi_data,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(bi_parts, facecolors=bi_colors, alpha=0.82)

    all_bi_values = rows["bi_per_request"].dropna().to_numpy()
    bi_ax.set_yscale("log")
    bi_ax.set_ylim(all_bi_values.min() * 0.75, all_bi_values.max() * 1.6)
    bi_ax.set_ylabel(r"BI$_{\mathrm{fu}}$ (species$\cdot$yr)", fontsize=FONTSIZE)
    bi_ax.tick_params(axis="x", labelbottom=False, width=0, length=0)
    bi_ax.set_xlim(min(positions) - 0.7, max(positions) + 0.7)
    _draw_group_separators(bi_ax, separator_positions)
    _style_axes(bi_ax)

    length_distributions = _build_length_distributions()
    combined_length_values = []
    for position, dataset in zip(positions, datasets, strict=True):
        input_values = length_distributions[dataset]["input"]
        if input_values.size > 0:
            box = length_ax.boxplot(
                [input_values],
                positions=[position - LENGTH_OFFSET],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
            )
            _style_boxplot(box, facecolor=INPUT_COLOR, alpha=0.45)
            combined_length_values.append(input_values)
        output_values = length_distributions[dataset]["output"]
        if output_values.size > 0:
            box = length_ax.boxplot(
                [output_values],
                positions=[position + LENGTH_OFFSET],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
            )
            _style_boxplot(box, facecolor=OUTPUT_COLOR, alpha=0.45)
            combined_length_values.append(output_values)

    all_length_values = np.concatenate(combined_length_values)
    length_ax.set_yscale("log")
    length_ax.set_ylim(all_length_values.min() * 0.75, all_length_values.max() * 1.35)
    length_ax.set_ylabel("Length (tokens)", fontsize=FONTSIZE)
    length_ax.set_xticks(positions)
    length_ax.set_xticklabels(tick_labels)
    length_ax.tick_params(axis="x", labelsize=TICK_FONTSIZE - 2, width=0, length=0, pad=10)
    length_ax.set_xlim(min(positions) - 0.7, max(positions) + 0.7)
    _draw_group_separators(length_ax, separator_positions)
    _draw_group_labels(length_ax, group_centers)
    _style_axes(length_ax)
    length_ax.legend(
        handles=[
            Patch(facecolor=INPUT_COLOR, edgecolor="black", alpha=0.45, label="Input"),
            Patch(facecolor=OUTPUT_COLOR, edgecolor="black", alpha=0.45, label="Output"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.02),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "figures" / "workload_comparison",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "derived" / "workload_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_energy_bi_dataset(args.results_dir, config=EnergyBiConfig())
    dataset = dataset[dataset["accelerator"] == "H100"].copy()
    rows = build_workload_distribution_rows(dataset)

    derived_path = args.derived_dir / "h100_workload_bi_per_request_length_box_source_rows.csv"
    figure_path = args.figures_dir / "h100_workload_bi_per_request_length_box.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(derived_path, index=False)
    plot_violin_with_length_boxes(rows, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
