"""Plot model size versus quality-normalized BI for selected model configs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.plot_model_bi_per_request_bars import (  # noqa: E402
    DEFAULT_SUPPLEMENT_GPU,
    GROUP_COLORS,
    GROUP_DISPLAY,
    GROUP_ORDER,
    build_selected_model_rows,
)
from bi_modeling.visualization.energy_bi_calculator import EnergyBiConfig  # noqa: E402
from bi_modeling.visualization.plot_style import (  # noqa: E402
    FONTSIZE,
    LEGEND_FONTSIZE,
    LINEWIDTH,
    MARKERSIZE,
    TICK_FONTSIZE,
    apply_academic_style,
)

MARKER_BY_MOE = {
    False: "o",
    True: "D",
}
MARKER_DISPLAY = {
    False: "Dense",
    True: "MoE",
}


def _normalize_quality_score(value: float) -> float:
    numeric = float(value)
    if numeric > 1.0:
        return numeric / 100.0
    return numeric


def _is_moe_model(model: str) -> bool:
    return bool(re.search(r"(?:^|[-_/])(a|e)\d+(?:\.\d+)?b", model.lower()))


def build_qnbi_rows(
    results_dir: Path,
    *,
    gpu: str,
    workload_substring: str,
    config: EnergyBiConfig,
    supplement_gpu: str | None,
) -> pd.DataFrame:
    rows = build_selected_model_rows(
        results_dir,
        gpu=gpu,
        workload_substring=workload_substring,
        config=config,
        supplement_gpu=supplement_gpu,
    ).copy()
    rows["normalized_quality_score"] = rows["quality_score"].apply(_normalize_quality_score)
    rows = rows[rows["normalized_quality_score"] > 0].copy()
    rows["qnbi_per_request"] = rows["bi_per_request"] / rows["normalized_quality_score"]
    rows["is_moe"] = rows["model"].map(_is_moe_model)
    return rows


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _size_tick(value: float, _: int) -> str:
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


def plot_qnbi_scatter(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(15, 8.5), constrained_layout=True)

    for family in GROUP_ORDER:
        family_rows = rows[rows["family"] == family]
        if family_rows.empty:
            continue
        for is_moe, marker in MARKER_BY_MOE.items():
            subset = family_rows[family_rows["is_moe"] == is_moe]
            if subset.empty:
                continue
            ax.scatter(
                subset["size_b"],
                subset["qnbi_per_request"],
                s=MARKERSIZE ** 2 * 2.2,
                marker=marker,
                facecolor=GROUP_COLORS[family],
                edgecolor="black",
                linewidth=LINEWIDTH * 0.22,
                alpha=0.9,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model size (B params)", fontsize=FONTSIZE)
    ax.set_ylabel("QNBI per req (species$\\cdot$yr)", fontsize=FONTSIZE)
    ax.xaxis.set_major_formatter(FuncFormatter(_size_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_scientific_tick))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=10)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[family],
            markeredgecolor="black",
            markeredgewidth=LINEWIDTH * 0.22,
            markersize=14,
            label=GROUP_DISPLAY[family],
        )
        for family in GROUP_ORDER
        if not rows[rows["family"] == family].empty
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER_BY_MOE[is_moe],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=LINEWIDTH * 0.22,
            markersize=14,
            label=MARKER_DISPLAY[is_moe],
        )
        for is_moe in (False, True)
        if not rows[rows["is_moe"] == is_moe].empty
    ]

    family_legend = ax.legend(
        handles=family_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=LEGEND_FONTSIZE - 4,
    )
    ax.add_artist(family_legend)
    ax.legend(
        handles=marker_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        frameon=False,
        fontsize=LEGEND_FONTSIZE - 4,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
    parser.add_argument("--gpu", type=str, default="h100")
    parser.add_argument("--supplement-gpu", type=str, default=DEFAULT_SUPPLEMENT_GPU)
    parser.add_argument("--workload", type=str, default="sharegpt")
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "derived",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_qnbi_rows(
        args.results_dir,
        gpu=args.gpu,
        workload_substring=args.workload,
        config=EnergyBiConfig(),
        supplement_gpu=args.supplement_gpu,
    )
    stem = f"{args.gpu.lower()}_{args.workload.lower()}_model_qnbi"
    derived_path = args.derived_dir / f"{stem}_selected_configs.csv"
    figure_path = args.figures_dir / f"{stem}_scatter.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(derived_path, index=False)
    plot_qnbi_scatter(rows, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
