"""Plot BI-per-request workload distributions across all successful energy rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

WORKLOAD_SPECS = [
    {
        "dataset": "sharegpt",
        "label": "ShareGPT",
        "group": "Chat",
        "patterns": ("sharegpt",),
        "color": "#4C956C",
    },
    {
        "dataset": "wildchat",
        "label": "WildChat",
        "group": "Chat",
        "patterns": ("wildchat",),
        "color": "#4C956C",
    },
    {
        "dataset": "mmlu-pro",
        "label": "MMLU-Pro",
        "group": "Reasoning",
        "patterns": ("mmlu-pro",),
        "color": "#577590",
    },
    {
        "dataset": "supergpqa",
        "label": "SuperGPQA",
        "group": "Reasoning",
        "patterns": ("supergpqa",),
        "color": "#577590",
    },
    {
        "dataset": "cceval",
        "label": "CCEval",
        "group": "Code",
        "patterns": ("crosscodeeval", "cceval"),
        "color": "#BC6C25",
    },
    {
        "dataset": "repobench",
        "label": "RepoBench",
        "group": "Code",
        "patterns": ("repobench",),
        "color": "#BC6C25",
    },
    {
        "dataset": "longbench_long_output_summarization",
        "label": "LB-LongSum",
        "group": "LongBench",
        "patterns": ("longbench_long_output_summarization",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_medium_output_summarization",
        "label": "LB-MedSum",
        "group": "LongBench",
        "patterns": ("longbench_medium_output_summarization",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_medium_answer_rag_qa",
        "label": "LB-RAG",
        "group": "LongBench",
        "patterns": ("longbench_medium_answer_rag_qa",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_short_answer_document_qa",
        "label": "LB-DocQA",
        "group": "LongBench",
        "patterns": ("longbench_short_answer_document_qa",),
        "color": "#7B5EA7",
    },
]

GROUP_ORDER = ("Chat", "Reasoning", "Code", "LongBench")
GROUP_GAP = 0.75
VIOLIN_WIDTH = 0.78


def _match_dataset(workload: str) -> dict | None:
    workload_lower = workload.lower()
    for spec in WORKLOAD_SPECS:
        if any(pattern in workload_lower for pattern in spec["patterns"]):
            return spec
    return None


def build_workload_distribution_rows(rows: pd.DataFrame) -> pd.DataFrame:
    dataset_specs = rows["workload"].map(_match_dataset)
    matched = rows.loc[dataset_specs.notna()].copy()
    matched["dataset"] = dataset_specs[dataset_specs.notna()].map(lambda spec: spec["dataset"])
    matched["dataset_label"] = dataset_specs[dataset_specs.notna()].map(lambda spec: spec["label"])
    matched["dataset_group"] = dataset_specs[dataset_specs.notna()].map(lambda spec: spec["group"])
    matched["dataset_color"] = dataset_specs[dataset_specs.notna()].map(lambda spec: spec["color"])
    matched["group_order"] = matched["dataset_group"].map(
        {group: idx for idx, group in enumerate(GROUP_ORDER)}
    )
    matched["dataset_order"] = matched["dataset"].map(
        {spec["dataset"]: idx for idx, spec in enumerate(WORKLOAD_SPECS)}
    )
    return matched.sort_values(["group_order", "dataset_order", "model", "accelerator", "request_rate"]).reset_index(
        drop=True
    )


def plot_violin(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(28, 8.8), constrained_layout=True)

    positions: list[float] = []
    tick_labels: list[str] = []
    violin_data: list[np.ndarray] = []
    colors: list[str] = []
    group_centers: dict[str, float] = {}
    separator_positions: list[float] = []

    current_x = 1.0
    for group in GROUP_ORDER:
        group_specs = [spec for spec in WORKLOAD_SPECS if spec["group"] == group]
        group_positions: list[float] = []
        for spec in group_specs:
            subset = rows.loc[rows["dataset"] == spec["dataset"], "bi_per_request"].dropna().to_numpy()
            if subset.size == 0:
                continue
            positions.append(current_x)
            group_positions.append(current_x)
            tick_labels.append(spec["label"])
            violin_data.append(subset)
            colors.append(spec["color"])
            current_x += 1.0
        if not group_positions:
            continue
        group_centers[group] = (group_positions[0] + group_positions[-1]) / 2.0
        separator_positions.append(group_positions[-1] + GROUP_GAP / 2.0)
        current_x += GROUP_GAP

    parts = ax.violinplot(
        violin_data,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.82)
        body.set_linewidth(LINEWIDTH * 0.45)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(LINEWIDTH * 0.5)

    all_values = rows["bi_per_request"].dropna().to_numpy()
    ax.set_yscale("log")
    ax.set_ylim(all_values.min() * 0.75, all_values.max() * 1.6)
    ax.set_ylabel(r"BI$_{\mathrm{fu}}$ (species$\cdot$yr)", fontsize=FONTSIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE - 2, width=0, length=0, pad=10)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    ax.set_xlim(min(positions) - 0.7, max(positions) + 0.7)

    for separator in separator_positions[:-1]:
        ax.axvline(
            separator,
            color="#444444",
            linewidth=LINEWIDTH * 0.2,
            alpha=0.55,
            linestyle=(0, (6, 6)),
        )
    for group, center in group_centers.items():
        ax.text(
            center,
            -0.25,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=TICK_FONTSIZE,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)

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
    violin_rows = build_workload_distribution_rows(dataset)

    derived_path = args.derived_dir / "all_gpu_workload_bi_per_request_violin_source_rows.csv"
    figure_path = args.figures_dir / "all_gpu_workload_bi_per_request_violin.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    violin_rows.to_csv(derived_path, index=False)
    plot_violin(violin_rows, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
