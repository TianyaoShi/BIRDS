"""Plot BI-per-request workload distributions across all successful energy rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
        "label": "MMLU-\nPro",
        "group": "Reasoning",
        "patterns": ("mmlu-pro",),
        "color": "#577590",
    },
    {
        "dataset": "supergpqa",
        "label": "Super\nGPQA",
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
        "label": "L-Sum",
        "group": "LongBench",
        "patterns": ("longbench_long_output_summarization",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_medium_output_summarization",
        "label": "M-Sum",
        "group": "LongBench",
        "patterns": ("longbench_medium_output_summarization",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_medium_answer_rag_qa",
        "label": "M-RAG\nQA",
        "group": "LongBench",
        "patterns": ("longbench_medium_answer_rag_qa",),
        "color": "#7B5EA7",
    },
    {
        "dataset": "longbench_short_answer_document_qa",
        "label": "S-Doc\nQA",
        "group": "LongBench",
        "patterns": ("longbench_short_answer_document_qa",),
        "color": "#7B5EA7",
    },
]

GROUP_ORDER = ("Chat", "Reasoning", "Code", "LongBench")
DATASET_STEP = 1.18
GROUP_GAP = 0.36
VIOLIN_WIDTH = 0.78
LENGTH_VIOLIN_WIDTH = 0.34
LENGTH_OFFSET = 0.18
INPUT_COLOR = "#4C78A8"
OUTPUT_COLOR = "#E07A5F"
REQUEST_LENGTHS_CSV = (
    REPO_ROOT / "results" / "workload_length_distributions" / "request_lengths.csv"
)
CHAT_REQUEST_LENGTHS_CSV = (
    REPO_ROOT / "results" / "workload_length_distributions" / "chat_quality_request_lengths.csv"
)
SUPERGPQA_OUTPUT_SUMMARY_CSV = (
    REPO_ROOT / "results" / "workload_length_distributions" / "supergpqa_real_output_length_summary.csv"
)
SUPERGPQA_FULL_OUTPUT_JSONL = (
    REPO_ROOT / "results" / "workload_length_distributions" / "supergpqa_full_real_output_lengths_by_request.jsonl"
)
SUPERGPQA_HARD_OUTPUT_JSONL = (
    REPO_ROOT / "results" / "workload_length_distributions" / "supergpqa_hard_real_output_lengths_by_request.jsonl"
)


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


def _build_layout_positions(rows: pd.DataFrame) -> tuple[
    list[float],
    list[str],
    list[str],
    dict[str, float],
    list[float],
]:
    positions: list[float] = []
    tick_labels: list[str] = []
    datasets: list[str] = []
    group_centers: dict[str, float] = {}
    separator_positions: list[float] = []

    current_x = 1.0
    for group in GROUP_ORDER:
        group_specs = [spec for spec in WORKLOAD_SPECS if spec["group"] == group]
        group_positions: list[float] = []
        for spec in group_specs:
            subset = rows.loc[rows["dataset"] == spec["dataset"]]
            if subset.size == 0:
                continue
            positions.append(current_x)
            group_positions.append(current_x)
            tick_labels.append(spec["label"])
            datasets.append(spec["dataset"])
            current_x += DATASET_STEP
        if not group_positions:
            continue
        group_centers[group] = (group_positions[0] + group_positions[-1]) / 2.0
        next_group_start = current_x + GROUP_GAP
        separator_positions.append((group_positions[-1] + next_group_start) / 2.0)
        current_x = next_group_start
    return positions, tick_labels, datasets, group_centers, separator_positions


def _style_violin_parts(parts, *, facecolors: list[str], alpha: float) -> None:
    for body, color in zip(parts["bodies"], facecolors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(alpha)
        body.set_linewidth(LINEWIDTH * 0.45)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(LINEWIDTH * 0.5)


def _draw_group_separators(ax: plt.Axes, separator_positions: list[float]) -> None:
    for separator in separator_positions[:-1]:
        ax.axvline(
            separator,
            color="#444444",
            linewidth=LINEWIDTH * 0.2,
            alpha=0.55,
            linestyle=(0, (6, 6)),
        )


def _draw_group_labels(ax: plt.Axes, group_centers: dict[str, float]) -> None:
    for group, center in group_centers.items():
        ax.text(
            center,
            -0.2,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=TICK_FONTSIZE,
        )


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _canonical_length_workload(name: str) -> str:
    lowered = str(name).strip().lower()
    mapping = {
        "crosscodeeval_rg1_unixcoder_cache_realistic": "cceval",
        "repobench_python_java_aggregate_cache_realistic_8k_drop": "repobench",
        "mmlu_pro_reasoning": "mmlu-pro",
        "supergpqa_reasoning": "supergpqa",
        "supergpqa_hard_reasoning": "supergpqa",
        "longbench_long_output_summarization_original_official_qwen3_8b": "longbench_long_output_summarization",
        "longbench_medium_output_summarization_original_official_qwen3_8b": "longbench_medium_output_summarization",
        "longbench_medium_answer_rag_qa_original_official_qwen3_8b": "longbench_medium_answer_rag_qa",
        "longbench_short_answer_document_qa_original_official_qwen3_8b": "longbench_short_answer_document_qa",
    }
    return mapping.get(lowered, lowered)


def _load_request_length_rows() -> pd.DataFrame:
    request_lengths = pd.read_csv(REQUEST_LENGTHS_CSV, low_memory=False)
    request_lengths["dataset"] = request_lengths["workload_name"].map(_canonical_length_workload)

    chat_lengths = pd.read_csv(CHAT_REQUEST_LENGTHS_CSV, low_memory=False)
    chat_lengths["dataset"] = chat_lengths["source"].map(
        {"sharegpt": "sharegpt", "wildchat": "wildchat"}
    )
    chat_lengths = chat_lengths.rename(columns={"source": "chat_source"})
    chat_lengths = chat_lengths[chat_lengths["input_tokens"] >= 10].copy()
    chat_lengths = chat_lengths[chat_lengths["output_tokens"] >= 5].copy()

    combined = pd.concat([request_lengths, chat_lengths], ignore_index=True, sort=False)
    return combined[combined["dataset"].notna()].copy()


def _load_supergpqa_output_lengths() -> np.ndarray:
    summary = pd.read_csv(SUPERGPQA_OUTPUT_SUMMARY_CSV)
    easy_models = set(summary.loc[summary["split"] == "easy_medium", "model"])
    hard_models = set(summary.loc[summary["split"] == "hard", "model"])
    shared_models = easy_models & hard_models

    lengths: list[int] = []
    for path in (SUPERGPQA_FULL_OUTPUT_JSONL, SUPERGPQA_HARD_OUTPUT_JSONL):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                for output in record.get("outputs", []):
                    if output.get("model") not in shared_models:
                        continue
                    if not output.get("success"):
                        continue
                    actual_output_len = output.get("actual_output_len")
                    if actual_output_len is None:
                        continue
                    lengths.append(int(actual_output_len))
    return np.asarray(lengths, dtype=float)


def _build_length_distributions() -> dict[str, dict[str, np.ndarray]]:
    rows = _load_request_length_rows()
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for spec in WORKLOAD_SPECS:
        dataset_rows = rows[rows["dataset"] == spec["dataset"]]
        input_values = dataset_rows["input_tokens"].dropna().to_numpy(dtype=float)
        if spec["dataset"] == "mmlu-pro":
            output_values = np.asarray([], dtype=float)
        elif spec["dataset"] == "supergpqa":
            output_values = _load_supergpqa_output_lengths()
        else:
            output_values = dataset_rows["output_tokens"].dropna().to_numpy(dtype=float)
        distributions[spec["dataset"]] = {
            "input": input_values,
            "output": output_values,
        }
    return distributions


def plot_violin(rows: pd.DataFrame, output_path: Path) -> None:
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

    parts = bi_ax.violinplot(
        bi_data,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(parts, facecolors=bi_colors, alpha=0.82)

    all_values = rows["bi_per_request"].dropna().to_numpy()
    bi_ax.set_yscale("log")
    bi_ax.set_ylim(all_values.min() * 0.75, all_values.max() * 1.6)
    bi_ax.set_ylabel(r"BI$_{\mathrm{fu}}$ (species$\cdot$yr)", fontsize=FONTSIZE)
    bi_ax.tick_params(axis="x", labelbottom=False, width=0, length=0)
    bi_ax.set_xlim(min(positions) - 0.7, max(positions) + 0.7)
    _draw_group_separators(bi_ax, separator_positions)
    _style_axes(bi_ax)

    length_distributions = _build_length_distributions()
    input_positions: list[float] = []
    input_data: list[np.ndarray] = []
    output_positions: list[float] = []
    output_data: list[np.ndarray] = []
    for position, dataset in zip(positions, datasets, strict=True):
        input_values = length_distributions[dataset]["input"]
        if input_values.size > 0:
            input_positions.append(position - LENGTH_OFFSET)
            input_data.append(input_values)
        output_values = length_distributions[dataset]["output"]
        if output_values.size > 0:
            output_positions.append(position + LENGTH_OFFSET)
            output_data.append(output_values)

    if input_data:
        input_parts = length_ax.violinplot(
            input_data,
            positions=input_positions,
            widths=LENGTH_VIOLIN_WIDTH,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
        _style_violin_parts(input_parts, facecolors=[INPUT_COLOR] * len(input_data), alpha=0.45)
    if output_data:
        output_parts = length_ax.violinplot(
            output_data,
            positions=output_positions,
            widths=LENGTH_VIOLIN_WIDTH,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
        _style_violin_parts(output_parts, facecolors=[OUTPUT_COLOR] * len(output_data), alpha=0.45)

    combined_length_values = np.concatenate(
        [
            values
            for distribution in length_distributions.values()
            for values in (distribution["input"], distribution["output"])
            if values.size > 0
        ]
    )
    length_ax.set_yscale("log")
    length_ax.set_ylim(combined_length_values.min() * 0.75, combined_length_values.max() * 1.35)
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
        loc="upper right",
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
