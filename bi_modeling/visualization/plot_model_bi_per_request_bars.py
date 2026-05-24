"""Plot BI per request for most energy-efficient model configs on a chosen GPU/workload."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.analyze_h100_lifecycle_stage_breakdown import (  # noqa: E402
    select_most_energy_efficient_config_per_model,
)
from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    EnergyBiConfig,
    add_biodiversity_metrics,
    load_energy_summary_rows,
)
from bi_modeling.visualization.plot_style import (  # noqa: E402
    FONTSIZE,
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)

GROUP_ORDER = ("llama-2", "llama-3", "gpt-oss", "qwen-3", "gemma-4")
GROUP_COLORS = {
    "llama-2": "#B56576",
    "llama-3": "#6D597A",
    "gpt-oss": "#355070",
    "qwen-3": "#2A9D8F",
    "gemma-4": "#BC6C25",
}
GROUP_DISPLAY = {
    "llama-2": "Llama-2",
    "llama-3": "Llama-3",
    "gpt-oss": "GPT-OSS",
    "qwen-3": "Qwen-3",
    "gemma-4": "Gemma-4",
}
DEFAULT_SUPPLEMENT_GPU = "a100"
SUPPLEMENT_MAX_SIZE_B = 3.0
BAR_WIDTH = 0.72
BAR_STEP = 1.12
GROUP_GAP = 0.42


def _summary_files_for_gpu(results_dir: Path, gpu: str) -> list[Path]:
    return sorted((results_dir / gpu.lower()).glob("**/summary_compact.csv"))


def _family_for_model(model: str) -> str | None:
    lower = model.lower()
    if "codellama" in lower:
        return None
    if "llama-2" in lower:
        return "llama-2"
    if "llama-3" in lower:
        return "llama-3"
    if "gpt-oss" in lower:
        return "gpt-oss"
    if "qwen3" in lower:
        return "qwen-3"
    if "gemma-4" in lower:
        return "gemma-4"
    return None


def _size_and_label(model: str, family: str) -> tuple[float, str]:
    lower = model.lower()
    if family == "gemma-4" and "e2b" in lower:
        return 2.0, "E2B"
    if family == "gemma-4" and "e4b" in lower:
        return 4.0, "E4B"

    match = re.search(r"(\d+(?:\.\d+)?)b", lower)
    if not match:
        return math.inf, model
    size = float(match.group(1))
    active_match = re.search(r"a(\d+(?:\.\d+)?)b", lower)
    if size.is_integer():
        size_label = f"{int(size)}B"
    else:
        size_label = f"{size:g}B"
    if active_match:
        active_size = float(active_match.group(1))
        if active_size.is_integer():
            active_label = f"A{int(active_size)}B"
        else:
            active_label = f"A{active_size:g}B"
        size_label = f"{size_label}-\n{active_label}"
    return size, size_label


def _load_filtered_rows(
    results_dir: Path,
    *,
    gpu: str,
    workload_substring: str,
    config: EnergyBiConfig,
) -> pd.DataFrame:
    rows = load_energy_summary_rows(_summary_files_for_gpu(results_dir, gpu))
    rows = add_biodiversity_metrics(rows, config=config)
    rows = rows[rows["workload"].str.contains(workload_substring, case=False, na=False)].copy()
    rows = rows[~rows["model"].str.contains("thinking", case=False, na=False)].copy()
    rows["family"] = rows["model"].map(_family_for_model)
    rows = rows[rows["family"].notna()].copy()
    rows["accelerator_source"] = gpu.upper()
    return rows


def build_selected_model_rows(
    results_dir: Path,
    *,
    gpu: str,
    workload_substring: str,
    config: EnergyBiConfig,
    supplement_gpu: str | None = DEFAULT_SUPPLEMENT_GPU,
) -> pd.DataFrame:
    primary_rows = _load_filtered_rows(
        results_dir,
        gpu=gpu,
        workload_substring=workload_substring,
        config=config,
    )
    best = select_most_energy_efficient_config_per_model(primary_rows)
    best["family"] = best["model"].map(_family_for_model)
    best[["size_b", "display_label"]] = best.apply(
        lambda row: pd.Series(_size_and_label(row["model"], row["family"])),
        axis=1,
    )

    if supplement_gpu and supplement_gpu.lower() != gpu.lower():
        supplement_rows = _load_filtered_rows(
            results_dir,
            gpu=supplement_gpu,
            workload_substring=workload_substring,
            config=config,
        )
        supplement_best = select_most_energy_efficient_config_per_model(supplement_rows)
        supplement_best["family"] = supplement_best["model"].map(_family_for_model)
        supplement_best[["size_b", "display_label"]] = supplement_best.apply(
            lambda row: pd.Series(_size_and_label(row["model"], row["family"])),
            axis=1,
        )
        primary_models = set(best["model"])
        supplement_best = supplement_best[
            (~supplement_best["model"].isin(primary_models))
            & (supplement_best["size_b"] <= SUPPLEMENT_MAX_SIZE_B)
        ].copy()
        best = pd.concat([best, supplement_best], ignore_index=True)

    best["group_order"] = best["family"].map({name: idx for idx, name in enumerate(GROUP_ORDER)})
    best = (
        best.sort_values(
            ["group_order", "size_b", "model"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )
    return best


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def plot_model_bars(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(28, 6.5), constrained_layout=True)

    positions = []
    colors = []
    current_x = 0.0
    group_centers: dict[str, float] = {}
    separator_positions = []
    group_starts = {}
    group_ends = {}
    for family in GROUP_ORDER:
        group_rows = rows[rows["family"] == family]
        if group_rows.empty:
            continue
        start = current_x
        for _ in range(len(group_rows)):
            positions.append(current_x)
            colors.append(GROUP_COLORS[family])
            current_x += BAR_STEP
        end = current_x - BAR_STEP
        group_starts[family] = start
        group_ends[family] = end
        group_centers[family] = (start + end) / 2.0
        next_group_start = current_x + GROUP_GAP
        separator_positions.append((end + next_group_start) / 2.0)
        current_x = next_group_start

    plot_rows = rows.copy()
    plot_rows["x"] = positions
    ax.bar(
        plot_rows["x"],
        plot_rows["bi_per_request"],
        color=colors,
        width=BAR_WIDTH,
        edgecolor="black",
        linewidth=LINEWIDTH * 0.25,
    )

    ax.set_yscale("log")
    ymin = float(plot_rows["bi_per_request"].min()) * 0.75
    ymax = float(plot_rows["bi_per_request"].max()) * 2.6
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("BI$_\\text{fu}$ (species$\\cdot$yr)", fontsize=FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(_scientific_tick))
    ax.set_xticks(plot_rows["x"], plot_rows["display_label"])
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE - 2, width=0, length=0, pad=12)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=10)
    ax.set_xlim(plot_rows["x"].min() - BAR_WIDTH * 1, plot_rows["x"].max() + BAR_WIDTH * 1)

    for family, center in group_centers.items():
        ax.text(
            center,
            -0.25,
            GROUP_DISPLAY[family],
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=TICK_FONTSIZE,
        )

    for separator in separator_positions[:-1]:
        ax.axvline(
            separator,
            color="#444444",
            linewidth=LINEWIDTH * 0.2,
            alpha=0.55,
            linestyle=(0, (6, 6)),
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
    selected = build_selected_model_rows(
        args.results_dir,
        gpu=args.gpu,
        workload_substring=args.workload,
        config=EnergyBiConfig(),
        supplement_gpu=args.supplement_gpu,
    )
    stem = f"{args.gpu.lower()}_{args.workload.lower()}_model_bi_per_request"
    derived_path = args.derived_dir / f"{stem}_selected_configs.csv"
    figure_path = args.figures_dir / f"{stem}_bar.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(derived_path, index=False)
    plot_model_bars(selected, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
