"""Plot Qwen superGPQA reasoning-vs-instruct case-study bars."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    LEGEND_FONTSIZE,
    LINEWIDTH,
    MARKERSIZE,
    TICK_FONTSIZE,
    apply_academic_style,
)

QUALITY_SCORES_PATH = REPO_ROOT / "results" / "quality_scores" / "quality_scores_compiled.csv"
SUPERGPQA_OUTPUT_SUMMARY_CSV = (
    REPO_ROOT / "results" / "workload_length_distributions" / "supergpqa_real_output_length_summary.csv"
)
TARGET_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
]
SIZE_ORDER = ["4B", "30B-A3B", "235B-A22B"]
SERIES_ORDER = ["Instruct", "Thinking"]
SERIES_COLORS = {
    "Instruct": "#4C78A8",
    "Thinking": "#E07A5F",
}


def _normalize_quality_score(value: float) -> float:
    numeric = float(value)
    if numeric > 1.0:
        return numeric / 100.0
    return numeric


def _size_label(model: str) -> str:
    lower = model.lower()
    size_match = re.search(r"(\d+(?:\.\d+)?)b", lower)
    if not size_match:
        return model
    size = size_match.group(1)
    active_match = re.search(r"a(\d+(?:\.\d+)?)b", lower)
    if active_match:
        return f"{size}B-A{active_match.group(1)}B"
    return f"{size}B"


def _series_label(model: str) -> str:
    return "Thinking" if "thinking" in model.lower() else "Instruct"


def build_case_study_rows(results_dir: Path) -> pd.DataFrame:
    rows = build_energy_bi_dataset(results_dir, config=EnergyBiConfig())
    rows = rows[
        (rows["accelerator"] == "H100")
        & (rows["workload"].str.contains("supergpqa", case=False, na=False))
        & (rows["model"].isin(TARGET_MODELS))
    ].copy()

    rows = rows.sort_values(["model", "bi_per_request", "request_rate"], ascending=[True, True, False])
    rows = rows.groupby("model", as_index=False).first()

    quality = pd.read_csv(QUALITY_SCORES_PATH)[["model", "supergpqa_all"]].rename(
        columns={"supergpqa_all": "quality_score"}
    )
    rows = rows.merge(quality, on="model", how="left")
    rows["normalized_quality_score"] = rows["quality_score"].apply(_normalize_quality_score)
    rows["qnbi_per_request"] = rows["bi_per_request"] / rows["normalized_quality_score"]
    rows["size_label"] = rows["model"].map(_size_label)
    rows["series_label"] = rows["model"].map(_series_label)

    output_summary = pd.read_csv(SUPERGPQA_OUTPUT_SUMMARY_CSV)
    output_summary = output_summary[
        (output_summary["split"] == "hard") & (output_summary["model"].isin(TARGET_MODELS))
    ][
        [
            "model",
            "output_tokens_mean",
            "output_tokens_p50",
            "output_tokens_p95",
        ]
    ].copy()
    rows = rows.merge(output_summary, on="model", how="left")
    rows["output_err_low"] = (rows["output_tokens_mean"] - rows["output_tokens_p50"]).clip(lower=0)
    rows["output_err_high"] = (rows["output_tokens_p95"] - rows["output_tokens_mean"]).clip(lower=0)

    rows["size_order"] = rows["size_label"].map({label: idx for idx, label in enumerate(SIZE_ORDER)})
    rows["series_order"] = rows["series_label"].map({label: idx for idx, label in enumerate(SERIES_ORDER)})
    rows = rows.sort_values(["size_order", "series_order"]).reset_index(drop=True)
    return rows


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _bar_positions(rows: pd.DataFrame):
    centers = list(range(len(SIZE_ORDER)))
    bar_width = 0.34
    offsets = {"Instruct": -bar_width / 2, "Thinking": bar_width / 2}
    series_positions = {}
    for series in SERIES_ORDER:
        subset = rows[rows["series_label"] == series]
        series_positions[series] = [
            centers[SIZE_ORDER.index(label)] + offsets[series] for label in subset["size_label"]
        ]
    return centers, bar_width, series_positions


def _plot_metric_bars(ax: plt.Axes, rows: pd.DataFrame, *, value_column: str, ylabel: str, log_scale: bool) -> None:
    centers, bar_width, series_positions = _bar_positions(rows)
    for series in SERIES_ORDER:
        subset = rows[rows["series_label"] == series]
        ax.bar(
            series_positions[series],
            subset[value_column],
            width=bar_width,
            color=SERIES_COLORS[series],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.45,
            label=series,
        )

    if log_scale:
        values = rows[value_column].to_numpy()
        ax.set_yscale("log")
        ax.set_ylim(values.min() * 0.75, values.max() * 1.6)
        ax.yaxis.set_major_formatter(FuncFormatter(_scientific_tick))

    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xticks(centers)
    ax.set_xticklabels(SIZE_ORDER)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, width=0, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _plot_output_and_quality(ax: plt.Axes, rows: pd.DataFrame) -> None:
    centers, bar_width, series_positions = _bar_positions(rows)
    right_ax = ax.twinx()

    for series in SERIES_ORDER:
        subset = rows[rows["series_label"] == series]
        ax.bar(
            series_positions[series],
            subset["output_tokens_mean"],
            width=bar_width,
            color=SERIES_COLORS[series],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.45,
            yerr=[subset["output_err_low"], subset["output_err_high"]],
            error_kw={
                "elinewidth": LINEWIDTH * 0.35,
                "capsize": 8,
                "capthick": LINEWIDTH * 0.35,
                "ecolor": "black",
            },
        )
        right_ax.plot(
            series_positions[series],
            subset["quality_score"],
            color=SERIES_COLORS[series],
            linewidth=LINEWIDTH * 0.7,
            marker="o",
            markersize=MARKERSIZE + 2,
            markerfacecolor="white",
            markeredgewidth=LINEWIDTH * 0.25,
            markeredgecolor=SERIES_COLORS[series],
            zorder=5,
        )

    ax.set_ylabel("Output length (tokens)", fontsize=FONTSIZE)
    ax.set_xticks(centers)
    ax.set_xticklabels(SIZE_ORDER)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, width=0, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    right_ax.set_ylabel(r"Q($\theta$) (%)", fontsize=FONTSIZE)
    right_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)
    for spine in right_ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def plot_case_study(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(30, 8), constrained_layout=True)

    _plot_metric_bars(
        axes[0],
        rows,
        value_column="bi_per_request",
        ylabel=r"BI$_{\mathrm{req}}$ (species$\cdot$yr)",
        log_scale=True,
    )
    _plot_output_and_quality(axes[1], rows)
    _plot_metric_bars(
        axes[2],
        rows,
        value_column="qnbi_per_request",
        ylabel=r"QNBI$_{\mathrm{req}}$ (species$\cdot$yr)",
        log_scale=True,
    )
    axes[0].legend(frameon=False, loc="upper left", fontsize=LEGEND_FONTSIZE - 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
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
    rows = build_case_study_rows(args.results_dir)
    derived_path = args.derived_dir / "qwen_supergpqa_case_study_rows.csv"
    figure_path = args.figures_dir / "qwen_supergpqa_case_study_1x3.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(derived_path, index=False)
    plot_case_study(rows, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
