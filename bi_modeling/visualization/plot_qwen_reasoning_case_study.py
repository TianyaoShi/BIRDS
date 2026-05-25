"""Plot Qwen reasoning-vs-instruct case-study bars for MMLU-Pro."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
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
    LEGEND_FONTSIZE,
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)

QUALITY_SCORES_PATH = REPO_ROOT / "results" / "quality_scores" / "quality_scores_compiled.csv"
MMLU_OUTPUT_LENGTHS_PATH = (
    REPO_ROOT / "results" / "workload_length_distributions" / "mmlu_pro_output_lengths_by_request.jsonl"
)
TARGET_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
]
SIZE_ORDER = ["4B", "30B", "235B"]
SERIES_ORDER = ["Instruct", "Thinking"]
SERIES_COLORS = {
    "Instruct": "#4C78A8",
    "Thinking": "#E07A5F",
}
QUALITY_LINE_COLOR = "#2F2F2F"
BI_AXIS_SCALE = 1e-13
SIZE_CENTER_OFFSETS = {
    "4B": 0.0,
    "30B": 0.0,
    "235B": 0.0,
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
    # if active_match:
    #     return f"{size}B-A{active_match.group(1)}B"
    return f"{size}B"


def _series_label(model: str) -> str:
    return "Thinking" if "thinking" in model.lower() else "Instruct"


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _plain_log_tick(value: float, _: int) -> str:
    if np.isclose(value, 1.0):
        return "1"
    if np.isclose(value, 10.0):
        return "10"
    return ""


def _k_token_tick(value: float, _: int) -> str:
    return f"{value / 1000.0:.0f}"


def _load_output_length_stats() -> pd.DataFrame:
    lengths_by_model: dict[str, list[int]] = {model: [] for model in TARGET_MODELS}
    with MMLU_OUTPUT_LENGTHS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            for output in record.get("outputs", []):
                model = output.get("model")
                if model not in lengths_by_model:
                    continue
                if not output.get("success"):
                    continue
                actual_output_len = output.get("actual_output_len")
                if actual_output_len is None:
                    continue
                lengths_by_model[model].append(int(actual_output_len))

    stats_rows = []
    for model in TARGET_MODELS:
        if "235B-A22B" in model:
            stats_rows.append(
                {
                    "model": model,
                    "output_length_mean": np.nan,
                    "output_length_p50": np.nan,
                    "output_length_p95": np.nan,
                    "output_length_count": 0,
                }
            )
            continue
        values = np.asarray(lengths_by_model[model], dtype=float)
        if values.size == 0:
            stats_rows.append(
                {
                    "model": model,
                    "output_length_mean": np.nan,
                    "output_length_p50": np.nan,
                    "output_length_p95": np.nan,
                    "output_length_count": 0,
                }
            )
            continue
        stats_rows.append(
            {
                "model": model,
                "output_length_mean": float(values.mean()),
                "output_length_p50": float(np.percentile(values, 50)),
                "output_length_p95": float(np.percentile(values, 95)),
                "output_length_count": int(values.size),
            }
        )
    return pd.DataFrame(stats_rows)


def build_case_study_rows(results_dir: Path) -> pd.DataFrame:
    rows = build_energy_bi_dataset(results_dir, config=EnergyBiConfig())
    rows = rows[
        (rows["accelerator"] == "H100")
        & (rows["workload"].str.contains("mmlu-pro", case=False, na=False))
        & (rows["model"].isin(TARGET_MODELS))
    ].copy()

    rows = rows.sort_values(["model", "bi_per_request", "request_rate"], ascending=[True, True, False])
    rows = rows.groupby("model", as_index=False).first()

    quality = pd.read_csv(QUALITY_SCORES_PATH)[["model", "mmlu_pro"]].rename(
        columns={"mmlu_pro": "quality_score"}
    )
    rows = rows.merge(quality, on="model", how="left")
    rows["normalized_quality_score"] = rows["quality_score"].apply(_normalize_quality_score)
    rows["qnbi_per_request"] = rows["bi_per_request"] / rows["normalized_quality_score"]

    output_stats = _load_output_length_stats()
    rows = rows.merge(output_stats, on="model", how="left")
    rows["output_length_err_low"] = rows["output_length_mean"] - rows["output_length_p50"]
    rows["output_length_err_high"] = rows["output_length_p95"] - rows["output_length_mean"]

    rows["size_label"] = rows["model"].map(_size_label)
    rows["series_label"] = rows["model"].map(_series_label)
    rows["size_order"] = rows["size_label"].map({label: idx for idx, label in enumerate(SIZE_ORDER)})
    rows["series_order"] = rows["series_label"].map({label: idx for idx, label in enumerate(SERIES_ORDER)})
    rows = rows.sort_values(["size_order", "series_order"]).reset_index(drop=True)
    return rows


def _base_positions() -> tuple[list[float], dict[str, float], dict[str, float], float]:
    x_centers = list(range(len(SIZE_ORDER)))
    bar_width = 0.34
    offsets = {"Instruct": -bar_width / 2, "Thinking": bar_width / 2}
    return x_centers, offsets, {label: idx for idx, label in enumerate(SIZE_ORDER)}, bar_width


def _tick_positions() -> list[float]:
    return [
        index + SIZE_CENTER_OFFSETS.get(size_label, 0.0)
        for index, size_label in enumerate(SIZE_ORDER)
    ]


def _bar_positions(rows: pd.DataFrame) -> list[float]:
    x_centers, offsets, size_index, _ = _base_positions()
    return [
        x_centers[size_index[size_label]] + SIZE_CENTER_OFFSETS.get(size_label, 0.0) + offsets[series_label]
        for size_label, series_label in zip(rows["size_label"], rows["series_label"], strict=True)
    ]


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE+2, width=0, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE-2, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _plot_grouped_bars(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    log_scale: bool,
    scale_factor: float = 1.0,
    exponent_note: str | None = None,
) -> None:
    x_centers, offsets, size_index, bar_width = _base_positions()
    for series in SERIES_ORDER:
        subset = rows[rows["series_label"] == series]
        positions = [
            x_centers[size_index[label]] + SIZE_CENTER_OFFSETS.get(label, 0.0) + offsets[series]
            for label in subset["size_label"]
        ]
        ax.bar(
            positions,
            subset[value_column] / scale_factor,
            width=bar_width,
            color=SERIES_COLORS[series],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.45,
            label=series,
        )

    if log_scale:
        values = rows[value_column].to_numpy(dtype=float) / scale_factor
        values = values[np.isfinite(values) & (values > 0)]
        ax.set_yscale("log")
        ax.set_ylim(values.min() * 0.75, values.max() * 1.6)
        ax.set_yticks([1, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))

    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xticks(_tick_positions())
    ax.set_xticklabels(SIZE_ORDER)
    _style_axis(ax)
    if exponent_note:
        ax.text(
            0.0,
            1.01,
            exponent_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=TICK_FONTSIZE - 8,
        )


def _plot_bi_with_quality(ax: plt.Axes, rows: pd.DataFrame) -> None:
    _plot_grouped_bars(
        ax,
        rows,
        value_column="bi_per_request",
        ylabel=r"BI$_{\mathrm{fu}}$ (species$\cdot$yr)",
        log_scale=True,
        scale_factor=BI_AXIS_SCALE,
        exponent_note=r"$10^{-13}$",
    )
    quality_ax = ax.twinx()
    positions = _bar_positions(rows)
    quality_ax.plot(
        positions,
        rows["normalized_quality_score"].to_numpy(),
        color=QUALITY_LINE_COLOR,
        marker="o",
        linewidth=LINEWIDTH * 0.7,
        markersize=14,
        zorder=5,
        label=r"$Q(\theta)$",
    )
    quality_values = rows["normalized_quality_score"].to_numpy(dtype=float)
    quality_ax.set_ylim(max(0.0, quality_values.min() - 0.08), min(1.0, quality_values.max() + 0.08))
    quality_ax.set_ylabel(r"$Q(\theta)$", fontsize=FONTSIZE)
    # quality_ax.yaxis.set_label_coords(1.03, 0.45)
    quality_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE-2, width=LINEWIDTH * 0.45, length=12)
    for spine in quality_ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _plot_output_lengths(ax: plt.Axes, rows: pd.DataFrame) -> None:
    x_centers, offsets, size_index, bar_width = _base_positions()
    for series in SERIES_ORDER:
        subset = rows[rows["series_label"] == series].copy()
        positions = [
            x_centers[size_index[label]] + SIZE_CENTER_OFFSETS.get(label, 0.0) + offsets[series]
            for label in subset["size_label"]
        ]
        means = subset["output_length_mean"].to_numpy(dtype=float)
        lower = subset["output_length_err_low"].to_numpy(dtype=float)
        upper = subset["output_length_err_high"].to_numpy(dtype=float)
        mask = np.isfinite(means)
        if not mask.any():
            continue
        ax.bar(
            np.asarray(positions)[mask],
            means[mask],
            width=bar_width,
            color=SERIES_COLORS[series],
            edgecolor="black",
            linewidth=LINEWIDTH * 0.45,
            yerr=np.vstack([lower[mask], upper[mask]]),
            error_kw={"elinewidth": LINEWIDTH * 0.45, "capsize": 7, "capthick": LINEWIDTH * 0.45, "ecolor": "black"},
        )

    valid_values = rows["output_length_p95"].to_numpy(dtype=float)
    valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0)]
    if valid_values.size:
        ax.set_ylim(0.0, valid_values.max() * 1.1)
    ax.set_ylabel("Output (k tokens)", fontsize=FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(_k_token_tick))
    ax.set_xticks(_tick_positions())
    ax.set_xticklabels(SIZE_ORDER)
    _style_axis(ax)


def plot_case_study(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    _plot_bi_with_quality(axes[0], rows)
    _plot_output_lengths(axes[1], rows)
    _plot_grouped_bars(
        axes[2],
        rows,
        value_column="qnbi_per_request",
        ylabel=r"QNBI (species$\cdot$yr)",
        log_scale=True,
        scale_factor=BI_AXIS_SCALE,
        exponent_note=r"$10^{-13}$",
    )

    fig.legend(
        handles=[
            Patch(facecolor=SERIES_COLORS["Instruct"], edgecolor="black", label="Instruct"),
            Patch(facecolor=SERIES_COLORS["Thinking"], edgecolor="black", label="Thinking"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE + 6,
        bbox_to_anchor=(0.52, 1.17),
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
    derived_path = args.derived_dir / "qwen_reasoning_mmlu_case_study_rows.csv"
    figure_path = args.figures_dir / "qwen_reasoning_mmlu_case_study_1x3.pdf"

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(derived_path, index=False)
    plot_case_study(rows, figure_path)
    print(f"Wrote {derived_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
