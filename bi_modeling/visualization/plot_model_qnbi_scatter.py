"""Plot model size versus quality-normalized BI for selected model configs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
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


def _classify_is_moe(row: pd.Series) -> bool:
    if row["family"] == "gpt-oss":
        return True
    if row["family"] == "gemma-4" and re.search(r"-e(?:2|4)b-", row["model"].lower()):
        return False
    return _is_moe_model(row["model"])


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
    rows["is_moe"] = rows.apply(_classify_is_moe, axis=1)
    return rows


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _size_tick(value: float, _: int) -> str:
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


def _scaled_log_features(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.column_stack(
        [
            np.log10(rows["size_b"].to_numpy(dtype=float)),
            np.log10(rows["qnbi_per_request"].to_numpy(dtype=float)),
        ]
    )
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0
    return (features - means) / stds, means, stds


def _find_linear_separator(rows: pd.DataFrame) -> tuple[np.ndarray, bool]:
    scaled_features, _, _ = _scaled_log_features(rows)
    augmented_features = np.column_stack([scaled_features, np.ones(len(rows), dtype=float)])
    labels = np.where(rows["is_moe"].to_numpy(dtype=bool), 1.0, -1.0)
    weights = np.zeros(augmented_features.shape[1], dtype=float)

    for _ in range(20_000):
        mistakes = 0
        for feature, label in zip(augmented_features, labels, strict=True):
            if label * np.dot(weights, feature) <= 0:
                weights += label * feature
                mistakes += 1
        if mistakes == 0:
            return weights, True
    return weights, False


def _optimize_separator(rows: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, float]:
    scaled_features, _, _ = _scaled_log_features(rows)
    labels = np.where(rows["is_moe"].to_numpy(dtype=bool), 1.0, -1.0)
    positive = scaled_features[labels > 0]
    negative = scaled_features[labels < 0]
    base_theta = float(np.arctan2(weights[1], weights[0]))

    best_normal: np.ndarray | None = None
    best_intercept: float | None = None
    best_margin = -np.inf
    best_angle_delta = np.inf

    for theta in np.linspace(base_theta - np.deg2rad(25), base_theta + np.deg2rad(25), 401):
        normal = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        lower_bound = float(np.max(-positive @ normal))
        upper_bound = float(np.min(-negative @ normal))
        if lower_bound >= upper_bound:
            continue
        intercept = 0.5 * (lower_bound + upper_bound)
        margin = 0.5 * (upper_bound - lower_bound)
        angle_delta = abs(theta - base_theta)
        if margin > best_margin + 1e-12 or (
            abs(margin - best_margin) <= 1e-12 and angle_delta < best_angle_delta
        ):
            best_normal = normal
            best_intercept = intercept
            best_margin = margin
            best_angle_delta = angle_delta

    if best_normal is not None and best_intercept is not None:
        return best_normal, best_intercept

    fallback_normal = weights[:2].astype(float)
    fallback_norm = np.linalg.norm(fallback_normal)
    if fallback_norm == 0:
        fallback_normal = np.array([1.0, 0.0], dtype=float)
    else:
        fallback_normal /= fallback_norm
    return fallback_normal, float(weights[2] / max(fallback_norm, 1.0))


def _plot_separator_if_separable(ax: plt.Axes, rows: pd.DataFrame) -> None:
    if rows["is_moe"].nunique() < 2:
        return

    weights, separable = _find_linear_separator(rows)
    if not separable:
        return
    normal, intercept = _optimize_separator(rows, weights)

    x_min = float(rows["size_b"].min())
    x_max = float(rows["size_b"].max())
    y_min = float(rows["qnbi_per_request"].min())
    y_max = float(rows["qnbi_per_request"].max())
    x_values = np.logspace(np.log10(x_min), np.log10(x_max), 256)
    _, means, stds = _scaled_log_features(rows)

    if abs(normal[1]) < 1e-12:
        if abs(normal[0]) < 1e-12:
            return
        x_boundary_scaled = -intercept / normal[0]
        x_boundary = 10 ** (x_boundary_scaled * stds[0] + means[0])
        ax.axvline(
            x_boundary,
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH * 0.5,
            alpha=0.8,
            zorder=2,
        )
        return

    scaled_log_x = (np.log10(x_values) - means[0]) / stds[0]
    scaled_log_y = -(normal[0] * scaled_log_x + intercept) / normal[1]
    log_y = scaled_log_y * stds[1] + means[1]
    y_values = 10 ** log_y
    in_range = (y_values >= y_min) & (y_values <= y_max)
    if not np.any(in_range):
        return
    ax.plot(
        x_values[in_range],
        y_values[in_range],
        color="black",
        linestyle="--",
        linewidth=LINEWIDTH * 0.5,
        alpha=0.8,
        zorder=2,
    )


def plot_qnbi_scatter(rows: pd.DataFrame, output_path: Path) -> None:
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(15, 8.5), constrained_layout=True)
    y_min = float(rows["qnbi_per_request"].min())
    y_max = float(rows["qnbi_per_request"].max())

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

    _plot_separator_if_separable(ax, rows)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model size (B params)", fontsize=FONTSIZE)
    ax.set_ylabel("QNBI($\\theta$) (species$\\cdot$yr)", fontsize=FONTSIZE)
    ax.set_ylim(y_min / 1.25, y_max * 1.25)
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
