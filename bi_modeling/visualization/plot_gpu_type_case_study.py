"""Plot GPU-type case-study grouped bars for selected Qwen and Gemma models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    TICK_FONTSIZE,
    apply_academic_style,
)

QUALITY_SCORES_PATH = REPO_ROOT / "results" / "quality_scores" / "quality_scores_compiled.csv"
GPU_SERIES = ("L40", "A100", "H100")
GPU_COLORS = {
    "L40": "#4C78A8",
    "A100": "#59A14F",
    "H100": "#E07A5F",
}
GPU_HATCHES = {
    "L40": "//",
    "A100": "xx",
    "H100": "..",
}
TP_MARKERS = {
    1: "o",
    2: "s",
    4: "D",
    8: "^",
}
MST_LINE_COLOR = "#2F2F2F"
MODEL_SELECTIONS = {
    "qwen": [
        ("Qwen/Qwen3-4B-Instruct-2507", "4B"),
        ("Qwen/Qwen3-8B", "8B"),
        ("Qwen/Qwen3-14B", "14B"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "30B-A3B"),
        ("Qwen/Qwen3-32B", "32B"),
    ],
    "gemma": [
        ("google/gemma-4-E4B-it", "4B"),
        ("google/gemma-4-26B-A4B-it", "26B-A4B"),
        ("google/gemma-4-31B-it", "31B"),
    ],
}
METRICS = {
    "bi_fu": {
        "column": "bi_per_request",
        "ylabel": r"BI$_{\mathrm{fu}}$ (species$\cdot$yr)",
        "log_scale": True,
    },
    "qnbi": {
        "column": "qnbi_per_request",
        "ylabel": r"QNBI (species$\cdot$yr)",
        "log_scale": True,
    },
}


def _quality_metric_for_workload(workload: str) -> str:
    workload_lower = workload.lower()
    if "sharegpt" in workload_lower or "wildchat" in workload_lower:
        return "chat_q"
    raise ValueError(f"No quality score mapping is defined for workload {workload!r}.")


def _normalize_quality_score(value: float) -> float:
    numeric = float(value)
    if numeric > 1.0:
        return numeric / 100.0
    return numeric


def _scientific_tick(value: float, _: int) -> str:
    return f"{value:.0e}"


def _mst_tick(value: float, _: int) -> str:
    return f"{value:.0f}"


def _figure_width(selection_key: str) -> float:
    return 15.5 if selection_key == "qwen" else 11.5


def build_case_study_rows(
    results_dir: Path,
    *,
    workload_substring: str,
    selection_key: str,
    config: EnergyBiConfig,
) -> pd.DataFrame:
    quality_metric = _quality_metric_for_workload(workload_substring)
    selected_models = MODEL_SELECTIONS[selection_key]
    model_order = {model: idx for idx, (model, _) in enumerate(selected_models)}
    display_labels = {model: label for model, label in selected_models}

    rows = build_energy_bi_dataset(results_dir, config=config)
    rows = rows[
        rows["accelerator"].isin(GPU_SERIES)
        & rows["workload"].str.contains(workload_substring, case=False, na=False)
        & rows["model"].isin(model_order)
    ].copy()

    rows = rows.sort_values(
        ["accelerator", "model", "incremental_energy_per_total_request_j", "request_rate"],
        ascending=[True, True, True, False],
    )
    rows = rows.groupby(["accelerator", "model"], as_index=False).first()

    quality = pd.read_csv(QUALITY_SCORES_PATH)[["model", quality_metric]].rename(
        columns={quality_metric: "quality_score"}
    )
    rows = rows.merge(quality, on="model", how="left")
    rows["normalized_quality_score"] = rows["quality_score"].apply(_normalize_quality_score)
    rows["qnbi_per_request"] = rows["bi_per_request"] / rows["normalized_quality_score"]
    rows["mst_value"] = rows["mst_rate"].fillna(rows["request_rate"])
    rows["display_label"] = rows["model"].map(display_labels)
    rows["model_order"] = rows["model"].map(model_order)
    rows["gpu_order"] = rows["accelerator"].map({gpu: idx for idx, gpu in enumerate(GPU_SERIES)})
    rows = rows.sort_values(["model_order", "gpu_order"]).reset_index(drop=True)

    expected_pairs = {(model, gpu) for model, _ in selected_models for gpu in GPU_SERIES}
    observed_pairs = set(zip(rows["model"], rows["accelerator"], strict=True))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        missing_text = ", ".join(f"{model} on {gpu}" for model, gpu in missing_pairs)
        raise ValueError(f"Missing selected GPU/model rows for {selection_key}: {missing_text}")

    return rows


def plot_gpu_case_study(
    rows: pd.DataFrame,
    *,
    selection_key: str,
    metric_key: str,
    output_path: Path,
) -> None:
    apply_academic_style()
    metric = METRICS[metric_key]
    selected_models = MODEL_SELECTIONS[selection_key]

    fig, ax = plt.subplots(figsize=(_figure_width(selection_key), 5.5), constrained_layout=True)
    mst_ax = ax.twinx()

    group_centers = list(range(len(selected_models)))
    bar_width = 0.22
    offsets = {
        "L40": -bar_width,
        "A100": 0.0,
        "H100": bar_width,
    }

    for gpu in GPU_SERIES:
        subset = rows[rows["accelerator"] == gpu].copy()
        subset = subset.sort_values("model_order")
        positions = [group_centers[idx] + offsets[gpu] for idx in subset["model_order"]]
        bars = ax.bar(
            positions,
            subset[metric["column"]],
            width=bar_width,
            color=GPU_COLORS[gpu],
            alpha=0.72,
            edgecolor="black",
            linewidth=LINEWIDTH * 0.55,
            hatch=GPU_HATCHES[gpu],
            label=gpu,
        )

    values = rows[metric["column"]].to_numpy(dtype=float)
    if metric["log_scale"]:
        ax.set_yscale("log")
        ax.set_ylim(values.min() * 0.75, values.max() * 1.3)
        ax.yaxis.set_major_formatter(FuncFormatter(_scientific_tick))

    for model_order in range(len(selected_models)):
        subset = rows[rows["model_order"] == model_order].sort_values("gpu_order")
        positions = [group_centers[model_order] + offsets[gpu] for gpu in subset["accelerator"]]
        mst_ax.plot(
            positions,
            subset["mst_value"].to_numpy(dtype=float),
            color=MST_LINE_COLOR,
            linewidth=LINEWIDTH * 0.55,
            zorder=4,
            alpha=0.85,
        )
        for position, (_, row) in zip(positions, subset.iterrows(), strict=True):
            tp = int(row["tensor_parallel_size"])
            mst_ax.scatter(
                [position],
                [row["mst_value"]],
                marker=TP_MARKERS.get(tp, "o"),
                s=180,
                facecolor="white",
                edgecolor=MST_LINE_COLOR,
                linewidth=LINEWIDTH * 0.55,
                zorder=5,
            )

    mst_values = rows["mst_value"].to_numpy(dtype=float)
    mst_ax.set_ylim(-1.5, mst_values.max() * 1.05)
    mst_ax.set_ylabel("MST (req/s)", fontsize=FONTSIZE)
    mst_ax.yaxis.set_major_formatter(FuncFormatter(_mst_tick))
    mst_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in mst_ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)

    ax.set_ylabel(metric["ylabel"], fontsize=FONTSIZE)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in selected_models])
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, width=0, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    ax.set_xlim(-0.6, len(selected_models) - 1 + 0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)

    gpu_handles = [
        Patch(
            facecolor=GPU_COLORS[gpu],
            alpha=0.72,
            edgecolor="black",
            hatch=GPU_HATCHES[gpu],
            label=gpu,
        )
        for gpu in GPU_SERIES
    ]
    tp_handles = [
        Line2D(
            [0],
            [0],
            color=MST_LINE_COLOR,
            marker=TP_MARKERS[tp],
            markerfacecolor="white",
            markeredgecolor=MST_LINE_COLOR,
            linewidth=0,
            markersize=15,
            label=f"TP{tp}",
        )
        for tp in sorted({int(tp) for tp in rows["tensor_parallel_size"].dropna().unique()})
    ]
    legend_handles = []
    for gpu_handle, tp_handle in zip(gpu_handles, tp_handles, strict=True):
        legend_handles.extend([gpu_handle, tp_handle])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE-2,
        bbox_to_anchor=(0.5, 1.3),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
    parser.add_argument("--workload", default="sharegpt")
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
    config = EnergyBiConfig()

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    for selection_key in MODEL_SELECTIONS:
        rows = build_case_study_rows(
            args.results_dir,
            workload_substring=args.workload,
            selection_key=selection_key,
            config=config,
        )
        derived_path = args.derived_dir / f"{args.workload}_{selection_key}_gpu_type_case_study_rows.csv"
        rows.to_csv(derived_path, index=False)
        print(f"Wrote {derived_path}")

        for metric_key in METRICS:
            figure_path = (
                args.figures_dir
                / f"{args.workload}_{selection_key}_gpu_type_case_study_{metric_key}.pdf"
            )
            plot_gpu_case_study(
                rows,
                selection_key=selection_key,
                metric_key=metric_key,
                output_path=figure_path,
            )
            print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
