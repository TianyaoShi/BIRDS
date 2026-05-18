from __future__ import annotations

import json
import os
from collections import Counter
from itertools import cycle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import MSTRow

if "MPLCONFIGDIR" not in os.environ:
    default_mpl_config = Path(__file__).resolve().parents[2] / ".mplconfig"
    default_mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(default_mpl_config)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_model_size_vs_mst(
    *,
    rows: Sequence[MSTRow | Mapping[str, Any]],
    output_path: str | Path,
    title: str = "Model Size vs MST",
    x_label: str = "Model Size (B)",
    y_label: str = "MST (rps)",
    x_scale: str = "log",
    annotate: bool = True,
) -> Path:
    points = [_coerce_point(row) for row in rows]
    points = [point for point in points if point["model_size_b"] is not None and point["mst_rps"] is not None]
    if not points:
        raise ValueError("model size vs MST plot requires at least one row with both model_size_b and mst_rps")

    points.sort(key=lambda item: (float(item["model_size_b"]), str(item["model"])))

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    x_values = [float(point["model_size_b"]) for point in points]
    y_values = [float(point["mst_rps"]) for point in points]

    classes = [str(point["bottleneck_class"] or "unknown") for point in points]
    ordered_classes = list(dict.fromkeys(classes))
    palette = cycle(
        [
            "#0f766e",
            "#2563eb",
            "#d97706",
            "#7c3aed",
            "#dc2626",
            "#16a34a",
            "#64748b",
        ]
    )
    class_to_color = {label: next(palette) for label in ordered_classes}
    colors = [class_to_color[label] for label in classes]

    axis.scatter(
        x_values,
        y_values,
        c=colors,
        s=54,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
    )

    if annotate:
        model_counts = Counter(str(point["model"]) for point in points)
        offsets = (
            (0, 10),
            (10, 6),
            (10, -12),
            (0, -14),
            (-10, -4),
            (-10, 12),
        )
        for index, point in enumerate(points):
            dx, dy = offsets[index % len(offsets)]
            label = _default_annotation_label(point, duplicate_model=model_counts[str(point["model"])] > 1)
            axis.annotate(
                label,
                xy=(float(point["model_size_b"]), float(point["mst_rps"])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if x_scale != "linear":
        axis.set_xscale(x_scale)
    axis.grid(True, alpha=0.3)

    if len(ordered_classes) > 1:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=class_to_color[label],
                markeredgecolor="white",
                label=label,
            )
            for label in ordered_classes
        ]
        axis.legend(handles=handles, title="Bottleneck Class", loc="best", frameon=True)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_model_size_vs_mst_from_json(
    *,
    mst_rows_json_path: str | Path,
    output_path: str | Path,
    title: str = "Model Size vs MST",
    x_label: str = "Model Size (B)",
    y_label: str = "MST (rps)",
    x_scale: str = "log",
    annotate: bool = True,
) -> Path:
    payload = json.loads(Path(mst_rows_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("mst_rows.json must contain a list of row mappings")
    return plot_model_size_vs_mst(
        rows=payload,
        output_path=output_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        annotate=annotate,
    )


def plot_model_size_vs_mst_from_orchestrator_run(
    *,
    orchestrator_run_root: str | Path | Sequence[str | Path],
    output_path: str | Path,
    title: str = "Model Size vs MST",
    x_label: str = "Model Size (B)",
    y_label: str = "MST (rps)",
    x_scale: str = "log",
    annotate: bool = True,
    exclude_models: tuple[str, ...] = (),
    exclude_experiment_ids: tuple[str, ...] = (),
    min_model_size_b: float | None = None,
) -> Path:
    from .extract import extract_runs

    if isinstance(orchestrator_run_root, (str, Path)):
        extracted = extract_runs(
            (orchestrator_run_root,),
            exclude_models=exclude_models,
            exclude_experiment_ids=exclude_experiment_ids,
            min_model_size_b=min_model_size_b,
        )
    else:
        extracted = extract_runs(
            tuple(orchestrator_run_root),
            exclude_models=exclude_models,
            exclude_experiment_ids=exclude_experiment_ids,
            min_model_size_b=min_model_size_b,
        )
    return plot_model_size_vs_mst(
        rows=extracted.rows,
        output_path=output_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        annotate=annotate,
    )


def _coerce_point(row: MSTRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, MSTRow):
        return {
            "model": row.model,
            "experiment_id": row.experiment_id,
            "model_size_b": _maybe_float(row.model_size_b),
            "mst_rps": _maybe_float(row.mst_rps),
            "bottleneck_class": row.bottleneck_class,
            "tensor_parallel_size": row.tensor_parallel_size,
            "gpu_count": row.gpu_count,
        }
    if isinstance(row, Mapping):
        return {
            "model": row.get("model"),
            "experiment_id": row.get("experiment_id"),
            "model_size_b": _maybe_float(row.get("model_size_b")),
            "mst_rps": _maybe_float(row.get("mst_rps")),
            "bottleneck_class": row.get("bottleneck_class"),
            "tensor_parallel_size": row.get("tensor_parallel_size"),
            "gpu_count": row.get("gpu_count"),
        }
    raise TypeError(f"unsupported row type for plotting: {type(row)!r}")


def _default_annotation_label(point: Mapping[str, Any], *, duplicate_model: bool) -> str:
    model = str(point.get("model") or "unknown")
    short_model = model.split("/")[-1]
    tp = point.get("tensor_parallel_size")
    if duplicate_model or (tp is not None and tp != 1):
        return f"{short_model}\ntp={tp if tp is not None else '-'}"
    return short_model


def _maybe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
