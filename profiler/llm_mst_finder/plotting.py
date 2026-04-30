from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

if "MPLCONFIGDIR" not in os.environ:
    default_mpl_config = Path(__file__).resolve().parents[2] / ".mplconfig"
    default_mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(default_mpl_config)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def plot_trial_windows(
    *,
    trial_dir: str | Path,
    x_values: Sequence[float],
    arrival_rate: Sequence[float],
    completion_rate: Sequence[float],
    outstanding: Sequence[float],
    ttft_p50_ms: Sequence[float | None],
    ttft_p90_ms: Sequence[float | None],
    ttft_p95_ms: Sequence[float | None],
    ttft_p99_ms: Sequence[float | None],
    tpot_p50_ms: Sequence[float | None],
    tpot_p90_ms: Sequence[float | None],
    tpot_p95_ms: Sequence[float | None],
    tpot_p99_ms: Sequence[float | None],
    output_tok_s: Sequence[float | None],
    kv_cache_usage: Sequence[float | None],
    num_running: Sequence[float | None],
    num_waiting: Sequence[float | None],
    num_swapped: Sequence[float | None],
) -> dict[str, str]:
    output_dir = Path(trial_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "arrival_vs_completion_rate": _plot_multi_line(
            output_dir / "arrival_vs_completion_rate.png",
            title="Arrival vs Completion Rate",
            x_label="Time (s)",
            y_label="Requests / s",
            x_values=x_values,
            series=(
                ("arrival_rate", arrival_rate, "#0f766e"),
                ("completion_rate", completion_rate, "#b45309"),
            ),
        ),
        "outstanding_requests": _plot_multi_line(
            output_dir / "outstanding_requests.png",
            title="Outstanding Requests",
            x_label="Time (s)",
            y_label="Requests",
            x_values=x_values,
            series=(("outstanding_mean", outstanding, "#1d4ed8"),),
        ),
        "ttft_percentiles": _plot_multi_line(
            output_dir / "ttft_percentiles.png",
            title="TTFT Percentiles",
            x_label="Time (s)",
            y_label="Milliseconds",
            x_values=x_values,
            series=(
                ("p50", ttft_p50_ms, "#16a34a"),
                ("p90", ttft_p90_ms, "#d97706"),
                ("p95", ttft_p95_ms, "#ea580c"),
                ("p99", ttft_p99_ms, "#dc2626"),
            ),
        ),
        "tpot_percentiles": _plot_multi_line(
            output_dir / "tpot_percentiles.png",
            title="TPOT Percentiles",
            x_label="Time (s)",
            y_label="Milliseconds",
            x_values=x_values,
            series=(
                ("p50", tpot_p50_ms, "#16a34a"),
                ("p90", tpot_p90_ms, "#d97706"),
                ("p95", tpot_p95_ms, "#ea580c"),
                ("p99", tpot_p99_ms, "#dc2626"),
            ),
        ),
        "output_tokens_per_s": _plot_multi_line(
            output_dir / "output_tokens_per_s.png",
            title="Output Tokens per Second",
            x_label="Time (s)",
            y_label="Tokens / s",
            x_values=x_values,
            series=(("generation_tok_s", output_tok_s, "#7c3aed"),),
        ),
        "kv_cache_usage": _plot_multi_line(
            output_dir / "kv_cache_usage.png",
            title="KV Cache Usage",
            x_label="Time (s)",
            y_label="Usage Fraction",
            x_values=x_values,
            series=(("kv_cache_usage_max", kv_cache_usage, "#0ea5e9"),),
            y_min=0.0,
            y_max=1.0,
        ),
        "server_queue_state": _plot_multi_line(
            output_dir / "server_queue_state.png",
            title="Running / Waiting / Swapped Requests",
            x_label="Time (s)",
            y_label="Requests",
            x_values=x_values,
            series=(
                ("num_running_mean", num_running, "#2563eb"),
                ("num_waiting_mean", num_waiting, "#f97316"),
                ("num_swapped_mean", num_swapped, "#dc2626"),
            ),
        ),
    }
    return {key: str(path) for key, path in outputs.items()}


def plot_search_results(
    *,
    output_dir: str | Path,
    request_rates: Sequence[float],
    classifications: Sequence[str],
    ttft_p90_ms: Sequence[float | None],
    tpot_p90_ms: Sequence[float | None],
    output_tok_s: Sequence[float | None],
    queue_drift: Sequence[float | None],
) -> dict[str, str]:
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "search_rate_vs_classification": _plot_classification_scatter(
            plots_dir / "search_rate_vs_classification.png",
            title="Tested Request Rate vs Classification",
            x_values=request_rates,
            labels=classifications,
        ),
        "search_rate_vs_ttft_p90": _plot_scatter(
            plots_dir / "search_rate_vs_ttft_p90.png",
            title="Request Rate vs TTFT p90",
            x_label="Request Rate (req/s)",
            y_label="TTFT p90 (ms)",
            x_values=request_rates,
            y_values=ttft_p90_ms,
            color="#d97706",
        ),
        "search_rate_vs_tpot_p90": _plot_scatter(
            plots_dir / "search_rate_vs_tpot_p90.png",
            title="Request Rate vs TPOT p90",
            x_label="Request Rate (req/s)",
            y_label="TPOT p90 (ms)",
            x_values=request_rates,
            y_values=tpot_p90_ms,
            color="#16a34a",
        ),
        "search_rate_vs_output_tokens": _plot_scatter(
            plots_dir / "search_rate_vs_output_tokens.png",
            title="Request Rate vs Output Token Throughput",
            x_label="Request Rate (req/s)",
            y_label="Output Tokens / s",
            x_values=request_rates,
            y_values=output_tok_s,
            color="#7c3aed",
        ),
        "search_rate_vs_queue_drift": _plot_scatter(
            plots_dir / "search_rate_vs_queue_drift.png",
            title="Request Rate vs Queue Drift",
            x_label="Request Rate (req/s)",
            y_label="Outstanding Slope (/s)",
            x_values=request_rates,
            y_values=queue_drift,
            color="#1d4ed8",
        ),
    }
    return {key: str(path) for key, path in outputs.items()}


def plot_result_comparison(
    *,
    output_dir: str | Path,
    labels: Sequence[str],
    max_sustainable_req_s: Sequence[float],
    max_output_tok_s: Sequence[float],
    bottleneck_classes: Sequence[str],
) -> dict[str, str]:
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "comparison_max_sustainable_rate": _plot_bar(
            plots_dir / "comparison_max_sustainable_rate.png",
            title="Max Sustainable Request Rate by Result Directory",
            x_labels=labels,
            values=max_sustainable_req_s,
            y_label="Requests / s",
            color="#0f766e",
        ),
        "comparison_max_output_tokens": _plot_bar(
            plots_dir / "comparison_max_output_tokens.png",
            title="Max Output Tokens / s by Result Directory",
            x_labels=labels,
            values=max_output_tok_s,
            y_label="Tokens / s",
            color="#7c3aed",
        ),
        "comparison_bottleneck_class": _plot_classification_bars(
            plots_dir / "comparison_bottleneck_class.png",
            title="Bottleneck Class by Result Directory",
            x_labels=labels,
            classes=bottleneck_classes,
        ),
    }
    return {key: str(path) for key, path in outputs.items()}


def _plot_multi_line(
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> Path:
    if not x_values:
        raise ValueError(f"{title} requires at least one x value")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    plotted = 0
    for label, y_values, color in series:
        filtered_x = [x for x, y in zip(x_values, y_values) if y is not None]
        filtered_y = [float(y) for y in y_values if y is not None]
        if not filtered_x:
            continue
        axis.plot(filtered_x, filtered_y, marker="o", linewidth=1.8, markersize=3.5, label=label, color=color)
        plotted += 1
    if plotted == 0:
        raise ValueError(f"{title} requires at least one non-empty data series")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if y_min is not None or y_max is not None:
        axis.set_ylim(bottom=y_min, top=y_max)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_scatter(
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: Sequence[float],
    y_values: Sequence[float | None],
    color: str,
) -> Path:
    filtered = [(x, float(y)) for x, y in zip(x_values, y_values) if y is not None]
    if not filtered:
        raise ValueError(f"{title} requires at least one point")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter([item[0] for item in filtered], [item[1] for item in filtered], color=color, s=36)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_classification_scatter(
    output_path: Path,
    *,
    title: str,
    x_values: Sequence[float],
    labels: Sequence[str],
) -> Path:
    if not x_values or len(x_values) != len(labels):
        raise ValueError(f"{title} requires matching non-empty x_values and labels")
    ordered_labels = list(dict.fromkeys(labels))
    positions = {label: idx for idx, label in enumerate(ordered_labels)}
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(x_values, [positions[label] for label in labels], color="#1d4ed8", s=36)
    axis.set_title(title)
    axis.set_xlabel("Request Rate (req/s)")
    axis.set_yticks(list(positions.values()), ordered_labels)
    axis.set_ylabel("Classification")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_bar(
    output_path: Path,
    *,
    title: str,
    x_labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
    color: str,
) -> Path:
    if not x_labels or len(x_labels) != len(values):
        raise ValueError(f"{title} requires matching non-empty labels and values")
    figure, axis = plt.subplots(figsize=(max(7.0, len(x_labels) * 1.4), 4.8))
    axis.bar(range(len(values)), values, color=color)
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xticks(range(len(x_labels)), x_labels, rotation=20, ha="right")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_classification_bars(
    output_path: Path,
    *,
    title: str,
    x_labels: Sequence[str],
    classes: Sequence[str],
) -> Path:
    if not x_labels or len(x_labels) != len(classes):
        raise ValueError(f"{title} requires matching non-empty labels and classes")
    ordered_classes = list(dict.fromkeys(classes))
    positions = {value: idx for idx, value in enumerate(ordered_classes)}
    figure, axis = plt.subplots(figsize=(max(7.0, len(x_labels) * 1.4), 4.8))
    axis.bar(range(len(classes)), [positions[value] for value in classes], color="#f97316")
    axis.set_title(title)
    axis.set_xticks(range(len(x_labels)), x_labels, rotation=20, ha="right")
    axis.set_yticks(list(positions.values()), ordered_classes)
    axis.set_ylabel("Bottleneck Class")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
