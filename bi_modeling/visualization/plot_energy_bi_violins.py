"""Plot BI per request and per token from energy summary_compact.csv files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.visualization.energy_bi_calculator import (  # noqa: E402
    EnergyBiConfig,
    build_energy_bi_dataset,
)
from bi_modeling.visualization.plot_style import (  # noqa: E402
    LEGEND_FONTSIZE,
    FONTSIZE,
    LINEWIDTH,
    TICK_FONTSIZE,
    apply_academic_style,
)

GOOGLE_TOKENS_CSV = REPO_ROOT / "bi_modeling" / "data" / "google_monthly_token_consumption" / "extracted_vals.csv"
OPENROUTER_TOKENS_CSV = (
    REPO_ROOT
    / "bi_modeling"
    / "data"
    / "openrouter_weekly_token_consumption"
    / "openrouter_total_consumption_plot_points.csv"
)
OPENROUTER_PROBLEM_START = pd.Timestamp("2025-01-27")
OPENROUTER_PROBLEM_END = pd.Timestamp("2025-04-21")
VIOLIN_COLORS = {
    "request": "#5B8C5A",
    "token": "#7B5EA7",
}
TREND_COLORS = {
    "google": "#1F77B4",
    "openrouter": "#C44E52",
}


def _plot_violin(ax, data, *, column: str, ylabel: str, color: str) -> None:
    values = [data[column].dropna().to_numpy()]
    parts = ax.violinplot(values, positions=[1], widths=0.34, showmeans=True, showmedians=True)

    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)
        body.set_linewidth(LINEWIDTH * 0.45)

    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(LINEWIDTH * 0.55)

    ax.set_xlim(0.76, 1.24)
    ax.set_xticks([1])
    ax.set_xticklabels([ylabel])
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)


def _bi_per_token_stats(data) -> dict[str, float]:
    values = data["bi_per_token"].dropna()
    return {
        "low": float(values.quantile(0.025)),
        "median": float(values.quantile(0.5)),
        "high": float(values.quantile(0.975)),
    }


def _load_platform_daily_token_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    google = pd.read_csv(GOOGLE_TOKENS_CSV)
    google["date"] = pd.to_datetime(google["month"])
    google["daily_tokens"] = (
        google["tokens_trillion_per_month"]
        * 1e12
        / google["date"].dt.days_in_month
    )
    google = google[["date", "daily_tokens"]].sort_values("date")

    openrouter = pd.read_csv(OPENROUTER_TOKENS_CSV)
    openrouter["date"] = pd.to_datetime(openrouter["week_start"])
    openrouter["daily_tokens"] = openrouter["total_tokens"] / 7.0
    google_start = google["date"].min()
    google_end = google["date"].max()
    keep_date_range = (openrouter["date"] >= google_start) & (openrouter["date"] <= google_end)
    drop_problem_range = (
        (openrouter["date"] >= OPENROUTER_PROBLEM_START)
        & (openrouter["date"] <= OPENROUTER_PROBLEM_END)
    )
    openrouter = openrouter.loc[keep_date_range & ~drop_problem_range, ["date", "daily_tokens"]]
    openrouter = openrouter.sort_values("date")
    return google, openrouter


def _apply_bi_token_stats(series: pd.DataFrame, stats: dict[str, float]) -> pd.DataFrame:
    result = series.copy()
    result["bi_daily_low"] = result["daily_tokens"] * stats["low"]
    result["bi_daily_median"] = result["daily_tokens"] * stats["median"]
    result["bi_daily_high"] = result["daily_tokens"] * stats["high"]
    return result


def _format_daily_tokens(tokens: float) -> str:
    if tokens >= 1e12:
        return f"{tokens / 1e12:.2f}".rstrip("0").rstrip(".") + " T\ntoken/day"
    if tokens >= 1e9:
        return f"{tokens / 1e9:.0f} B\ntoken/day"
    return f"{tokens:.0f}\ntoken/day"


def _annotate_may_token_rates(ax, frame: pd.DataFrame, *, label: str, color: str, y_multiplier: float) -> None:
    for year in (2024, 2025, 2026):
        target = pd.Timestamp(year=year, month=5, day=1)
        if frame.empty:
            continue
        nearest_idx = (frame["date"] - target).abs().idxmin()
        row = frame.loc[nearest_idx]
        if abs((row["date"] - target).days) > 14:
            continue
        x_offset = 0
        if year == 2024:
            x_offset = 28
        elif year == 2026:
            x_offset = -28
        ax.annotate(
            _format_daily_tokens(row["daily_tokens"]),
            xy=(row["date"], row["bi_daily_median"]),
            xytext=(x_offset, y_multiplier),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right" if x_offset < 0 else "center",
            va="bottom" if y_multiplier > 0 else "top",
            fontsize=TICK_FONTSIZE - 3,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": LINEWIDTH * 0.25},
        )


def _plot_platform_trends(ax, data) -> None:
    stats = _bi_per_token_stats(data)
    google, openrouter = (
        _apply_bi_token_stats(series, stats)
        for series in _load_platform_daily_token_series()
    )

    for frame, label, color in (
        (google, "Google", TREND_COLORS["google"]),
        (openrouter, "OpenRouter", TREND_COLORS["openrouter"]),
    ):
        dates = mdates.date2num(frame["date"].dt.to_pydatetime())
        ax.fill_between(
            dates,
            frame["bi_daily_low"].to_numpy(),
            frame["bi_daily_high"].to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            dates,
            frame["bi_daily_median"].to_numpy(),
            color=color,
            linewidth=LINEWIDTH * 0.75,
            label=label,
        )

    _annotate_may_token_rates(ax, google, label="Google", color=TREND_COLORS["google"], y_multiplier=44)
    _annotate_may_token_rates(
        ax,
        openrouter,
        label="OR",
        color=TREND_COLORS["openrouter"],
        y_multiplier=-50,
    )

    ax.set_ylabel(r"Daily BI (species$\cdot$yr/day)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(google["date"].min(), google["date"].max())
    ax.margins(x=0)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=LINEWIDTH * 0.45, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH * 0.45)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")


def plot_energy_bi_violins(data, output_path: Path) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.55, 0.55, 2.0]},
    )
    _plot_violin(
        axes[0],
        data,
        column="bi_per_request",
        ylabel="BI per req",
        color=VIOLIN_COLORS["request"],
    )
    axes[0].set_ylabel(r"BI (species$\cdot$yr)", fontsize=FONTSIZE)
    _plot_violin(
        axes[1],
        data,
        column="bi_per_token",
        ylabel="BI per token",
        color=VIOLIN_COLORS["token"],
    )
    axes[1].set_ylabel("")
    _plot_platform_trends(axes[2], data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "energy")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "bi_modeling" / "visualization" / "figures" / "energy_bi_combined.pdf")
    parser.add_argument("--derived-csv", type=Path, default=REPO_ROOT / "bi_modeling" / "visualization" / "derived" / "energy_bi_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EnergyBiConfig()
    data = build_energy_bi_dataset(args.results_dir, config=config)
    args.derived_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.derived_csv, index=False)
    plot_energy_bi_violins(data, args.output)
    print(f"Wrote {args.derived_csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
