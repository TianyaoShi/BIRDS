"""Shared plotting style for biodiversity-impact figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager

LINEWIDTH = 5
MARKERSIZE = 14
FONTSIZE = 40
LEGEND_FONTSIZE = 30
TICK_FONTSIZE = FONTSIZE - 8
SERIF_FONT_FAMILY = [
    "Times New Roman",
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Tinos",
    "Liberation Serif",
]


def _first_available_font(font_families: list[str]) -> str:
    for font_family in font_families:
        try:
            font_manager.findfont(font_family, fallback_to_default=False)
        except ValueError:
            continue
        return font_family
    return "serif"


def apply_academic_style() -> None:
    plt.rcParams["font.family"] = _first_available_font(SERIF_FONT_FAMILY)
    plt.rcParams["axes.linewidth"] = LINEWIDTH
    plt.rcParams["lines.linewidth"] = LINEWIDTH
    plt.rcParams["lines.markersize"] = MARKERSIZE
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["axes.labelsize"] = FONTSIZE
    plt.rcParams["xtick.labelsize"] = TICK_FONTSIZE
    plt.rcParams["ytick.labelsize"] = TICK_FONTSIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONTSIZE
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
