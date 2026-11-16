"""Shared chart style: colours, type sizes and save helpers.

Every figure is drawn twice from the same code:
    report variant  -> results/figures/<name>.png   (title, subtitle and source note inside the image)
    slides variant  -> slides/assets/<name>.png     (larger type; no title or source note:
                                                     the slide supplies both)

Colours come from one validated palette (colour-blind safe for the first three
categorical slots; one-hue blue ramp for magnitude). Text is always dark ink,
never the series colour.
"""
from __future__ import annotations

from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config as C

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"        # titles, values
INK2 = "#52514e"       # labels, subtitles
MUTED = "#898781"      # axis ticks, notes
GRID = "#e1e0d9"       # hairline grid
AXIS = "#c3c2b7"       # baselines; also the de-emphasis grey for context marks

BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2
AQUA = "#1baf7a"       # categorical slot 3
RED = "#e34948"
NEUTRAL = "#f0efec"

BLUE_RAMP = ["#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_FROM_ZERO = LinearSegmentedColormap.from_list("seq_blue0", ["#eef4fd", "#cde2fb"] + BLUE_RAMP)

SLIDES_DIR = C.ROOT / "slides" / "assets"
DPI = 200


def rc(variant: str) -> dict:
    base = 11.5 if variant == "report" else 16.5
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.titlepad": 8,
        "axes.labelsize": base,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": base - 1,
        "legend.labelcolor": INK2,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "text.color": INK,
    }


def render(name: str, draw: Callable, *, title: str, subtitle: str | None = None,
           source: str | None = None, size: tuple[float, float] = (9.6, 5.4)) -> None:
    """Draw one figure in both variants and save it.

    `draw(fig, variant)` creates the axes and marks. The title block and the
    source note are added here so they look the same on every figure.
    """
    for variant in ("report", "slides"):
        with plt.rc_context(rc(variant)):
            fig = plt.figure(figsize=size, layout="constrained")
            h = size[1]
            report = variant == "report"
            top = (0.78 if subtitle else 0.50) / h if report else 0.0
            bottom = (0.34 / h) if source and report else 0.0
            fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - top - bottom), w_pad=0.12, h_pad=0.10)
            draw(fig, variant)
            if variant == "report":
                fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14.5,
                         fontweight="bold", color=INK)
                if subtitle:
                    fig.text(0.012, 1 - 0.44 / h, subtitle, ha="left", va="top", fontsize=11, color=INK2)
            if source and report:
                fig.text(0.012, 0.10 / h, source, ha="left", va="bottom", fontsize=9, color=MUTED)
            out = (C.FIGURES if variant == "report" else SLIDES_DIR) / f"{name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
    print(f"  figure  {name}")


def dots(ax, x, y, color=BLUE, size=52, zorder=3, **kw):
    """Scatter markers with a thin surface-coloured ring so overlaps stay readable."""
    return ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.1, zorder=zorder, **kw)


def label_points(ax, df, xcol, ycol, names, offsets=None, fontsize=None):
    """Directly label a few named counties (used sparingly)."""
    offsets = offsets or {}
    for name in names:
        row = df.loc[df["county"].eq(name)].iloc[0]
        dx, dy, *align = offsets.get(name, (6, 6))      # optional third item: horizontal alignment
        ha = align[0] if align else ("left" if dx >= 0 else "right")
        va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
        ax.annotate(name, (row[xcol], row[ycol]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=INK2, ha=ha, va=va, zorder=5)


def zero_line(ax, axis: str = "x") -> None:
    """Solid baseline at zero (a reference, so it uses the axis colour)."""
    (ax.axvline if axis == "x" else ax.axhline)(0, color=AXIS, linewidth=1.2, zorder=1)


SOURCE_PLACES = ("Source: CDC PLACES county data, 2025 release (model-based estimates from 2023 BRFSS); "
                 "100 North Carolina counties.")
SOURCE_MAP = ("Sources: CDC PLACES county data, 2025 release (2023 BRFSS); "
              "county outlines from the U.S. Census Bureau cartographic boundary file, 2023.")
SOURCE_BOTH = ("Sources: CDC PLACES county data, 2025 release (2023 BRFSS); "
               "EPA AirData annual AQI by county, 2023.")
