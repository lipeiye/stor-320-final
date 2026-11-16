"""Layered figures for the slide deck.

    python -I slides/build_figures.py            # every figure -> slides/layers/*.png
    python -I slides/build_figures.py s05 s09    # only the figures whose name starts with these

These images are presentation material, not analysis: every number they show is read from
results/results.json, results/tables/*.csv or the processed county table, so run the pipeline
first (run_all.py). Nothing here writes to results/.

How the builds work. A figure that is revealed in steps is saved as several transparent PNGs of
the same size: a base layer and one layer per click. Every layer is drawn on identical axes
(fixed position, fixed limits), so stacking them in the slide gives the full chart. One figure
unit is one pixel of the 1920 x 1080 slide canvas; files are written at twice that resolution.

Style: the paper background comes from the slide, grey is context, blue is smoking and
inactivity (and results adjusted for them), orange is the social indicators and highlights,
text is always an ink colour, Inter throughout (the deck's body face; the font files and their
licences are in slides/fonts/, so the figures look the same on any machine). No text is smaller
than 32 slide pixels (16 pt on a 13.33 inch slide), so every label can be read from the back of a room.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

import config as C

OUT = ROOT / "slides" / "layers"
SCALE = 2                      # PNG pixels per canvas pixel
T = 32                         # smallest text, in slide pixels

INK = "#14233b"                # values and labels that carry the message
BODY = "#3b4656"               # axis titles, ordinary labels
MUTED = "#5f6b7a"              # tick labels, notes
GREY = "#b0b7c3"               # context marks
GREY_DARK = "#7d8796"          # context marks that must stay readable on their own
RULE = "#d5dae2"               # axis lines
ZERO = "#8a94a3"               # reference line at zero
BLUE = "#1c5cab"               # smoking, inactivity, results adjusted for them
ORANGE = "#eb6834"             # social indicators, highlights
NAVY = "#0f2440"
PAPER = "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("ink", ["#e3e9f2", "#a9bcd6", "#5d7fae", "#27508a", NAVY])

R = C.read_json(C.RESULTS_JSON)
D = pd.read_csv(C.PROCESSED_TABLE, dtype={"fips": str})
NEED = {"transport": "No transport", "housing": "Housing insecurity", "uninsured": "Uninsured"}
NEED_ORDER = ["transport", "housing", "uninsured"]     # the collinear pair first, side by side


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLES / f"{name}.csv")


# --------------------------------------------------------------------------
# Small drawing kit
# --------------------------------------------------------------------------
def pt(pixels: float) -> float:
    """Slide pixels to points (figures are laid out at 100 dpi)."""
    return pixels * 0.72


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def plain(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


class Fig:
    """One canvas of W x H slide pixels. Axes are placed by pixel rectangle so that layers line up."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self.fig.patch.set_alpha(0)

    def axes(self, left: float, top: float, width: float, height: float, frame: bool = True):
        ax = self.fig.add_axes([left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h])
        ax.patch.set_alpha(0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
            ax.spines[side].set_linewidth(pt(2))
        ax.tick_params(length=0, pad=pt(10), labelsize=pt(T), labelcolor=MUTED)
        if not frame:
            ax.axis("off")
        return ax

    def text(self, x: float, y: float, s: str, size: float = 32, color: str = INK, ha: str = "left",
             va: str = "top", weight: str = "normal", **kw):
        """Text placed in slide pixels measured from the figure's top-left corner."""
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=pt(size), color=color, ha=ha, va=va,
                             fontweight=weight, **kw)

    def swatch(self, x: float, y: float, color: str, kind: str = "dot") -> None:
        """A small key mark at slide-pixel position (x, y): a dot, a bar or a hollow diamond."""
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")
        ax.patch.set_alpha(0)
        if kind == "dot":
            ax.scatter([x], [y], s=pt(20) ** 2, color=color, linewidth=0)
        elif kind == "bar":
            ax.plot([x - 20, x + 20], [y, y], color=color, linewidth=pt(7), solid_capstyle="round")
        else:
            ax.scatter([x], [y], s=pt(20) ** 2, marker="D", facecolor=PAPER, edgecolor=color, linewidth=pt(3.5))

    def save(self, name: str) -> None:
        # a label that runs past the canvas is cut off without any warning, so check every text box
        renderer, (width, height), outside = self.fig.canvas.get_renderer(), self.fig.bbox.size, []
        for text in self.fig.findobj(Text):
            if text.get_visible() and text.get_text().strip():
                box = text.get_window_extent(renderer)
                over = max(-box.x0, -box.y0, box.x1 - width, box.y1 - height) / SCALE
                if over > 0.5:
                    outside.append(f"{text.get_text()!r} by {over:.1f} px")
        if outside:
            raise SystemExit(f"{name}: text runs past the edge of the figure: " + "; ".join(outside))
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / f"{name}.png", dpi=100 * SCALE, transparent=True, metadata={"Software": None})
        plt.close(self.fig)
        print(f"  layer   {name}")


def halo(width: float = 6):
    return [pe.withStroke(linewidth=pt(width), foreground=PAPER)]


def county_paths() -> dict:
    geometry = {f["properties"]["GEOID"]: f["geometry"] for f in C.read_json(C.SNAPSHOT_BOUNDARIES)["features"]}

    def outline(geom: dict) -> MplPath:
        verts, codes = [], []
        for ring in C.polygon_rings(geom):
            verts.extend(ring)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
        return MplPath(verts, codes)

    return {fips: outline(g) for fips, g in geometry.items()}


MAP_XLIM, MAP_YLIM = (-84.45, -75.35), (33.75, 36.65)
MAP_ASPECT = 1 / np.cos(np.radians(35.4))     # a degree of latitude is longer than a degree of longitude here


def map_axes(f: Fig, left: float, top: float, width: float):
    """Axes holding the state outline at its true proportions. Returns the axes and its pixel height."""
    height = width * (MAP_YLIM[1] - MAP_YLIM[0]) * MAP_ASPECT / (MAP_XLIM[1] - MAP_XLIM[0])
    ax = f.axes(left, top, width, height, frame=False)
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    return ax, height


def leader(ax, start: tuple, end: tuple) -> None:
    ax.annotate("", start, xytext=end, zorder=6,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=pt(2), shrinkA=0, shrinkB=0))


def interval(ax, y: float, low: float, high: float, estimate: float, color: str, width: float = 6) -> None:
    ax.plot([low, high], [y, y], color=color, linewidth=pt(width), solid_capstyle="round", zorder=3)
    ax.scatter([estimate], [y], s=pt(20) ** 2, color=color, edgecolor=PAPER, linewidth=pt(2.5), zorder=4)


def slope_axis(ax, limits: tuple, ticks: list, bottom: float, top: float = -0.8) -> None:
    """Shared look of the slope charts: zero line, a few signed ticks, no y axis."""
    ax.set_xlim(*limits)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(ticks, ["0" if t == 0 else signed(t, 1) for t in ticks])
    ax.axvline(0, color=ZERO, linewidth=pt(2.5), zorder=1)


# --------------------------------------------------------------------------
# Cover: county outlines as a quiet backdrop
# --------------------------------------------------------------------------
def cover() -> None:
    f = Fig(1500, 600)
    ax, _ = map_axes(f, 0, 6, 1500)
    for path in county_paths().values():
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor="#9cc7f5", linewidth=pt(1.6), alpha=0.28))
    f.save("cover_outline")


# --------------------------------------------------------------------------
# Slide 2: the map, then the three unadjusted scatterplots
# --------------------------------------------------------------------------
def s02() -> None:
    paths = county_paths()
    norm = Normalize(vmin=D["copd"].min(), vmax=D["copd"].max())
    hi, lo = R["eda"]["copd_highest"], R["eda"]["copd_lowest"]

    f = Fig(980, 600)
    ax, _ = map_axes(f, 0, 92, 980)
    for row in D.itertuples():
        ax.add_patch(PathPatch(paths[row.fips], facecolor=RAMP(norm(row.copd)), edgecolor=PAPER,
                               linewidth=pt(1.4), zorder=2))
    for county in (hi["county"], lo["county"]):
        row = D.loc[D["county"].eq(county)].iloc[0]
        ax.add_patch(PathPatch(paths[row["fips"]], facecolor="none", edgecolor=ORANGE, linewidth=pt(4.5), zorder=4))
    low_row = D.loc[D["county"].eq(lo["county"])].iloc[0]
    leader(ax, (low_row["lon"] + 0.06, low_row["lat"] + 0.22), (-77.72, 36.98))
    f.text(736, 44, f"{lo['county']} {lo['copd']:.1f}%", size=38, weight="bold", va="center")
    f.text(736, 88, "lowest", size=T, color=MUTED, va="center", path_effects=halo())
    high_row = D.loc[D["county"].eq(hi["county"])].iloc[0]
    leader(ax, (high_row["lon"] - 0.14, high_row["lat"] - 0.24), (-80.75, 33.62))
    f.text(388, 510, f"{hi['county']} {hi['copd']:.1f}%", size=38, weight="bold", ha="right", va="center")
    f.text(388, 556, "highest", size=T, color=MUTED, ha="right", va="center")
    key = f.axes(612, 524, 330, 20, frame=False)
    key.imshow(np.linspace(0, 1, 256)[None, :], cmap=RAMP, aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1))
    for value in (5, 7.5, 10):
        f.text(612 + 330 * (value - norm.vmin) / (norm.vmax - norm.vmin), 554, f"{value:g}%", size=T,
               color=MUTED, ha="center")
    f.text(612, 496, "Adults with COPD", size=T, color=BODY, va="center")
    f.save("s02_map")

    labels = {"transport": "No reliable\ntransportation", "uninsured": "No health\ninsurance",
              "housing": "Housing\ninsecurity"}
    f = Fig(620, 600)
    for k, col in enumerate(C.SOCIAL):
        top = 4 + k * 166
        ax = f.axes(8, top, 290, 136)
        ax.scatter(D[col], D["copd"], s=pt(9) ** 2, color=ORANGE, alpha=0.75, linewidth=0, zorder=3)
        slope, intercept = np.polyfit(D[col], D["copd"], 1)
        xs = np.array([D[col].min(), D[col].max()])
        ax.plot(xs, intercept + slope * xs, color=INK, linewidth=pt(3), zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(3.9, 10.9)
        f.text(326, top, labels[col], size=T, color=BODY, linespacing=1.12)
        f.text(326, top + 88, f"r = {R['eda']['corr_copd'][col]:.2f}", size=46, weight="bold")
    f.text(8, 500, "Each dot is one county.\nHigher means more COPD.", size=T, color=MUTED, linespacing=1.25)
    f.save("s02_scatter")


# --------------------------------------------------------------------------
# Slide 4: every county estimate with its CDC interval
# --------------------------------------------------------------------------
def s04() -> None:
    s = D.sort_values("copd").reset_index(drop=True)
    overlap = R["eda"]["copd_interval_overlap"]
    med = overlap["median_estimate"]
    apart = (s["copd_lcl"] > med) | (s["copd_ucl"] < med)
    assert int(apart.sum()) == overlap["n_interval_excludes_median"]
    x = np.arange(1, len(s) + 1)

    for layer in ("base", "apart"):
        f = Fig(1040, 620)
        ax = f.axes(112, 70, 904, 452)
        ax.set_xlim(-1.5, 102.5)
        ax.set_ylim(2.4, 12.3)
        if layer == "base":
            ax.set_xticks([])
            ax.set_yticks([4, 6, 8, 10, 12], ["4%", "6%", "8%", "10%", "12%"])
            ax.vlines(x, s["copd_lcl"], s["copd_ucl"], color=GREY, linewidth=pt(4), zorder=2)
            ax.scatter(x, s["copd"], s=pt(9) ** 2, color=GREY_DARK, linewidth=0, zorder=3)
            ax.axhline(med, color=INK, linewidth=pt(2), zorder=1)
            ax.annotate(f"median county {med:.1f}%", (0, med), xytext=(0, pt(10)), textcoords="offset points",
                        fontsize=pt(T), color=INK, va="bottom", path_effects=halo())
            first, last = s.iloc[0], s.iloc[-1]
            ax.annotate(f"{first['county']} {first['copd']:.1f}%", (1, first["copd_lcl"]), xytext=(0, -pt(8)),
                        textcoords="offset points", fontsize=pt(T), color=INK, va="top", ha="left")
            ax.annotate(f"{last['county']} {last['copd']:.1f}%", (100, last["copd_ucl"]), xytext=(0, pt(8)),
                        textcoords="offset points", fontsize=pt(T), color=INK, va="bottom", ha="right")
            f.text(112, 6, "Adults with COPD: CDC estimate and its 95% interval", size=T, color=BODY)
            f.text(564, 538, "100 counties, from lowest to highest estimate", size=T, color=MUTED, ha="center")
        else:
            ax.axis("off")
            ax.vlines(x[apart], s.loc[apart, "copd_lcl"], s.loc[apart, "copd_ucl"], color=BLUE,
                      linewidth=pt(4.6), zorder=4)
            ax.scatter(x[apart], s.loc[apart, "copd"], s=pt(10) ** 2, color=BLUE, linewidth=0, zorder=5)
            f.swatch(148, 150, BLUE, "bar")
            f.text(182, 150, f"Only {int(apart.sum())} intervals clear the median", size=34, weight="bold",
                   va="center")
            copd = R["eda"]["descriptives"]["copd"]
            f.text(1000, 404, f"Typical interval: ±{copd['cdc_ci_halfwidth_mean']:.2f} points\n"
                              f"Spread between counties: {copd['sd']:.2f}", size=T, color=BODY, ha="right",
                   linespacing=1.3)
        f.save(f"s04_intervals_{layer}")


# --------------------------------------------------------------------------
# Slide 5: fit, then the slopes under three sets of adjustment variables
# --------------------------------------------------------------------------
def s05_fit() -> None:
    fit = R["q1"]["fit"]["copd"]
    rows = [("Smoking alone", R["q1"]["smoking_only_r2"], None),
            ("+ physical inactivity", fit["baseline"]["r2"], None),
            ("+ 3 social indicators", fit["expanded"]["r2"], fit["baseline"]["r2"])]
    f = Fig(380, 600)
    ax = f.fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 380)
    ax.set_ylim(600, 0)
    ax.axis("off")
    ax.patch.set_alpha(0)
    f.text(0, 0, "Fit of the model (R²)", size=T, color=MUTED)
    track = 256
    for k, (label, value, previous) in enumerate(rows):
        top = 70 + k * 112
        f.text(0, top, label, size=T, color=BODY)
        ax.add_patch(Rectangle((0, top + 46), track, 30, facecolor="#e7ebf1", edgecolor="none"))
        ax.add_patch(Rectangle((0, top + 46), track * (previous or value), 30, facecolor=BLUE, edgecolor="none"))
        if previous:
            ax.add_patch(Rectangle((track * previous, top + 46), track * (value - previous), 30,
                                   facecolor=ORANGE, edgecolor="none"))
        f.text(track + 16, top + 61, f"{value:.3f}", size=36, weight="bold", va="center")
    f.text(0, 400, signed(R["q1"]["comparison"]["copd"]["delta_r2"], 3), size=92, weight="bold")
    f.text(0, 504, "gained by all three\nsocial indicators", size=T, color=BODY, linespacing=1.2)
    f.save("s05_fit")


def specification_forest(outcome: str, name: str, limits: tuple, ticks: list, layered: bool) -> None:
    spec = table("q1_specifications")
    spec = spec[spec["outcome"].eq(outcome)].set_index(["term", "specification"])
    steps = [("alone", "Each indicator alone", GREY_DARK),
             ("with smoking and inactivity", "+ smoking and inactivity", BLUE),
             ("with all five predictors", "All five together", ORANGE)]
    key_x = [56, 452, 912]
    top, height, span = 80, 404, 10.55

    def frame(f: Fig):
        ax = f.axes(316, top, 760, height)
        slope_axis(ax, limits, ticks, bottom=9.75)
        return ax

    def row_px(y: float) -> float:
        return top + height * (y + 0.8) / span

    def draw_step(f: Fig, ax, k: int) -> None:
        specification, label, color = steps[k]
        f.swatch(key_x[k], 30, color)
        f.text(key_x[k] + 24, 30, label, size=T, color=BODY, va="center")
        for g, term in enumerate(NEED_ORDER):
            row = spec.loc[(term, specification)]
            y = g * 3.5 + k
            interval(ax, y, row["hc3_low"], row["hc3_high"], row["estimate"], color)
            f.text(1104, row_px(y), signed(row["estimate"]), size=T, va="center")

    def draw_frame(f: Fig) -> None:
        for g, term in enumerate(NEED_ORDER):
            f.text(296, row_px(g * 3.5 + 1), NEED[term], size=T, weight="bold", ha="right", va="center")
        f.text(696, 552, f"Difference in {'COPD' if outcome == 'copd' else 'asthma'} (points) per 1 point of the indicator",
               size=T, color=MUTED, ha="center")

    if not layered:
        f = Fig(1236, 600)
        ax = frame(f)
        draw_frame(f)
        for k in range(3):
            draw_step(f, ax, k)
        f.save(name)
        return
    for k in range(3):
        f = Fig(1236, 600)
        ax = frame(f)
        if k == 0:
            draw_frame(f)
        else:
            ax.axis("off")
        draw_step(f, ax, k)
        f.save(f"{name}_{k}")


def s05() -> None:
    s05_fit()
    specification_forest("copd", "s05_slopes", (-0.5, 0.62), [-0.4, -0.2, 0, 0.2, 0.4, 0.6], layered=True)


# --------------------------------------------------------------------------
# Slide 6: two indicators that are almost one, and the same model re-expressed
# --------------------------------------------------------------------------
def s06() -> None:
    f = Fig(620, 600)
    ax = f.axes(116, 16, 488, 462)
    odd = D["county"].eq(R["q1"]["most_influential_county"])
    ax.scatter(D.loc[~odd, "transport"], D.loc[~odd, "housing"], s=pt(13) ** 2, color=GREY_DARK, alpha=0.8,
               linewidth=0, zorder=3)
    ax.scatter(D.loc[odd, "transport"], D.loc[odd, "housing"], s=pt(17) ** 2, color=ORANGE, edgecolor=PAPER,
               linewidth=pt(2), zorder=4)
    row = D.loc[odd].iloc[0]
    ax.annotate(row["county"], (row["transport"], row["housing"]), xytext=(18.45, 20.9), textcoords="data",
                fontsize=pt(T), color=INK, ha="right", va="top", zorder=5,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=pt(2), shrinkA=pt(2), shrinkB=pt(12),
                                relpos=(0.86, 1)))
    ax.set_xlim(5, 18.6)
    ax.set_ylim(7.2, 25.6)
    ax.set_xticks([6, 10, 14, 18], ["6%", "10%", "14%", "18%"])
    ax.set_yticks([10, 15, 20, 25], ["10%", "15%", "20%", "25%"])
    f.text(144, 34, f"r = {R['eda']['corr_transport_housing']:.2f}", size=48, weight="bold")
    f.text(360, 540, "No reliable transportation", size=T, color=MUTED, ha="center")
    f.text(20, 247, "Housing insecurity", size=T, color=MUTED, ha="center", va="center", rotation=90)
    vif = R["q1"]["vif"]["expanded"]
    own = 100 / np.mean([vif["transport"], vif["housing"]])
    f.text(596, 340, f"VIF {vif['transport']:.0f} and {vif['housing']:.0f}:\nabout {own:.0f}% of each\nvaries on its own",
           size=T, color=BODY, ha="right", linespacing=1.25)
    f.save("s06_pair")

    coefs = table("q1_coefficients")
    coefs = coefs[coefs["outcome"].eq("copd") & coefs["model"].eq("expanded")].set_index("term")
    pair = table("q1_transport_housing_pair").set_index("term")
    rows = [(0, "No transport", coefs.loc["transport"], ORANGE, "base"),
            (1, "Housing insecurity", coefs.loc["housing"], ORANGE, "base"),
            (3.1, "Average of the two", pair.loc["mean of transport and housing (per 1 point)"], INK, "pair"),
            (4.1, "Gap between them", pair.loc["transport minus housing (per 1 point)"], INK, "pair")]
    top, height, span = 26, 452, 5.8
    for layer in ("base", "pair"):
        f = Fig(980, 600)
        ax = f.axes(380, top, 440, height)
        slope_axis(ax, (-0.5, 0.62), [-0.4, 0, 0.4], bottom=5.0)
        if layer == "base":
            f.text(600, 548, "Difference in COPD (points) per 1 point", size=T, color=MUTED, ha="center")
        else:
            ax.axis("off")
            f.text(358, top + height * (2.1 + 0.8) / span, "Same model, rewritten:", size=T, color=MUTED,
                   ha="right", va="center")
        for y, label, row, color, where in rows:
            if where != layer:
                continue
            interval(ax, y, row["hc3_low"], row["hc3_high"], row["estimate"], color)
            ypx = top + height * (y + 0.8) / span
            f.text(358, ypx, label, size=T, weight="bold", ha="right", va="center")
            digits = 3 if abs(row["estimate"]) < 0.01 else 2
            f.text(848, ypx, signed(row["estimate"], digits), size=T, weight="bold", va="center")
        f.save(f"s06_rewrite_{layer}")


# --------------------------------------------------------------------------
# Slide 7: COPD against smoking, with CDC's own limits around every county
# --------------------------------------------------------------------------
def s07() -> None:
    f = Fig(740, 600)
    ax = f.axes(116, 16, 608, 462)
    ax.errorbar(D["smoking"], D["copd"],
                xerr=[D["smoking"] - D["smoking_lcl"], D["smoking_ucl"] - D["smoking"]],
                yerr=[D["copd"] - D["copd_lcl"], D["copd_ucl"] - D["copd"]],
                fmt="none", ecolor=GREY, elinewidth=pt(1.6), alpha=0.75, zorder=1)
    slope, intercept = np.polyfit(D["smoking"], D["copd"], 1)
    xs = np.array([D["smoking"].min(), D["smoking"].max()])
    ax.plot(xs, intercept + slope * xs, color=INK, linewidth=pt(2.5), zorder=2)
    ax.scatter(D["smoking"], D["copd"], s=pt(13) ** 2, color=BLUE, edgecolor=PAPER, linewidth=pt(1.5), zorder=3)
    ax.set_xlim(6.5, 28)
    ax.set_ylim(2.6, 12.2)
    ax.set_xticks([10, 15, 20, 25], ["10%", "15%", "20%", "25%"])
    ax.set_yticks([4, 6, 8, 10, 12], ["4%", "6%", "8%", "10%", "12%"])
    f.text(144, 34, f"r = {R['eda']['corr_copd']['smoking']:.2f}", size=48, weight="bold")
    f.text(712, 424, "grey lines: CDC 95% intervals", size=T, color=MUTED, ha="right", path_effects=halo())
    f.text(420, 540, "Adults who smoke", size=T, color=MUTED, ha="center")
    f.text(20, 247, "Adults with COPD", size=T, color=MUTED, ha="center", va="center", rotation=90)
    f.save("s07_smoking")


# --------------------------------------------------------------------------
# Slide 8: three kinds of 95% interval, and the same slopes without one county
# --------------------------------------------------------------------------
def s08() -> None:
    coefs = table("q1_coefficients")
    coefs = coefs[coefs["outcome"].eq("copd") & coefs["model"].eq("expanded")].set_index("term")
    county = R["q1"]["most_influential_county"]
    without = {row["term"]: row["estimate"] for row in R["q1"]["sensitivity"]
               if row["analysis"].startswith(f"without {county}")}
    kinds = [("ci", "Classical", GREY_DARK, "base"), ("hc3", "Robust (HC3)", ORANGE, "more"),
             ("boot", "County bootstrap", BLUE, "more")]
    key_x = {"ci": 150, "hc3": 392, "boot": 682}
    terms = ["transport", "housing"]
    top, height, span = 80, 404, 9.4

    def row_px(y: float) -> float:
        return top + height * (y + 0.8) / span

    for layer in ("base", "more", "county"):
        f = Fig(1000, 600)
        ax = f.axes(316, top, 664, height)
        slope_axis(ax, (-0.56, 0.78), [-0.4, 0, 0.4], bottom=8.6)
        if layer == "base":
            for g, term in enumerate(terms):
                f.text(296, row_px(g * 4.6 + 1.0), NEED[term], size=T, weight="bold", ha="right", va="center")
                f.text(296, row_px(g * 4.6 + 2.0), f"slope {signed(coefs.loc[term, 'estimate'])}", size=T,
                       color=BODY, ha="right", va="center")
            f.text(648, 552, "COPD difference per 1 point of the indicator", size=T, color=MUTED,
                   ha="center")
        else:
            ax.axis("off")
        for k, (stem, label, color, where) in enumerate(kinds):
            if where != layer:
                continue
            f.swatch(key_x[stem] + 20, 30, color, "bar")
            f.text(key_x[stem] + 52, 30, label, size=T, color=BODY, va="center")
            for g, term in enumerate(terms):
                row = coefs.loc[term]
                interval(ax, g * 4.6 + k, row[f"{stem}_low"], row[f"{stem}_high"], row["estimate"], color)
        if layer == "county":
            for g, term in enumerate(terms):
                y = g * 4.6 + 3.05
                est, new = coefs.loc[term, "estimate"], without[term]
                ax.annotate("", (new, y), xytext=(est, y), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=pt(3), shrinkA=0,
                                            shrinkB=pt(9), mutation_scale=pt(26)))
                ax.scatter([est], [y], s=pt(12) ** 2, color=INK, linewidth=0, zorder=5)
                ax.scatter([new], [y], s=pt(20) ** 2, marker="D", facecolor=PAPER, edgecolor=INK,
                           linewidth=pt(3.5), zorder=6)
                side = 1 if new > est else -1
                ax.annotate(signed(new), (new, y), xytext=(side * pt(22), 0), textcoords="offset points",
                            fontsize=pt(T), color=INK, va="center", ha="left" if side > 0 else "right")
                f.text(296, row_px(y), f"without {county}", size=T, color=BODY, ha="right", va="center")
        f.save(f"s08_intervals_{layer}")


# --------------------------------------------------------------------------
# Slide 9: median AQI, before and after adjusting for the two behaviors
# --------------------------------------------------------------------------
def s09() -> None:
    a = D[D["has_aqi"].eq(1)].reset_index(drop=True)
    q2 = R["q2"]
    size = (pt(11) + pt(30) * (np.log10(a["population"]) - 3.6) / 2.5) ** 2      # dot area grows with county size
    X = np.column_stack([np.ones(len(a)), a[C.BEHAVIORAL].to_numpy()])
    res_y = a["copd"].to_numpy() - X @ np.linalg.lstsq(X, a["copd"].to_numpy(), rcond=None)[0]
    res_x = a["aqi_median"].to_numpy() - X @ np.linalg.lstsq(X, a["aqi_median"].to_numpy(), rcond=None)[0]
    span = 2.9

    f = Fig(800, 600)
    f.text(104, 4, "Before adjustment", size=T, weight="bold")
    f.text(776, 4, "dot size = county size", size=T, color=MUTED, ha="right")
    ax = f.axes(104, 64, 672, 424)
    ax.scatter(a["aqi_median"], a["copd"], s=size, color=GREY_DARK, alpha=0.7, edgecolor=PAPER,
               linewidth=pt(1.5), zorder=3)
    xs = np.array([a["aqi_median"].min(), a["aqi_median"].max()])
    b0 = a["copd"].mean() - q2["unadjusted"]["estimate"] * a["aqi_median"].mean()
    ax.plot(xs, b0 + q2["unadjusted"]["estimate"] * xs, color=ORANGE, linewidth=pt(5), zorder=4,
            solid_capstyle="round")
    for county, dx, dy, ha, va in (("Wake", -26, 0, "right", "center"), ("Hyde", -40, 16, "left", "bottom")):
        row = a.loc[a["county"].eq(county)].iloc[0]
        people = (f"{row['population'] / 1e6:.1f} million" if row["population"] >= 1e6
                  else f"{round(row['population'], -2):,.0f}")
        ax.annotate(f"{county}: {people} people", (row["aqi_median"], row["copd"]), xytext=(pt(dx), pt(dy)),
                    textcoords="offset points", fontsize=pt(T), color=INK, ha=ha, va=va, zorder=6,
                    path_effects=halo())
    ax.set_xlim(23.5, 54.5)
    ax.set_ylim(a["copd"].mean() - span, a["copd"].mean() + span)
    ax.set_xticks([30, 40, 50])
    ax.set_yticks([5, 6, 7, 8, 9], ["5%", "6%", "7%", "8%", "9%"])
    f.text(752, 80, f"r = {plain(q2['corr_copd_aqi_median'])}", size=46, weight="bold", ha="right")
    f.text(440, 550, "Median daily AQI, 2023", size=T, color=MUTED, ha="center")
    f.text(20, 276, "Adults with COPD", size=T, color=MUTED, ha="center", va="center", rotation=90)
    f.save("s09_before")

    f = Fig(800, 600)
    f.text(104, 4, "After adjusting for smoking and inactivity", size=T, weight="bold")
    ax = f.axes(104, 64, 672, 424)
    ax.axhline(0, color=RULE, linewidth=pt(2), zorder=1)
    ax.scatter(res_x, res_y, s=size, color=GREY_DARK, alpha=0.7, edgecolor=PAPER, linewidth=pt(1.5), zorder=3)
    xr = np.array([res_x.min(), res_x.max()])
    ax.plot(xr, q2["adjusted"]["estimate"] * xr, color=BLUE, linewidth=pt(5), zorder=4, solid_capstyle="round")
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-span, span)
    ax.set_xticks([-10, 0, 10], ["−10", "0", "+10"])
    ax.set_yticks([-2, -1, 0, 1, 2], ["−2", "−1", "0", "+1", "+2"])
    f.text(752, 80, f"slope {plain(q2['adjusted']['estimate'], 3)}", size=46, weight="bold", ha="right")
    f.text(752, 140, f"95% interval {plain(q2['adjusted']['ci_low'], 3)} to {plain(q2['adjusted']['ci_high'], 3)}",
           size=T, color=BODY, ha="right")
    f.text(752, 416, f"{q2['ge300']['n']} counties with {C.AQI_MIN_DAYS}+ days: slope "
                     f"{plain(q2['ge300']['estimate'], 3)}", size=T, color=MUTED, ha="right")
    f.text(440, 550, "AQI not explained by smoking and inactivity", size=T, color=MUTED, ha="center")
    f.text(20, 276, "COPD not explained (points)", size=T, color=MUTED, ha="center", va="center", rotation=90)
    f.save("s09_after")


# --------------------------------------------------------------------------
# Slide 10: the two classifiers over 50 partitions, and the counties both miss
# --------------------------------------------------------------------------
def s10() -> None:
    q3 = R["q3"]
    wide = table("q3_repeats").pivot(index="repeat", columns="model", values="roc_auc")
    diff = (wide["penalized"] - wide["baseline"]).to_numpy()
    summary = {(row["metric"], row["quantity"]): row for row in q3["summary"]}
    gap = summary[("roc_auc", "penalized minus baseline")]
    assert abs(diff.mean() - gap["repeats_mean"]) < 1e-4 and len(diff) == q3["n_repeats"]

    step = 0.0008
    bins = np.round(diff / step).astype(int)
    f = Fig(960, 520)
    ax = f.axes(40, 124, 880, 268)
    for b in np.unique(bins):
        n = int((bins == b).sum())
        ax.scatter(np.full(n, b * step), np.arange(n) + 0.5, s=pt(20) ** 2, color=GREY_DARK, linewidth=0, zorder=3)
    ax.axvline(0, color=INK, linewidth=pt(2.5), zorder=2)
    ax.set_xlim(-0.0125, 0.0105)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-0.01, -0.005, 0, 0.005, 0.01], ["−0.010", "−0.005", "0", "+0.005", "+0.010"])
    ax.scatter([gap["repeats_mean"]], [10.5], s=pt(24) ** 2, marker="v", color=ORANGE, linewidth=0, zorder=5)
    ax.annotate(f"average {signed(gap['repeats_mean'], 3)}", (gap["repeats_mean"], 10.5), xytext=(pt(22), 0),
                textcoords="offset points", fontsize=pt(T), color=INK, va="center")
    base, pen = q3["prespecified"]["baseline"]["roc_auc"], q3["prespecified"]["penalized"]["roc_auc"]
    assert f"{base:.3f}" == f"{pen:.3f}"
    f.text(40, 2, f"ROC AUC on held-out counties: {base:.3f} for both models", size=T, weight="bold")
    f.text(40, 52, f"Each dot is one of {q3['n_repeats']} random splits of the counties", size=T, color=MUTED)
    f.text(60, 462, "← smoking alone better", size=T, color=BODY)
    f.text(900, 462, "five indicators better →", size=T, color=BODY, ha="right")
    f.save("s10_partitions")

    paths = county_paths()
    oof = table("q3_out_of_fold_predictions").set_index("county")
    f = Fig(640, 420)
    ax, height = map_axes(f, 0, 100, 640)
    missed = []
    for row in D.itertuples():
        o = oof.loc[row.county]
        if o["high"] == 1 and o["flag_baseline"] == 0 and o["flag_penalized"] == 0:
            color, z = ORANGE, 3
            missed.append(row)
        elif o["high"] == 1:
            color, z = "#27508a", 2
        else:
            color, z = "#e3e7ee", 2
        ax.add_patch(PathPatch(paths[row.fips], facecolor=color, edgecolor=PAPER, linewidth=pt(1.1), zorder=z))
    assert sorted(m.county for m in missed) == sorted(m["county"] for m in q3["missed_counties"])
    f.swatch(16, 22, "#27508a")
    f.text(42, 22, "high COPD, flagged", size=T, color=BODY, va="center")
    f.swatch(16, 68, ORANGE)
    f.text(42, 68, f"high COPD, missed by both ({len(missed)})", size=T, color=BODY, va="center")
    f.text(0, 100 + height + 16, ", ".join(sorted(m.county for m in missed)), size=T, weight="bold")
    f.save("s10_missed")


# --------------------------------------------------------------------------
# Backup figures
# --------------------------------------------------------------------------
def backup() -> None:
    specification_forest("asthma", "b1_asthma", (-0.42, 0.74), [-0.4, -0.2, 0, 0.2, 0.4, 0.6], layered=False)

    cover_rows = table("eda_aqi_coverage").set_index("variable").loc["population"]
    groups = [(1, f"With AQI records ({int(cover_rows['n_with_aqi'])})", cover_rows["with_aqi_mean"], BLUE),
              (0, f"No AQI records ({int(cover_rows['n_without_aqi'])})", cover_rows["without_aqi_mean"], GREY_DARK)]
    jitter = np.random.default_rng(C.SEED).uniform(-0.26, 0.26, size=len(D))
    f = Fig(1000, 440)
    ax = f.axes(384, 52, 590, 270)
    for k, (flag, label, mean, color) in enumerate(groups):
        m = D["has_aqi"].eq(flag).to_numpy()
        ax.scatter(np.log10(D.loc[m, "population"]), k + jitter[m], s=pt(15) ** 2, color=color, alpha=0.7,
                   linewidth=0, zorder=3)
        ax.plot([np.log10(mean)] * 2, [k - 0.42, k + 0.42], color=INK, linewidth=pt(4), zorder=4)
        ax.annotate(f"mean {mean:,.0f}", (np.log10(mean), k + (0.42 if k else -0.42)),
                    xytext=(pt(10), pt(2) if k else -pt(2)), textcoords="offset points", fontsize=pt(T),
                    color=INK, va="top" if k else "bottom", path_effects=halo())
        f.text(360, 52 + 270 * (k + 0.6) / 2.45, label, size=T, weight="bold", ha="right", va="center")
    ax.set_ylim(1.85, -0.6)                # room under the lower mean label, clear of the axis
    ax.set_xlim(3.4, 6.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([4, 5, 6], ["10,000", "100,000", "1,000,000"])
    f.text(679, 384, "County population (log scale)", size=T, color=MUTED, ha="center")
    f.save("b3_coverage")


FIGURES = {"cover": cover, "s02": s02, "s04": s04, "s05": s05, "s06": s06, "s07": s07, "s08": s08,
           "s09": s09, "s10": s10, "backup": backup}


def main() -> None:
    for path in sorted((ROOT / "slides" / "fonts").glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "Inter", "axes.unicode_minus": True, "text.color": INK})
    wanted = sys.argv[1:]
    for name, build in FIGURES.items():
        if not wanted or any(name.startswith(w) for w in wanted):
            build()


if __name__ == "__main__":
    main()
