"""Step 02 - describe the data before any modeling.

Tables (results/tables)
    eda_descriptives.csv            spread of each indicator across counties and the width of CDC's intervals
    eda_correlations.csv            Pearson correlations among the seven indicators
    eda_correlations_spearman.csv   rank correlations (check on the Pearson table)
    eda_copd_extremes.csv           five highest and five lowest counties with CDC limits
    eda_aqi_coverage.csv            counties with versus without EPA AQI records

Figures (results/figures and slides/assets)
    fig01_copd_map  fig02_copd_intervals  fig03_correlations
    fig04_copd_vs_smoking  fig05_copd_vs_social
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

import config as C
import plotstyle as P

INDICATORS = [m["col"] for m in C.MEASURES.values()]


def load() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_TABLE, dtype={"fips": str})


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def descriptives(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in INDICATORS:
        x = d[col]
        half = (d[f"{col}_ucl"] - d[f"{col}_lcl"]) / 2
        rows.append({
            "variable": col, "indicator": C.LABELS[col], "n": int(x.notna().sum()),
            "mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(), "q1": x.quantile(0.25),
            "median": x.median(), "q3": x.quantile(0.75), "max": x.max(),
            "cdc_ci_halfwidth_mean": half.mean(),
            "halfwidth_to_sd": half.mean() / x.std(ddof=1),
        })
    out = pd.DataFrame(rows)
    out.round(3).to_csv(C.TABLES / "eda_descriptives.csv", index=False)
    return out


def interval_overlap(d: pd.DataFrame) -> dict:
    """How well do CDC's own intervals separate counties on COPD?"""
    med = d["copd"].median()
    excludes = (d["copd_lcl"] > med) | (d["copd_ucl"] < med)
    lo, hi = d["copd_lcl"].to_numpy(), d["copd_ucl"].to_numpy()
    pairs = list(combinations(range(len(d)), 2))
    overlap = sum(lo[i] <= hi[j] and lo[j] <= hi[i] for i, j in pairs)
    return {
        "median_estimate": float(med),
        "n_interval_excludes_median": int(excludes.sum()),
        "n_interval_above_median": int((d["copd_lcl"] > med).sum()),
        "n_interval_below_median": int((d["copd_ucl"] < med).sum()),
        "n_pairs": len(pairs),
        "share_pairs_overlapping": overlap / len(pairs),
    }


def aqi_coverage(d: pd.DataFrame) -> pd.DataFrame:
    """Counties with and without AQI records: who is in the air-quality subset?"""
    rows = []
    for col in ["population"] + INDICATORS:
        a, b = d.loc[d["has_aqi"].eq(1), col], d.loc[d["has_aqi"].eq(0), col]
        pooled = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
        rows.append({
            "variable": col,
            "with_aqi_mean": a.mean(), "with_aqi_median": a.median(),
            "without_aqi_mean": b.mean(), "without_aqi_median": b.median(),
            "difference_in_means": a.mean() - b.mean(),
            "standardized_difference": (a.mean() - b.mean()) / pooled,
        })
    out = pd.DataFrame(rows)
    out.insert(1, "n_with_aqi", int(d["has_aqi"].sum()))
    out.insert(2, "n_without_aqi", int((d["has_aqi"] == 0).sum()))
    out.round(3).to_csv(C.TABLES / "eda_aqi_coverage.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_map(d: pd.DataFrame) -> None:
    """County choropleth. Outlines come from the Census boundary snapshot; colour is the COPD estimate."""
    geometry = {f["properties"]["GEOID"]: f["geometry"]
                for f in C.read_json(C.SNAPSHOT_BOUNDARIES)["features"]}
    norm = Normalize(vmin=d["copd"].min(), vmax=d["copd"].max())
    # label position (lon, lat) outside the state, joined to the county by a thin leader line
    labels = {"Wake": (-78.20, 37.05), "Buncombe": (-83.85, 36.50), "Mecklenburg": (-82.05, 34.35),
              "Robeson": (-80.45, 33.92), "New Hanover": (-76.65, 33.80)}

    def outline(geom: dict) -> MplPath:
        verts, codes = [], []
        for ring in C.polygon_rings(geom):
            verts.extend(ring)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
        return MplPath(verts, codes)

    def draw(fig, variant):
        ax = fig.add_subplot()
        for row in d.itertuples():
            # a thin surface-coloured edge separates neighbouring counties
            ax.add_patch(PathPatch(outline(geometry[row.fips]), facecolor=P.SEQ(norm(row.copd)),
                                   edgecolor=P.SURFACE, linewidth=0.9, zorder=2))
        ax.set_xlim(-84.75, -75.15)
        ax.set_ylim(33.45, 37.30)
        ax.set_aspect(1 / np.cos(np.radians(35.4)))
        ax.axis("off")
        for name, (lx, ly) in labels.items():
            r = d.loc[d["county"].eq(name)].iloc[0]
            ax.annotate(f"{name} {r['copd']:.1f}%", (r["lon"], r["lat"]), xytext=(lx, ly),
                        textcoords="data", ha="center", va="center", color=P.INK, zorder=6,
                        fontsize=10.5 if variant == "report" else 13.5,
                        arrowprops=dict(arrowstyle="-", color=P.INK, linewidth=0.9, shrinkA=3, shrinkB=0))
            ax.scatter([r["lon"]], [r["lat"]], s=14, color=P.INK, edgecolor=P.SURFACE, linewidth=0.8, zorder=7)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=P.SEQ), ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.02, shrink=0.36, anchor=(0.10, 1.0))
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        cb.set_label("Age-adjusted COPD prevalence among adults (%)" if variant == "report"
                     else "COPD prevalence (%)", color=P.INK2)

    P.render("fig01_copd_map", draw,
             title="Modeled COPD prevalence by county, North Carolina, 2023",
             subtitle="Counties shaded by the age-adjusted estimate; darker = higher prevalence.",
             source=P.SOURCE_MAP, size=(9.6, 5.0))


def fig_intervals(d: pd.DataFrame, overlap: dict) -> None:
    s = d.sort_values("copd").reset_index(drop=True)
    med = overlap["median_estimate"]
    apart = (s["copd_lcl"] > med) | (s["copd_ucl"] < med)

    def draw(fig, variant):
        ax = fig.add_subplot()
        x = np.arange(1, len(s) + 1)
        ax.vlines(x, s["copd_lcl"], s["copd_ucl"], color=P.AXIS, linewidth=1.3, zorder=2)
        ax.scatter(x[~apart], s.loc[~apart, "copd"], s=26, color=P.MUTED, zorder=3,
                   label=f"Interval includes the median county's estimate ({int((~apart).sum())})")
        ax.scatter(x[apart], s.loc[apart, "copd"], s=30, color=P.BLUE, zorder=4,
                   label=f"Interval lies entirely above or below it ({int(apart.sum())})")
        ax.axhline(med, color=P.INK2, linewidth=1.0, zorder=1)
        ax.annotate(f"median county estimate {med:.1f}%", (1, med), xytext=(2, 5),
                    textcoords="offset points", color=P.INK2, va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground=P.SURFACE)],
                    fontsize=10 if variant == "report" else 13)
        for name, dx, ha in ((s.loc[0, "county"], 8, "left"), (s.loc[len(s) - 1, "county"], -8, "right")):
            r = s.loc[s["county"].eq(name)].iloc[0]
            ax.annotate(f"{name} {r['copd']:.1f}%", (x[s["county"].eq(name)][0], r["copd"]),
                        xytext=(dx, 0), textcoords="offset points", ha=ha, va="center", color=P.INK,
                        path_effects=[pe.withStroke(linewidth=3, foreground=P.SURFACE)],
                        fontsize=10 if variant == "report" else 13)
        ax.set_xlim(-1, len(s) + 2)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("100 counties, ordered from lowest to highest estimate")
        ax.set_ylabel("COPD prevalence (%)")
        ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)

    P.render("fig02_copd_intervals", draw,
             title="County COPD estimates with the 95% confidence limits published by CDC",
             subtitle="Dot = age-adjusted estimate; vertical line = CDC lower to upper confidence limit.",
             source=P.SOURCE_PLACES, size=(9.6, 5.2))


def fig_correlations(corr: pd.DataFrame) -> None:
    names = [C.SHORT_LABELS[c] for c in corr.columns]
    k = len(names)
    mask = np.triu(np.ones((k, k), dtype=bool))            # hide the diagonal and upper triangle
    values = np.ma.masked_where(mask, corr.to_numpy())

    def draw(fig, variant):
        ax = fig.add_subplot()
        mesh = ax.pcolormesh(values, cmap=P.SEQ_FROM_ZERO, vmin=0, vmax=1,
                             edgecolors=P.SURFACE, linewidth=3)
        for i in range(k):
            for j in range(i):
                v = corr.iat[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                        color="#ffffff" if v >= 0.55 else P.INK,
                        fontsize=11 if variant == "report" else 14)
        ax.set_xlim(0, k - 1)
        ax.set_ylim(k, 1)
        ax.set_xticks(np.arange(k - 1) + 0.5, names[:-1], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(1, k) + 0.5, names[1:])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_aspect("equal")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03, shrink=0.7)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        cb.set_label("Pearson correlation across 100 counties", color=P.INK2)

    P.render("fig03_correlations", draw,
             title="Correlations among the seven county indicators",
             subtitle="Lower triangle of the correlation matrix; all seven are model-based PLACES estimates.",
             source=P.SOURCE_PLACES, size=(8.2, 6.2))


def fig_copd_vs_smoking(d: pd.DataFrame) -> None:
    slope, intercept = np.polyfit(d["smoking"], d["copd"], 1)
    r = d["smoking"].corr(d["copd"])

    def draw(fig, variant):
        ax = fig.add_subplot()
        ax.errorbar(d["smoking"], d["copd"],
                    xerr=[d["smoking"] - d["smoking_lcl"], d["smoking_ucl"] - d["smoking"]],
                    yerr=[d["copd"] - d["copd_lcl"], d["copd_ucl"] - d["copd"]],
                    fmt="none", ecolor=P.AXIS, elinewidth=0.7, alpha=0.75, zorder=1)
        xs = np.array([d["smoking"].min(), d["smoking"].max()])
        ax.plot(xs, intercept + slope * xs, color=P.INK2, linewidth=1.6, zorder=2)
        P.dots(ax, d["smoking"], d["copd"], size=46 if variant == "report" else 60)
        fs = 10 if variant == "report" else 13
        P.label_points(ax, d, "smoking", "copd", ["Robeson", "Wake"],
                       offsets={"Robeson": (-9, 4), "Wake": (9, -6)}, fontsize=fs)
        ax.annotate(f"least-squares line, r = {r:.2f}", (19.4, intercept + slope * 19.4),
                    xytext=(20.6, 6.1), textcoords="data", color=P.INK2, fontsize=fs, va="center",
                    arrowprops=dict(arrowstyle="-", color=P.INK2, linewidth=0.9, shrinkA=2, shrinkB=4))
        ax.set_xlabel("Current cigarette smoking, age-adjusted prevalence (%)")
        ax.set_ylabel("COPD, age-adjusted prevalence (%)")

    P.render("fig04_copd_vs_smoking", draw,
             title="COPD and smoking estimates by county, with CDC confidence limits",
             subtitle="Dot = county estimate; grey lines = CDC 95% confidence limits in each direction.",
             source=P.SOURCE_PLACES, size=(9.6, 5.6))


def fig_copd_vs_social(d: pd.DataFrame) -> None:
    def draw(fig, variant):
        axes = fig.subplots(1, 3, sharey=True)
        for ax, col in zip(axes, C.SOCIAL):
            slope, intercept = np.polyfit(d[col], d["copd"], 1)
            xs = np.array([d[col].min(), d[col].max()])
            ax.plot(xs, intercept + slope * xs, color=P.INK2, linewidth=1.6, zorder=2)
            P.dots(ax, d[col], d["copd"], size=34 if variant == "report" else 44)
            ax.set_title(f"{C.SHORT_LABELS[col]}  (r = {d[col].corr(d['copd']):.2f})",
                         fontweight="normal", color=P.INK)
            ax.set_xlabel(f"{C.LABELS[col]} (%)" if variant == "report" else "Share of adults (%)")
        axes[0].set_ylabel("COPD prevalence (%)")

    P.render("fig05_copd_vs_social", draw,
             title="COPD against each social indicator, before any adjustment",
             subtitle="Age-adjusted prevalence estimates for 100 counties; line = least-squares fit.",
             source=P.SOURCE_PLACES, size=(11.0, 4.6))


# --------------------------------------------------------------------------
def main() -> None:
    C.ensure_dirs()
    d = load()

    desc = descriptives(d)
    corr = d[INDICATORS].corr()
    corr.round(3).to_csv(C.TABLES / "eda_correlations.csv")
    d[INDICATORS].corr(method="spearman").round(3).to_csv(C.TABLES / "eda_correlations_spearman.csv")

    cols = ["county", "copd", "copd_lcl", "copd_ucl", "smoking", "population"]
    ranked = d.sort_values("copd", ascending=False)
    extremes = pd.concat([ranked.head(5).assign(group="highest five"),
                          ranked.tail(5).assign(group="lowest five")])[["group"] + cols]
    extremes.to_csv(C.TABLES / "eda_copd_extremes.csv", index=False)

    overlap = interval_overlap(d)
    coverage = aqi_coverage(d)

    print(desc.round(2).to_string(index=False))
    print("\nCOPD interval overlap:", {k: round(v, 3) if isinstance(v, float) else v for k, v in overlap.items()})
    print("\n", coverage.round(2).to_string(index=False))

    fig_map(d)
    fig_intervals(d, overlap)
    fig_correlations(corr)
    fig_copd_vs_smoking(d)
    fig_copd_vs_social(d)

    off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    pop_aqi = d.loc[d["has_aqi"].eq(1), "population"].sum()
    C.update_results("eda", {
        "descriptives": desc.set_index("variable").drop(columns="indicator").round(3).to_dict(orient="index"),
        "corr_copd": corr["copd"].drop("copd").round(3).to_dict(),
        "corr_min": float(off_diag.min()),
        "corr_max": float(off_diag.max()),
        "corr_transport_housing": float(corr.loc["transport", "housing"]),
        "corr_inactivity_transport": float(corr.loc["inactivity", "transport"]),
        "copd_highest": ranked.iloc[0][cols].to_dict(),
        "copd_lowest": ranked.iloc[-1][cols].to_dict(),
        "copd_interval_overlap": overlap,
        "aqi_population_share": float(pop_aqi / d["population"].sum()),
        "aqi_coverage": coverage.set_index("variable").round(3).to_dict(orient="index"),
    })


if __name__ == "__main__":
    main()
