"""Step 03 - Question 1: social indicators and COPD, after smoking and inactivity.

Models (ordinary least squares, one row per county, n = 100)
    baseline   outcome ~ smoking + inactivity
    expanded   outcome ~ smoking + inactivity + transport + uninsured + housing
    outcome is COPD (primary) or current asthma (secondary)

Uncertainty for every slope
    classical t interval, heteroskedasticity-robust HC3 interval, and a county
    bootstrap (2,000 resamples, percentile interval, seed fixed in config.py)

Checks before trusting the slopes
    collinearity, heteroskedasticity, influential counties, leave-one-county-out
    slopes, spatial autocorrelation of residuals, leave-one-out prediction error,
    and a plausible-value simulation drawn from CDC's own confidence limits

Outputs: results/tables/q1_*.csv, figures fig06-fig08, section "q1" of results.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as C
import plotstyle as P

N_PERM = 9999          # permutations for Moran's I
N_SIM = 2000           # plausible-value draws
SPECS = {"baseline": C.BEHAVIORAL, "expanded": C.ALL_PREDICTORS}


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def fit(d: pd.DataFrame, y: str, cols: list[str], robust: bool = False):
    X = sm.add_constant(d[cols])
    model = sm.OLS(d[y], X)
    return model.fit(cov_type="HC3", use_t=True) if robust else model.fit()


def bootstrap_slopes(d: pd.DataFrame, y: str, cols: list[str], idx: np.ndarray) -> np.ndarray:
    """Refit the model on resampled counties. `idx` has one row of county positions per resample."""
    X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
    yv = d[y].to_numpy()
    out = np.empty((idx.shape[0], X.shape[1]))
    for b, rows in enumerate(idx):
        out[b] = np.linalg.lstsq(X[rows], yv[rows], rcond=None)[0]
    return out


def coefficient_table(d, y, name, cols, boot) -> pd.DataFrame:
    """One row per term: estimate and the three kinds of 95% interval."""
    res, rob = fit(d, y, cols), fit(d, y, cols, robust=True)
    ci, ci_r = res.conf_int(), rob.conf_int()
    rows = []
    for k, term in enumerate(["const"] + cols):
        rows.append({
            "outcome": y, "model": name, "term": term,
            "estimate": res.params[term],
            "estimate_per_sd": res.params[term] * d[term].std(ddof=1) if term != "const" else np.nan,
            "se": res.bse[term], "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "p": res.pvalues[term],
            "se_hc3": rob.bse[term], "hc3_low": ci_r.loc[term, 0], "hc3_high": ci_r.loc[term, 1],
            "p_hc3": rob.pvalues[term],
            "boot_low": np.percentile(boot[:, k], 2.5), "boot_high": np.percentile(boot[:, k], 97.5),
            "boot_share_positive": float((boot[:, k] > 0).mean()),
        })
    return pd.DataFrame(rows)


def loo_errors(res) -> np.ndarray:
    """Leave-one-county-out prediction errors from the hat matrix (no refitting needed)."""
    h = res.get_influence().hat_matrix_diag
    return res.resid.to_numpy() / (1 - h)


def fit_summary(d, y) -> tuple[pd.DataFrame, dict]:
    rows, fits = [], {}
    for name, cols in SPECS.items():
        res = fit(d, y, cols)
        fits[name] = res
        rows.append({
            "outcome": y, "model": name, "n": int(res.nobs), "predictors": len(cols),
            "r2": res.rsquared, "adj_r2": res.rsquared_adj, "residual_sd": np.sqrt(res.mse_resid),
            "aic": res.aic, "bic": res.bic,
            "loo_rmse": float(np.sqrt(np.mean(loo_errors(res) ** 2))),
        })
    f_stat, f_p, _ = fits["expanded"].compare_f_test(fits["baseline"])
    wald = fit(d, y, C.ALL_PREDICTORS, robust=True).wald_test(
        ", ".join(f"{c} = 0" for c in C.SOCIAL), use_f=True, scalar=True)
    comparison = {
        "delta_r2": fits["expanded"].rsquared - fits["baseline"].rsquared,
        "f_stat": float(f_stat), "f_p": float(f_p),
        "wald_hc3_f": float(wald.statistic), "wald_hc3_p": float(wald.pvalue),
    }
    return pd.DataFrame(rows), comparison


def specification_table(d, y) -> pd.DataFrame:
    """Each social indicator under three sets of adjustment variables."""
    rows = []
    for term in C.SOCIAL:
        for label, cols in (("alone", [term]),
                            ("with smoking and inactivity", [term] + C.BEHAVIORAL),
                            ("with all five predictors", C.ALL_PREDICTORS)):
            res, rob = fit(d, y, cols), fit(d, y, cols, robust=True)
            rows.append({
                "outcome": y, "term": term, "specification": label,
                "estimate": res.params[term],
                "ci_low": res.conf_int().loc[term, 0], "ci_high": res.conf_int().loc[term, 1],
                "hc3_low": rob.conf_int().loc[term, 0], "hc3_high": rob.conf_int().loc[term, 1],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def collinearity(d) -> tuple[pd.DataFrame, dict]:
    rows = []
    for name, cols in SPECS.items():
        X = sm.add_constant(d[cols]).to_numpy()
        for k, col in enumerate(cols, start=1):
            rows.append({"model": name, "term": col, "vif": variance_inflation_factor(X, k)})
    z = (d[C.ALL_PREDICTORS] - d[C.ALL_PREDICTORS].mean()) / d[C.ALL_PREDICTORS].std(ddof=1)
    eig = np.linalg.eigvalsh(np.corrcoef(z.to_numpy(), rowvar=False))[::-1]
    facts = {"condition_number_standardized": float(np.sqrt(eig[0] / eig[-1])),
             "pc1_share_of_variance": float(eig[0] / eig.sum()),
             "pc1_pc2_share_of_variance": float(eig[:2].sum() / eig.sum())}
    return pd.DataFrame(rows), facts


def influence_tables(d, res) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    infl = res.get_influence()
    cooks = pd.Series(infl.cooks_distance[0], index=d.index)
    dfbeta = pd.DataFrame(infl.dfbeta, columns=res.params.index, index=d.index)
    top = (pd.DataFrame({
        "county": d["county"], "population": d["population"], "cooks_d": cooks,
        "leverage": infl.hat_matrix_diag, "studentized_residual": infl.resid_studentized_external,
        **{f"slope_change_if_dropped_{c}": -dfbeta[c] for c in C.SOCIAL},
    }).sort_values("cooks_d", ascending=False).head(10))

    rows = []
    for term in C.SOCIAL:
        loo = res.params[term] - dfbeta[term]            # slope when each county is left out
        rows.append({
            "term": term, "estimate_all_counties": res.params[term],
            "loo_min": loo.min(), "loo_min_dropped": d.loc[loo.idxmin(), "county"],
            "loo_max": loo.max(), "loo_max_dropped": d.loc[loo.idxmax(), "county"],
            "n_sign_changes": int((np.sign(loo) != np.sign(res.params[term])).sum()),
        })
    return top, pd.DataFrame(rows), cooks


def reexpressed_pair(d, boot: np.ndarray) -> pd.DataFrame:
    """Same expanded model, with transport and housing replaced by their mean and difference.

    b_t * transport + b_h * housing = (b_t + b_h) * mean + ((b_t - b_h) / 2) * difference,
    so the fit is identical; only the question asked of the two collinear indicators changes.
    """
    data = d.assign(pair_mean=(d["transport"] + d["housing"]) / 2,
                    pair_difference=d["transport"] - d["housing"])
    cols = C.BEHAVIORAL + ["uninsured", "pair_mean", "pair_difference"]
    res, rob = fit(data, C.OUTCOME, cols), fit(data, C.OUTCOME, cols, robust=True)
    k_t, k_h = 1 + C.ALL_PREDICTORS.index("transport"), 1 + C.ALL_PREDICTORS.index("housing")
    draws = {"pair_mean": boot[:, k_t] + boot[:, k_h], "pair_difference": (boot[:, k_t] - boot[:, k_h]) / 2}
    labels = {"pair_mean": "mean of transport and housing (per 1 point)",
              "pair_difference": "transport minus housing (per 1 point)"}
    return pd.DataFrame([{
        "term": labels[t], "estimate": res.params[t],
        "ci_low": res.conf_int().loc[t, 0], "ci_high": res.conf_int().loc[t, 1],
        "hc3_low": rob.conf_int().loc[t, 0], "hc3_high": rob.conf_int().loc[t, 1],
        "boot_low": np.percentile(draws[t], 2.5), "boot_high": np.percentile(draws[t], 97.5),
        "sd_across_counties": data[t].std(ddof=1),
    } for t in labels])


def bootstrap_by_county(d, boot: np.ndarray, idx: np.ndarray, county: str) -> pd.DataFrame:
    """Bootstrap social slopes grouped by how often one county was drawn into the resample."""
    pos = int(np.flatnonzero(d["county"].eq(county).to_numpy())[0])
    times = np.minimum((idx == pos).sum(axis=1), 2)
    rows = []
    for t, label in ((0, "not drawn"), (1, "drawn once"), (2, "drawn twice or more")):
        row = {"county": county, "times_in_resample": label, "share_of_resamples": float((times == t).mean())}
        for term in C.SOCIAL:
            row[f"mean_slope_{term}"] = boot[times == t, 1 + C.ALL_PREDICTORS.index(term)].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_table(d, cooks) -> pd.DataFrame:
    """Refits that probe how much the social slopes depend on a few counties or on collinearity."""
    order = cooks.sort_values(ascending=False).index
    variants = {
        "all 100 counties": d,
        f"without {d.loc[order[0], 'county']} (largest Cook's distance)": d.drop(order[:1]),
        "without the three most influential counties": d.drop(order[:3]),
    }
    rows = []
    for label, data in variants.items():
        rob = fit(data, C.OUTCOME, C.ALL_PREDICTORS, robust=True)
        for term in C.SOCIAL:
            rows.append({"analysis": label, "n": len(data), "term": term, "estimate": rob.params[term],
                         "hc3_low": rob.conf_int().loc[term, 0], "hc3_high": rob.conf_int().loc[term, 1]})

    # Composite index (added after the collinearity diagnosis; not in the proposal):
    # average of the three standardized social indicators, rescaled to SD = 1.
    z = (d[C.SOCIAL] - d[C.SOCIAL].mean()) / d[C.SOCIAL].std(ddof=1)
    index = z.mean(axis=1)
    data = d.assign(social_index=(index - index.mean()) / index.std(ddof=1))
    for label, cols in (("composite social index, alone", ["social_index"]),
                        ("composite social index, with smoking and inactivity", C.BEHAVIORAL + ["social_index"])):
        rob = fit(data, C.OUTCOME, cols, robust=True)
        rows.append({"analysis": label, "n": len(data), "term": "social_index (per 1 SD)",
                     "estimate": rob.params["social_index"],
                     "hc3_low": rob.conf_int().loc["social_index", 0],
                     "hc3_high": rob.conf_int().loc["social_index", 1]})
    return pd.DataFrame(rows)


def knn_weights(lon: np.ndarray, lat: np.ndarray, k: int = 5) -> np.ndarray:
    """Row-standardized weights: each county's k nearest centroids (great-circle distance)."""
    lam, phi = np.radians(lon), np.radians(lat)
    dphi, dlam = phi[:, None] - phi[None, :], lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros_like(dist)
    nearest = np.argsort(dist, axis=1)[:, :k]
    W[np.arange(len(lon))[:, None], nearest] = 1.0 / k
    return W


def morans_i(values: np.ndarray, W: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Moran's I and a one-sided permutation p-value (positive spatial autocorrelation)."""
    def stat(v):
        v = v - v.mean()
        return (len(v) / W.sum()) * (v @ W @ v) / (v @ v)
    observed = stat(values)
    perms = np.array([stat(rng.permutation(values)) for _ in range(N_PERM)])
    return float(observed), float((1 + (perms >= observed).sum()) / (1 + N_PERM))


def plausible_values(d, rng) -> tuple[pd.DataFrame, dict]:
    """Refit the models on draws from CDC's published limits.

    Each draw replaces every county estimate by estimate + Normal(0, se), with
    se = (upper - lower) / (2 * 1.96), independently across counties and measures.
    Independence is an assumption made for this stress test, not a property of PLACES.
    """
    cols = [C.OUTCOME] + C.ALL_PREDICTORS
    est = d[cols].to_numpy()
    se = np.column_stack([(d[f"{c}_ucl"] - d[f"{c}_lcl"]) / (2 * 1.96) for c in cols])
    ones = np.ones(len(d))
    slopes, r2_base, r2_exp, r_smoke = [], [], [], []
    for _ in range(N_SIM):
        draw = est + rng.normal(0.0, se)
        y = draw[:, 0]
        Xe = np.column_stack([ones, draw[:, 1:]])
        Xb = Xe[:, :3]
        be = np.linalg.lstsq(Xe, y, rcond=None)[0]
        bb = np.linalg.lstsq(Xb, y, rcond=None)[0]
        tss = ((y - y.mean()) ** 2).sum()
        slopes.append(be[1:])
        r2_exp.append(1 - ((y - Xe @ be) ** 2).sum() / tss)
        r2_base.append(1 - ((y - Xb @ bb) ** 2).sum() / tss)
        r_smoke.append(np.corrcoef(y, draw[:, 1])[0, 1])
    slopes = np.array(slopes)
    observed = fit(d, C.OUTCOME, C.ALL_PREDICTORS).params
    table = pd.DataFrame({
        "term": C.ALL_PREDICTORS,
        "estimate_published_values": [observed[c] for c in C.ALL_PREDICTORS],
        "sim_median": np.median(slopes, axis=0),
        "sim_low": np.percentile(slopes, 2.5, axis=0),
        "sim_high": np.percentile(slopes, 97.5, axis=0),
        "sim_share_positive": (slopes > 0).mean(axis=0),
    })
    facts = {
        "n_draws": N_SIM,
        "r2_baseline_median": float(np.median(r2_base)), "r2_expanded_median": float(np.median(r2_exp)),
        "r2_gain_median": float(np.median(np.array(r2_exp) - np.array(r2_base))),
        "corr_copd_smoking_median": float(np.median(r_smoke)),
        "corr_copd_smoking_low": float(np.percentile(r_smoke, 2.5)),
        "corr_copd_smoking_high": float(np.percentile(r_smoke, 97.5)),
    }
    return table, facts


def slice_table(d, fits) -> pd.DataFrame:
    """Leave-one-out prediction error by county size, by longitude and by AQI coverage.

    Error = observed minus predicted, so a positive mean says the model predicts too low.
    """
    groups = {
        "population third": pd.qcut(d["population"], 3, labels=["smallest third", "middle third", "largest third"]),
        "longitude third": pd.qcut(d["lon"], 3, labels=["western third", "central third", "eastern third"]),
        "AQI records": d["has_aqi"].map({1: "with AQI records", 0: "without AQI records"}),
    }
    rows = []
    for slice_name, labels in groups.items():
        for name, res in fits.items():
            err = pd.Series(loo_errors(res), index=d.index)
            for level, e in err.groupby(labels, observed=True):
                rows.append({"slice": slice_name, "group": level, "model": name, "n": len(e),
                             "mean_loo_error": e.mean(), "mean_abs_loo_error": e.abs().mean(),
                             "max_abs_loo_error": e.abs().max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_specifications(spec: pd.DataFrame) -> None:
    rows = ["alone", "with smoking and inactivity", "with all five predictors"]
    row_labels = ["Indicator alone", "+ smoking, inactivity", "+ other two social indicators"]
    s = spec[spec["outcome"].eq(C.OUTCOME)]

    def draw(fig, variant):
        axes = fig.subplots(1, 3, sharex=True, sharey=True)
        fs = 10.5 if variant == "report" else 13.5
        for ax, term in zip(axes, C.SOCIAL):
            t = s[s["term"].eq(term)].set_index("specification").loc[rows]
            y = np.arange(len(rows))[::-1]
            P.zero_line(ax)
            ax.hlines(y, t["hc3_low"], t["hc3_high"], color=P.BLUE, linewidth=2.2, zorder=3)
            P.dots(ax, t["estimate"], y, size=80 if variant == "report" else 110, zorder=4)
            for yi, (est, hi) in zip(y, zip(t["estimate"], t["hc3_high"])):
                ax.annotate(f"{est:+.2f}", (est, yi), xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom", color=P.INK, fontsize=fs)
            ax.set_title(C.LABELS[term] if variant == "report" else C.SHORT_LABELS[term], fontweight="normal")
            ax.set_yticks(y, row_labels)
            ax.set_ylim(-0.6, len(rows) - 0.35)
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
        axes[1].set_xlabel("Difference in COPD prevalence (percentage points) per 1-point difference in the indicator"
                           if variant == "report" else "COPD difference (points) per 1 point of the indicator")

    P.render("fig06_q1_specifications", draw,
             title="Social-indicator slopes for COPD under three sets of adjustment variables",
             subtitle="Dot = least-squares slope; bar = heteroskedasticity-robust (HC3) 95% interval; n = 100 counties.",
             source=P.SOURCE_PLACES, size=(11.6, 4.3))


def fig_intervals(coefs: pd.DataFrame) -> None:
    t = coefs[coefs["outcome"].eq(C.OUTCOME) & coefs["model"].eq("expanded") & coefs["term"].ne("const")]
    t = t.set_index("term").loc[C.ALL_PREDICTORS]
    kinds = [("Classical", "ci_low", "ci_high", P.BLUE, 0.24),
             ("Robust (HC3)", "hc3_low", "hc3_high", P.ORANGE, 0.0),
             ("County bootstrap", "boot_low", "boot_high", P.AQUA, -0.24)]

    def draw(fig, variant):
        ax = fig.add_subplot()
        y = np.arange(len(t))[::-1]
        P.zero_line(ax)
        for label, lo, hi, color, off in kinds:
            ax.hlines(y + off, t[lo], t[hi], color=color, linewidth=3.2, label=label, zorder=3,
                      capstyle="round")
        ax.scatter(t["estimate"], y, s=70 if variant == "report" else 95, color=P.INK,
                   edgecolor=P.SURFACE, linewidth=1.2, zorder=5, label="Slope estimate")
        ax.set_yticks(y, [C.LABELS[c] for c in t.index])
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Difference in COPD prevalence (percentage points) per 1-point difference in the predictor"
                      if variant == "report" else "COPD difference (points) per 1 point of the predictor")
        ax.legend(loc="upper left", title="95% interval", title_fontsize=ax.xaxis.label.get_size() - 1,
                  alignment="left")

    P.render("fig07_q1_intervals", draw,
             title="Expanded COPD model: three 95% intervals for each slope",
             subtitle="Same least-squares estimates; intervals differ in how they treat unequal error variance and influential counties.",
             source=P.SOURCE_PLACES, size=(10.4, 4.9))


def fig_diagnostics(d, res, cooks) -> None:
    infl = res.get_influence()
    fitted, resid = res.fittedvalues, res.resid
    stud = np.sort(infl.resid_studentized_external)
    theo = stats.t.ppf((np.arange(1, len(stud) + 1) - 0.5) / len(stud), df=res.df_resid - 1)
    top = cooks.sort_values(ascending=False).head(3)

    def draw(fig, variant):
        ax1, ax2, ax3 = fig.subplots(1, 3)
        fs = 10 if variant == "report" else 12.5
        size = 34 if variant == "report" else 44

        P.zero_line(ax1, "y")
        P.dots(ax1, fitted, resid, size=size)
        for i in top.index:
            ax1.annotate(d.loc[i, "county"], (fitted[i], resid[i]), xytext=(6, 5),
                         textcoords="offset points", fontsize=fs, color=P.INK2)
        short = variant == "slides"
        ax1.set_title("Residuals vs fitted" if short else "Residuals against fitted values", fontweight="normal")
        ax1.set_xlabel("Fitted COPD prevalence (%)")
        ax1.set_ylabel("Residual (percentage points)")

        lim = max(abs(stud).max(), abs(theo).max()) * 1.05
        ax2.plot([-lim, lim], [-lim, lim], color=P.AXIS, linewidth=1.2, zorder=1)
        P.dots(ax2, theo, stud, size=size)
        ax2.set_title("Normal quantile plot" if short else "Studentized residuals: normal quantile plot",
                      fontweight="normal")
        ax2.set_xlabel("Theoretical quantile")
        ax2.set_ylabel("Studentized residual")

        ranked = cooks.sort_values(ascending=False).head(8)[::-1]
        pos = np.arange(len(ranked))
        ax3.hlines(pos, 0, ranked.to_numpy(), color=P.BLUE, linewidth=2.2, zorder=3)
        P.dots(ax3, ranked.to_numpy(), pos, size=size + 14, zorder=4)
        ax3.axvline(4 / len(cooks), color=P.INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax3.annotate("4 / n", (4 / len(cooks), -0.55), xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=fs, color=P.INK2)
        ax3.set_yticks(pos, d.loc[ranked.index, "county"])
        ax3.set_ylim(-0.9, len(ranked) - 0.5)
        ax3.set_xlim(left=0)
        ax3.set_title("Cook's distance, top 8" if short else "Cook's distance, eight largest", fontweight="normal")
        ax3.set_xlabel("Cook's distance")
        ax3.grid(axis="y", visible=False)
        ax3.spines["left"].set_visible(False)

    P.render("fig08_q1_diagnostics", draw,
             title="Diagnostics for the expanded COPD model",
             subtitle="Ordinary least squares with five predictors, n = 100 counties; labelled counties have the largest Cook's distance.",
             source=P.SOURCE_PLACES, size=(12.4, 4.4))


# --------------------------------------------------------------------------
def main() -> None:
    C.ensure_dirs()
    d = pd.read_csv(C.PROCESSED_TABLE, dtype={"fips": str})
    rng = np.random.default_rng(C.SEED)
    boot_idx = rng.integers(0, len(d), size=(C.N_BOOT, len(d)))   # shared by every model

    boot = {(y, name): bootstrap_slopes(d, y, cols, boot_idx)
            for y in (C.OUTCOME, C.SECONDARY_OUTCOME) for name, cols in SPECS.items()}
    coefs = pd.concat([coefficient_table(d, y, name, cols, boot[(y, name)])
                       for y in (C.OUTCOME, C.SECONDARY_OUTCOME) for name, cols in SPECS.items()],
                      ignore_index=True)
    coefs.round(5).to_csv(C.TABLES / "q1_coefficients.csv", index=False)

    fit_rows, comparisons = [], {}
    for y in (C.OUTCOME, C.SECONDARY_OUTCOME):
        table, comparisons[y] = fit_summary(d, y)
        fit_rows.append(table)
    fits_table = pd.concat(fit_rows, ignore_index=True)
    fits_table.round(5).to_csv(C.TABLES / "q1_model_fit.csv", index=False)
    pd.DataFrame(comparisons).T.rename_axis("outcome").reset_index().to_csv(
        C.TABLES / "q1_model_comparison.csv", index=False)

    spec = pd.concat([specification_table(d, y) for y in (C.OUTCOME, C.SECONDARY_OUTCOME)], ignore_index=True)
    spec.round(5).to_csv(C.TABLES / "q1_specifications.csv", index=False)

    vif, collin = collinearity(d)
    vif.round(3).to_csv(C.TABLES / "q1_collinearity.csv", index=False)

    fits = {name: fit(d, C.OUTCOME, cols) for name, cols in SPECS.items()}
    expanded = fits["expanded"]
    top, loo, cooks = influence_tables(d, expanded)
    top.round(5).to_csv(C.TABLES / "q1_influence.csv", index=False)
    loo.round(5).to_csv(C.TABLES / "q1_leave_one_out.csv", index=False)
    sens = sensitivity_table(d, cooks)
    sens.round(5).to_csv(C.TABLES / "q1_sensitivity.csv", index=False)
    pair = reexpressed_pair(d, boot[(C.OUTCOME, "expanded")])
    pair.round(5).to_csv(C.TABLES / "q1_transport_housing_pair.csv", index=False)
    most_influential = d.loc[cooks.idxmax(), "county"]
    by_county = bootstrap_by_county(d, boot[(C.OUTCOME, "expanded")], boot_idx, most_influential)
    by_county.round(5).to_csv(C.TABLES / "q1_bootstrap_by_county.csv", index=False)

    bp_stat, bp_p, _, _ = het_breuschpagan(expanded.resid, expanded.model.exog)
    W = knn_weights(d["lon"].to_numpy(), d["lat"].to_numpy(), k=5)
    spatial = []
    for label, values in (("COPD estimate", d[C.OUTCOME].to_numpy()),
                          ("baseline model residuals", fits["baseline"].resid.to_numpy()),
                          ("expanded model residuals", expanded.resid.to_numpy())):
        i, p = morans_i(values, W, rng)
        spatial.append({"series": label, "morans_i": i, "p_permutation_one_sided": p,
                        "weights": "5 nearest county centroids", "permutations": N_PERM})
    spatial = pd.DataFrame(spatial)
    spatial.round(5).to_csv(C.TABLES / "q1_spatial.csv", index=False)

    sim, sim_facts = plausible_values(d, rng)
    sim.round(5).to_csv(C.TABLES / "q1_plausible_values.csv", index=False)
    slices = slice_table(d, fits)
    slices.round(5).to_csv(C.TABLES / "q1_slices.csv", index=False)

    fig_specifications(spec)
    fig_intervals(coefs)
    fig_diagnostics(d, expanded, cooks)

    # ---- console summary and key numbers ----------------------------------
    pd.set_option("display.width", 220)
    show = ["outcome", "model", "term", "estimate", "ci_low", "ci_high", "hc3_low", "hc3_high", "boot_low", "boot_high"]
    print(coefs[show].round(3).to_string(index=False))
    print("\n", fits_table.round(4).to_string(index=False))
    print("\n", pd.DataFrame(comparisons).T.round(6).to_string())
    print("\n", spec.round(3).to_string(index=False))
    print("\n", vif.round(1).to_string(index=False), "\n", {k: round(v, 3) for k, v in collin.items()})
    print("\n", top.round(3).to_string(index=False))
    print("\n", loo.round(3).to_string(index=False))
    print("\n", sens.round(3).to_string(index=False))
    print("\n", pair.round(3).to_string(index=False))
    print("\n", by_county.round(3).to_string(index=False))
    print("\nBreusch-Pagan:", round(bp_stat, 2), "p =", round(bp_p, 5))
    print("\n", spatial.round(4).to_string(index=False))
    print("\n", sim.round(3).to_string(index=False), "\n", {k: round(v, 3) for k, v in sim_facts.items()})
    print("\n", slices.round(3).to_string(index=False))

    def records(frame, **filters):
        for k, v in filters.items():
            frame = frame[frame[k].eq(v)]
        return frame.drop(columns=list(filters)).set_index("term").round(5).to_dict(orient="index")

    C.update_results("q1", {
        "n": len(d), "n_boot": C.N_BOOT, "seed": C.SEED,
        "coefficients": {y: {m: records(coefs, outcome=y, model=m) for m in SPECS}
                         for y in (C.OUTCOME, C.SECONDARY_OUTCOME)},
        "fit": {y: fits_table[fits_table["outcome"].eq(y)].drop(columns="outcome").set_index("model")
                .round(5).to_dict(orient="index") for y in (C.OUTCOME, C.SECONDARY_OUTCOME)},
        "comparison": comparisons,
        "specifications": {y: {t: spec[spec["outcome"].eq(y) & spec["term"].eq(t)]
                               .drop(columns=["outcome", "term"]).set_index("specification")
                               .round(5).to_dict(orient="index") for t in C.SOCIAL}
                           for y in (C.OUTCOME, C.SECONDARY_OUTCOME)},
        "vif": {m: vif[vif["model"].eq(m)].set_index("term")["vif"].round(2).to_dict() for m in SPECS},
        "collinearity": collin,
        "smoking_only_r2": float(fit(d, C.OUTCOME, ["smoking"]).rsquared),
        "breusch_pagan": {"statistic": float(bp_stat), "p": float(bp_p)},
        "cooks_threshold": 4 / len(d),
        "n_cooks_above_threshold": int((cooks > 4 / len(d)).sum()),
        "influence_top": top.head(5).round(4).to_dict(orient="records"),
        "leave_one_out": loo.set_index("term").round(5).to_dict(orient="index"),
        "sensitivity": sens.round(5).to_dict(orient="records"),
        "transport_housing_pair": pair.round(5).to_dict(orient="records"),
        "most_influential_county": most_influential,
        "bootstrap_by_county": by_county.round(5).to_dict(orient="records"),
        "spatial": spatial.set_index("series")[["morans_i", "p_permutation_one_sided"]].round(5).to_dict(orient="index"),
        "plausible_values": {"slopes": sim.set_index("term").round(5).to_dict(orient="index"), **sim_facts},
        "slices": slices.round(4).to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
