"""Step 04 - Question 2: median AQI and COPD among counties with 2023 AQI records.

Only counties with an EPA annual AQI summary enter this step (n = 35). Missing AQI
values are never imputed.

Models (ordinary least squares on the same counties)
    behaviors            copd ~ smoking + inactivity
    + median AQI         copd ~ smoking + inactivity + aqi_median      (main comparison)
    + days above 100     copd ~ smoking + inactivity + aqi_pct_gt100   (separate sensitivity measure)
    coverage restriction the first two models among counties with >= 300 monitored days

Outputs: results/tables/q2_*.csv, figure fig09, section "q2" of results.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C
import plotstyle as P


def fit(d, cols, robust=False):
    model = sm.OLS(d[C.OUTCOME], sm.add_constant(d[cols]))
    return model.fit(cov_type="HC3", use_t=True) if robust else model.fit()


def bootstrap(d, cols, rng) -> np.ndarray:
    X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
    y = d[C.OUTCOME].to_numpy()
    idx = rng.integers(0, len(d), size=(C.N_BOOT, len(d)))
    return np.array([np.linalg.lstsq(X[r], y[r], rcond=None)[0] for r in idx])


def coefficient_rows(d, label, cols, rng) -> list[dict]:
    res, rob, boot = fit(d, cols), fit(d, cols, robust=True), bootstrap(d, cols, rng)
    h = res.get_influence().hat_matrix_diag
    rows = []
    for k, term in enumerate(["const"] + cols):
        rows.append({
            "model": label, "n": len(d), "term": term, "estimate": res.params[term],
            "ci_low": res.conf_int().loc[term, 0], "ci_high": res.conf_int().loc[term, 1], "p": res.pvalues[term],
            "hc3_low": rob.conf_int().loc[term, 0], "hc3_high": rob.conf_int().loc[term, 1],
            "boot_low": np.percentile(boot[:, k], 2.5), "boot_high": np.percentile(boot[:, k], 97.5),
            "r2": res.rsquared, "adj_r2": res.rsquared_adj,
            "loo_rmse": float(np.sqrt(np.mean((res.resid / (1 - h)) ** 2))),
        })
    return rows


def monitoring_profile(a: pd.DataFrame) -> pd.DataFrame:
    """Which pollutants set the daily AQI in each county (main-pollutant day counts)."""
    def profile(r):
        main = [name for name, col in (("ozone", "aqi_days_ozone"), ("PM2.5", "aqi_days_pm25"),
                                       ("PM10", "aqi_days_pm10"), ("NO2", "aqi_days_no2"),
                                       ("CO", "aqi_days_co")) if r[col] > 0]
        if len(main) == 1:
            return f"{main[0]} only"
        if "ozone" in main and "PM2.5" in main:
            return "ozone and PM2.5 (some also PM10 or NO2)"
        return " and ".join(main)
    a = a.assign(profile=a.apply(profile, axis=1))
    out = (a.groupby("profile")
           .agg(n_counties=("county", "size"), median_days_with_aqi=("aqi_days", "median"),
                min_days=("aqi_days", "min"), max_days=("aqi_days", "max"),
                median_of_median_aqi=("aqi_median", "median"), mean_copd=("copd", "mean"))
           .reset_index())
    return out, a["profile"]


def fig_aqi(a: pd.DataFrame, slope_raw: float, slope_adj: float) -> None:
    behaviors = sm.add_constant(a[C.BEHAVIORAL])
    res_y = sm.OLS(a[C.OUTCOME], behaviors).fit().resid       # COPD not explained by behaviors
    res_x = sm.OLS(a["aqi_median"], behaviors).fit().resid    # AQI not explained by behaviors
    full = a["aqi_ge300_days"].eq(1)
    groups = [(full, P.BLUE, f"{C.AQI_MIN_DAYS}+ monitored days ({int(full.sum())})"),
              (~full, P.ORANGE, f"Under {C.AQI_MIN_DAYS} days ({int((~full).sum())})")]

    def draw(fig, variant):
        ax1, ax2 = fig.subplots(1, 2)
        fs = 10 if variant == "report" else 12.5
        size = 60 if variant == "report" else 80

        xs = np.array([a["aqi_median"].min(), a["aqi_median"].max()])
        b0 = a[C.OUTCOME].mean() - slope_raw * a["aqi_median"].mean()
        ax1.plot(xs, b0 + slope_raw * xs, color=P.INK2, linewidth=1.6, zorder=2)
        for mask, color, label in groups:
            P.dots(ax1, a.loc[mask, "aqi_median"], a.loc[mask, C.OUTCOME], color=color, size=size, label=label)
        P.label_points(ax1, a, "aqi_median", C.OUTCOME, ["Hyde", "Forsyth", "Wake"],
                       offsets={"Hyde": (4, 9), "Forsyth": (0, 9, "center"), "Wake": (-9, 0)},
                       fontsize=fs)
        ax1.set_title("Before adjustment", fontweight="normal")
        ax1.set_xlabel("Median daily AQI, 2023")
        ax1.set_ylabel("COPD prevalence (%)")
        ax1.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.3, title="Counties by coverage",
                   title_fontsize=fs, alignment="left")

        xr = np.array([res_x.min(), res_x.max()])
        P.zero_line(ax2, "y")
        ax2.plot(xr, slope_adj * xr, color=P.INK2, linewidth=1.6, zorder=2)
        for mask, color, _ in groups:
            P.dots(ax2, res_x[mask], res_y[mask], color=color, size=size)
        ax2.set_title("After removing smoking and inactivity from both" if variant == "report"
                      else "After adjusting for smoking, inactivity", fontweight="normal")
        ax2.set_xlabel("Median AQI, residual")
        ax2.set_ylabel("COPD prevalence, residual (points)" if variant == "report" else "COPD residual (points)")
        # same vertical span as the left panel so the two slopes can be compared by eye
        span = (a[C.OUTCOME].max() - a[C.OUTCOME].min()) / 2 * 1.08
        ax2.set_ylim(-span, span)

    P.render("fig09_q2_aqi", draw,
             title="COPD and median AQI among the 35 counties with 2023 AQI records",
             subtitle="Left: county estimates. Right: added-variable plot from the model with smoking and inactivity.",
             source=P.SOURCE_BOTH, size=(11.4, 4.9))


def main() -> None:
    C.ensure_dirs()
    d = pd.read_csv(C.PROCESSED_TABLE, dtype={"fips": str})
    a = d[d["has_aqi"].eq(1)].copy().reset_index(drop=True)
    a["aqi_pct_gt100"] = 100 * a["aqi_prop_gt100"]            # percent of monitored days above 100
    a["log10_population"] = np.log10(a["population"])
    sub = a[a["aqi_ge300_days"].eq(1)].reset_index(drop=True)
    rng = np.random.default_rng(C.SEED)

    specs = [
        (a, "median AQI alone", ["aqi_median"]),
        (a, "behaviors", C.BEHAVIORAL),
        (a, "behaviors + median AQI", C.BEHAVIORAL + ["aqi_median"]),
        (a, "behaviors + percent of days above 100", C.BEHAVIORAL + ["aqi_pct_gt100"]),
        (sub, f">= {C.AQI_MIN_DAYS} days: behaviors", C.BEHAVIORAL),
        (sub, f">= {C.AQI_MIN_DAYS} days: behaviors + median AQI", C.BEHAVIORAL + ["aqi_median"]),
    ]
    coefs = pd.DataFrame([row for data, label, cols in specs for row in coefficient_rows(data, label, cols, rng)])
    coefs.round(5).to_csv(C.TABLES / "q2_coefficients.csv", index=False)

    # Does adding median AQI improve on the behaviors-only model (same 35 counties)?
    base, plus = fit(a, C.BEHAVIORAL), fit(a, C.BEHAVIORAL + ["aqi_median"])
    f_stat, f_p, _ = plus.compare_f_test(base)

    # Influence on the AQI slope: leave each county out in turn.
    infl = plus.get_influence()
    loo = plus.params["aqi_median"] - infl.dfbeta[:, list(plus.params.index).index("aqi_median")]
    influence = (pd.DataFrame({"county": a["county"], "aqi_days": a["aqi_days"], "aqi_median": a["aqi_median"],
                               "cooks_d": infl.cooks_distance[0], "leverage": infl.hat_matrix_diag,
                               "aqi_slope_if_dropped": loo})
                 .sort_values("cooks_d", ascending=False).head(8))
    influence.round(5).to_csv(C.TABLES / "q2_influence.csv", index=False)

    profile, labels = monitoring_profile(a)
    profile.round(3).to_csv(C.TABLES / "q2_monitoring_profile.csv", index=False)

    corr = a[["aqi_median", "aqi_pct_gt100", "aqi_days", "log10_population", C.OUTCOME] + C.BEHAVIORAL].corr()
    corr.round(3).to_csv(C.TABLES / "q2_correlations.csv")

    raw_slope = coefs.query("model == 'median AQI alone' and term == 'aqi_median'")["estimate"].iloc[0]
    fig_aqi(a, raw_slope, plus.params["aqi_median"])

    pd.set_option("display.width", 220)
    print(coefs.drop(columns=["p", "adj_r2"]).round(4).to_string(index=False))
    print("\nF test for median AQI:", round(float(f_stat), 3), "p =", round(float(f_p), 4))
    print("\n", influence.round(4).to_string(index=False))
    print("\n", profile.round(2).to_string(index=False))
    print("\n", corr.round(2).to_string())

    def term(model, name):
        r = coefs[(coefs["model"] == model) & (coefs["term"] == name)].iloc[0]
        return {k: float(r[k]) for k in ("estimate", "ci_low", "ci_high", "hc3_low", "hc3_high",
                                         "boot_low", "boot_high", "r2", "loo_rmse")} | {"n": int(r["n"])}

    sub_label = f">= {C.AQI_MIN_DAYS} days: behaviors + median AQI"
    C.update_results("q2", {
        "n": len(a), "n_ge300": len(sub), "n_boot": C.N_BOOT,
        "corr_copd_aqi_median": float(corr.loc[C.OUTCOME, "aqi_median"]),
        "corr_copd_aqi_pct_gt100": float(corr.loc[C.OUTCOME, "aqi_pct_gt100"]),
        "corr_smoking_aqi_median": float(corr.loc["smoking", "aqi_median"]),
        "corr_logpop_aqi_median": float(corr.loc["log10_population", "aqi_median"]),
        "corr_aqi_median_pct_gt100": float(corr.loc["aqi_median", "aqi_pct_gt100"]),
        "aqi_median_min": float(a["aqi_median"].min()), "aqi_median_max": float(a["aqi_median"].max()),
        "aqi_median_sd": float(a["aqi_median"].std(ddof=1)),
        "pct_gt100_max": float(a["aqi_pct_gt100"].max()),
        "n_counties_any_day_gt100": int((a["aqi_days_gt100"] > 0).sum()),
        "unadjusted": term("median AQI alone", "aqi_median"),
        "behaviors_r2": float(base.rsquared),
        "behaviors_loo_rmse": term("behaviors", "smoking")["loo_rmse"],
        "adjusted": term("behaviors + median AQI", "aqi_median"),
        "adjusted_per_10_aqi": 10 * float(plus.params["aqi_median"]),
        "f_test": {"statistic": float(f_stat), "p": float(f_p)},
        "pct_gt100": term("behaviors + percent of days above 100", "aqi_pct_gt100"),
        "ge300": term(sub_label, "aqi_median"),
        "loo_slope_min": float(loo.min()), "loo_slope_max": float(loo.max()),
        "loo_slope_min_dropped": a.loc[int(np.argmin(loo)), "county"],
        "loo_slope_max_dropped": a.loc[int(np.argmax(loo)), "county"],
        "monitoring_profile": profile.round(3).to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
