"""Step 06 - render the report from its Markdown source.

    report/report.md  ->  report/report.html   self-contained (figures embedded)
                      ->  report/report.pdf    with --pdf, if Chrome or Chromium is installed

The Markdown source holds the team's text. Everything computed is pulled in at build
time, so numbers, tables and figures in the rendered report always match the results:

    {{figure:NAME}}           figure from results/figures, numbered in order of appearance
    {{table:NAME}}            table built here from results/tables or results/audit
    {{ref:NAME}}              "Figure 3" or "Table 5" for a figure or table used in the report
    {{path.to.number|.2f}}    a value from results/results.json with a Python format
    {{show:path|.2f}}         the value followed by its token (handy while drafting)
    ::: write ... :::         a block the team still has to write (highlighted in the draft)

Usage
    python -I scripts/06_build_report.py            # draft
    python -I scripts/06_build_report.py --final    # refuses to build while a "::: write" block remains
    python -I scripts/06_build_report.py --pdf      # also print to PDF
"""
from __future__ import annotations

import argparse
import ast
import base64
import html
import platform
import re
import shutil
import subprocess
import sys
from email.utils import parsedate_to_datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import markdown
import pandas as pd

import config as C

SOURCE = C.REPORT / "report.md"
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "markdown", "pyshp"]
MINUS = "−"


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def num(x, nd: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s.replace("-", MINUS)


def interval(lo, hi, nd: int = 2) -> str:
    return f"{num(lo, nd)} to {num(hi, nd)}"


def pval(p: float) -> str:
    """'p < 0.001' or 'p = 0.040'."""
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


WORD = {"smoking": "smoking", "inactivity": "inactivity", "transport": "transportation",
        "uninsured": "uninsured", "housing": "housing"}


def table_html(header: list[str], rows: list[list[str]], left=1) -> str:
    """Plain HTML table. `left` = number of leading left-aligned columns, or the set of
    left-aligned column positions; all other columns are right-aligned."""
    left_cols = set(range(left)) if isinstance(left, int) else set(left)

    def cell(tag, text, i):
        cls = "l" if i in left_cols else "r"
        return f'<{tag} class="{cls}">{text}</{tag}>'
    head = "".join(cell("th", h, i) for i, h in enumerate(header))
    body = "\n".join("<tr>" + "".join(cell("td", c, i) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>"


def label(term: str) -> str:
    return {"const": "Intercept"}.get(term, C.LABELS.get(term, term))


# --------------------------------------------------------------------------
# Tables: each function returns (caption, html, note)
# --------------------------------------------------------------------------
def t_sources(R):
    m = C.read_json(C.SNAPSHOT_MANIFEST)
    src, files, acq = m["sources"], m["files"], R["acquisition"]
    cdc, epa, census = acq["comparison"]
    fresh = acq["cdc_fresh_retrieved_utc"][:10]

    def http_date(text: str) -> str:
        return parsedate_to_datetime(text).strftime("%Y-%m-%d")

    def same(c: dict) -> str:
        return f"identical: {c['cells_different']} of {c['cells_compared']:,} values differ"

    def checksum(key: str) -> str:
        name = src[key]["file"]
        return f"<code>{name}</code><br><code>{files[name]['sha256'][:16]}&hellip;</code>"

    rows = [
        ["Original publisher"] + [src[k]["publisher"] for k in ("cdc", "epa", "census")],
        ["Dataset and version"] + [src[k]["dataset"] for k in ("cdc", "epa", "census")],
        ["Access", f"data.cdc.gov dataset <code>{src['cdc']['dataset_id']}</code>, API query <code>stateabbr=NC</code>",
         "<code>annual_aqi_by_county_2023.zip</code> from aqs.epa.gov/aqsweb/airdata, North Carolina rows",
         f"<code>{C.CENSUS_ZIP_STEM}.zip</code> from www2.census.gov/geo/tiger/GENZ2023/shp, North Carolina polygons"],
        ["Row in the source", "county &times; measure &times; value type (model-based estimate with 95% limits)",
         "county &times; year (summary of daily AQI values)", "county polygon (generalized outline)"],
        ["Records retrieved", f"{cdc['rows_snapshot']:,} (100 counties &times; 40 measures &times; 2 value types)",
         f"{epa['rows_snapshot']} counties with AQI records in 2023", f"{census['rows_snapshot']} county polygons"],
        ["Use in this project", "outcomes and predictors", "air-quality measures (Question 2)",
         "outlines of the county map only; no model uses it"],
        ["Analysis snapshot retrieved"] + [src[k]["retrieved_local_date"] for k in ("cdc", "epa", "census")],
        ["Source last changed", acq["cdc_rows_updated_utc"][:10] + " (rows updated)",
         http_date(acq["epa_zip_last_modified"]) + " (file last modified)",
         http_date(acq["census_zip_last_modified"]) + " (file last modified)"],
        [f"Re-download on {fresh}", same(cdc), same(epa), "same download as the snapshot"],
        ["Terms", "Public domain (dataset metadata)", "U.S. federal government public data",
         "U.S. Census Bureau public data product"],
        ["Snapshot file, SHA-256", checksum("cdc"), checksum("epa"), checksum("census")],
    ]
    return ("Data sources and retrieval record",
            table_html(["", "CDC PLACES", "EPA AirData", "Census boundary file"], rows, left=4),
            "Full checksums and retrieval details: data/raw/SNAPSHOT_MANIFEST.json and data/fresh/.")


def t_variables(R):
    dd = pd.read_csv(C.DATA_DICTIONARY).set_index("variable")
    rows = []
    for m in C.MEASURES.values():
        desc = dd.loc[m["col"], "description"].split(":")[0]
        rows.append([f"<code>{m['col']}</code>", desc, "percent, age-adjusted", str(int(dd.loc[m["col"], "year"])), m["role"]])
    for var, role in (("aqi_median", "air-quality measure (Question 2)"),
                      ("aqi_prop_gt100", "air-quality sensitivity measure"),
                      ("aqi_days", "monitoring coverage"),
                      ("population", "context, county size")):
        rows.append([f"<code>{var}</code>", dd.loc[var, "description"], dd.loc[var, "unit"],
                     str(int(dd.loc[var, "year"])), role])
    return ("Variables used in the analysis",
            table_html(["Variable", "Definition (publisher's measure name)", "Unit", "Year", "Role"], rows, left=5),
            "PLACES measures are model-based estimates for adults aged 18 or older (18-64 for health insurance); "
            "each has CDC lower and upper 95% limits in columns ending _lcl and _ucl. "
            "All columns: data/processed/data_dictionary.csv.")


def t_row_counts(R):
    d = pd.read_csv(C.AUDIT / "row_counts.csv")
    rows = [[r.step, f"{r.rows:,}", str(r.counties), r.note] for r in d.itertuples()]
    return ("Rows kept at each preparation step",
            table_html(["Step", "Rows", "Counties", "Note"], rows, left={0, 3}),
            "Source: results/audit/row_counts.csv, written by scripts/01_clean_audit.py.")


def t_audit_summary(R):
    d = pd.read_csv(C.AUDIT / "audit_checks.csv")
    rows = []
    for area, g in d.groupby("area", sort=False):
        rows.append([area, f"{int(g['result'].eq('PASS').sum())} of {len(g)}",
                     "; ".join(s if s[:2].isupper() else s[0].lower() + s[1:] for s in g["check"])])
    rows.append(["<strong>All checks</strong>", f"<strong>{R['audit']['n_passed']} of {R['audit']['n_checks']}</strong>", ""])
    return ("Executable audit checks, by area", table_html(["Area", "Passed", "What is checked"], rows, left=3),
            "Each check records an expected and an observed value; the pipeline stops if any check fails. "
            "Full list in Appendix A.")


def t_audit_full(R):
    d = pd.read_csv(C.AUDIT / "audit_checks.csv")
    rows = [[r.id, r.area, r.check, html.escape(str(r.expected)), html.escape(str(r.observed)), r.result]
            for r in d.itertuples()]
    return ("All audit checks with expected and observed values",
            table_html(["ID", "Area", "Check", "Expected", "Observed", "Result"], rows, left=6),
            "Source: results/audit/audit_checks.csv.")


def t_descriptives(R):
    d = pd.read_csv(C.TABLES / "eda_descriptives.csv")
    rows = [[r.indicator, num(r["mean"]), num(r.sd), num(r["min"], 1), num(r["median"], 1), num(r["max"], 1),
             num(r.cdc_ci_halfwidth_mean), num(r.halfwidth_to_sd)] for _, r in d.iterrows()]
    return ("County indicators: spread across counties and width of CDC's intervals (percent, n = 100)",
            table_html(["Indicator", "Mean", "SD", "Min", "Median", "Max",
                        "Mean half-width of CDC 95% interval", "Half-width &divide; SD"], rows),
            "Half-width = (upper limit &minus; lower limit) / 2 for one county, averaged over counties. "
            "Last column = that mean half-width divided by the SD across counties.")


def t_design(R):
    q3 = R["q3"]
    rows = [
        ["Q1 Social indicators and COPD", "100", "COPD (asthma secondary)",
         "Baseline: smoking + inactivity.<br>Expanded: baseline + transport + uninsured + housing.",
         f"Classical, HC3 robust and county-bootstrap ({C.N_BOOT:,} resamples) 95% intervals",
         "VIF, Breusch-Pagan, Cook's distance, leave-one-county-out slopes, Moran's I of residuals, "
         "leave-one-out prediction error"],
        ["Q2 Median AQI and COPD", str(R["q2"]["n"]), "COPD",
         "Behaviors only versus behaviors + median AQI on the same counties.",
         "Same three intervals",
         f"Share of days above 100 as a separate measure; counties with at least {C.AQI_MIN_DAYS} monitored "
         f"days (n = {R['q2']['n_ge300']}); leave-one-county-out AQI slope"],
        ["Q3 Flagging high-COPD counties", "100", "COPD at or above the training-set 80th percentile",
         "Logistic regression on smoking only versus penalized logistic regression on five indicators.",
         f"Nested cross-validation ({q3['n_outer']} outer &times; {q3['n_inner']} inner folds); "
         f"{q3['n_repeats']} other random partitions",
         "Cutoff, scaling and tuning from training counties only; shuffled-outcome check; errors by county group"],
    ]
    return ("Analysis design",
            table_html(["Question", "Counties", "Outcome", "Comparison", "Uncertainty", "Checks"], rows, left=6),
            f"Random seed {C.SEED} for every resampling step. Models are ordinary least squares (Q1, Q2) "
            "and logistic regression (Q3).")


def _coef_table(R, outcome: str):
    c = R["q1"]["coefficients"][outcome]
    rows = []
    for term in C.ALL_PREDICTORS:
        b, e = c["baseline"].get(term), c["expanded"][term]
        rows.append([
            label(term),
            num(b["estimate"], 3, sign=True) if b else "—",
            interval(b["hc3_low"], b["hc3_high"], 3) if b else "—",
            num(e["estimate"], 3, sign=True),
            interval(e["ci_low"], e["ci_high"], 3),
            interval(e["hc3_low"], e["hc3_high"], 3),
            interval(e["boot_low"], e["boot_high"], 3),
        ])
    header = ["Predictor", "Baseline slope", "Baseline HC3 95%", "Expanded slope",
              "Classical 95%", "HC3 robust 95%", "Bootstrap 95%"]
    return table_html(header, rows)


def t_q1_coefs(R):
    return (f"{C.LABELS[C.OUTCOME]}: slopes from the baseline and expanded models (n = 100)", _coef_table(R, C.OUTCOME),
            "Slopes are percentage-point differences in modeled COPD prevalence per one-point difference in the "
            f"predictor. Bootstrap: {C.N_BOOT:,} county resamples, percentile interval.")


def t_q1_asthma(R):
    return ("Current asthma (secondary outcome): slopes from the baseline and expanded models (n = 100)",
            _coef_table(R, C.SECONDARY_OUTCOME),
            "Same models and intervals as the COPD table, with current asthma prevalence as the outcome.")


def t_q1_fit(R):
    rows = []
    for y in (C.OUTCOME, C.SECONDARY_OUTCOME):
        for m in ("baseline", "expanded"):
            f = R["q1"]["fit"][y][m]
            rows.append([C.LABELS[y], m.capitalize(), str(f["predictors"]), num(f["r2"], 3), num(f["adj_r2"], 3),
                         num(f["residual_sd"], 3), num(f["aic"], 1), num(f["loo_rmse"], 3)])
    k = R["q1"]["comparison"]
    note = "; ".join(
        f"{C.LABELS[y]}: R&sup2; gain {num(k[y]['delta_r2'], 3)}, F = {num(k[y]['f_stat'], 2)} "
        f"({pval(k[y]['f_p'])}), HC3 Wald F = {num(k[y]['wald_hc3_f'], 2)} ({pval(k[y]['wald_hc3_p'])})"
        for y in (C.OUTCOME, C.SECONDARY_OUTCOME))
    return ("Model fit with and without the three social indicators",
            table_html(["Outcome", "Model", "Predictors", "R&sup2;", "Adj. R&sup2;", "Residual SD",
                        "AIC", "Leave-one-out RMSE"], rows, left=2),
            "Residual SD and RMSE in percentage points. Joint test of the three social indicators &mdash; " + note + ".")


def t_q1_checks(R):
    q = R["q1"]
    vif = q["vif"]["expanded"]
    top = q["influence_top"][0]
    sp = q["spatial"]
    loo = q["leave_one_out"]
    rows = [
        ["Collinearity", "Variance inflation factor, expanded model",
         ", ".join(f"{WORD[t]} {num(vif[t], 1)}" for t in C.ALL_PREDICTORS)],
        ["Collinearity", "Share of predictor variance on the first principal component",
         f"{100 * q['collinearity']['pc1_share_of_variance']:.0f}%"],
        ["Collinearity", "R&sup2; of COPD on smoking alone", num(q["smoking_only_r2"], 3)],
        ["Unequal variance", "Breusch-Pagan test, expanded model",
         f"statistic {num(q['breusch_pagan']['statistic'], 2)}, {pval(q['breusch_pagan']['p'])}"],
        ["Influence", "Largest Cook's distance",
         f"{top['county']}: {num(top['cooks_d'], 2)} (leverage {num(top['leverage'], 2)}); "
         f"{q['n_cooks_above_threshold']} counties above 4/n = {num(q['cooks_threshold'], 2)}"],
        ["Influence", "Leave-one-county-out range of the slope",
         "; ".join(f"{WORD[t]} {interval(loo[t]['loo_min'], loo[t]['loo_max'], 3)}" for t in C.SOCIAL)],
        ["Spatial dependence", "Moran's I of residuals, 5 nearest centroids",
         f"baseline {num(sp['baseline model residuals']['morans_i'], 2)} "
         f"({pval(sp['baseline model residuals']['p_permutation_one_sided'])}); "
         f"expanded {num(sp['expanded model residuals']['morans_i'], 2)} "
         f"({pval(sp['expanded model residuals']['p_permutation_one_sided'])})"],
    ]
    return ("Checks on the COPD models", table_html(["Concern", "Check", "Result"], rows, left=3),
            "Moran's I p-values are one-sided, from 9,999 permutations. Diagnostic plots are in Appendix B.")


def t_q1_sensitivity(R):
    q = R["q1"]
    rows = []
    for s in q["sensitivity"]:
        term = label(s["term"]) if s["term"] in C.LABELS else "Composite social index (per 1 SD)"
        rows.append([cap(s["analysis"]), str(s["n"]), term, num(s["estimate"], 3, sign=True),
                     interval(s["hc3_low"], s["hc3_high"], 3) + " (HC3)"])
    for p in q["transport_housing_pair"]:
        rows.append(["Expanded model with transport and housing re-expressed", "100", cap(p["term"]),
                     num(p["estimate"], 3, sign=True), interval(p["hc3_low"], p["hc3_high"], 3) + " (HC3); "
                     + interval(p["boot_low"], p["boot_high"], 3) + " (bootstrap)"])
    pv = q["plausible_values"]
    for t in C.SOCIAL:
        s = pv["slopes"][t]
        rows.append([f"Plausible-value draws from CDC limits ({pv['n_draws']:,} draws)", "100", label(t),
                     num(s["sim_median"], 3, sign=True) + " (median)",
                     interval(s["sim_low"], s["sim_high"], 3) + " (2.5th to 97.5th percentile)"])
    return ("Sensitivity of the social-indicator slopes for COPD",
            table_html(["Analysis", "n", "Term", "Slope", "95% interval"], rows, left={0, 2, 4}),
            "The composite index, the re-expression of transport and housing, and the plausible-value draws were "
            "added after the collinearity diagnosis and were not in the proposal. Plausible-value draws assume "
            "independent errors across counties and measures.")


def t_aqi_coverage(R):
    d = pd.read_csv(C.TABLES / "eda_aqi_coverage.csv")
    rows = []
    for _, r in d.iterrows():
        nd = 0 if r.variable == "population" else 2
        name = "Population (mean)" if r.variable == "population" else C.LABELS[r.variable] + " (%)"
        rows.append([name, num(r.with_aqi_mean, nd), num(r.without_aqi_mean, nd),
                     num(r.difference_in_means, nd, sign=True), num(r.standardized_difference, 2, sign=True)])
    n1, n0 = int(d.n_with_aqi.iloc[0]), int(d.n_without_aqi.iloc[0])
    return ("Counties with and without 2023 AQI records",
            table_html(["", f"With AQI records (n = {n1})", f"Without (n = {n0})", "Difference",
                        "Standardized difference"], rows),
            f"Counties with AQI records hold {100 * R['eda']['aqi_population_share']:.0f}% of the state population. "
            "Standardized difference = difference in means divided by the pooled SD.")


def t_q2_models(R):
    d = pd.read_csv(C.TABLES / "q2_coefficients.csv")
    names = {"aqi_median": "Median AQI (per 1 AQI point)", "aqi_pct_gt100": "Days with AQI above 100 (per 1 % of days)"}
    rows = []
    for model, g in d.groupby("model", sort=False):
        first = g.iloc[0]
        aqi = g[g["term"].isin(names)]
        if len(aqi):
            r = aqi.iloc[0]
            cells = [names[r.term], num(r.estimate, 4, sign=True), interval(r.ci_low, r.ci_high, 4),
                     interval(r.hc3_low, r.hc3_high, 4), interval(r.boot_low, r.boot_high, 4)]
        else:
            cells = ["none", "—", "—", "—", "—"]
        rows.append([cap(model).replace(">=", "&ge;"), str(int(first.n))] + cells
                    + [num(first.r2, 3), num(first.loo_rmse, 3)])
    return ("Air-quality models for COPD among counties with AQI records",
            table_html(["Model", "n", "Air-quality term", "Slope", "Classical 95%", "HC3 robust 95%",
                        "Bootstrap 95%", "R&sup2;", "Leave-one-out RMSE"], rows, left={0, 2}),
            "Behaviors = smoking and inactivity. Slopes are percentage-point differences in COPD prevalence. "
            "F test for adding median AQI to the behaviors model: "
            f"F = {num(R['q2']['f_test']['statistic'], 2)}, {pval(R['q2']['f_test']['p'])}.")


def t_q3_cv(R):
    names = {"roc_auc": "ROC AUC", "average_precision": "Average precision",
             "sensitivity": f"Sensitivity at probability {R['q3']['threshold']}"}
    who = {"baseline": "Smoking only", "penalized": "Five indicators, penalized",
           "penalized minus baseline": "Difference (penalized &minus; smoking only)"}
    rows = []
    for s in R["q3"]["summary"]:
        sd = s.get("sd_across_its_folds")
        sign = s["quantity"].startswith("penalized minus")
        rows.append([names[s["metric"]], who[s["quantity"]],
                     num(s["prespecified_partition"], 3, sign=sign) + (f" ({num(sd, 3)})" if sd == sd and sd is not None else ""),
                     num(s["repeats_mean"], 3, sign=sign),
                     interval(s["repeats_p2_5"], s["repeats_p97_5"], 3)])
    q = R["q3"]
    return ("Held-out performance for flagging high-COPD counties",
            table_html(["Metric", "Model", "Prespecified partition: mean over 5 folds (SD)",
                        f"Mean over {q['n_repeats']} other partitions", "2.5th to 97.5th percentile"], rows, left=2),
            f"High COPD = at or above the training-set 80th percentile ({q['n_high_held_out_total']} held-out counties "
            f"labelled high across the five folds). Shuffled-outcome check: mean ROC AUC "
            f"{num(q['shuffled_outcome']['baseline']['mean_roc_auc'], 2)} (smoking only) and "
            f"{num(q['shuffled_outcome']['penalized']['mean_roc_auc'], 2)} (penalized) over {q['n_permutations']} shuffles.")


def t_q3_folds(R):
    d = pd.read_csv(C.TABLES / "q3_cv_folds.csv")
    rows = [[str(r.fold), "Smoking only" if r.model == "baseline" else "Penalized", num(r.cutoff, 2),
             f"{r.n_high} of {r.n_test}", str(r.false_negatives), str(r.false_positives), num(r.roc_auc, 3),
             num(r.average_precision, 3), num(r.sensitivity, 2),
             "—" if pd.isna(r.C) else f"C = {r.C:g}, L1 ratio = {r.l1_ratio:g}"] for r in d.itertuples()]
    return ("Prespecified partition: results by outer fold",
            table_html(["Fold", "Model", "Training cutoff (%)", "Held-out high / all", "False negatives",
                        "False positives", "ROC AUC", "Average precision", "Sensitivity", "Tuned penalty"], rows, left=2),
            "Source: results/tables/q3_cv_folds.csv.")


def t_q1_slices(R):
    d = pd.read_csv(C.TABLES / "q1_slices.csv")
    rows = []
    for (grouping, group), part in d.groupby(["slice", "group"], sort=False):
        b, e = (part[part["model"].eq(m)].iloc[0] for m in ("baseline", "expanded"))
        rows.append([cap(grouping), cap(group), str(int(b["n"])),
                     num(b["mean_loo_error"], 3, sign=True), num(e["mean_loo_error"], 3, sign=True),
                     num(b["mean_abs_loo_error"], 3), num(e["mean_abs_loo_error"], 3)])
    return ("COPD models: leave-one-out prediction error by county group",
            table_html(["Grouping", "Group", "Counties", "Baseline mean error", "Expanded mean error",
                        "Baseline mean |error|", "Expanded mean |error|"], rows, left=2),
            "Error = observed minus leave-one-out prediction, in percentage points. Thirds are equal-count "
            "groups by 2023 population and by centroid longitude.")


def t_q3_slices(R):
    d = pd.read_csv(C.TABLES / "q3_slices.csv")
    rows = [[cap(r.slice), cap(r.group), str(r.n), str(r.n_high), str(r.false_negatives_baseline),
             str(r.false_negatives_penalized), str(r.false_positives_baseline), str(r.false_positives_penalized)]
            for r in d.itertuples()]
    missed = ", ".join(m["county"] for m in R["q3"]["missed_counties"])
    return ("Held-out classification errors by county group",
            table_html(["Grouping", "Group", "Held-out counties", "Labelled high", "Missed: smoking only",
                        "Missed: penalized", "False alarms: smoking only", "False alarms: penalized"], rows, left=2),
            f"Prespecified partition, probability threshold {R['q3']['threshold']}. "
            f"High-COPD counties missed by either model: {missed}.")


def t_q1_influence(R):
    d = pd.read_csv(C.TABLES / "q1_influence.csv").head(8)
    rows = [[r.county, f"{r.population:,}", num(r.cooks_d, 3), num(r.leverage, 3), num(r.studentized_residual, 2),
             num(r.slope_change_if_dropped_transport, 3, sign=True), num(r.slope_change_if_dropped_housing, 3, sign=True)]
            for r in d.itertuples()]
    return ("Most influential counties in the expanded COPD model",
            table_html(["County", "Population", "Cook's distance", "Leverage", "Studentized residual",
                        "Change in transport slope if dropped", "Change in housing slope if dropped"], rows),
            "Source: results/tables/q1_influence.csv.")


def t_software(R):
    rows = [["Python", platform.python_version()]] + [[p, metadata.version(p)] for p in PACKAGES]
    return ("Software used to produce this report", table_html(["Component", "Version"], rows, left=2),
            "Exact versions of every installed package: requirements-lock.txt.")


def t_files(R):
    rows = []
    for path in sorted((C.ROOT / "scripts").glob("0*.py")):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        first = doc.split("\n")[0]
        rows.append([f"<code>scripts/{path.name}</code>", re.sub(r"^Step \d+ - ", "", first).rstrip(".")])
    rows.append(["<code>run_all.py</code>", "Entry point: runs every step in order from a clean start"])
    return ("Entry points and pipeline steps", table_html(["File", "Purpose"], rows, left=2),
            "Shared settings: scripts/config.py (paths, seed, variable roles) and scripts/plotstyle.py (figure style).")


TABLES = {f.__name__[2:]: f for f in (
    t_sources, t_variables, t_row_counts, t_audit_summary, t_audit_full, t_descriptives, t_design,
    t_q1_coefs, t_q1_asthma, t_q1_fit, t_q1_checks, t_q1_sensitivity, t_aqi_coverage, t_q2_models,
    t_q3_cv, t_q3_folds, t_q3_slices, t_q1_influence, t_q1_slices, t_software, t_files)}

FIGURES = {
    "fig01_copd_map": "Modeled age-adjusted COPD prevalence by county, North Carolina, 2023.",
    "fig02_copd_intervals": "County COPD estimates ordered from lowest to highest, with CDC 95% confidence limits.",
    "fig03_correlations": "Pearson correlations among the seven county indicators.",
    "fig04_copd_vs_smoking": "COPD against smoking by county, with CDC 95% confidence limits in both directions.",
    "fig05_copd_vs_social": "COPD against each social indicator before adjustment.",
    "fig06_q1_specifications": "Slopes of the three social indicators for COPD under three sets of adjustment variables.",
    "fig07_q1_intervals": "Expanded COPD model: classical, robust and bootstrap 95% intervals for each slope.",
    "fig08_q1_diagnostics": "Residual and influence diagnostics for the expanded COPD model.",
    "fig09_q2_aqi": "COPD and median AQI among counties with AQI records, before and after adjustment.",
    "fig10_q3_cv": "Held-out performance of the two classifiers across cross-validation partitions.",
}


# --------------------------------------------------------------------------
# Markdown -> HTML
# --------------------------------------------------------------------------
TOKEN = re.compile(r"\{\{([^{}]+)\}\}")


def lookup(R: dict, path: str):
    node = R
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def value(R: dict, spec: str) -> str:
    path, _, fmt = spec.partition("|")
    try:
        v = lookup(R, path.strip())
    except (KeyError, IndexError, ValueError) as err:
        raise SystemExit(f"Unknown result reference in report.md: {{{{{spec}}}}} ({err})")
    return format(v, fmt.strip()).replace("-", MINUS) if fmt else str(v)


def build(final: bool) -> tuple[str, int]:
    R = C.read_json(C.RESULTS_JSON)
    text = SOURCE.read_text(encoding="utf-8")

    # blocks still to be written by the team
    n_todo = len(re.findall(r"^::: write\s*$", text, flags=re.MULTILINE))
    if final and n_todo:
        raise SystemExit(f"{n_todo} '::: write' block(s) remain in report/report.md. "
                         "Replace each with the team's own text before building the final report.")
    text = re.sub(r"^::: write\s*$", '<div class="todo" markdown="1">\n<p class="todo-tag">To write (team)</p>\n',
                  text, flags=re.MULTILINE)
    text = re.sub(r"^:::\s*$", "\n</div>", text, flags=re.MULTILINE)

    # number figures and tables in order of first appearance
    numbers, counts = {}, {"figure": 0, "table": 0}
    for kind, name in re.findall(r"\{\{(figure|table):([\w]+)\}\}", text):
        if (kind, name) not in numbers:
            counts[kind] += 1
            numbers[(kind, name)] = counts[kind]

    def replace(match: re.Match) -> str:
        spec = match.group(1).strip()
        if spec.startswith("figure:"):
            name = spec[7:]
            if name not in FIGURES:
                raise SystemExit(f"Unknown figure in report.md: {name}")
            png = (C.FIGURES / f"{name}.png").read_bytes()
            n = numbers[("figure", name)]
            return (f'\n<figure id="{name}">\n<img alt="{html.escape(FIGURES[name])}" '
                    f'src="data:image/png;base64,{base64.b64encode(png).decode()}">\n'
                    f"<figcaption><strong>Figure {n}.</strong> {FIGURES[name]}</figcaption>\n</figure>\n")
        if spec.startswith("table:"):
            name = spec[6:]
            if name not in TABLES:
                raise SystemExit(f"Unknown table in report.md: {name}")
            caption, body, note = TABLES[name](R)
            n = numbers[("table", name)]
            return (f'\n<div class="table" id="{name}">\n<p class="caption"><strong>Table {n}.</strong> {caption}</p>\n'
                    f'{body}\n<p class="note">{note}</p>\n</div>\n')
        if spec.startswith("ref:"):
            name = spec[4:]
            for kind in ("figure", "table"):
                if (kind, name) in numbers:
                    return f'<a href="#{name}">{kind.capitalize()} {numbers[(kind, name)]}</a>'
            raise SystemExit(f"{{{{ref:{name}}}}} points to a figure or table that is not in the report.")
        if spec.startswith("show:"):
            return f'{value(R, spec[5:])} <code class="token">{{{{{spec[5:]}}}}}</code>'
        return value(R, spec)

    text = TOKEN.sub(replace, text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html", "sane_lists"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
    banner = (f'<div class="draft">Draft &mdash; {n_todo} block(s) marked &ldquo;To write (team)&rdquo; '
              "still need the team&rsquo;s own text.</div>" if n_todo else "")
    title = re.search(r"^# (.+)$", SOURCE.read_text(encoding="utf-8"), flags=re.MULTILINE).group(1)
    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n<style>\n{CSS}\n</style>\n</head>\n<body>\n{banner}\n"
            f"<main>\n{body}\n</main>\n</body>\n</html>\n")
    return page, n_todo


CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --accent:#184f95; }
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; color: var(--ink); line-height: 1.58;
       font: 16px/1.58 -apple-system, "Helvetica Neue", "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 40px 28px 80px; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .3rem; }
h2 { font-size: 1.35rem; margin: 2.6rem 0 .6rem; padding-top: .9rem; border-top: 1px solid var(--line); }
h3 { font-size: 1.08rem; margin: 1.8rem 0 .4rem; }
p, li { max-width: 78ch; }
a { color: var(--accent); }
code { font: .86em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f4f4f1;
       padding: .08em .3em; border-radius: 3px; }
pre { background: #f4f4f1; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
.subtitle { color: var(--ink2); margin: 0 0 1.4rem; }
.toc { font-size: .93rem; background: #f9f9f7; border: 1px solid var(--line); border-radius: 6px;
       padding: 10px 18px; margin: 1.2rem 0 0; }
.toc ul { margin: .2rem 0; padding-left: 1.1rem; }
figure { margin: 1.5rem 0 1.7rem; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; }
figcaption, .caption, .note { font-size: .9rem; color: var(--ink2); max-width: none; }
figcaption { margin-top: .5rem; }
.table { margin: 1.5rem 0 1.7rem; overflow-x: auto; }
.caption { margin: 0 0 .4rem; color: var(--ink); }
.note { margin: .45rem 0 0; color: var(--muted); font-size: .84rem; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; font-variant-numeric: tabular-nums; }
th { text-align: right; font-weight: 600; color: var(--ink2); border-bottom: 1.5px solid #c3c2b7;
     padding: 6px 9px; vertical-align: bottom; }
td { padding: 5px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
th.l, td.l { text-align: left; }
th.r, td.r { text-align: right; white-space: nowrap; }
.todo { background: #fff8e1; border-left: 4px solid #f0b400; padding: 4px 16px 8px; margin: 1.1rem 0;
        font-size: .92rem; border-radius: 0 6px 6px 0; }
.todo-tag { font-weight: 700; text-transform: uppercase; letter-spacing: .04em; font-size: .78rem;
            color: #8a6500; margin: .5rem 0 .2rem; }
.todo li, .todo p { max-width: none; }
.token { color: var(--muted); background: none; font-size: .78em; }
.draft { background: #f0b400; color: #1c1500; text-align: center; font-weight: 600; padding: 8px 12px;
         font-size: .92rem; }
@media print {
  body { font-size: 10.5pt; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; } h3 { break-after: avoid; }
  figure, .todo, tr { break-inside: avoid; }
  .caption { break-after: avoid; }
  .table { overflow: visible; }
  table { font-size: 8pt; }
  th, td { padding: 4px 5px; }
  code { font-size: .82em; }
  a { color: inherit; text-decoration: none; }
  .toc { break-after: page; }
  @page { size: letter; margin: 0.8in 0.75in; }
}
"""


def print_pdf(html_path: Path) -> None:
    candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Chromium.app/Contents/MacOS/Chromium",
                  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    found = next((c for c in candidates if Path(c).exists()), None) or next(
        (shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")
         if shutil.which(n)), None)
    if not found:
        print("No Chrome, Chromium or Edge found: open report/report.html in a browser and print to PDF.")
        return
    pdf = html_path.with_suffix(".pdf")
    subprocess.run([found, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Wrote {C.rel(pdf)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--final", action="store_true", help="fail if any '::: write' block remains")
    parser.add_argument("--pdf", action="store_true", help="also print the report to PDF with a headless browser")
    args = parser.parse_args()

    page, n_todo = build(args.final)
    out = C.REPORT / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {C.rel(out)}" + (f"  (draft: {n_todo} block(s) still to be written by the team)" if n_todo else ""))
    if args.pdf:
        print_pdf(out)


if __name__ == "__main__":
    main()
