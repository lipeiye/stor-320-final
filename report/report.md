# Respiratory Health and Social Needs in North Carolina Counties

STOR 320-003 Methods of Data Science, Fall 2026 &middot; Final project report
{: .subtitle }

::: write
Replace this block with the author line: every team member's name, and the submission date.
:::

[TOC]

## 1. Question, background, and scenario

::: write
**Rubric: "Question, purpose, and scope" (2) and "Originality and contribution" (2).** Roughly 250-350 words. Section 1 of your approved proposal is your own text and covers most of this; update it to past tense and to what was actually done.

- The approved question, stated once and exactly: across North Carolina's 100 counties, how are transportation difficulties, lack of health insurance and housing insecurity associated with age-adjusted COPD prevalence after adjusting for smoking and physical inactivity? Then the two secondary questions (median AQI among counties with records; whether the indicators can flag high-COPD counties better than smoking alone).
- Why it matters and to whom: name the audience (North Carolina public health practitioners, county health planners) and one concrete use.
- Scope in one or two sentences: cross-sectional, ecological, counties rather than people, associations rather than causes, modeled estimates rather than counts.
- What is distinctive about this project and why that adds value. Candidates to choose from and explain yourselves: the health-related social needs measures in PLACES combined with EPA AQI; CDC's own confidence limits treated as evidence next to the model's uncertainty; checking whether adjusted slopes survive collinearity, influential counties and three kinds of interval.
- One-sentence preview of the answer you reach in Section 6.
:::

## 2. Data and provenance

{{table:sources}}

{{table:variables}}

::: write
**Rubric: "Data provenance, audit, and preparation" (2).** Roughly 250-350 words explaining the two tables above. A reader should be able to answer: who made the data, what one row is, who and when are covered, how it was obtained, and whether you may use it.

- Original publisher of each source (not the hosting page), version, and the unit of observation.
- What a PLACES value is: a model-based small-area estimate (multilevel regression and poststratification from BRFSS, Census county population estimates and ACS), not a direct county count. Say which predictors CDC's model uses (age, sex, race/ethnicity, education, county poverty; state and county random effects) because this matters for Section 5. Cite the methodology page.
- Coverage and exclusions: adults 18+ (18-64 for insurance); survey year 2023; children and individual records are outside the study; the two social-needs measures come from an optional BRFSS module.
- EPA AQI: what "days with AQI" and "median AQI" mean; only {{show:audit.n_with_aqi}} of 100 counties have 2023 records, holding {{show:eda.aqi_population_share|.0%}} of the state population; a missing county means no AQI summary, not clean air.
- Acquisition: snapshot retrieved {{show:acquisition.snapshot_date}}; re-downloaded on 2026-11-15 and compared cell by cell ({{ref:sources}}); checksums kept.
- Map outlines: the county shapes in {{ref:fig01_copd_map}} come from a U.S. Census Bureau cartographic boundary file (2023, 1:5,000,000), retrieved {{show:acquisition.boundaries_date}}. Say that this third source was not in the proposal's data plan, that it was added only to draw the map, and that no model uses it. The outlines are generalized and are not suitable for measuring area or distance.
- Permissions: public domain / public federal data; aggregate records only; no personal data. Mention CDC's own cautions on use (FAQ page).
:::

## 3. Data audit and preparation

{{table:row_counts}}

{{table:audit_summary}}

{{table:descriptives}}

{{figure:fig02_copd_intervals}}

::: write
**Rubric: "Data provenance, audit, and preparation" (2).** Roughly 300-400 words. Explain what was checked, what changed the data, and the decision to proceed.

- Preparation decisions and why: seven prespecified measures only; age-adjusted rather than crude prevalence; long-to-wide pivot; EPA joined by county name within North Carolina, then keyed by CDC FIPS; left join so all 100 counties stay; AQI never imputed; map outlines matched to counties by FIPS code and checked, not merged into the table. Use {{ref:row_counts}}.
- What the executable checks cover ({{ref:audit_summary}}, Appendix A): keys and duplicates, units, ranges, limit ordering, missingness, join (one-to-one, zero unmatched), timing (both 2023), privacy. {{show:audit.n_passed}} of {{show:audit.n_checks}} passed.
- Missingness: none in the seven health indicators; AQI absent for {{show:audit.n_without_aqi}} counties by design of the monitoring network. Days with AQI range {{show:audit.aqi_days_min}} to {{show:audit.aqi_days_max}}; {{show:audit.n_aqi_ge300}} counties have at least 300.
- The audit finding that shapes everything after it: CDC's 95% interval for one county is about as wide as the spread between counties ({{ref:descriptives}}, {{ref:fig02_copd_intervals}}). COPD: SD {{show:eda.descriptives.copd.sd|.2f}} points, mean half-width {{show:eda.descriptives.copd.cdc_ci_halfwidth_mean|.2f}} points; {{show:eda.copd_interval_overlap.share_pairs_overlapping|.0%}} of county pairs have overlapping intervals; only {{show:eda.copd_interval_overlap.n_interval_excludes_median}} counties have an interval that excludes the median county's estimate.
- Decision: proceed with the approved question, and carry this measurement uncertainty into the interpretation. Say what was revised or added compared with the proposal (see Section 4 list).
- Disclose the exploratory work done on these data in September before the proposal (an OLS and Gamma model with four predictors, and an in-sample elastic net), as promised in the proposal.
:::

## 4. Methods and evaluation

{{table:design}}

::: write
**Rubric: "Methods, evaluation, and uncertainty" (2).** Roughly 400-500 words. Justify each choice in {{ref:design}}; do not just list it.

- Why ordinary least squares is the transparent starting point for Question 1, what the baseline is and why (smoking and inactivity are established behavioral correlates), and what the slopes mean (percentage-point difference in modeled COPD per one-point difference in the predictor).
- Three intervals and what each assumes: classical (equal variance, independent counties), HC3 (unequal variance, sensitive to leverage), county bootstrap with {{show:q1.n_boot}} resamples. Explain that the 100 counties are the whole state, so intervals describe conditional model uncertainty, not sampling of counties.
- Checks run before interpreting slopes, and why each matters here: collinearity, heteroskedasticity, influential counties, leave-one-county-out, spatial autocorrelation, leave-one-out prediction error.
- Complexity added only after a diagnosed need (say so explicitly, these were not in the proposal): composite social index and the re-expression of transport and housing as mean and difference (after the collinearity diagnosis); plausible-value draws from CDC's limits (after the audit finding in Section 3); 50 repeated partitions, the shuffled-outcome check and the west/central/east error slices for Question 3.
- Question 2 design: same counties in both models; one AQI variable at a time; the 300-day restriction is a sensitivity check, not an EPA standard.
- Question 3 evaluation design and leakage protection: nested cross-validation; the 80th-percentile cutoff, scaling and tuning computed from training counties only; average precision as the prespecified tuning metric; threshold fixed at 0.5; only held-out predictions scored. Explain why a shuffled outcome should give AUC near 0.5.
- State the random seed ({{show:q1.seed}}) and that results are deterministic.
:::

## 5. Results

### 5.1 What the county estimates look like

{{figure:fig01_copd_map}}

{{figure:fig03_correlations}}

{{figure:fig04_copd_vs_smoking}}

::: write
**Rubric: "Results and interpretation" (2), "Figures and tables" (2).** Roughly 150-200 words. Interpret the three figures; do not read values aloud.

- Where modeled COPD is high and low ({{ref:fig01_copd_map}}): highest {{show:eda.copd_highest.county}} {{show:eda.copd_highest.copd}}%, lowest {{show:eda.copd_lowest.county}} {{show:eda.copd_lowest.copd}}%; the urban/rural pattern you see.
- How strongly the seven indicators move together ({{ref:fig03_correlations}}): correlations from {{show:eda.corr_min|.2f}} to {{show:eda.corr_max|.2f}}; COPD with smoking {{show:eda.corr_copd.smoking|.2f}}; transport with housing {{show:eda.corr_transport_housing|.2f}}.
- What {{ref:fig04_copd_vs_smoking}} shows at the same time: the estimates line up almost perfectly, yet each one carries a wide CDC interval. Connect this to how PLACES estimates are produced (Section 2). Flag clearly what is a finding and what is your explanation for it.
:::

### 5.2 Question 1: social indicators and COPD

{{figure:fig05_copd_vs_social}}

{{table:q1_coefs}}

{{table:q1_fit}}

{{figure:fig06_q1_specifications}}

{{figure:fig07_q1_intervals}}

::: write
**Rubric: "Results and interpretation" (2), "Methods, evaluation, and uncertainty" (2).** Roughly 350-450 words. This is the main result; report all prespecified slopes, not only the "significant" ones.

- Before adjustment ({{ref:fig05_copd_vs_social}}): direction and strength of each social indicator with COPD.
- Baseline versus expanded fit ({{ref:q1_fit}}): R&sup2; {{show:q1.fit.copd.baseline.r2|.3f}} to {{show:q1.fit.copd.expanded.r2|.3f}}; smoking alone {{show:q1.smoking_only_r2|.3f}}; leave-one-out RMSE {{show:q1.fit.copd.baseline.loo_rmse|.3f}} to {{show:q1.fit.copd.expanded.loo_rmse|.3f}} points. Put the size of that gain next to the CDC half-width from Section 3.
- The adjusted slopes with all three intervals ({{ref:q1_coefs}}, {{ref:fig07_q1_intervals}}): transport {{show:q1.coefficients.copd.expanded.transport.estimate|+.3f}}, uninsured {{show:q1.coefficients.copd.expanded.uninsured.estimate|+.3f}}, housing {{show:q1.coefficients.copd.expanded.housing.estimate|+.3f}}. Say plainly where the three intervals agree and where they disagree about zero.
- What happens to each slope as adjustment variables are added ({{ref:fig06_q1_specifications}}): compare sign and size across the three rows for each indicator.
- Interpret in context: what a slope of this size would mean in percentage points, and whether the data can separate the three indicators from each other and from smoking.
:::

{{table:q1_checks}}

{{table:q1_sensitivity}}

{{table:q1_asthma}}

::: write
**Rubric: "Methods, evaluation, and uncertainty" (2), "Results and interpretation" (2).** Roughly 300-400 words.

- Collinearity ({{ref:q1_checks}}): VIF {{show:q1.vif.expanded.transport|.0f}} for transport and {{show:q1.vif.expanded.housing|.0f}} for housing; what that implies for reading their separate slopes.
- The re-expression in {{ref:q1_sensitivity}}: slope on the mean of transport and housing {{show:q1.transport_housing_pair.0.estimate|+.3f}}; slope on their difference {{show:q1.transport_housing_pair.1.estimate|+.3f}}. Explain in your own words what the expanded model is actually using.
- Influence: {{show:q1.most_influential_county}} County (Cook's distance {{show:q1.influence_top.0.cooks_d|.2f}}, leverage {{show:q1.influence_top.0.leverage|.2f}}); how the slopes change without it; why HC3 and bootstrap intervals differ here.
- Spatial dependence: Moran's I of expanded-model residuals {{show:q1.spatial.expanded model residuals.morans_i|.2f}}; what it means for every interval in this section (the proposal said inference would be qualified if dependence is material).
- Composite index and plausible-value draws: what each adds, and its stated assumption.
- Secondary outcome ({{ref:q1_asthma}}): what is similar and what differs for asthma (for example the uninsured slope {{show:q1.coefficients.asthma.expanded.uninsured.estimate|+.3f}} and the R&sup2; gain {{show:q1.comparison.asthma.delta_r2|.3f}}). Keep any explanation labelled as a possibility, not a finding.
- One-paragraph answer to Question 1 that is no broader than this evidence.
:::

### 5.3 Question 2: air quality in counties with records

{{table:aqi_coverage}}

{{table:q2_models}}

{{figure:fig09_q2_aqi}}

::: write
**Rubric: "Results and interpretation" (2).** Roughly 200-300 words.

- Who is in the subset ({{ref:aqi_coverage}}): the {{show:q2.n}} counties with AQI records are larger and have lower smoking and COPD estimates than the {{show:audit.n_without_aqi}} without; this comparison describes selection, it does not correct for it.
- Unadjusted association: r = {{show:q2.corr_copd_aqi_median|.2f}}, slope {{show:q2.unadjusted.estimate|+.3f}} points per AQI point. Note the direction.
- Adjusted association ({{ref:q2_models}}, {{ref:fig09_q2_aqi}}): slope {{show:q2.adjusted.estimate|+.4f}} per AQI point, that is {{show:q2.adjusted_per_10_aqi|+.2f}} points per 10 AQI points, with its intervals; F test p = {{show:q2.f_test.p|.2f}}.
- Why the sign of the raw association is not evidence that worse air goes with less COPD: median AQI rises with county size (correlation with log population {{show:q2.corr_logpop_aqi_median|.2f}}) while smoking falls.
- Sensitivity: share of days above 100; the {{show:q2.n_ge300}} counties with at least 300 monitored days.
- Limits specific to this question: 35 counties; same-year AQI is not lifetime exposure; counties monitor different pollutants and seasons (results/tables/q2_monitoring_profile.csv).
:::

### 5.4 Question 3: can the indicators flag high-COPD counties?

{{table:q3_cv}}

{{figure:fig10_q3_cv}}

{{table:q3_slices}}

::: write
**Rubric: "Methods, evaluation, and uncertainty" (2), "Results and interpretation" (2).** Roughly 200-300 words.

- Baseline versus penalized model on held-out counties ({{ref:q3_cv}}): ROC AUC {{show:q3.prespecified.baseline.roc_auc|.3f}} versus {{show:q3.prespecified.penalized.roc_auc|.3f}}; average precision {{show:q3.prespecified.baseline.average_precision|.3f}} versus {{show:q3.prespecified.penalized.average_precision|.3f}}; sensitivity {{show:q3.prespecified.baseline.sensitivity|.2f}} for both.
- How much results move across partitions ({{ref:fig10_q3_cv}}) and whether the difference between models is distinguishable from that variation.
- Leak check: shuffled-outcome AUC {{show:q3.shuffled_outcome.baseline.mean_roc_auc|.2f}}.
- Failure cases ({{ref:q3_slices}}): {{show:q3.prespecified.baseline.false_negatives}} high-COPD counties missed at threshold 0.5; where they are and what they have in common (their COPD and smoking values are in results/tables/q3_out_of_fold_predictions.csv). Compare with the leave-one-out errors by group in Appendix B.
- What this does and does not show: both the label and the predictors are PLACES estimates for the same year, so this is internal validation of how much information the indicators share, not a tool for finding unknown high-burden counties. About {{show:q3.n_high_statewide}} counties are "high", so fold-level numbers are unstable.
:::

## 6. Conclusion and limitations

::: write
**Rubric: "Limitations and responsible use" (1), "Question, purpose, and scope" (2), "Results and interpretation" (2).** Roughly 350-450 words.

- Answer each of the three questions in one or two sentences, for the stated audience, no broader than the evidence.
- What the evidence does not establish. Work through: ecological design (counties, not people); cross-sectional (no time order); PLACES values are model outputs that share inputs, so associations among them partly reflect the model; CDC's intervals are as wide as county differences and are not propagated by our intervals; spatial dependence; one influential county; AQI subset is not representative and AQI is not personal exposure; results describe North Carolina in 2023 only.
- Responsible use: what a county health planner should and should not do with these results (CDC cautions against using PLACES to evaluate programs or to rank overall health).
- Next steps that would actually address the limits, for example: direct (not modeled) measures of social needs or respiratory outcomes; demographic composition as a covariate; a spatial model; other years or states as an external check.
:::

## 7. Reproducibility and disclosure

**Reproduce from a clean start** (Python 3.12; commands run from the project folder):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_all.py
```

{{table:software}}

{{table:files}}

::: write
**Rubric: "Reproducibility, organization, and writing" (2).** Roughly 150-250 words plus the disclosure note.

- Data access: the analysis snapshot ships in data/raw with checksums; `run_all.py --refresh` re-downloads both sources into data/fresh and repeats the comparison.
- What is deterministic (seed {{show:q1.seed}}; fixed partitions) and how long a full run takes.
- Collaboration: who did what.
- Sources: point to the reference list.
- **AI disclosure, required by the syllabus:** name the tool, explain the assistance, identify what was incorporated, and state how the result was checked. The factual record is in team_notes/ai_use_log.md; write the note yourselves from it. The rubric does not allow AI-generated prose in this report, so every paragraph here must be the team's own writing.
:::

## References

1. Centers for Disease Control and Prevention. *PLACES: Local Data for Better Health, County Data, 2025 release* [dataset `swc5-untb`]. Retrieved 2026-09-28 from <https://data.cdc.gov/resource/swc5-untb.json?stateabbr=NC&$limit=50000>; metadata at <https://data.cdc.gov/api/views/swc5-untb.json>.
2. Centers for Disease Control and Prevention. *PLACES: Methodology.* <https://www.cdc.gov/places/methodology/index.html>. Accessed 2026-11-15.
3. Centers for Disease Control and Prevention. *PLACES: Measure Definitions.* <https://www.cdc.gov/places/measure-definitions/index.html>. Accessed 2026-11-15.
4. Centers for Disease Control and Prevention. *PLACES: Frequently Asked Questions.* <https://www.cdc.gov/places/faqs/index.html>. Accessed 2026-11-15.
5. Centers for Disease Control and Prevention. *About PLACES.* <https://www.cdc.gov/places/about/index.html>. Accessed 2026-11-15.
6. U.S. Environmental Protection Agency. *AirData pre-generated data files: Annual AQI by County, 2023* (`annual_aqi_by_county_2023.zip`). Retrieved 2026-09-28 from <https://aqs.epa.gov/aqsweb/airdata/download_files.html>.
7. U.S. Environmental Protection Agency. *About AirData Reports* (Air Quality Index Report definitions). <https://www.epa.gov/outdoor-air-quality-data/about-airdata-reports>. Accessed 2026-11-15.
8. U.S. Census Bureau. *Cartographic Boundary Files, 2023: Counties, 1:5,000,000* (`cb_2023_us_county_5m.zip`). Retrieved 2026-11-15 from <https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip>; description at <https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html>.

## Appendix A. All audit checks

{{table:audit_full}}

## Appendix B. Diagnostics for the expanded COPD model

{{figure:fig08_q1_diagnostics}}

{{table:q1_influence}}

{{table:q1_slices}}

## Appendix C. Question 3 by fold

{{table:q3_folds}}
