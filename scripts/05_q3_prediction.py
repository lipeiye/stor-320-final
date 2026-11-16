"""Step 05 - Question 3 (predictive extension): flagging counties with high modeled COPD.

Task        classify a county as "high COPD": at or above the 80th percentile
Predictors  smoking, inactivity, transport, uninsured, housing (no disease outcome is a predictor)
Models      baseline   logistic regression on smoking only, no penalty
            penalized  logistic regression on all five predictors with an L1 or elastic-net
                       penalty; strength and mixing chosen by inner cross-validation
                       on average precision

Evaluation  nested cross-validation, 5 stratified outer folds x 5 inner folds.
            Inside every training set, and only there, the script computes
              - the 80th-percentile COPD cutoff that defines "high",
              - the centring and scaling of the predictors,
              - the penalty settings.
            Held-out counties are labelled and scored with those training quantities.
            Sensitivity and false negatives use a fixed probability threshold of 0.5.

Stability   the whole procedure is repeated on 50 other random partitions.
Leak check  the same procedure on shuffled COPD values, which should score at chance.

Outputs: results/tables/q3_*.csv, figure fig10, section "q3" of results.json
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import plotstyle as P

N_OUTER, N_INNER = 5, 5
GRID = [(c, l1) for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0) for l1 in (0.5, 1.0)]
THRESHOLD = 0.5
N_REPEATS = 50
N_PERMUTATIONS = 100
METRICS = ["roc_auc", "average_precision", "sensitivity"]
MODELS = ["baseline", "penalized"]


def high(values: np.ndarray, cutoff: float) -> np.ndarray:
    return (values >= cutoff).astype(int)


def balance_labels(values: np.ndarray) -> np.ndarray:
    """Top-quintile flag used only to keep the folds balanced, never as a training label."""
    return high(values, np.quantile(values, C.HIGH_QUANTILE))


def penalized(c: float, l1_ratio: float):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, l1_ratio=l1_ratio, solver="saga",
                                            max_iter=10000, random_state=C.SEED))


def baseline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))


def tune(X: np.ndarray, copd: np.ndarray, seed: int) -> tuple[float, float]:
    """Choose the penalty by inner cross-validation, using only the outer training counties."""
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    scores = {g: [] for g in GRID}
    for tr, va in inner.split(X, balance_labels(copd)):
        cutoff = np.quantile(copd[tr], C.HIGH_QUANTILE)            # inner training counties only
        y_tr, y_va = high(copd[tr], cutoff), high(copd[va], cutoff)
        if y_va.min() == y_va.max() or y_tr.min() == y_tr.max():
            continue                                               # no positives to rank
        for c, l1 in GRID:
            p = penalized(c, l1).fit(X[tr], y_tr).predict_proba(X[va])[:, 1]
            scores[(c, l1)].append(average_precision_score(y_va, p))
    mean = {g: float(np.mean(v)) for g, v in scores.items()}
    best = max(mean.values())
    # ties: prefer the stronger penalty (smaller C), then the sparser one (larger l1_ratio)
    return sorted((g for g, m in mean.items() if m >= best - 1e-12), key=lambda g: (g[0], -g[1]))[0]


def score(y: np.ndarray, p: np.ndarray) -> dict:
    flagged, positive = p >= THRESHOLD, y == 1
    n_pos = int(positive.sum())
    return {
        "n_test": len(y), "n_high": n_pos,
        "false_negatives": int((~flagged & positive).sum()),
        "false_positives": int((flagged & ~positive).sum()),
        "roc_auc": roc_auc_score(y, p) if 0 < n_pos < len(y) else np.nan,
        "average_precision": average_precision_score(y, p) if n_pos else np.nan,
        "sensitivity": float((flagged & positive).sum() / n_pos) if n_pos else np.nan,
    }


def nested_cv(X: np.ndarray, copd: np.ndarray, split_seed: int):
    """One full nested cross-validation. Returns fold metrics, held-out predictions, fitted details."""
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=split_seed)
    folds, predictions, details = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X, balance_labels(copd)), start=1):
        assert not set(tr) & set(te), "a county appears in both training and held-out sets"
        cutoff = np.quantile(copd[tr], C.HIGH_QUANTILE)            # outer training counties only
        y_tr, y_te = high(copd[tr], cutoff), high(copd[te], cutoff)

        c, l1 = tune(X[tr], copd[tr], seed=split_seed + fold)
        fitted = {"baseline": baseline().fit(X[tr][:, :1], y_tr),   # column 0 is smoking
                  "penalized": penalized(c, l1).fit(X[tr], y_tr)}
        probs = {"baseline": fitted["baseline"].predict_proba(X[te][:, :1])[:, 1],
                 "penalized": fitted["penalized"].predict_proba(X[te])[:, 1]}
        for name in MODELS:
            folds.append({"fold": fold, "model": name, "cutoff": cutoff, "n_train": len(tr),
                          "n_train_high": int(y_tr.sum()), **score(y_te, probs[name])})
        predictions.append(pd.DataFrame({"row": te, "fold": fold, "cutoff": cutoff, "high": y_te,
                                         "p_baseline": probs["baseline"], "p_penalized": probs["penalized"]}))
        details.append({"fold": fold, "C": c, "l1_ratio": l1,
                        "coef": fitted["penalized"][-1].coef_[0].copy()})
    return pd.DataFrame(folds), pd.concat(predictions, ignore_index=True), details


def fold_means(folds: pd.DataFrame) -> pd.DataFrame:
    return folds.groupby("model")[METRICS].mean().loc[MODELS]


# --------------------------------------------------------------------------
def fig_cv(repeats: pd.DataFrame, primary: pd.DataFrame) -> None:
    titles = {"roc_auc": "ROC AUC", "average_precision": "Average precision",
              "sensitivity": f"Sensitivity at probability {THRESHOLD}"}
    labels = {"baseline": "Smoking only", "penalized": "Five indicators,\npenalized"}
    colors = {"baseline": P.BLUE, "penalized": P.ORANGE}
    jitter = np.random.default_rng(C.SEED).uniform(-0.2, 0.2, size=N_REPEATS)
    low = np.floor(10 * min(repeats[METRICS].min().min(), primary[METRICS].min().min())) / 10

    def draw(fig, variant):
        # three metric panels, and a thin row underneath that only holds the legend
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.09])
        axes = [fig.add_subplot(grid[0, 0])]
        axes += [fig.add_subplot(grid[0, k], sharex=axes[0], sharey=axes[0]) for k in (1, 2)]
        legend_ax = fig.add_subplot(grid[1, :])
        legend_ax.axis("off")
        for ax, metric in zip(axes, METRICS):
            for y, name in zip((1, 0), MODELS):
                r = repeats.loc[repeats["model"].eq(name), metric].to_numpy()
                ax.scatter(r, y + jitter, s=30 if variant == "report" else 38, color=colors[name],
                           alpha=0.55, linewidth=0, zorder=3)
                ax.scatter([primary.loc[name, metric]], [y], marker="D", s=95 if variant == "report" else 120,
                           color=P.INK, edgecolor=P.SURFACE, linewidth=1.3, zorder=5)
            ax.set_title(titles[metric] if variant == "report" else titles[metric].replace("probability ", ""),
                         fontweight="normal")
            ax.set_yticks([1, 0], [labels[m] for m in MODELS])
            ax.tick_params(labelleft=ax is axes[0])
            ax.set_ylim(-0.6, 1.6)
            ax.set_xlim(low, 1.015)
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
        axes[1].set_xlabel("Mean over the five held-out folds")
        handles = [Line2D([], [], marker="o", linestyle="none", color=P.MUTED, markersize=7,
                          label=f"One random partition ({N_REPEATS} per model)"),
                   Line2D([], [], marker="D", linestyle="none", color=P.INK, markeredgecolor=P.SURFACE,
                          markersize=10, label="Prespecified partition")]
        legend_ax.legend(handles=handles, loc="center", ncols=2, handletextpad=0.3, columnspacing=2.5,
                         borderaxespad=0)

    P.render("fig10_q3_cv", draw,
             title="Held-out performance for flagging high-COPD counties",
             subtitle="Nested 5-fold cross-validation; cutoff, scaling and tuning use training counties only; n = 100.",
             source=P.SOURCE_PLACES, size=(11.6, 4.7))


def main() -> None:
    C.ensure_dirs()
    d = pd.read_csv(C.PROCESSED_TABLE, dtype={"fips": str})
    X = d[C.ALL_PREDICTORS].to_numpy()
    copd = d[C.OUTCOME].to_numpy()
    assert C.ALL_PREDICTORS[0] == "smoking" and C.OUTCOME not in C.ALL_PREDICTORS

    # ---- prespecified partition -------------------------------------------
    folds, preds, details = nested_cv(X, copd, split_seed=C.SEED)
    folds = folds.merge(pd.DataFrame([{k: v for k, v in x.items() if k != "coef"} for x in details]), on="fold")
    folds.loc[folds["model"].eq("baseline"), ["C", "l1_ratio"]] = np.nan
    folds.round(5).to_csv(C.TABLES / "q3_cv_folds.csv", index=False)
    primary = fold_means(folds)
    primary_sd = folds.groupby("model")[METRICS].std(ddof=1).loc[MODELS]

    preds = preds.assign(county=d.loc[preds["row"], "county"].to_numpy(),
                         population=d.loc[preds["row"], "population"].to_numpy(),
                         lon=d.loc[preds["row"], "lon"].to_numpy(),
                         copd=copd[preds["row"]], smoking=d.loc[preds["row"], "smoking"].to_numpy())
    for name in MODELS:
        preds[f"flag_{name}"] = (preds[f"p_{name}"] >= THRESHOLD).astype(int)
    preds = preds.drop(columns="row").sort_values(["fold", "county"]).reset_index(drop=True)
    preds.round(5).to_csv(C.TABLES / "q3_out_of_fold_predictions.csv", index=False)

    groupings = {
        "population third": pd.qcut(preds["population"], 3, labels=["smallest third", "middle third", "largest third"]),
        "longitude third": pd.qcut(preds["lon"], 3, labels=["western third", "central third", "eastern third"]),
    }
    slice_rows = []
    for slice_name, labels in groupings.items():
        for level, g in preds.groupby(labels, observed=True):
            row = {"slice": slice_name, "group": level, "n": len(g), "n_high": int(g["high"].sum())}
            for m in MODELS:
                row[f"false_negatives_{m}"] = int(((g[f"flag_{m}"] == 0) & (g["high"] == 1)).sum())
                row[f"false_positives_{m}"] = int(((g[f"flag_{m}"] == 1) & (g["high"] == 0)).sum())
            slice_rows.append(row)
    slices = pd.DataFrame(slice_rows)
    slices.to_csv(C.TABLES / "q3_slices.csv", index=False)

    # ---- 50 other partitions: how much do the results move? ------------------
    rows, chosen, nonzero = [], Counter(), np.zeros(len(C.ALL_PREDICTORS))
    for r in range(1, N_REPEATS + 1):
        f, _, det = nested_cv(X, copd, split_seed=C.SEED + r)
        rows.append(fold_means(f).assign(repeat=r, n_undefined_folds=int(f[METRICS].isna().any(axis=1).sum())))
        for x in det:
            chosen[(x["C"], x["l1_ratio"])] += 1
            nonzero += (np.abs(x["coef"]) > 1e-8)
    repeats = pd.concat(rows).reset_index()
    repeats.round(5).to_csv(C.TABLES / "q3_repeats.csv", index=False)
    n_fits = N_REPEATS * N_OUTER

    wide = repeats.pivot(index="repeat", columns="model", values=METRICS)
    summary = []
    for metric in METRICS:
        for name in MODELS:
            v = wide[(metric, name)]
            summary.append({"metric": metric, "quantity": name,
                            "prespecified_partition": primary.loc[name, metric],
                            "sd_across_its_folds": primary_sd.loc[name, metric],
                            "repeats_mean": v.mean(), "repeats_p2_5": v.quantile(0.025),
                            "repeats_p97_5": v.quantile(0.975)})
        diff = wide[(metric, "penalized")] - wide[(metric, "baseline")]
        summary.append({"metric": metric, "quantity": "penalized minus baseline",
                        "prespecified_partition": primary.loc["penalized", metric] - primary.loc["baseline", metric],
                        "repeats_mean": diff.mean(), "repeats_p2_5": diff.quantile(0.025),
                        "repeats_p97_5": diff.quantile(0.975),
                        "share_repeats_penalized_better": float((diff > 1e-12).mean()),
                        "share_repeats_tied": float((diff.abs() <= 1e-12).mean()),
                        "share_repeats_baseline_better": float((diff < -1e-12).mean())})
    summary = pd.DataFrame(summary)
    summary.round(5).to_csv(C.TABLES / "q3_cv_summary.csv", index=False)

    tuning = pd.DataFrame([{"C": c, "l1_ratio": l1, "times_chosen": n, "share": n / n_fits}
                           for (c, l1), n in sorted(chosen.items())])
    tuning.round(4).to_csv(C.TABLES / "q3_tuning_choices.csv", index=False)
    kept = pd.DataFrame({"predictor": C.ALL_PREDICTORS, "share_of_fits_with_nonzero_coefficient": nonzero / n_fits})
    kept.round(4).to_csv(C.TABLES / "q3_predictors_kept.csv", index=False)

    # ---- leak check: shuffled outcome must score at chance --------------------
    rng = np.random.default_rng(C.SEED)
    shuffled = []
    for _ in range(N_PERMUTATIONS):
        f, _, _ = nested_cv(X, rng.permutation(copd), split_seed=C.SEED)
        shuffled.append(fold_means(f)["roc_auc"])
    shuffled = pd.DataFrame(shuffled).reset_index(drop=True)
    leak = pd.DataFrame([{"model": m, "permutations": N_PERMUTATIONS, "mean_roc_auc": shuffled[m].mean(),
                          "p2_5": shuffled[m].quantile(0.025), "p97_5": shuffled[m].quantile(0.975),
                          "observed_roc_auc": primary.loc[m, "roc_auc"],
                          "share_shuffled_at_or_above_observed": float((shuffled[m] >= primary.loc[m, "roc_auc"]).mean())}
                         for m in MODELS])
    leak.round(5).to_csv(C.TABLES / "q3_shuffled_outcome_check.csv", index=False)

    fig_cv(repeats, primary)

    pd.set_option("display.width", 220)
    print(folds.round(3).to_string(index=False))
    print("\n", summary.round(3).to_string(index=False))
    print("\n", tuning.to_string(index=False), "\n\n", kept.to_string(index=False))
    print("\n", leak.round(3).to_string(index=False))
    print("\n", slices.to_string(index=False))
    missed = preds[(preds["high"] == 1) & ((preds["flag_baseline"] == 0) | (preds["flag_penalized"] == 0))]
    print("\nHigh-COPD counties missed at threshold 0.5 by either model:\n",
          missed[["county", "fold", "cutoff", "copd", "smoking", "p_baseline", "p_penalized"]].round(3).to_string(index=False))

    statewide_cut = float(np.quantile(copd, C.HIGH_QUANTILE))
    C.update_results("q3", {
        "n": len(d), "n_outer": N_OUTER, "n_inner": N_INNER, "n_repeats": N_REPEATS,
        "n_permutations": N_PERMUTATIONS, "threshold": THRESHOLD, "grid_size": len(GRID),
        "statewide_cutoff": statewide_cut, "n_high_statewide": int((copd >= statewide_cut).sum()),
        "fold_cutoffs": sorted(set(folds["cutoff"].round(3))),
        "fold_n_high": folds[folds["model"].eq("baseline")]["n_high"].tolist(),
        "prespecified": {m: {**primary.loc[m].round(5).to_dict(),
                             "false_negatives": int(folds.loc[folds["model"].eq(m), "false_negatives"].sum()),
                             "false_positives": int(folds.loc[folds["model"].eq(m), "false_positives"].sum())}
                         for m in MODELS},
        "prespecified_fold_sd": {m: primary_sd.loc[m].round(5).to_dict() for m in MODELS},
        "n_high_held_out_total": int(folds.loc[folds["model"].eq("baseline"), "n_high"].sum()),
        "summary": summary.round(5).to_dict(orient="records"),
        "tuning_most_common": tuning.sort_values("times_chosen", ascending=False).head(3).to_dict(orient="records"),
        "predictors_kept": kept.set_index("predictor").iloc[:, 0].round(4).to_dict(),
        "shuffled_outcome": leak.set_index("model").round(5).to_dict(orient="index"),
        "missed_counties": missed[["county", "copd", "cutoff", "p_baseline", "p_penalized"]].round(4).to_dict(orient="records"),
        "slices": slices.to_dict(orient="records"),
        "undefined_folds_in_repeats": int(repeats["n_undefined_folds"].sum() / len(MODELS)),
    })


if __name__ == "__main__":
    main()
