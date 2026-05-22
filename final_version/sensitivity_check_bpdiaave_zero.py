#!/usr/bin/env python3
"""Sensitivity check for the four BPDiaAve = 0 rows.

The final analysis keeps the four rows with impossible diastolic blood pressure
values and reports them as a data-quality limitation. This script reruns the
main numerical analyses after dropping those four rows and writes side-by-side
comparison tables under final_version/outputs/sensitivity.
"""
from __future__ import annotations

import os
import sys
import warnings
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "final_version"
VENDOR_DIR = FINAL_DIR / "vendor"
OUTPUT_DIR = FINAL_DIR / "outputs" / "sensitivity"

CACHE_DIR = FINAL_DIR / "_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
from kmodes.kmodes import KModes


DATA_PATH = PROJECT_ROOT / "data" / "nhanes_health.csv"

PCA_COLS = [
    "Age",
    "Poverty",
    "Weight",
    "Height",
    "BMI",
    "Pulse",
    "BPSysAve",
    "BPDiaAve",
    "DirectChol",
    "TotChol",
    "AlcoholYear",
    "SleepHrsNight",
]

CLUSTER_FEATURES = [
    "Gender",
    "Race1",
    "Education",
    "MaritalStatus",
    "HHIncome",
    "HomeOwn",
    "HealthGen",
    "PhysActive",
    "SmokeNow",
]
OUTCOME_VARS = ["Diabetes", "HighBP"]
BEST_K = 4
N_INIT = 10
RAND_SEED = 42

LOGIT_NUMERIC_COLS = ["Age", "BMI", "DirectChol", "TotChol", "Poverty"]
LOGIT_CATEGORICAL_COLS = ["Gender", "HighBP", "PhysActive", "SmokeNow"]
LOGIT_CATEGORICAL_LEVELS = [
    ["female", "male"],
    ["Normal", "High"],
    ["No", "Yes"],
    ["No", "Yes"],
]
LOGIT_FEATURES = LOGIT_NUMERIC_COLS + LOGIT_CATEGORICAL_COLS
LOGIT_TARGET = "Diabetes"
LOGIT_THRESHOLD = 0.5
LOGIT_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_logit_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), LOGIT_NUMERIC_COLS),
            (
                "categorical",
                OneHotEncoder(
                    categories=LOGIT_CATEGORICAL_LEVELS,
                    drop="first",
                    handle_unknown="ignore",
                ),
                LOGIT_CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                ),
            ),
        ]
    )


def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_PATH)
    baseline = raw.drop_duplicates().reset_index(drop=True)
    impossible = baseline[baseline["BPDiaAve"].eq(0)].copy()
    sensitivity = baseline[~baseline["BPDiaAve"].eq(0)].reset_index(drop=True)
    return baseline, impossible, sensitivity


def analysis_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "All": df.reset_index(drop=True),
        "White": df[df["Race1"].eq("White")].reset_index(drop=True),
        "Black": df[df["Race1"].eq("Black")].reset_index(drop=True),
    }


def race_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "White": df[df["Race1"].eq("White")].reset_index(drop=True),
        "Black": df[df["Race1"].eq("Black")].reset_index(drop=True),
    }


def descriptive_summary(label: str, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for race, data in race_subsets(df).items():
        records.append(
            {
                "analysis": label,
                "race": race,
                "n": len(data),
                "diabetes_yes_n": int(data["Diabetes"].eq("Yes").sum()),
                "diabetes_yes_pct": data["Diabetes"].eq("Yes").mean() * 100,
                "highbp_high_n": int(data["HighBP"].eq("High").sum()),
                "highbp_high_pct": data["HighBP"].eq("High").mean() * 100,
                "mean_bpsysave": data["BPSysAve"].mean(),
                "mean_bpdiaave": data["BPDiaAve"].mean(),
                "mean_bmi": data["BMI"].mean(),
                "mean_age": data["Age"].mean(),
            }
        )
    return pd.DataFrame(records)


def run_pca(label: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    variance_records = []
    loadings_by_subset: dict[str, pd.DataFrame] = {}
    for subset_name, data in analysis_subsets(df).items():
        x = StandardScaler().fit_transform(data[PCA_COLS])
        pca = PCA()
        pca.fit(x)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=PCA_COLS,
            columns=[f"PC{i + 1}" for i in range(len(PCA_COLS))],
        )
        loadings_by_subset[subset_name] = loadings
        expl = pca.explained_variance_ratio_
        cum = np.cumsum(expl)
        variance_records.append(
            {
                "analysis": label,
                "subset": subset_name,
                "n": len(data),
                "pc1_variance_pct": expl[0] * 100,
                "pc1_pc2_cumulative_pct": cum[1] * 100,
                "pc1_pc4_cumulative_pct": cum[3] * 100,
            }
        )
    return pd.DataFrame(variance_records), loadings_by_subset


def max_aligned_pca_loading_change(
    baseline_loadings: dict[str, pd.DataFrame],
    sensitivity_loadings: dict[str, pd.DataFrame],
    n_components: int = 4,
) -> pd.DataFrame:
    records = []
    for subset_name in baseline_loadings:
        base = baseline_loadings[subset_name].iloc[:, :n_components].copy()
        sens = sensitivity_loadings[subset_name].iloc[:, :n_components].copy()
        for col in base.columns:
            if float(np.dot(base[col], sens[col])) < 0:
                sens[col] = -sens[col]
        diff = (sens - base).abs()
        max_location = diff.stack().idxmax()
        records.append(
            {
                "subset": subset_name,
                "max_abs_loading_delta_pc1_pc4": float(diff.to_numpy().max()),
                "variable": max_location[0],
                "component": max_location[1],
            }
        )
    return pd.DataFrame(records)


def run_factor_analysis(label: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records = []
    communality_records = []
    for subset_name, data in analysis_subsets(df).items():
        x = StandardScaler().fit_transform(data[PCA_COLS])
        fa = FactorAnalysis(n_components=4, rotation="varimax", random_state=42)
        fa.fit(x)
        communalities = pd.Series(1 - fa.noise_variance_, index=PCA_COLS)
        summary_records.append(
            {
                "analysis": label,
                "subset": subset_name,
                "n": len(data),
                "mean_communality": communalities.mean(),
                "min_communality": communalities.min(),
                "min_communality_variable": communalities.idxmin(),
                "bpdiaave_communality": communalities["BPDiaAve"],
            }
        )
        for variable, communality in communalities.items():
            communality_records.append(
                {
                    "analysis": label,
                    "subset": subset_name,
                    "variable": variable,
                    "communality": communality,
                }
            )
    return pd.DataFrame(summary_records), pd.DataFrame(communality_records)


def run_kmodes(label: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_records = []
    mode_records = []
    outcome_records = []
    for race, data in race_subsets(df).items():
        x = data[CLUSTER_FEATURES].copy()
        costs = {}
        for k in range(2, 7):
            km_tmp = KModes(n_clusters=k, init="Huang", n_init=N_INIT, random_state=RAND_SEED)
            km_tmp.fit(x.values)
            costs[k] = float(km_tmp.cost_)
            cost_records.append({"analysis": label, "race": race, "k": k, "cost": costs[k]})

        km = KModes(n_clusters=BEST_K, init="Huang", n_init=N_INIT, random_state=RAND_SEED)
        labels = km.fit_predict(x.values)
        cluster_names = pd.Series([f"C{i + 1}" for i in labels], name="cluster")

        modes = pd.DataFrame(km.cluster_centroids_, columns=CLUSTER_FEATURES)
        modes.insert(0, "cluster", [f"C{i + 1}" for i in range(BEST_K)])
        modes.insert(0, "race", race)
        modes.insert(0, "analysis", label)
        mode_records.extend(modes.to_dict("records"))

        outcomes = data[OUTCOME_VARS].copy()
        outcomes["cluster"] = cluster_names
        for cluster in sorted(outcomes["cluster"].unique()):
            sub = outcomes[outcomes["cluster"].eq(cluster)]
            outcome_records.append(
                {
                    "analysis": label,
                    "race": race,
                    "cluster": cluster,
                    "n": len(sub),
                    "pct_of_race": len(sub) / len(outcomes) * 100,
                    "diabetes_yes_pct": sub["Diabetes"].eq("Yes").mean() * 100,
                    "highbp_high_pct": sub["HighBP"].eq("High").mean() * 100,
                }
            )
    return (
        pd.DataFrame(cost_records),
        pd.DataFrame(mode_records),
        pd.DataFrame(outcome_records),
    )


def hamming_distance(left: pd.Series, right: pd.Series) -> int:
    return int((left[CLUSTER_FEATURES].astype(str) != right[CLUSTER_FEATURES].astype(str)).sum())


def align_kmodes_clusters(
    modes: pd.DataFrame,
    outcomes: pd.DataFrame,
    sensitivity_label: str = "drop_bpdiaave_zero",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align sensitivity cluster labels to baseline labels by centroid modes.

    KModes can return equivalent or near-equivalent clusters in a different
    order. With k=4, a brute-force assignment over 24 permutations is clearer
    than pulling in another dependency.
    """
    aligned_modes = modes.copy()
    aligned_outcomes = outcomes.copy()
    alignment_records = []

    for race in sorted(modes["race"].unique()):
        base = (
            modes[modes["analysis"].eq("baseline") & modes["race"].eq(race)]
            .set_index("cluster")
            .sort_index()
        )
        sens = (
            modes[modes["analysis"].eq(sensitivity_label) & modes["race"].eq(race)]
            .set_index("cluster")
            .sort_index()
        )
        base_clusters = list(base.index)
        sens_clusters = list(sens.index)

        best_total = None
        best_mapping = None
        for perm in permutations(base_clusters):
            mapping = dict(zip(sens_clusters, perm))
            total = sum(hamming_distance(base.loc[mapping[s_cluster]], sens.loc[s_cluster]) for s_cluster in sens_clusters)
            if best_total is None or total < best_total:
                best_total = total
                best_mapping = mapping

        if best_mapping is None:
            raise RuntimeError(f"Could not align k-modes clusters for {race}")

        for sens_cluster, baseline_cluster in best_mapping.items():
            dist = hamming_distance(base.loc[baseline_cluster], sens.loc[sens_cluster])
            alignment_records.append(
                {
                    "race": race,
                    "sensitivity_cluster_original": sens_cluster,
                    "baseline_cluster_aligned": baseline_cluster,
                    "mode_hamming_distance": dist,
                }
            )

        mode_mask = aligned_modes["analysis"].eq(sensitivity_label) & aligned_modes["race"].eq(race)
        outcome_mask = aligned_outcomes["analysis"].eq(sensitivity_label) & aligned_outcomes["race"].eq(race)
        aligned_modes.loc[mode_mask, "cluster_original"] = aligned_modes.loc[mode_mask, "cluster"]
        aligned_outcomes.loc[outcome_mask, "cluster_original"] = aligned_outcomes.loc[outcome_mask, "cluster"]
        aligned_modes.loc[mode_mask, "cluster"] = aligned_modes.loc[mode_mask, "cluster"].map(best_mapping)
        aligned_outcomes.loc[outcome_mask, "cluster"] = aligned_outcomes.loc[outcome_mask, "cluster"].map(best_mapping)

    aligned_modes["cluster_original"] = aligned_modes["cluster_original"].fillna(aligned_modes["cluster"])
    aligned_outcomes["cluster_original"] = aligned_outcomes["cluster_original"].fillna(aligned_outcomes["cluster"])
    alignment = pd.DataFrame(alignment_records)
    return aligned_modes, aligned_outcomes, alignment


def run_logit(label: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_records = []
    confusion_records = []
    coef_records = []
    for race, data in race_subsets(df).items():
        x = data[LOGIT_FEATURES].copy()
        y = data[LOGIT_TARGET].eq("Yes").astype(int)
        class_counts = y.value_counts().reindex([0, 1], fill_value=0)
        if class_counts.min() < LOGIT_CV.n_splits:
            raise ValueError(
                f"{race} has too few rows in a diabetes class for "
                f"{LOGIT_CV.n_splits}-fold CV: {class_counts.to_dict()}"
            )

        pipe = make_logit_pipeline()
        y_prob = cross_val_predict(pipe, x, y, cv=LOGIT_CV, method="predict_proba")[:, 1]
        y_pred = (y_prob >= LOGIT_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        performance_records.append(
            {
                "analysis": label,
                "race": race,
                "n": len(data),
                "diabetes_yes_n": int(y.sum()),
                "diabetes_yes_pct": y.mean() * 100,
                "roc_auc": roc_auc_score(y, y_prob),
                "accuracy_at_0_5": accuracy_score(y, y_pred),
                "sensitivity_at_0_5": sensitivity,
                "specificity_at_0_5": specificity,
                "predicted_positive_pct_at_0_5": y_pred.mean() * 100,
            }
        )
        confusion_records.extend(
            [
                {
                    "analysis": label,
                    "race": race,
                    "true_class": "No",
                    "predicted_class": "No",
                    "n": int(tn),
                },
                {
                    "analysis": label,
                    "race": race,
                    "true_class": "No",
                    "predicted_class": "Yes",
                    "n": int(fp),
                },
                {
                    "analysis": label,
                    "race": race,
                    "true_class": "Yes",
                    "predicted_class": "No",
                    "n": int(fn),
                },
                {
                    "analysis": label,
                    "race": race,
                    "true_class": "Yes",
                    "predicted_class": "Yes",
                    "n": int(tp),
                },
            ]
        )

        final_model = make_logit_pipeline().fit(x, y)
        feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
        coefs = final_model.named_steps["model"].coef_[0]
        for feature, coef in zip(feature_names, coefs):
            coef_records.append(
                {
                    "analysis": label,
                    "race": race,
                    "feature": feature,
                    "coefficient": coef,
                    "odds_ratio": np.exp(coef),
                }
            )
    return (
        pd.DataFrame(performance_records),
        pd.DataFrame(confusion_records),
        pd.DataFrame(coef_records),
    )


def add_delta(
    df: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    baseline_label: str = "baseline",
    sensitivity_label: str = "drop_bpdiaave_zero",
) -> pd.DataFrame:
    base = df[df["analysis"].eq(baseline_label)][key_cols + value_cols].copy()
    sens = df[df["analysis"].eq(sensitivity_label)][key_cols + value_cols].copy()
    merged = base.merge(sens, on=key_cols, suffixes=("_baseline", "_drop_zero"))
    for col in value_cols:
        merged[f"{col}_delta"] = merged[f"{col}_drop_zero"] - merged[f"{col}_baseline"]
    return merged


def format_report_table(df: pd.DataFrame) -> str:
    rounded = df.copy()
    for col in rounded.select_dtypes(include=[np.number]).columns:
        rounded[col] = rounded[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    return rounded.to_markdown(index=False)


def main() -> None:
    baseline, impossible, sensitivity = load_analysis_data()

    summary = pd.DataFrame(
        [
            {"metric": "baseline_deduplicated_rows", "value": len(baseline)},
            {"metric": "bpdiaave_zero_rows_removed", "value": len(impossible)},
            {"metric": "sensitivity_rows", "value": len(sensitivity)},
            {
                "metric": "removed_rows_share_pct",
                "value": len(impossible) / len(baseline) * 100,
            },
        ]
    )
    removed_profile = (
        impossible[
            [
                "Race1",
                "Gender",
                "Age",
                "BMI",
                "BPSysAve",
                "BPDiaAve",
                "HighBP",
                "Diabetes",
                "Poverty",
            ]
        ]
        .sort_values(["Race1", "Diabetes", "Age"])
        .reset_index(drop=True)
    )

    desc = pd.concat(
        [
            descriptive_summary("baseline", baseline),
            descriptive_summary("drop_bpdiaave_zero", sensitivity),
        ],
        ignore_index=True,
    )
    desc_delta = add_delta(
        desc,
        ["race"],
        [
            "n",
            "diabetes_yes_n",
            "diabetes_yes_pct",
            "highbp_high_n",
            "highbp_high_pct",
            "mean_bpsysave",
            "mean_bpdiaave",
            "mean_bmi",
            "mean_age",
        ],
    )

    pca_base, pca_loadings_base = run_pca("baseline", baseline)
    pca_sens, pca_loadings_sens = run_pca("drop_bpdiaave_zero", sensitivity)
    pca = pd.concat([pca_base, pca_sens], ignore_index=True)
    pca_delta = add_delta(
        pca,
        ["subset"],
        ["n", "pc1_variance_pct", "pc1_pc2_cumulative_pct", "pc1_pc4_cumulative_pct"],
    )
    pca_loading_delta = max_aligned_pca_loading_change(pca_loadings_base, pca_loadings_sens)

    fa_summary_base, fa_comm_base = run_factor_analysis("baseline", baseline)
    fa_summary_sens, fa_comm_sens = run_factor_analysis("drop_bpdiaave_zero", sensitivity)
    fa_summary = pd.concat([fa_summary_base, fa_summary_sens], ignore_index=True)
    fa_summary_delta = add_delta(
        fa_summary,
        ["subset"],
        ["n", "mean_communality", "min_communality", "bpdiaave_communality"],
    )
    fa_comm = pd.concat([fa_comm_base, fa_comm_sens], ignore_index=True)
    fa_comm_delta = add_delta(fa_comm, ["subset", "variable"], ["communality"])
    max_fa_comm_delta = (
        fa_comm_delta.assign(abs_delta=lambda x: x["communality_delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(10)
        .drop(columns=["abs_delta"])
    )

    km_cost_base, km_modes_base, km_out_base = run_kmodes("baseline", baseline)
    km_cost_sens, km_modes_sens, km_out_sens = run_kmodes("drop_bpdiaave_zero", sensitivity)
    km_cost = pd.concat([km_cost_base, km_cost_sens], ignore_index=True)
    km_modes = pd.concat([km_modes_base, km_modes_sens], ignore_index=True)
    km_outcomes = pd.concat([km_out_base, km_out_sens], ignore_index=True)
    km_modes_aligned, km_outcomes_aligned, km_alignment = align_kmodes_clusters(
        km_modes, km_outcomes
    )
    km_outcome_delta = add_delta(
        km_outcomes_aligned,
        ["race", "cluster"],
        ["n", "pct_of_race", "diabetes_yes_pct", "highbp_high_pct"],
    )

    logit_perf_base, logit_conf_base, logit_coef_base = run_logit("baseline", baseline)
    logit_perf_sens, logit_conf_sens, logit_coef_sens = run_logit(
        "drop_bpdiaave_zero", sensitivity
    )
    logit_perf = pd.concat([logit_perf_base, logit_perf_sens], ignore_index=True)
    logit_perf_delta = add_delta(
        logit_perf,
        ["race"],
        [
            "n",
            "diabetes_yes_n",
            "diabetes_yes_pct",
            "roc_auc",
            "accuracy_at_0_5",
            "sensitivity_at_0_5",
            "specificity_at_0_5",
            "predicted_positive_pct_at_0_5",
        ],
    )
    logit_conf = pd.concat([logit_conf_base, logit_conf_sens], ignore_index=True)
    logit_coef = pd.concat([logit_coef_base, logit_coef_sens], ignore_index=True)
    logit_coef_delta = add_delta(logit_coef, ["race", "feature"], ["coefficient", "odds_ratio"])
    largest_logit_coef_delta = (
        logit_coef_delta.assign(abs_delta=lambda x: x["coefficient_delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(12)
        .drop(columns=["abs_delta"])
    )

    # Full-precision tables for auditability.
    summary.to_csv(OUTPUT_DIR / "bpdiaave_zero_summary.csv", index=False)
    removed_profile.to_csv(OUTPUT_DIR / "bpdiaave_zero_removed_rows.csv", index=False)
    desc.to_csv(OUTPUT_DIR / "bpdiaave_zero_descriptive_full.csv", index=False)
    desc_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_descriptive_delta.csv", index=False)
    pca.to_csv(OUTPUT_DIR / "bpdiaave_zero_pca_full.csv", index=False)
    pca_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_pca_delta.csv", index=False)
    pca_loading_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_pca_loading_delta.csv", index=False)
    fa_summary.to_csv(OUTPUT_DIR / "bpdiaave_zero_fa_summary_full.csv", index=False)
    fa_summary_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_fa_summary_delta.csv", index=False)
    fa_comm_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_fa_communality_delta.csv", index=False)
    max_fa_comm_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_fa_largest_communality_deltas.csv", index=False)
    km_cost.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_costs.csv", index=False)
    km_modes.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_modes.csv", index=False)
    km_modes_aligned.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_modes_aligned.csv", index=False)
    km_alignment.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_alignment.csv", index=False)
    km_outcomes.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_outcomes_full.csv", index=False)
    km_outcomes_aligned.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_outcomes_aligned.csv", index=False)
    km_outcome_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_kmodes_outcome_delta.csv", index=False)
    logit_perf.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_performance_full.csv", index=False)
    logit_perf_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_performance_delta.csv", index=False)
    logit_conf.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_confusion_full.csv", index=False)
    logit_coef.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_coefficients_full.csv", index=False)
    logit_coef_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_coefficient_delta.csv", index=False)
    largest_logit_coef_delta.to_csv(OUTPUT_DIR / "bpdiaave_zero_logit_largest_coefficient_deltas.csv", index=False)

    report = f"""# Sensitivity check: dropping rows with BPDiaAve = 0

The baseline analysis uses the deduplicated NHANES file with {len(baseline):,} rows.
There are {len(impossible):,} rows where `BPDiaAve = 0`, which is {len(impossible) / len(baseline) * 100:.3f}% of the analysis data.
The sensitivity analysis reruns the same descriptive summaries, PCA, four-factor varimax factor analysis, race-stratified k-modes clustering, and race-specific logistic regression after removing those rows.

## Removed rows

{format_report_table(removed_profile)}

## Descriptive race-level changes

{format_report_table(desc_delta)}

## PCA changes

{format_report_table(pca_delta)}

The largest sign-aligned absolute PCA loading changes among PC1-PC4 are:

{format_report_table(pca_loading_delta)}

## Factor-analysis changes

{format_report_table(fa_summary_delta)}

The largest communality changes are:

{format_report_table(max_fa_comm_delta)}

## K-modes cluster outcome changes

Sensitivity clusters are aligned to the baseline clusters by matching categorical centroid modes before computing deltas.

{format_report_table(km_alignment)}

{format_report_table(km_outcome_delta)}

## Logistic-regression performance changes

{format_report_table(logit_perf_delta)}

## Logistic-regression coefficient changes

{format_report_table(largest_logit_coef_delta)}

## Bottom line

The four impossible diastolic blood-pressure rows are too small a share of the sample to alter the main descriptive, multivariate, clustering, or predictive conclusions. The largest changes occur in small Black k-modes clusters and in the Black logistic regression because the Black subset is much smaller, so each removed row has more leverage. Even there, the qualitative result remains the same: the Black subset has higher observed diabetes prevalence, PCA and factor analysis continue to show that no single risk axis captures the full profile, and the supervised model still predicts diabetes better for Black respondents than for White respondents in AUC terms.
"""
    (OUTPUT_DIR / "bpdiaave_zero_sensitivity_report.md").write_text(report)

    print("Sensitivity check complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Removed rows:")
    print(removed_profile.to_string(index=False))
    print()
    print("PCA delta:")
    print(pca_delta.round(4).to_string(index=False))
    print()
    print("Logistic performance delta:")
    print(logit_perf_delta.round(4).to_string(index=False))
    print()
    print("K-modes cluster alignment:")
    print(km_alignment.to_string(index=False))
    print()
    print("Largest aligned k-modes outcome deltas:")
    km_delta_ranked = km_outcome_delta.assign(
        max_abs_pct_delta=lambda x: x[
            ["diabetes_yes_pct_delta", "highbp_high_pct_delta"]
        ].abs().max(axis=1)
    ).sort_values("max_abs_pct_delta", ascending=False)
    print(km_delta_ranked.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
