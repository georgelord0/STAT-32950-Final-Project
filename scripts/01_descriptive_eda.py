#!/usr/bin/env python3
"""Descriptive EDA for the NHANES adult health project data.

Scope: summary tables and visualizations only. This script intentionally avoids
statistical tests, dimensionality reduction, supervised learning, and prediction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = PROJECT_ROOT / "data" / "nhanes_health.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "descriptive_eda"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = PROJECT_ROOT / "reports" / "01_descriptive_eda.md"

REQUIRED_COLUMNS = [
    "Gender",
    "Age",
    "Race1",
    "Education",
    "MaritalStatus",
    "HHIncome",
    "Poverty",
    "HomeOwn",
    "Weight",
    "Height",
    "BMI",
    "Pulse",
    "BPSysAve",
    "BPDiaAve",
    "DirectChol",
    "TotChol",
    "Diabetes",
    "HealthGen",
    "PhysActive",
    "SmokeNow",
    "AlcoholYear",
    "SleepHrsNight",
    "HighBP",
]

RACE_ORDER = ["White", "Black", "Mexican", "Hispanic", "Other"]
DIABETES_ORDER = ["No", "Yes"]
HIGHBP_ORDER = ["Normal", "High"]
EDUCATION_ORDER = [
    "8th Grade",
    "9 - 11th Grade",
    "High School",
    "Some College",
    "College Grad",
]
INCOME_ORDER = [
    " 0-4999",
    " 5000-9999",
    "10000-14999",
    "15000-19999",
    "20000-24999",
    "25000-34999",
    "35000-44999",
    "45000-54999",
    "55000-64999",
    "65000-74999",
    "75000-99999",
    "more 99999",
]
INCOME_LABELS = {
    " 0-4999": "$0-4,999",
    " 5000-9999": "$5,000-9,999",
    "10000-14999": "$10,000-14,999",
    "15000-19999": "$15,000-19,999",
    "20000-24999": "$20,000-24,999",
    "25000-34999": "$25,000-34,999",
    "35000-44999": "$35,000-44,999",
    "45000-54999": "$45,000-54,999",
    "55000-64999": "$55,000-64,999",
    "65000-74999": "$65,000-74,999",
    "75000-99999": "$75,000-99,999",
    "more 99999": "$100,000+",
}
INCOME_BAND_ORDER = ["<$25k", "$25k-$64,999", "$65k-$99,999", "$100k+"]
INCOME_BANDS = {
    " 0-4999": "<$25k",
    " 5000-9999": "<$25k",
    "10000-14999": "<$25k",
    "15000-19999": "<$25k",
    "20000-24999": "<$25k",
    "25000-34999": "$25k-$64,999",
    "35000-44999": "$25k-$64,999",
    "45000-54999": "$25k-$64,999",
    "55000-64999": "$25k-$64,999",
    "65000-74999": "$65k-$99,999",
    "75000-99999": "$65k-$99,999",
    "more 99999": "$100k+",
}
HEALTH_ORDER = ["Poor", "Fair", "Good", "Vgood", "Excellent"]

RACE_PALETTE = {
    "White": "#4C78A8",
    "Black": "#F58518",
    "Mexican": "#54A24B",
    "Hispanic": "#B279A2",
    "Other": "#9D755D",
}
OUTCOME_PALETTE = {"No": "#4C78A8", "Yes": "#E45756", "Normal": "#72B7B2", "High": "#F58518"}
INCOME_PALETTE = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def pct(series: pd.Series, value: object) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == value).mean() * 100)


def median_iqr(series: pd.Series) -> str:
    q1, med, q3 = series.quantile([0.25, 0.50, 0.75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def ordered_counts(series: pd.Series, order: Iterable[str] | None = None) -> pd.Series:
    counts = series.value_counts(dropna=False)
    if order is None:
        return counts
    ordered = counts.reindex([x for x in order if x in counts.index]).dropna()
    remaining = counts.drop(index=ordered.index, errors="ignore")
    return pd.concat([ordered, remaining])


def save_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    highbp_expected = np.where((df["BPSysAve"] >= 130) | (df["BPDiaAve"] >= 80), "High", "Normal")
    overview = pd.DataFrame(
        [
            ("rows", len(df)),
            ("columns", df.shape[1]),
            ("missing_cells", int(df.isna().sum().sum())),
            ("exact_duplicate_rows", int(df.duplicated().sum())),
            ("unique_full_profiles", int(len(df.drop_duplicates()))),
            ("bpdiaave_zero_values", int((df["BPDiaAve"] == 0).sum())),
            ("highbp_derivation_mismatches", int((highbp_expected != df["HighBP"]).sum())),
            ("white_rows", int((df["Race1"] == "White").sum())),
            ("black_rows", int((df["Race1"] == "Black").sum())),
        ],
        columns=["metric", "value"],
    )
    overview.to_csv(TABLE_DIR / "table_01_dataset_overview.csv", index=False)
    return overview


def save_variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        record: dict[str, object] = {
            "variable": col,
            "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "missing_n": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique_n": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe(percentiles=[0.25, 0.5, 0.75])
            record.update(
                {
                    "mean": round(float(desc["mean"]), 3),
                    "sd": round(float(desc["std"]), 3),
                    "min": round(float(desc["min"]), 3),
                    "q1": round(float(desc["25%"]), 3),
                    "median": round(float(desc["50%"]), 3),
                    "q3": round(float(desc["75%"]), 3),
                    "max": round(float(desc["max"]), 3),
                    "top_level": "",
                    "top_n": "",
                    "top_pct": "",
                }
            )
        else:
            counts = series.value_counts(dropna=False)
            top_level = counts.index[0]
            top_n = int(counts.iloc[0])
            record.update(
                {
                    "mean": "",
                    "sd": "",
                    "min": "",
                    "q1": "",
                    "median": "",
                    "q3": "",
                    "max": "",
                    "top_level": str(top_level),
                    "top_n": top_n,
                    "top_pct": round(top_n / len(series) * 100, 2),
                }
            )
        records.append(record)

    variable_summary = pd.DataFrame(records)
    variable_summary.to_csv(TABLE_DIR / "table_02_variable_summary.csv", index=False)
    return variable_summary


def save_categorical_distributions(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(exclude="number").columns
    records: list[dict[str, object]] = []
    order_map = {
        "Race1": RACE_ORDER,
        "Diabetes": DIABETES_ORDER,
        "HighBP": HIGHBP_ORDER,
        "Education": EDUCATION_ORDER,
        "HHIncome": INCOME_ORDER,
        "HealthGen": HEALTH_ORDER,
    }
    for col in cat_cols:
        counts = ordered_counts(df[col], order_map.get(col))
        for level, count in counts.items():
            records.append(
                {
                    "variable": col,
                    "level": level,
                    "n": int(count),
                    "pct": round(float(count / len(df) * 100), 2),
                }
            )
    categorical = pd.DataFrame(records)
    categorical.to_csv(TABLE_DIR / "table_03_categorical_distributions.csv", index=False)
    return categorical


def save_continuous_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    summary = (
        df[numeric_cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.rename(columns={"50%": "median", "25%": "q1", "75%": "q3"})
        .reset_index(names="variable")
    )
    summary = summary[["variable", "count", "mean", "std", "min", "q1", "median", "q3", "max"]]
    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(3)
    summary.to_csv(TABLE_DIR / "table_04_continuous_summary.csv", index=False)
    return summary


def save_race_risk_rates(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for race in RACE_ORDER:
        sub = df[df["Race1"] == race]
        if sub.empty:
            continue
        records.append(
            {
                "race": race,
                "n": len(sub),
                "diabetes_yes_n": int((sub["Diabetes"] == "Yes").sum()),
                "diabetes_yes_pct": round(pct(sub["Diabetes"], "Yes"), 2),
                "highbp_high_n": int((sub["HighBP"] == "High").sum()),
                "highbp_high_pct": round(pct(sub["HighBP"], "High"), 2),
                "age_median": round(float(sub["Age"].median()), 2),
                "bmi_median": round(float(sub["BMI"].median()), 2),
                "poverty_median": round(float(sub["Poverty"].median()), 2),
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(TABLE_DIR / "table_05_race_risk_rates.csv", index=False)
    return table


def save_white_black_profile(df: pd.DataFrame) -> pd.DataFrame:
    def profile_for(sub: pd.DataFrame) -> dict[str, str]:
        return {
            "n": f"{len(sub):,}",
            "Age median [IQR]": median_iqr(sub["Age"]),
            "BMI median [IQR]": median_iqr(sub["BMI"]),
            "BPSysAve median [IQR]": median_iqr(sub["BPSysAve"]),
            "BPDiaAve median [IQR]": median_iqr(sub["BPDiaAve"]),
            "TotChol median [IQR]": median_iqr(sub["TotChol"]),
            "Poverty median [IQR]": median_iqr(sub["Poverty"]),
            "Diabetes Yes (%)": f"{pct(sub['Diabetes'], 'Yes'):.1f}",
            "HighBP High (%)": f"{pct(sub['HighBP'], 'High'):.1f}",
            "College graduate (%)": f"{pct(sub['Education'], 'College Grad'):.1f}",
            "Household income $75k+ (%)": f"{sub['HHIncome'].isin(['75000-99999', 'more 99999']).mean() * 100:.1f}",
            "Physically active Yes (%)": f"{pct(sub['PhysActive'], 'Yes'):.1f}",
            "Smoke now Yes (%)": f"{pct(sub['SmokeNow'], 'Yes'):.1f}",
        }

    groups = {
        "All": df,
        "White": df[df["Race1"] == "White"],
        "Black": df[df["Race1"] == "Black"],
    }
    rows = []
    for metric in profile_for(df).keys():
        row = {"metric": metric}
        for group_name, sub in groups.items():
            row[group_name] = profile_for(sub)[metric]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "table_06_white_black_profile.csv", index=False)
    return table


def save_duplicate_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_counts = df.value_counts(dropna=False).reset_index(name="profile_count")
    multiplicity = (
        profile_counts["profile_count"]
        .value_counts()
        .sort_index()
        .rename_axis("rows_per_profile")
        .reset_index(name="number_of_profiles")
    )
    multiplicity["total_rows"] = multiplicity["rows_per_profile"] * multiplicity["number_of_profiles"]
    multiplicity.to_csv(TABLE_DIR / "table_07_duplicate_multiplicity.csv", index=False)

    top_profiles = profile_counts.head(20)
    top_profiles.to_csv(TABLE_DIR / "table_08_top_duplicate_profiles.csv", index=False)
    return multiplicity, top_profiles


def add_bar_labels(ax: plt.Axes, values: pd.Series, total: int | None = None, *, pct_format: bool = False) -> None:
    y_max = max(values) if len(values) else 0
    for patch, value in zip(ax.patches, values):
        if pct_format:
            label = f"{value:.1f}%"
        elif total:
            label = f"{int(value):,}\n({value / total * 100:.1f}%)"
        else:
            label = f"{int(value):,}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax.set_ylim(0, y_max * 1.18 if y_max else 1)


def plot_sample_composition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    specs = [
        ("Race1", "Self-identified race", RACE_ORDER, RACE_PALETTE),
        ("Diabetes", "Diabetes status", DIABETES_ORDER, OUTCOME_PALETTE),
        ("HighBP", "Derived blood pressure status", HIGHBP_ORDER, OUTCOME_PALETTE),
    ]
    for ax, (col, title, order, palette) in zip(axes, specs):
        counts = ordered_counts(df[col], order)
        colors = [palette.get(level, "#777777") for level in counts.index]
        ax.bar(counts.index, counts.values, color=colors)
        add_bar_labels(ax, counts, total=len(df))
        ax.set_title(title)
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("NHANES sample composition", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "fig_01_sample_composition.png", dpi=200)
    plt.close(fig)


def plot_race_risk_rates(race_rates: pd.DataFrame) -> None:
    plot_df = race_rates.melt(
        id_vars=["race", "n"],
        value_vars=["diabetes_yes_pct", "highbp_high_pct"],
        var_name="measure",
        value_name="percent",
    )
    plot_df["measure"] = plot_df["measure"].map(
        {
            "diabetes_yes_pct": "Diabetes = Yes",
            "highbp_high_pct": "HighBP = High",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="race",
        y="percent",
        hue="measure",
        order=[race for race in RACE_ORDER if race in race_rates["race"].tolist()],
        palette=["#E45756", "#F58518"],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=3)
    ax.set_title("Descriptive diabetes and high blood pressure rates by race")
    ax.set_xlabel("Race1")
    ax.set_ylabel("Rows in category (%)")
    ax.set_ylim(0, max(plot_df["percent"]) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    displayed_races = [race for race in RACE_ORDER if race in race_rates["race"].tolist()]
    counts = race_rates.set_index("race")["n"]
    ax.set_xticks(range(len(displayed_races)))
    ax.set_xticklabels([f"{race}\n(n={counts[race]:,})" for race in displayed_races], rotation=0)
    fig.savefig(FIGURE_DIR / "fig_02_race_diabetes_highbp_rates.png", dpi=200)
    plt.close(fig)


def plot_cardiometabolic_by_diabetes(df: pd.DataFrame) -> None:
    variables = [
        ("Age", "Age"),
        ("BMI", "Body mass index"),
        ("BPSysAve", "Systolic BP"),
        ("BPDiaAve", "Diastolic BP"),
        ("DirectChol", "Direct cholesterol"),
        ("TotChol", "Total cholesterol"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, (col, label) in zip(axes.ravel(), variables):
        sns.boxplot(
            data=df,
            x="Diabetes",
            y=col,
            order=DIABETES_ORDER,
            hue="Diabetes",
            hue_order=DIABETES_ORDER,
            palette=[OUTCOME_PALETTE["No"], OUTCOME_PALETTE["Yes"]],
            width=0.55,
            fliersize=2,
            legend=False,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Diabetes")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cardiometabolic variable distributions by diabetes status", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "fig_03_cardiometabolic_by_diabetes.png", dpi=200)
    plt.close(fig)


def plot_socioeconomic_by_race(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["IncomeBand"] = plot_df["HHIncome"].map(INCOME_BANDS)
    income = (
        plot_df.groupby(["Race1", "IncomeBand"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    income["pct"] = income["n"] / income.groupby("Race1")["n"].transform("sum") * 100
    income_pivot = (
        income.pivot(index="Race1", columns="IncomeBand", values="pct")
        .reindex(index=[race for race in RACE_ORDER if race in plot_df["Race1"].unique()])
        .reindex(columns=INCOME_BAND_ORDER)
        .fillna(0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="Race1",
        y="Poverty",
        order=[race for race in RACE_ORDER if race in plot_df["Race1"].unique()],
        hue="Race1",
        hue_order=[race for race in RACE_ORDER if race in plot_df["Race1"].unique()],
        palette=RACE_PALETTE,
        width=0.6,
        fliersize=2,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Poverty ratio by race")
    axes[0].set_xlabel("Race1")
    axes[0].set_ylabel("Poverty ratio")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    bottom = np.zeros(len(income_pivot))
    x = np.arange(len(income_pivot))
    for idx, band in enumerate(INCOME_BAND_ORDER):
        vals = income_pivot[band].values
        axes[1].bar(x, vals, bottom=bottom, label=band, color=INCOME_PALETTE[idx])
        bottom += vals
    axes[1].set_xticks(x, income_pivot.index, rotation=30)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Household income band composition by race")
    axes[1].set_xlabel("Race1")
    axes[1].set_ylabel("Rows in income band (%)")
    axes[1].legend(title="Income band", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Socioeconomic descriptors across race groups", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "fig_04_socioeconomic_by_race.png", dpi=200)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    numeric_cols = [
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
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Descriptive correlation among numeric variables")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.savefig(FIGURE_DIR / "fig_05_numeric_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    clean = df.copy().astype(str)
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(row[col] for col in headers) + " |")
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    overview: pd.DataFrame,
    race_rates: pd.DataFrame,
    white_black_profile: pd.DataFrame,
    duplicate_multiplicity: pd.DataFrame,
) -> None:
    overview_map = dict(zip(overview["metric"], overview["value"]))
    race_counts = df["Race1"].value_counts()
    diabetes_yes_pct = pct(df["Diabetes"], "Yes")
    highbp_high_pct = pct(df["HighBP"], "High")
    black_diabetes = race_rates.loc[race_rates["race"] == "Black", "diabetes_yes_pct"].iloc[0]
    white_diabetes = race_rates.loc[race_rates["race"] == "White", "diabetes_yes_pct"].iloc[0]
    black_highbp = race_rates.loc[race_rates["race"] == "Black", "highbp_high_pct"].iloc[0]
    white_highbp = race_rates.loc[race_rates["race"] == "White", "highbp_high_pct"].iloc[0]

    report = f"""# Descriptive EDA: NHANES Adult Health Data

## Scope

This note covers project component 2 only: descriptive data exploration and visualization. It does not run PCA, factor analysis, clustering, regression, classification, hypothesis tests, or prediction models.

## Data

The source file is `data/nhanes_health.csv`. The project instructions describe this as a simplified complete-case adult NHANES sample without survey weights, so all summaries below describe this project sample rather than official U.S. population estimates.

Key dimensions:

- Rows: {int(overview_map["rows"]):,}
- Columns: {int(overview_map["columns"]):,}
- Missing cells: {int(overview_map["missing_cells"]):,}
- Exact duplicate rows across all selected variables: {int(overview_map["exact_duplicate_rows"]):,}
- Unique full profiles after exact-row collapse: {int(overview_map["unique_full_profiles"]):,}
- `BPDiaAve` values equal to 0: {int(overview_map["bpdiaave_zero_values"]):,}
- `HighBP` derivation mismatches from `BPSysAve >= 130` or `BPDiaAve >= 80`: {int(overview_map["highbp_derivation_mismatches"]):,}

## Reproducibility

Run the EDA from the repository root:

```bash
python3 scripts/01_descriptive_eda.py
```

The script regenerates all files in `outputs/descriptive_eda/figures/`, `outputs/descriptive_eda/tables/`, and this report.

## Preprocessing Choices

- The raw CSV is read as supplied.
- No rows are dropped and no values are imputed.
- Exact duplicate rows are documented but retained because the file has no respondent identifier that would justify automatic deduplication.
- Household income is grouped into broad bands only for the socioeconomic composition plot.
- Race-specific summaries use `Race1`; White and Black are highlighted because they align with the planned research question, while the broader EDA still reports all available race categories.
- All comparisons are descriptive and unweighted.

## Main Descriptive Findings

The sample is heavily White: {race_counts.get("White", 0):,} White rows and {race_counts.get("Black", 0):,} Black rows out of {len(df):,}. This imbalance matters for later race-specific methods because the Black subgroup is much smaller than the White subgroup.

Overall, {diabetes_yes_pct:.1f}% of rows have `Diabetes = Yes`, and {highbp_high_pct:.1f}% have `HighBP = High`. By race, the descriptive diabetes percentage is {black_diabetes:.1f}% for Black rows and {white_diabetes:.1f}% for White rows; the descriptive high blood pressure percentage is {black_highbp:.1f}% for Black rows and {white_highbp:.1f}% for White rows. These are sample percentages, not population estimates.

The cardiometabolic distributions show visible differences by diabetes status for age and BMI, with additional spread in blood pressure and cholesterol measurements. These plots are descriptive only and should be used to motivate later formal modeling rather than to claim independent effects.

Socioeconomic descriptors vary across `Race1`, especially the poverty ratio and broad household-income bands. These variables should be handled explicitly in later analyses because socioeconomic composition is part of the planned research question and may be associated with health markers.

## Selected Tables

Race-level descriptive rates:

{markdown_table(race_rates)}

White/Black profile table:

{markdown_table(white_black_profile)}

Duplicate multiplicity summary:

{markdown_table(duplicate_multiplicity)}

## Generated Figures

- `outputs/descriptive_eda/figures/fig_01_sample_composition.png`
- `outputs/descriptive_eda/figures/fig_02_race_diabetes_highbp_rates.png`
- `outputs/descriptive_eda/figures/fig_03_cardiometabolic_by_diabetes.png`
- `outputs/descriptive_eda/figures/fig_04_socioeconomic_by_race.png`
- `outputs/descriptive_eda/figures/fig_05_numeric_correlation_heatmap.png`

## Handoff Notes

- Later modeling should decide explicitly whether to retain exact duplicate rows. The current EDA retains them and reports the issue.
- The four `BPDiaAve = 0` rows are retained for this descriptive pass but should be reviewed before blood-pressure modeling.
- Because survey weights are absent, avoid language about national prevalence or U.S. population-level inference.
- The White/Black sample-size imbalance should be acknowledged in race-specific factor analysis or prediction work.

## AI Use Acknowledgment Draft

AI assistance was used to help organize reproducible code, generate descriptive tables and figures, and draft this EDA handoff note. The group should review all code, figures, and interpretations before incorporating them into the final report.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )

    df = load_data()
    overview = save_dataset_overview(df)
    save_variable_summary(df)
    save_categorical_distributions(df)
    save_continuous_summary(df)
    race_rates = save_race_risk_rates(df)
    white_black_profile = save_white_black_profile(df)
    duplicate_multiplicity, _ = save_duplicate_tables(df)

    plot_sample_composition(df)
    plot_race_risk_rates(race_rates)
    plot_cardiometabolic_by_diabetes(df)
    plot_socioeconomic_by_race(df)
    plot_numeric_correlation(df)
    build_report(df, overview, race_rates, white_black_profile, duplicate_multiplicity)

    print(f"Generated descriptive EDA outputs in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Generated report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
