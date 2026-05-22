#!/usr/bin/env python3
"""Centralized final analysis runner for the NHANES group project.

This file centralizes the submission-facing notebook analysis without changing the
statistical workflow. It reads data/nhanes_health.csv, removes exact duplicates,
regenerates descriptive EDA, PCA, factor analysis, k-modes clustering, and
race-specific logistic regression outputs, and writes them under final_version/outputs.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / 'final_version'
VENDOR_DIR = FINAL_DIR / 'vendor'

CACHE_DIR = FINAL_DIR / '_cache'
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_DIR / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR / 'xdg'))
(CACHE_DIR / 'matplotlib').mkdir(parents=True, exist_ok=True)
(CACHE_DIR / 'xdg').mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    from IPython.display import Image, display as ipy_display
except Exception:  # pragma: no cover - only used when running outside notebooks
    Image = None
    def ipy_display(*_args, **_kwargs):
        return None
display = ipy_display

DATA_PATH = PROJECT_ROOT / 'data' / 'nhanes_health.csv'
OUTPUT_ROOT = FINAL_DIR / 'outputs'
OUTPUT_DIR = OUTPUT_ROOT / 'descriptive_eda'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'
PCA_OUTPUT_DIR = OUTPUT_ROOT / 'pca'
PCA_FIGURE_DIR = PCA_OUTPUT_DIR / 'figures'
PCA_TABLE_DIR = PCA_OUTPUT_DIR / 'tables'
FA_OUTPUT_DIR = OUTPUT_ROOT / 'fa'
FA_FIGURE_DIR = FA_OUTPUT_DIR / 'figures'
FA_TABLE_DIR = FA_OUTPUT_DIR / 'tables'
CLUSTER_OUTPUT_DIR = OUTPUT_ROOT / 'clustering'
CLUSTER_FIGURE_DIR = CLUSTER_OUTPUT_DIR / 'figures'
CLUSTER_TABLE_DIR = CLUSTER_OUTPUT_DIR / 'tables'
REPORT_PATH = FINAL_DIR / 'reports' / '01_descriptive_eda.md'

for path in [
    OUTPUT_DIR, FIGURE_DIR, TABLE_DIR,
    PCA_OUTPUT_DIR, PCA_FIGURE_DIR, PCA_TABLE_DIR,
    FA_OUTPUT_DIR, FA_FIGURE_DIR, FA_TABLE_DIR,
    CLUSTER_OUTPUT_DIR, CLUSTER_FIGURE_DIR, CLUSTER_TABLE_DIR,
    REPORT_PATH.parent,
]:
    path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.titleweight': 'bold',
    'font.size': 10,
    'savefig.bbox': 'tight',
})

print(f'Project root: {PROJECT_ROOT}')
print(f'Data path:    {DATA_PATH}')
print(f'Output root:  {OUTPUT_ROOT}')



# %% Notebook cell 2
REQUIRED_COLUMNS = [
    'Gender', 'Age', 'Race1', 'Education', 'MaritalStatus',
    'HHIncome', 'Poverty', 'HomeOwn', 'Weight', 'Height',
    'BMI', 'Pulse', 'BPSysAve', 'BPDiaAve', 'DirectChol',
    'TotChol', 'Diabetes', 'HealthGen', 'PhysActive',
    'SmokeNow', 'AlcoholYear', 'SleepHrsNight', 'HighBP',
]

RACE_ORDER = ['White', 'Black', 'Mexican', 'Hispanic', 'Other']
DIABETES_ORDER = ['No', 'Yes']
HIGHBP_ORDER = ['Normal', 'High']
EDUCATION_ORDER = ['8th Grade', '9 - 11th Grade', 'High School', 'Some College', 'College Grad']
INCOME_ORDER = [
    ' 0-4999', ' 5000-9999', '10000-14999', '15000-19999',
    '20000-24999', '25000-34999', '35000-44999', '45000-54999',
    '55000-64999', '65000-74999', '75000-99999', 'more 99999',
]
INCOME_LABELS = {
    ' 0-4999': '$0-4,999', ' 5000-9999': '$5,000-9,999',
    '10000-14999': '$10,000-14,999', '15000-19999': '$15,000-19,999',
    '20000-24999': '$20,000-24,999', '25000-34999': '$25,000-34,999',
    '35000-44999': '$35,000-44,999', '45000-54999': '$45,000-54,999',
    '55000-64999': '$55,000-64,999', '65000-74999': '$65,000-74,999',
    '75000-99999': '$75,000-99,999', 'more 99999': '$100,000+',
}
INCOME_BAND_ORDER = ['<$25k', '$25k-$64,999', '$65k-$99,999', '$100k+']
INCOME_BANDS = {
    ' 0-4999': '<$25k', ' 5000-9999': '<$25k', '10000-14999': '<$25k',
    '15000-19999': '<$25k', '20000-24999': '<$25k',
    '25000-34999': '$25k-$64,999', '35000-44999': '$25k-$64,999',
    '45000-54999': '$25k-$64,999', '55000-64999': '$25k-$64,999',
    '65000-74999': '$65k-$99,999', '75000-99999': '$65k-$99,999',
    'more 99999': '$100k+',
}
HEALTH_ORDER = ['Poor', 'Fair', 'Good', 'Vgood', 'Excellent']
GENDER_ORDER = ['male', 'female']
MARITAL_ORDER = ['Married', 'LivePartner', 'NeverMarried', 'Divorced', 'Separated', 'Widowed']
PHYSACTIVE_ORDER = ['Yes', 'No']
SMOKENOW_ORDER = ['Yes', 'No']

RACE_PALETTE = {
    'White': '#4C78A8', 'Black': '#F58518', 'Mexican': '#54A24B',
    'Hispanic': '#B279A2', 'Other': '#9D755D',
}
OUTCOME_PALETTE = {'No': '#4C78A8', 'Yes': '#E45756', 'Normal': '#72B7B2', 'High': '#F58518'}
INCOME_PALETTE = ['#4C78A8', '#54A24B', '#F58518', '#B279A2']



# %% Notebook cell 3
def pct(series: pd.Series, value: object) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == value).mean() * 100)


def median_iqr(series: pd.Series) -> str:
    q1, med, q3 = series.quantile([0.25, 0.50, 0.75])
    return f'{med:.2f} [{q1:.2f}, {q3:.2f}]'


def ordered_counts(series: pd.Series, order: Iterable[str] | None = None) -> pd.Series:
    counts = series.value_counts(dropna=False)
    if order is None:
        return counts
    ordered = counts.reindex([x for x in order if x in counts.index]).dropna()
    remaining = counts.drop(index=ordered.index, errors='ignore')
    return pd.concat([ordered, remaining])


def add_bar_labels(ax: plt.Axes, values: pd.Series, total: int | None = None, *, pct_format: bool = False) -> None:
    y_max = max(values) if len(values) else 0
    for patch, value in zip(ax.patches, values):
        if pct_format:
            label = f'{value:.1f}%'
        elif total:
            label = f'{int(value):,}\n({value / total * 100:.1f}%)'
        else:
            label = f'{int(value):,}'
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha='center', va='bottom', fontsize=8,
            xytext=(0, 3), textcoords='offset points',
        )
    ax.set_ylim(0, y_max * 1.18 if y_max else 1)


def markdown_table(df: pd.DataFrame) -> str:
    clean = df.copy().astype(str)
    headers = list(clean.columns)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for _, row in clean.iterrows():
        lines.append('| ' + ' | '.join(row[col] for col in headers) + ' |')
    return '\n'.join(lines)



# %% Notebook cell 4
df = pd.read_csv(DATA_PATH)
raw_df = df.copy()
missing_cols = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_cols:
    raise ValueError(f'Missing expected columns: {missing_cols}')

print(f'Shape: {df.shape}')
print(f'Missing cells: {df.isna().sum().sum():,}')
df.head()



# %% Notebook cell 5
# Tables 07 & 08 — Duplicate-row analysis (run on raw df BEFORE deduplication)
profile_counts = df.value_counts(dropna=False).reset_index(name='profile_count')
duplicate_multiplicity = (
    profile_counts['profile_count']
    .value_counts()
    .sort_index()
    .rename_axis('rows_per_profile')
    .reset_index(name='number_of_profiles')
)
duplicate_multiplicity['total_rows'] = (
    duplicate_multiplicity['rows_per_profile'] * duplicate_multiplicity['number_of_profiles']
)
duplicate_multiplicity.to_csv(TABLE_DIR / 'table_07_duplicate_multiplicity.csv', index=False)

top_profiles = profile_counts.head(20)
top_profiles.to_csv(TABLE_DIR / 'table_08_top_duplicate_profiles.csv', index=False)

print('Duplicate multiplicity (raw data):')
display(duplicate_multiplicity)
print('\nTop-20 repeated profiles (raw data):')
top_profiles



# %% Notebook cell 6
# Deduplication — remove exact duplicate rows once so all downstream cells
# (tables, figures, PCA) run on the clean dataset without further changes.
# NOTE: This permanently reassigns df; 2,412 → 1,508 unique profiles.
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
n_after  = len(df)
print(f'Rows before deduplication : {n_before:,}')
print(f'Rows after  deduplication : {n_after:,}  ({n_before - n_after:,} exact duplicates removed)')



# %% Notebook cell 7
# Table 01 — Dataset overview
highbp_expected = np.where((df['BPSysAve'] >= 130) | (df['BPDiaAve'] >= 80), 'High', 'Normal')
overview = pd.DataFrame(
    [
        ('rows', len(df)),
        ('columns', df.shape[1]),
        ('missing_cells', int(df.isna().sum().sum())),
        ('exact_duplicate_rows', int(df.duplicated().sum())),
        ('unique_full_profiles', int(len(df.drop_duplicates()))),
        ('bpdiaave_zero_values', int((df['BPDiaAve'] == 0).sum())),
        ('highbp_derivation_mismatches', int((highbp_expected != df['HighBP']).sum())),
        ('white_rows', int((df['Race1'] == 'White').sum())),
        ('black_rows', int((df['Race1'] == 'Black').sum())),
    ],
    columns=['metric', 'value'],
)
overview.to_csv(TABLE_DIR / 'table_01_dataset_overview.csv', index=False)
overview



# %% Notebook cell 8
# Table 02 — Per-variable summary
records = []
for col in df.columns:
    series = df[col]
    record = {
        'variable': col,
        'type': 'numeric' if pd.api.types.is_numeric_dtype(series) else 'categorical',
        'missing_n': int(series.isna().sum()),
        'missing_pct': round(float(series.isna().mean() * 100), 2),
        'unique_n': int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        record.update({
            'mean': round(float(desc['mean']), 3),
            'sd': round(float(desc['std']), 3),
            'min': round(float(desc['min']), 3),
            'q1': round(float(desc['25%']), 3),
            'median': round(float(desc['50%']), 3),
            'q3': round(float(desc['75%']), 3),
            'max': round(float(desc['max']), 3),
            'top_level': '', 'top_n': '', 'top_pct': '',
        })
    else:
        counts = series.value_counts(dropna=False)
        top_level = counts.index[0]
        top_n = int(counts.iloc[0])
        record.update({
            'mean': '', 'sd': '', 'min': '', 'q1': '', 'median': '', 'q3': '', 'max': '',
            'top_level': str(top_level),
            'top_n': top_n,
            'top_pct': round(top_n / len(series) * 100, 2),
        })
    records.append(record)

variable_summary = pd.DataFrame(records)
variable_summary.to_csv(TABLE_DIR / 'table_02_variable_summary.csv', index=False)
variable_summary



# %% Notebook cell 9
# Table 03 — Categorical distributions
cat_cols = df.select_dtypes(exclude='number').columns
order_map = {
    'Race1': RACE_ORDER, 'Diabetes': DIABETES_ORDER, 'HighBP': HIGHBP_ORDER,
    'Education': EDUCATION_ORDER, 'HHIncome': INCOME_ORDER, 'HealthGen': HEALTH_ORDER,
    'Gender': GENDER_ORDER, 'MaritalStatus': MARITAL_ORDER,
    'PhysActive': PHYSACTIVE_ORDER, 'SmokeNow': SMOKENOW_ORDER,
}
cat_records = []
for col in cat_cols:
    counts = ordered_counts(df[col], order_map.get(col))
    for level, count in counts.items():
        cat_records.append({
            'variable': col,
            'level': level,
            'n': int(count),
            'pct': round(float(count / len(df) * 100), 2),
        })
categorical = pd.DataFrame(cat_records)
categorical.to_csv(TABLE_DIR / 'table_03_categorical_distributions.csv', index=False)
categorical.head(30)



# %% Notebook cell 10
# Table 04 — Continuous variable summary
numeric_cols = df.select_dtypes(include='number').columns
cont_summary = (
    df[numeric_cols]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T.rename(columns={'50%': 'median', '25%': 'q1', '75%': 'q3'})
    .reset_index(names='variable')
)
cont_summary = cont_summary[['variable', 'count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max']]
num_cols_round = cont_summary.select_dtypes(include='number').columns
cont_summary[num_cols_round] = cont_summary[num_cols_round].round(3)
cont_summary.to_csv(TABLE_DIR / 'table_04_continuous_summary.csv', index=False)
cont_summary



# %% Notebook cell 11
# Table 05 — Race-level descriptive risk rates
race_records = []
for race in RACE_ORDER:
    sub = df[df['Race1'] == race]
    if sub.empty:
        continue
    race_records.append({
        'race': race,
        'n': len(sub),
        'diabetes_yes_n': int((sub['Diabetes'] == 'Yes').sum()),
        'diabetes_yes_pct': round(pct(sub['Diabetes'], 'Yes'), 2),
        'highbp_high_n': int((sub['HighBP'] == 'High').sum()),
        'highbp_high_pct': round(pct(sub['HighBP'], 'High'), 2),
        'age_median': round(float(sub['Age'].median()), 2),
        'bmi_median': round(float(sub['BMI'].median()), 2),
        'poverty_median': round(float(sub['Poverty'].median()), 2),
    })
race_rates = pd.DataFrame(race_records)
race_rates.to_csv(TABLE_DIR / 'table_05_race_risk_rates.csv', index=False)
race_rates



# %% Notebook cell 12
# Table 06 — White / Black descriptive profile comparison
def profile_for(sub: pd.DataFrame) -> dict:
    return {
        'n': f'{len(sub):,}',
        'Age median [IQR]': median_iqr(sub['Age']),
        'BMI median [IQR]': median_iqr(sub['BMI']),
        'BPSysAve median [IQR]': median_iqr(sub['BPSysAve']),
        'BPDiaAve median [IQR]': median_iqr(sub['BPDiaAve']),
        'TotChol median [IQR]': median_iqr(sub['TotChol']),
        'Poverty median [IQR]': median_iqr(sub['Poverty']),
        'Diabetes Yes (%)': f"{pct(sub['Diabetes'], 'Yes'):.1f}",
        'HighBP High (%)': f"{pct(sub['HighBP'], 'High'):.1f}",
        'College graduate (%)': f"{pct(sub['Education'], 'College Grad'):.1f}",
        'Household income $75k+ (%)': f"{sub['HHIncome'].isin(['75000-99999', 'more 99999']).mean() * 100:.1f}",
        'Physically active Yes (%)': f"{pct(sub['PhysActive'], 'Yes'):.1f}",
        'Smoke now Yes (%)': f"{pct(sub['SmokeNow'], 'Yes'):.1f}",
    }

groups = {'All': df, 'White': df[df['Race1'] == 'White'], 'Black': df[df['Race1'] == 'Black']}
wb_rows = []
for metric in profile_for(df).keys():
    row = {'metric': metric}
    for group_name, sub in groups.items():
        row[group_name] = profile_for(sub)[metric]
    wb_rows.append(row)
white_black_profile = pd.DataFrame(wb_rows)
white_black_profile.to_csv(TABLE_DIR / 'table_06_white_black_profile.csv', index=False)
white_black_profile



# %% Notebook cell 13
# Figure 01 — Sample composition
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
specs = [
    ('Race1', 'Self-identified race', RACE_ORDER, RACE_PALETTE),
    ('Diabetes', 'Diabetes status', DIABETES_ORDER, OUTCOME_PALETTE),
    ('HighBP', 'Derived blood pressure status', HIGHBP_ORDER, OUTCOME_PALETTE),
]
for ax, (col, title, order, palette) in zip(axes, specs):
    counts = ordered_counts(df[col], order)
    colors = [palette.get(level, '#777777') for level in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    add_bar_labels(ax, counts, total=len(df))
    ax.set_title(title)
    ax.set_ylabel('Rows')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('NHANES sample composition', fontsize=15, fontweight='bold')
fig.savefig(FIGURE_DIR / 'fig_01_sample_composition.png', dpi=200)
plt.close(fig)



# %% Notebook cell 14
# Figure 02 — Diabetes and high-BP rates by race
plot_df = race_rates.melt(
    id_vars=['race', 'n'],
    value_vars=['diabetes_yes_pct', 'highbp_high_pct'],
    var_name='measure', value_name='percent',
)
plot_df['measure'] = plot_df['measure'].map({
    'diabetes_yes_pct': 'Diabetes = Yes',
    'highbp_high_pct': 'HighBP = High',
})

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
sns.barplot(
    data=plot_df, x='race', y='percent', hue='measure',
    order=[r for r in RACE_ORDER if r in race_rates['race'].tolist()],
    palette=['#E45756', '#F58518'], ax=ax,
)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=3)
ax.set_title('Descriptive diabetes and high blood pressure rates by race')
ax.set_xlabel('Race1')
ax.set_ylabel('Rows in category (%)')
ax.set_ylim(0, max(plot_df['percent']) * 1.25)
ax.grid(axis='y', alpha=0.25)
ax.legend(title='')
displayed_races = [r for r in RACE_ORDER if r in race_rates['race'].tolist()]
counts_by_race = race_rates.set_index('race')['n']
ax.set_xticks(range(len(displayed_races)))
ax.set_xticklabels([f'{r}\n(n={counts_by_race[r]:,})' for r in displayed_races], rotation=0)
fig.savefig(FIGURE_DIR / 'fig_02_race_diabetes_highbp_rates.png', dpi=200)
plt.close(fig)



# %% Notebook cell 15
# Figure 03 — Cardiometabolic distributions by diabetes status
variables = [
    ('Age', 'Age'), ('BMI', 'Body mass index'),
    ('BPSysAve', 'Systolic BP'), ('BPDiaAve', 'Diastolic BP'),
    ('DirectChol', 'Direct cholesterol'), ('TotChol', 'Total cholesterol'),
]
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
for ax, (col, label) in zip(axes.ravel(), variables):
    sns.boxplot(
        data=df, x='Diabetes', y=col,
        order=DIABETES_ORDER, hue='Diabetes', hue_order=DIABETES_ORDER,
        palette=[OUTCOME_PALETTE['No'], OUTCOME_PALETTE['Yes']],
        width=0.55, fliersize=2, legend=False, ax=ax,
    )
    ax.set_title(label)
    ax.set_xlabel('Diabetes')
    ax.set_ylabel(label)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('Cardiometabolic variable distributions by diabetes status', fontsize=15, fontweight='bold')
fig.savefig(FIGURE_DIR / 'fig_03_cardiometabolic_by_diabetes.png', dpi=200)
plt.close(fig)



# %% Notebook cell 16
# Figure 04 — Socioeconomic descriptors by race
plot_df2 = df.copy()
plot_df2['IncomeBand'] = plot_df2['HHIncome'].map(INCOME_BANDS)
income = (
    plot_df2.groupby(['Race1', 'IncomeBand'], observed=False)
    .size().rename('n').reset_index()
)
income['pct'] = income['n'] / income.groupby('Race1')['n'].transform('sum') * 100
income_pivot = (
    income.pivot(index='Race1', columns='IncomeBand', values='pct')
    .reindex(index=[r for r in RACE_ORDER if r in plot_df2['Race1'].unique()])
    .reindex(columns=INCOME_BAND_ORDER)
    .fillna(0)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
sns.boxplot(
    data=plot_df2, x='Race1', y='Poverty',
    order=[r for r in RACE_ORDER if r in plot_df2['Race1'].unique()],
    hue='Race1', hue_order=[r for r in RACE_ORDER if r in plot_df2['Race1'].unique()],
    palette=RACE_PALETTE, width=0.6, fliersize=2, legend=False, ax=axes[0],
)
axes[0].set_title('Poverty ratio by race')
axes[0].set_xlabel('Race1')
axes[0].set_ylabel('Poverty ratio')
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(axis='y', alpha=0.25)

bottom = np.zeros(len(income_pivot))
x = np.arange(len(income_pivot))
for idx, band in enumerate(INCOME_BAND_ORDER):
    vals = income_pivot[band].values
    axes[1].bar(x, vals, bottom=bottom, label=band, color=INCOME_PALETTE[idx])
    bottom += vals
axes[1].set_xticks(x, income_pivot.index, rotation=30)
axes[1].set_ylim(0, 100)
axes[1].set_title('Household income band composition by race')
axes[1].set_xlabel('Race1')
axes[1].set_ylabel('Rows in income band (%)')
axes[1].legend(title='Income band', bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].grid(axis='y', alpha=0.25)
fig.suptitle('Socioeconomic descriptors across race groups', fontsize=15, fontweight='bold')
fig.savefig(FIGURE_DIR / 'fig_04_socioeconomic_by_race.png', dpi=200)
plt.close(fig)



# %% Notebook cell 17
# Figure 05 — Numeric correlation heatmap (includes binary categoricals as 0/1)
numeric_cols_corr = [
    'Age', 'Poverty', 'Weight', 'Height', 'BMI', 'Pulse',
    'BPSysAve', 'BPDiaAve', 'DirectChol', 'TotChol', 'AlcoholYear', 'SleepHrsNight',
]
binary_encodings = {
    'Gender_male':    df['Gender'].eq('male').astype(int),
    'Diabetes_yes':   df['Diabetes'].eq('Yes').astype(int),
    'HighBP_high':    df['HighBP'].eq('High').astype(int),
    'PhysActive_yes': df['PhysActive'].eq('Yes').astype(int),
    'SmokeNow_yes':   df['SmokeNow'].eq('Yes').astype(int),
}
corr_df = pd.concat([df[numeric_cols_corr], pd.DataFrame(binary_encodings)], axis=1)
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
sns.heatmap(
    corr, cmap='vlag', center=0, vmin=-1, vmax=1,
    linewidths=0.5, annot=True, fmt='.2f', annot_kws={'fontsize': 7},
    cbar_kws={'label': 'Pearson correlation'}, ax=ax,
)
ax.set_title('Descriptive correlation among numeric and binary variables')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
fig.savefig(FIGURE_DIR / 'fig_05_numeric_correlation_heatmap.png', dpi=200)
plt.close(fig)



# %% Notebook cell 18
# PCA — fit on all three datasets
# NOTE: df is not modified; all arrays are local to PCA cells.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PCA_COLS = [
    'Age', 'Poverty', 'Weight', 'Height', 'BMI', 'Pulse',
    'BPSysAve', 'BPDiaAve', 'DirectChol', 'TotChol', 'AlcoholYear', 'SleepHrsNight',
]

subsets = {
    'All':   df,
    'White': df[df['Race1'] == 'White'].reset_index(drop=True),
    'Black': df[df['Race1'] == 'Black'].reset_index(drop=True),
}

pca_results = {}

# Legacy full-sample PCA tables are intentionally generated on the raw
# pre-deduplication data because those tables already exist in root outputs/.
# The race-stratified PCA below follows the notebook workflow and uses the
# deduplicated analysis data.
X_raw_pca = StandardScaler().fit_transform(raw_df[PCA_COLS])
raw_pca = PCA()
raw_pca.fit(X_raw_pca)
raw_expl = raw_pca.explained_variance_ratio_
raw_cum = np.cumsum(raw_expl)
raw_loadings = pd.DataFrame(
    raw_pca.components_.T,
    index=PCA_COLS,
    columns=[f'PC{i+1}' for i in range(len(PCA_COLS))],
)
pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(PCA_COLS))],
    'eigenvalue': raw_pca.explained_variance_.round(3),
    'variance_pct': (raw_expl * 100).round(2),
    'cumulative_pct': (raw_cum * 100).round(2),
}).to_csv(PCA_TABLE_DIR / 'table_09_pca_variance.csv', index=False)
raw_loadings.round(3).to_csv(PCA_TABLE_DIR / 'table_10_pca_loadings.csv')

for name, data in subsets.items():
    X = StandardScaler().fit_transform(data[PCA_COLS])
    pca = PCA()
    scores = pca.fit_transform(X)
    n = len(PCA_COLS)
    expl = pca.explained_variance_ratio_
    cum  = np.cumsum(expl)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_COLS,
        columns=[f'PC{i+1}' for i in range(n)],
    )
    pca_results[name] = {
        'data': data, 'pca': pca, 'scores': scores,
        'expl_var': expl, 'cum_var': cum, 'loadings': loadings,
        'n_rows': len(data),
    }
    pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n)],
        'eigenvalue':     pca.explained_variance_.round(3),
        'variance_pct':   (expl * 100).round(2),
        'cumulative_pct': (cum  * 100).round(2),
    }).to_csv(PCA_TABLE_DIR / f'table_09_pca_variance_{name.lower()}.csv', index=False)
    loadings.round(3).to_csv(PCA_TABLE_DIR / f'table_10_pca_loadings_{name.lower()}.csv')

print(f"{'Dataset':<8} {'n':>6}  {'PC1':>6}  {'PC1+2':>7}  {'PC1-4':>7}")
print('-' * 40)
for name, r in pca_results.items():
    print(f"{name:<8} {r['n_rows']:>6,}  {r['expl_var'][0]:>6.1%}  {r['cum_var'][1]:>7.1%}  {r['cum_var'][3]:>7.1%}")



# %% Notebook cell 19
# Figure 06 — Scree plots side by side (All / White / Black)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

for ax, (name, r) in zip(axes, pca_results.items()):
    expl = r['expl_var']
    cum  = r['cum_var']
    n    = len(expl)
    colors = ['#4C78A8' if i < 4 else '#AABFD0' for i in range(n)]

    bars = ax.bar(range(1, n + 1), expl * 100, color=colors, zorder=2)
    for bar, v in zip(bars, expl * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=6)

    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum * 100, marker='o', markersize=4,
             color='#E45756', linewidth=1.5, zorder=3)
    ax2.axhline(80, color='#F58518', linestyle='--', linewidth=0.8)
    ax2.axhline(90, color='#B279A2', linestyle='--', linewidth=0.8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Cumulative %', fontsize=8)
    ax2.tick_params(axis='y', labelsize=7)

    ax.set_title(f'{name}  (n = {r["n_rows"]:,})', fontweight='bold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xticks(range(1, n + 1))
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    ax.set_ylim(0, expl[0] * 100 * 1.2)

fig.suptitle('PCA — Scree Plots', fontsize=14, fontweight='bold')
fig.savefig(PCA_FIGURE_DIR / 'fig_06_pca_scree.png', dpi=200)
plt.close(fig)



# %% Notebook cell 20
# Figure 07 — Loading heatmaps side by side (PC1–PC4 for All / White / Black)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, (name, r) in zip(axes, pca_results.items()):
    sns.heatmap(
        r['loadings'].iloc[:, :4], cmap='vlag', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'fontsize': 7.5},
        linewidths=0.4, cbar=(name == 'Black'),
        cbar_kws={'label': 'Loading', 'shrink': 0.8} if name == 'Black' else {},
        ax=ax,
    )
    ax.set_title(f'{name}  (n = {r["n_rows"]:,})', fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

fig.suptitle('PCA — Variable Loadings (PC1–PC4)', fontsize=14, fontweight='bold')
fig.savefig(PCA_FIGURE_DIR / 'fig_07_pca_loadings.png', dpi=200)
plt.close(fig)



# %% Notebook cell 21
# Figure 08 — Biplots side by side (PC1 vs PC2 for All / White / Black)
# Arrows scaled to the 95th percentile of the score spread in each dataset.
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for ax, (name, r) in zip(axes, pca_results.items()):
    sc = r['scores']
    ld = r['loadings']
    arrow_scale = np.percentile(np.abs(sc[:, :2]), 95) * 0.9

    ax.scatter(sc[:, 0], sc[:, 1], alpha=0.12, s=8, color='#888888', zorder=1)

    for var in PCA_COLS:
        xend = ld.loc[var, 'PC1'] * arrow_scale
        yend = ld.loc[var, 'PC2'] * arrow_scale
        ax.annotate('', xy=(xend, yend), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#E45756', lw=1.4), zorder=3)
        ax.text(xend * 1.12, yend * 1.12, var, fontsize=7, color='#E45756',
                ha='center', va='center', zorder=4)

    ax.axhline(0, color='grey', linewidth=0.4, zorder=0)
    ax.axvline(0, color='grey', linewidth=0.4, zorder=0)
    ax.set_xlabel(f'PC1 ({r["expl_var"][0]:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({r["expl_var"][1]:.1%})', fontsize=9)
    ax.set_title(f'{name}  (n = {r["n_rows"]:,})', fontweight='bold')
    ax.grid(alpha=0.15)

fig.suptitle('PCA — Biplots (PC1 vs PC2)', fontsize=14, fontweight='bold')
fig.savefig(PCA_FIGURE_DIR / 'fig_08_pca_biplot.png', dpi=200)
plt.close(fig)



# %% Notebook cell 22
# Figure 09 — Score plots: 3 datasets Ãƒâ€” 2 outcomes (Diabetes, HighBP)
# Race is omitted for White/Black subsets since it is uniform within each.
fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

outcome_specs = [
    ('Diabetes', DIABETES_ORDER, OUTCOME_PALETTE),
    ('HighBP',   HIGHBP_ORDER,   OUTCOME_PALETTE),
]

for row, (name, r) in enumerate(pca_results.items()):
    sc   = r['scores']
    data = r['data']
    for col, (outcome, order, palette) in enumerate(outcome_specs):
        ax = axes[row, col]
        for level in order:
            mask = data[outcome] == level
            ax.scatter(sc[mask, 0], sc[mask, 1],
                       alpha=0.35, s=12,
                       color=palette.get(level, '#777777'), label=level)
        ax.axhline(0, color='grey', linewidth=0.4)
        ax.axvline(0, color='grey', linewidth=0.4)
        ax.set_xlabel(f'PC1 ({r["expl_var"][0]:.1%})', fontsize=8)
        ax.set_ylabel(f'PC2 ({r["expl_var"][1]:.1%})', fontsize=8)
        ax.set_title(f'{name} — by {outcome}  (n = {r["n_rows"]:,})', fontweight='bold')
        ax.legend(title=outcome, fontsize=7, markerscale=2, framealpha=0.7)
        ax.grid(alpha=0.2)

fig.suptitle('PCA — Score Plots by Diabetes and HighBP', fontsize=14, fontweight='bold')
fig.savefig(PCA_FIGURE_DIR / 'fig_09_pca_scores.png', dpi=200)
plt.close(fig)



# %% Notebook cell 23
# Factor Analysis — fit 4-factor varimax model on all three datasets
# NOTE: subsets dict reused from PCA setup; df is not modified.
from sklearn.decomposition import FactorAnalysis

N_FACTORS = 4

fa_results = {}
for name, data in subsets.items():
    X = StandardScaler().fit_transform(data[PCA_COLS])
    fa = FactorAnalysis(n_components=N_FACTORS, rotation='varimax', random_state=42)
    fa_scores = fa.fit_transform(X)

    loadings = pd.DataFrame(
        fa.components_.T,          # shape: (n_features, n_factors)
        index=PCA_COLS,
        columns=[f'F{i+1}' for i in range(N_FACTORS)],
    )
    communalities = pd.Series(1 - fa.noise_variance_, index=PCA_COLS, name='communality')

    fa_results[name] = {
        'data': data, 'fa': fa, 'scores': fa_scores,
        'loadings': loadings, 'communalities': communalities,
        'n_rows': len(data),
    }
    loadings.round(3).to_csv(FA_TABLE_DIR / f'table_11_fa_loadings_{name.lower()}.csv')
    communalities.round(3).to_frame().to_csv(FA_TABLE_DIR / f'table_12_fa_communalities_{name.lower()}.csv')

print(f'Factor Analysis: {N_FACTORS} factors, varimax rotation\n')
print(f"{'Dataset':<8} {'n':>6}  {'Mean communality':>18}  {'Min communality':>16}")
print('-' * 52)
for name, r in fa_results.items():
    c = r['communalities']
    print(f"{name:<8} {r['n_rows']:>6,}  {c.mean():>18.3f}  {c.min():>16.3f}  ({c.idxmin()})")



# %% Notebook cell 24
# Figure 10 — FA loading heatmaps side by side (All / White / Black)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, (name, r) in zip(axes, fa_results.items()):
    sns.heatmap(
        r['loadings'], cmap='vlag', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'fontsize': 8},
        linewidths=0.4, cbar=(name == 'Black'),
        cbar_kws={'label': 'Loading', 'shrink': 0.8} if name == 'Black' else {},
        ax=ax,
    )
    ax.set_title(f'{name}  (n = {r["n_rows"]:,})', fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

fig.suptitle('Factor Analysis — Varimax Loadings (4 Factors)', fontsize=14, fontweight='bold')
fig.savefig(FA_FIGURE_DIR / 'fig_10_fa_loadings.png', dpi=200)
plt.close(fig)



# %% Notebook cell 25
# Figure 11 — Communality bar charts side by side (All / White / Black)
# Communality = proportion of each variable's variance explained by the 4 factors.
# Dashed line at 0.5 marks the conventional "adequately explained" threshold.
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

for ax, (name, r) in zip(axes, fa_results.items()):
    comm = r['communalities'].sort_values()
    colors = ['#E45756' if v < 0.5 else '#4C78A8' for v in comm]
    bars = ax.barh(comm.index, comm.values, color=colors)
    for bar, v in zip(bars, comm.values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.2f}', va='center', fontsize=7.5)
    ax.axvline(0.5, color='grey', linestyle='--', linewidth=1, label='0.5 threshold')
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Communality')
    ax.set_title(f'{name}  (n = {r["n_rows"]:,})\nmean = {comm.mean():.2f}', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='x', alpha=0.2)

fig.suptitle('Factor Analysis — Communalities (4 Factors, Varimax)\nBlue ≥ 0.5 (well explained)  |  Red < 0.5 (high uniqueness)',
             fontsize=13, fontweight='bold')
fig.savefig(FA_FIGURE_DIR / 'fig_11_fa_communalities.png', dpi=200)
plt.close(fig)



# %% Notebook cell 26
# Figure 12 — FA score plots: 3 datasets Ãƒâ€” 2 outcomes (Diabetes, HighBP)
# F1 vs F2 scores coloured by outcome — mirrors the PCA score plot layout.
outcome_specs_fa = [
    ('Diabetes', DIABETES_ORDER, OUTCOME_PALETTE),
    ('HighBP',   HIGHBP_ORDER,   OUTCOME_PALETTE),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

for row, (name, r) in enumerate(fa_results.items()):
    sc   = r['scores']
    data = r['data']
    for col, (outcome, order, palette) in enumerate(outcome_specs_fa):
        ax = axes[row, col]
        for level in order:
            mask = data[outcome] == level
            ax.scatter(sc[mask, 0], sc[mask, 1],
                       alpha=0.35, s=12,
                       color=palette.get(level, '#777777'), label=level)
        ax.axhline(0, color='grey', linewidth=0.4)
        ax.axvline(0, color='grey', linewidth=0.4)
        ax.set_xlabel('F1', fontsize=9)
        ax.set_ylabel('F2', fontsize=9)
        ax.set_title(f'{name} — by {outcome}  (n = {r["n_rows"]:,})', fontweight='bold')
        ax.legend(title=outcome, fontsize=7, markerscale=2, framealpha=0.7)
        ax.grid(alpha=0.2)

fig.suptitle('Factor Analysis — F1 vs F2 Score Plots by Diabetes and HighBP',
             fontsize=13, fontweight='bold')
fig.savefig(FA_FIGURE_DIR / 'fig_12_fa_scores.png', dpi=200)
plt.close(fig)



# %% Notebook cell 27
# Race-stratified K-Modes — elbow + final model per race
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
from kmodes.kmodes import KModes

RAND_SEED = 42
N_INIT = 10
BEST_K = 4
CLUSTER_FEATURES = [
    'Gender', 'Race1', 'Education', 'MaritalStatus', 'HHIncome',
    'HomeOwn', 'HealthGen', 'PhysActive', 'SmokeNow',
]
OUTCOME_VARS = ['Diabetes', 'HighBP']
CLUSTER_PALETTE = ['#4C78A8', '#54A24B', '#F58518', '#B279A2']
_order_map = {
    'Race1': RACE_ORDER, 'Diabetes': DIABETES_ORDER, 'HighBP': HIGHBP_ORDER,
    'Education': EDUCATION_ORDER, 'HHIncome': INCOME_ORDER, 'HealthGen': HEALTH_ORDER,
    'Gender': GENDER_ORDER, 'MaritalStatus': MARITAL_ORDER,
    'PhysActive': PHYSACTIVE_ORDER, 'SmokeNow': SMOKENOW_ORDER,
}

RACE_SUBSETS = {
    'White': df[df['Race1'] == 'White'].reset_index(drop=True),
    'Black': df[df['Race1'] == 'Black'].reset_index(drop=True),
}

race_km_results = {}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

for ax, (race, rdf) in zip(axes, RACE_SUBSETS.items()):
    X_r = rdf[CLUSTER_FEATURES].copy()
    costs = {}
    for k in range(2, 7):
        km_tmp = KModes(n_clusters=k, init='Huang', n_init=N_INIT, random_state=RAND_SEED)
        km_tmp.fit(X_r.values)
        costs[k] = float(km_tmp.cost_)
    ks, vals = list(costs.keys()), list(costs.values())
    ax.plot(ks, vals, marker="o", color=RACE_PALETTE[race], linewidth=2)
    ax.axvline(BEST_K, color='#E45756', linestyle='--', linewidth=1.4,
               label=f'Selected k={BEST_K}')
    for k, v in zip(ks, vals):
        ax.annotate(f'{v:.0f}', (k, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_title(f'"{race}” K-Modes elbow (n={len(rdf):,})', fontweight='bold')
    ax.set_xlabel('k'); ax.set_ylabel('Total cost')
    ax.legend(); ax.grid(alpha=0.3)

    km_r = KModes(n_clusters=BEST_K, init='Huang', n_init=N_INIT, random_state=RAND_SEED)
    km_r.fit(X_r.values)
    race_km_results[race] = {'km': km_r, 'df': rdf, 'labels': km_r.labels_, 'X': X_r}
    print(f'{race}: cost={km_r.cost_:.0f}')

fig.suptitle('K-Modes elbow by race', fontsize=13, fontweight='bold')
fig.savefig(CLUSTER_FIGURE_DIR / 'fig_18_race_kmodes_elbow.png', dpi=200)
plt.close(fig)



# %% Notebook cell 28
# Tables 16-17 ” race-stratified cluster modes and outcome rates
race_outcomes_list = []

for race, res in race_km_results.items():
    rdf     = res['df']
    labels  = res['labels']
    X_r     = res['X'].copy()
    X_r['cluster'] = [f'C{l+1}' for l in labels]

    # Table 16 ” cluster modes
    mode_r = pd.DataFrame(res["km"].cluster_centroids_, columns=CLUSTER_FEATURES)
    mode_r.insert(0, 'cluster', [f'C{i+1}' for i in range(BEST_K)])
    mode_r.insert(0, 'race', race)
    mode_r.to_csv(CLUSTER_TABLE_DIR / f'table_16_cluster_modes_{race.lower()}.csv', index=False)

    # Table 17 ” outcome rates per cluster
    df_out_r = rdf[OUTCOME_VARS].copy()
    df_out_r['cluster'] = X_r['cluster']
    out_records = []
    for clust in sorted(df_out_r['cluster'].unique()):
        sub = df_out_r[df_out_r['cluster'] == clust]
        rec = {
            'race': race, 'cluster': clust, 'n': len(sub),
            'pct_of_race': round(len(sub) / len(df_out_r) * 100, 1),
            'diabetes_yes_pct': round(float((sub['Diabetes'] == 'Yes').mean() * 100), 1),
            'highbp_high_pct':  round(float((sub['HighBP']   == 'High').mean() * 100), 1),
        }
        out_records.append(rec)
        race_outcomes_list.append(rec)
    out_r = pd.DataFrame(out_records)
    out_r.to_csv(CLUSTER_TABLE_DIR / f'table_17_cluster_outcomes_{race.lower()}.csv', index=False)
    print(f'\n{race} cluster outcomes:')
    display(out_r)

race_outcomes_df = pd.DataFrame(race_outcomes_list)



# %% Notebook cell 29
# Figure 19 â€” Cluster sizes per race (side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

for ax, (race, res) in zip(axes, race_km_results.items()):
    labels  = res['labels']
    counts  = pd.Series([f'C{l+1}' for l in labels]).value_counts().sort_index()
    colors  = [CLUSTER_PALETTE[i] for i in range(BEST_K)]
    bars    = ax.bar(counts.index, counts.values, color=colors)
    for bar, val in zip(bars, counts.values):
        ax.annotate(
            f'{val:,}\n({val/len(labels)*100:.1f}%)',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=9,
            xytext=(0, 4), textcoords='offset points')
    ax.set_title(f'{race} cluster sizes (k={BEST_K}, n={len(labels):,})', fontweight='bold')
    ax.set_xlabel('Cluster'); ax.set_ylabel('n')
    ax.set_ylim(0, counts.max() * 1.28); ax.grid(axis='y', alpha=0.3)

fig.suptitle('K-Modes cluster sizes by race', fontsize=13, fontweight='bold')
fig.savefig(CLUSTER_FIGURE_DIR / 'fig_19_race_cluster_sizes.png', dpi=200)
plt.close(fig)



# %% Notebook cell 30
# Figure 20 â€” Health outcome rates by cluster for each race
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
w = 0.35

for ax, race in zip(axes, ['White', 'Black']):
    sub = race_outcomes_df[race_outcomes_df['race'] == race].reset_index(drop=True)
    x = np.arange(len(sub))
    b1 = ax.bar(x - w/2, sub['diabetes_yes_pct'], w, color='#E45756', label='Diabetes = Yes')
    b2 = ax.bar(x + w/2, sub['highbp_high_pct'],  w, color='#F58518', label='HighBP = High')
    ax.bar_label(b1, fmt='%.1f%%', fontsize=8, padding=3)
    ax.bar_label(b2, fmt='%.1f%%', fontsize=8, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={int(n):,})' for c, n in zip(sub['cluster'], sub['n'])])
    ax.set_title(f'"{race}” health outcome rates by cluster', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ymax = max(sub['diabetes_yes_pct'].max(), sub['highbp_high_pct'].max())
    ax.set_ylim(0, ymax * 1.35)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

fig.suptitle('K-Modes cluster health outcomes  White vs Black', fontsize=13, fontweight='bold')
fig.savefig(CLUSTER_FIGURE_DIR / 'fig_20_race_cluster_outcomes.png', dpi=200)
plt.close(fig)



# %% Notebook cell 31
# Figure 21 â€” Cluster profile heatmaps: White (top) vs Black (bottom)
KEY_FEATS = ['Education', 'HHIncome', 'HealthGen', 'PhysActive']

fig, axes = plt.subplots(2, len(KEY_FEATS), figsize=(16, 9), constrained_layout=True)

for row_idx, (race, res) in enumerate(race_km_results.items()):
    X_tmp2 = res['X'].copy()
    X_tmp2['cluster'] = [f'C{l+1}' for l in res['labels']]
    cluster_cols = sorted(X_tmp2['cluster'].unique())

    for col_idx, feat in enumerate(KEY_FEATS):
        ax = axes[row_idx, col_idx]
        records = []
        for level in X_tmp2[feat].dropna().unique():
            for clust in cluster_cols:
                sub = X_tmp2[X_tmp2['cluster'] == clust]
                records.append({'level': level, 'cluster': clust,
                                 'pct': round(float((sub[feat] == level).mean() * 100), 1)})
        piv = (pd.DataFrame(records)
               .pivot(index='level', columns='cluster', values='pct')
               .fillna(0).reindex(columns=cluster_cols))
        if feat in _order_map:
            ordered = [l for l in _order_map[feat] if l in piv.index]
            piv = piv.reindex(ordered)
        else:
            piv = piv.sort_index()
        sns.heatmap(piv, ax=ax, cmap='Blues', vmin=0, vmax=100,
                    annot=True, fmt='.0f', annot_kws={'fontsize': 7},
                    linewidths=0.4, cbar=False)
        if row_idx == 0:
            ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('Cluster'); ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=7); ax.tick_params(axis='x', labelsize=8)
        if col_idx == 0:
            ax.set_ylabel(race, fontsize=11, fontweight='bold', rotation=90, labelpad=10)

fig.suptitle('Cluster profile heatmaps (% within cluster) â€” White vs Black',
             fontsize=13, fontweight='bold')
fig.savefig(CLUSTER_FIGURE_DIR / 'fig_21_race_profile_heatmaps.png', dpi=200)
plt.close(fig)



# %% Notebook cell 32
# Logistic regression setup — race-specific diabetes prediction
# NOTE: df is not modified; preprocessing happens inside the sklearn Pipeline.
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGIT_OUTPUT_DIR = OUTPUT_ROOT / 'logistic_regression'
LOGIT_FIGURE_DIR = LOGIT_OUTPUT_DIR / 'figures'
LOGIT_TABLE_DIR = LOGIT_OUTPUT_DIR / 'tables'
for path in [LOGIT_OUTPUT_DIR, LOGIT_FIGURE_DIR, LOGIT_TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

LOGIT_NUMERIC_COLS = ['Age', 'BMI', 'DirectChol', 'TotChol', 'Poverty']
LOGIT_CATEGORICAL_COLS = ['Gender', 'HighBP', 'PhysActive', 'SmokeNow']
LOGIT_CATEGORICAL_LEVELS = [
    ['female', 'male'],
    ['Normal', 'High'],
    ['No', 'Yes'],
    ['No', 'Yes'],
]
LOGIT_FEATURES = LOGIT_NUMERIC_COLS + LOGIT_CATEGORICAL_COLS
LOGIT_TARGET = 'Diabetes'
LOGIT_RACE_SUBSETS = {
    'White': df[df['Race1'] == 'White'].reset_index(drop=True),
    'Black': df[df['Race1'] == 'Black'].reset_index(drop=True),
}
LOGIT_THRESHOLD = 0.5
LOGIT_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_logit_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), LOGIT_NUMERIC_COLS),
            (
                'categorical',
                OneHotEncoder(
                    categories=LOGIT_CATEGORICAL_LEVELS,
                    drop='first',
                    handle_unknown='ignore',
                ),
                LOGIT_CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('model', LogisticRegression(max_iter=5000, solver='lbfgs', penalty='l2', C=1.0)),
        ]
    )

for race, data in LOGIT_RACE_SUBSETS.items():
    class_counts = data[LOGIT_TARGET].value_counts().reindex(DIABETES_ORDER, fill_value=0)
    if class_counts.min() < LOGIT_CV.n_splits:
        raise ValueError(f'{race} has too few rows in a diabetes class for {LOGIT_CV.n_splits}-fold CV: {class_counts.to_dict()}')
    print(f'{race}: n={len(data):,}; Diabetes Yes={int(class_counts["Yes"]):,}; Diabetes No={int(class_counts["No"]):,}')



# %% Notebook cell 33
# Tables 18-19 — Cross-validated prediction performance and confusion matrices
logit_results = {}
performance_records = []
confusion_records = []

for race, data in LOGIT_RACE_SUBSETS.items():
    X = data[LOGIT_FEATURES].copy()
    y = data[LOGIT_TARGET].eq('Yes').astype(int)
    pipe = make_logit_pipeline()
    y_prob = cross_val_predict(pipe, X, y, cv=LOGIT_CV, method='predict_proba')[:, 1]
    y_pred = (y_prob >= LOGIT_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    auc = roc_auc_score(y, y_prob)
    accuracy = accuracy_score(y, y_pred)

    logit_results[race] = {
        'data': data,
        'X': X,
        'y': y,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'auc': auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }
    performance_records.append({
        'race': race,
        'n': len(data),
        'diabetes_yes_n': int(y.sum()),
        'diabetes_yes_pct': round(float(y.mean() * 100), 2),
        'roc_auc': round(float(auc), 3),
        'accuracy_at_0_5': round(float(accuracy), 3),
        'sensitivity_at_0_5': round(float(sensitivity), 3),
        'specificity_at_0_5': round(float(specificity), 3),
        'predicted_positive_pct_at_0_5': round(float(y_pred.mean() * 100), 2),
    })
    confusion_records.extend([
        {'race': race, 'true_class': 'No', 'predicted_class': 'No', 'n': int(tn)},
        {'race': race, 'true_class': 'No', 'predicted_class': 'Yes', 'n': int(fp)},
        {'race': race, 'true_class': 'Yes', 'predicted_class': 'No', 'n': int(fn)},
        {'race': race, 'true_class': 'Yes', 'predicted_class': 'Yes', 'n': int(tp)},
    ])

logit_performance = pd.DataFrame(performance_records)
logit_confusion = pd.DataFrame(confusion_records)
logit_performance.to_csv(LOGIT_TABLE_DIR / 'table_18_logit_cv_performance.csv', index=False)
logit_confusion.to_csv(LOGIT_TABLE_DIR / 'table_19_logit_confusion_matrices.csv', index=False)

print('Cross-validated logistic regression performance:')
display(logit_performance)
print('Confusion matrix counts at threshold 0.5:')
display(logit_confusion)



# %% Notebook cell 34
# Figure 22 — ROC curves by race
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for race, res in logit_results.items():
    fpr, tpr, _ = roc_curve(res['y'], res['y_prob'])
    ax.plot(fpr, tpr, linewidth=2, label=f'{race} (AUC = {res["auc"]:.3f})')
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Random')
ax.set_title('Race-specific logistic regression ROC curves')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc='lower right')
ax.grid(alpha=0.25)
fig.savefig(LOGIT_FIGURE_DIR / 'fig_22_logit_roc_by_race.png', dpi=200)
plt.close(fig)



# %% Notebook cell 35
# Figure 23 — Out-of-fold predicted probabilities by true diabetes status
prob_records = []
for race, res in logit_results.items():
    for prob, y_val in zip(res['y_prob'], res['y']):
        prob_records.append({
            'race': race,
            'predicted_probability': float(prob),
            'Diabetes': 'Yes' if y_val == 1 else 'No',
        })
prob_plot = pd.DataFrame(prob_records)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharex=True, sharey=True)
for ax, race in zip(axes, ['White', 'Black']):
    sns.histplot(
        data=prob_plot[prob_plot['race'] == race],
        x='predicted_probability', hue='Diabetes', hue_order=DIABETES_ORDER,
        bins=np.linspace(0, 1, 21), stat='density', common_norm=False,
        element='step', fill=True, alpha=0.35,
        palette=[OUTCOME_PALETTE['No'], OUTCOME_PALETTE['Yes']], ax=ax,
    )
    ax.axvline(LOGIT_THRESHOLD, color='black', linestyle='--', linewidth=1, label='0.5 threshold')
    ax.set_title(f'{race} (n={len(LOGIT_RACE_SUBSETS[race]):,})', fontweight='bold')
    ax.set_xlabel('Out-of-fold predicted Pr(Diabetes = Yes)')
    ax.set_ylabel('Density')
    ax.grid(alpha=0.25)
fig.suptitle('Logistic regression predicted probability distributions by race', fontsize=13, fontweight='bold')
fig.savefig(LOGIT_FIGURE_DIR / 'fig_23_logit_predicted_probabilities_by_race.png', dpi=200)
plt.close(fig)



# %% Notebook cell 36
# Table 20 and Figure 24 — Final full-subset logistic coefficients
coef_records = []
for race, data in LOGIT_RACE_SUBSETS.items():
    X = data[LOGIT_FEATURES].copy()
    y = data[LOGIT_TARGET].eq('Yes').astype(int)
    final_model = make_logit_pipeline().fit(X, y)
    feature_names = final_model.named_steps['preprocess'].get_feature_names_out()
    coefs = final_model.named_steps['model'].coef_[0]
    for feature, coef in zip(feature_names, coefs):
        coef_records.append({
            'race': race,
            'feature': feature,
            'coefficient': round(float(coef), 4),
            'odds_ratio': round(float(np.exp(coef)), 4),
        })

logit_coefficients = pd.DataFrame(coef_records)
logit_coefficients['absolute_coefficient'] = logit_coefficients['coefficient'].abs()
logit_coefficients['absolute_coefficient_rank'] = (
    logit_coefficients
    .groupby('race')['absolute_coefficient']
    .rank(ascending=False, method='first')
    .astype(int)
)
logit_coefficients = logit_coefficients.sort_values(['race', 'absolute_coefficient_rank'])
logit_coefficients.to_csv(LOGIT_TABLE_DIR / 'table_20_logit_coefficients.csv', index=False)

display(logit_coefficients)

feature_order = (
    logit_coefficients.groupby('feature')['absolute_coefficient']
    .max().sort_values().index.tolist()
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
for ax, race in zip(axes, ['White', 'Black']):
    sub = logit_coefficients[logit_coefficients['race'] == race].set_index('feature').reindex(feature_order)
    colors = ['#E45756' if v > 0 else '#4C78A8' for v in sub['coefficient']]
    ax.barh(sub.index, sub['coefficient'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'{race} full-subset model', fontweight='bold')
    ax.set_xlabel('Logistic coefficient')
    ax.grid(axis='x', alpha=0.25)
axes[0].set_ylabel('Predictor')
fig.suptitle('Race-specific logistic regression coefficients for Diabetes = Yes - Positive values indicate higher predicted diabetes log-odds', fontsize=13, fontweight='bold')
fig.savefig(LOGIT_FIGURE_DIR / 'fig_24_logit_coefficients_by_race.png', dpi=200)
plt.close(fig)



# %% Notebook cell 37
overview_map = dict(zip(overview['metric'], overview['value']))
race_counts = df['Race1'].value_counts()
diabetes_yes_pct = pct(df['Diabetes'], 'Yes')
highbp_high_pct = pct(df['HighBP'], 'High')
black_diabetes = race_rates.loc[race_rates['race'] == 'Black', 'diabetes_yes_pct'].iloc[0]
white_diabetes = race_rates.loc[race_rates['race'] == 'White', 'diabetes_yes_pct'].iloc[0]
black_highbp  = race_rates.loc[race_rates['race'] == 'Black', 'highbp_high_pct'].iloc[0]
white_highbp  = race_rates.loc[race_rates['race'] == 'White', 'highbp_high_pct'].iloc[0]

report = f'''# Descriptive EDA: NHANES Adult Health Data

## Scope

This note covers project component 2 only: descriptive data exploration and visualization.
It does not run PCA, factor analysis, clustering, regression, classification,
hypothesis tests, or prediction models.

## Data

The source file is `data/nhanes_health.csv`. The project instructions describe this as a
simplified complete-case adult NHANES sample without survey weights, so all summaries below
describe this project sample rather than official U.S. population estimates.

Key dimensions:

- Rows: {int(overview_map["rows"]):,}
- Columns: {int(overview_map["columns"]):,}
- Missing cells: {int(overview_map["missing_cells"]):,}
- Exact duplicate rows across all selected variables: {int(overview_map["exact_duplicate_rows"]):,}
- Unique full profiles after exact-row collapse: {int(overview_map["unique_full_profiles"]):,}
- `BPDiaAve` values equal to 0: {int(overview_map["bpdiaave_zero_values"]):,}
- `HighBP` derivation mismatches: {int(overview_map["highbp_derivation_mismatches"]):,}

## Main Descriptive Findings

The sample is heavily White: {race_counts.get("White", 0):,} White rows and
{race_counts.get("Black", 0):,} Black rows out of {len(df):,}.

Overall, {diabetes_yes_pct:.1f}% of rows have `Diabetes = Yes` and
{highbp_high_pct:.1f}% have `HighBP = High`. By race, diabetes is
{black_diabetes:.1f}% for Black rows vs. {white_diabetes:.1f}% for White rows;
high BP is {black_highbp:.1f}% vs. {white_highbp:.1f}%.

## Selected Tables

Race-level descriptive rates:

{markdown_table(race_rates)}

White/Black profile:

{markdown_table(white_black_profile)}

## Handoff Notes

- Later modeling should decide whether to retain exact duplicate rows.
- The `BPDiaAve = 0` rows should be reviewed before blood-pressure modeling.
- Avoid language about national prevalence — survey weights are absent.
- The White/Black sample-size imbalance should be acknowledged in race-specific analyses.
'''

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report, encoding='utf-8')
print(f'Report written to {REPORT_PATH.relative_to(PROJECT_ROOT)}')



# %% Notebook cell 38



if __name__ == '__main__':
    print('Final centralized analysis finished.')
