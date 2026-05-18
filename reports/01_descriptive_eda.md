# Descriptive EDA: NHANES Adult Health Data

## Scope

This note covers project component 2 only: descriptive data exploration and visualization. It does not run PCA, factor analysis, clustering, regression, classification, hypothesis tests, or prediction models.

## Data

The source file is `data/nhanes_health.csv`. The project instructions describe this as a simplified complete-case adult NHANES sample without survey weights, so all summaries below describe this project sample rather than official U.S. population estimates.

Key dimensions:

- Rows: 2,412
- Columns: 23
- Missing cells: 0
- Exact duplicate rows across all selected variables: 904
- Unique full profiles after exact-row collapse: 1,508
- `BPDiaAve` values equal to 0: 4
- `HighBP` derivation mismatches from `BPSysAve >= 130` or `BPDiaAve >= 80`: 0

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

The sample is heavily White: 1,846 White rows and 213 Black rows out of 2,412. This imbalance matters for later race-specific methods because the Black subgroup is much smaller than the White subgroup.

Overall, 12.0% of rows have `Diabetes = Yes`, and 37.5% have `HighBP = High`. By race, the descriptive diabetes percentage is 17.8% for Black rows and 11.2% for White rows; the descriptive high blood pressure percentage is 41.8% for Black rows and 38.1% for White rows. These are sample percentages, not population estimates.

The cardiometabolic distributions show visible differences by diabetes status for age and BMI, with additional spread in blood pressure and cholesterol measurements. These plots are descriptive only and should be used to motivate later formal modeling rather than to claim independent effects.

Socioeconomic descriptors vary across `Race1`, especially the poverty ratio and broad household-income bands. These variables should be handled explicitly in later analyses because socioeconomic composition is part of the planned research question and may be associated with health markers.

## Selected Tables

Race-level descriptive rates:

| race | n | diabetes_yes_n | diabetes_yes_pct | highbp_high_n | highbp_high_pct | age_median | bmi_median | poverty_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| White | 1846 | 206 | 11.16 | 704 | 38.14 | 50.0 | 27.37 | 3.16 |
| Black | 213 | 38 | 17.84 | 89 | 41.78 | 47.0 | 28.6 | 1.74 |
| Mexican | 141 | 21 | 14.89 | 50 | 35.46 | 42.0 | 29.76 | 1.36 |
| Hispanic | 91 | 8 | 8.79 | 34 | 37.36 | 38.0 | 27.66 | 1.7 |
| Other | 121 | 16 | 13.22 | 27 | 22.31 | 43.0 | 25.53 | 2.05 |

White/Black profile table:

| metric | All | White | Black |
| --- | --- | --- | --- |
| n | 2,412 | 1,846 | 213 |
| Age median [IQR] | 49.00 [35.00, 61.00] | 50.00 [37.00, 63.00] | 47.00 [30.00, 59.00] |
| BMI median [IQR] | 27.54 [24.00, 32.00] | 27.37 [23.80, 31.79] | 28.60 [24.48, 34.80] |
| BPSysAve median [IQR] | 120.00 [111.00, 131.00] | 120.00 [111.00, 131.00] | 122.00 [112.00, 133.00] |
| BPDiaAve median [IQR] | 71.00 [64.00, 78.00] | 72.00 [64.00, 78.00] | 71.00 [64.00, 79.00] |
| TotChol median [IQR] | 4.99 [4.27, 5.72] | 5.07 [4.33, 5.79] | 4.60 [3.96, 5.38] |
| Poverty median [IQR] | 2.75 [1.31, 4.76] | 3.16 [1.61, 5.00] | 1.74 [1.02, 3.40] |
| Diabetes Yes (%) | 12.0 | 11.2 | 17.8 |
| HighBP High (%) | 37.5 | 38.1 | 41.8 |
| College graduate (%) | 22.6 | 25.4 | 8.0 |
| Household income $75k+ (%) | 32.7 | 36.6 | 16.9 |
| Physically active Yes (%) | 49.5 | 49.7 | 47.4 |
| Smoke now Yes (%) | 44.8 | 40.5 | 62.4 |

Duplicate multiplicity summary:

| rows_per_profile | number_of_profiles | total_rows |
| --- | --- | --- |
| 1 | 943 | 943 |
| 2 | 337 | 674 |
| 3 | 148 | 444 |
| 4 | 54 | 216 |
| 5 | 21 | 105 |
| 6 | 5 | 30 |

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
