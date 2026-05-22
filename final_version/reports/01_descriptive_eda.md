# Descriptive EDA: NHANES Adult Health Data

## Scope

This note covers project component 2 only: descriptive data exploration and visualization.
It does not run PCA, factor analysis, clustering, regression, classification,
hypothesis tests, or prediction models.

## Data

The source file is `data/nhanes_health.csv`. The project instructions describe this as a
simplified complete-case adult NHANES sample without survey weights, so all summaries below
describe this project sample rather than official U.S. population estimates.

Key dimensions:

- Rows: 1,504
- Columns: 23
- Missing cells: 0
- Exact duplicate rows across all selected variables: 0
- Unique full profiles after exact-row collapse: 1,504
- `BPDiaAve` values equal to 0: 0
- `HighBP` derivation mismatches: 0

## Main Descriptive Findings

The sample is heavily White: 1,052 White rows and
183 Black rows out of 1,504.

Overall, 13.5% of rows have `Diabetes = Yes` and
38.3% have `HighBP = High`. By race, diabetes is
18.6% for Black rows vs. 12.3% for White rows;
high BP is 44.3% vs. 38.1%.

## Selected Tables

Race-level descriptive rates:

| race | n | diabetes_yes_n | diabetes_yes_pct | highbp_high_n | highbp_high_pct | age_median | bmi_median | poverty_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| White | 1052 | 129 | 12.26 | 401 | 38.12 | 51.0 | 27.45 | 2.76 |
| Black | 183 | 34 | 18.58 | 81 | 44.26 | 48.0 | 28.6 | 1.79 |
| Mexican | 115 | 20 | 17.39 | 45 | 39.13 | 41.0 | 29.63 | 1.35 |
| Hispanic | 71 | 7 | 9.86 | 29 | 40.85 | 40.0 | 28.0 | 1.57 |
| Other | 83 | 13 | 15.66 | 20 | 24.1 | 43.0 | 25.53 | 1.84 |

White/Black profile:

| metric | All | White | Black |
| --- | --- | --- | --- |
| n | 1,504 | 1,052 | 183 |
| Age median [IQR] | 49.00 [34.00, 62.00] | 51.00 [36.75, 64.00] | 48.00 [32.00, 60.00] |
| BMI median [IQR] | 27.70 [24.20, 32.14] | 27.45 [24.00, 31.91] | 28.60 [24.52, 34.65] |
| BPSysAve median [IQR] | 120.00 [111.00, 131.00] | 120.00 [111.00, 131.00] | 123.00 [112.00, 134.00] |
| BPDiaAve median [IQR] | 71.00 [63.00, 78.00] | 71.00 [63.00, 78.00] | 72.00 [64.00, 79.50] |
| TotChol median [IQR] | 4.97 [4.27, 5.72] | 5.04 [4.29, 5.77] | 4.63 [4.03, 5.46] |
| Poverty median [IQR] | 2.43 [1.19, 4.35] | 2.76 [1.31, 4.79] | 1.79 [1.02, 3.45] |
| Diabetes Yes (%) | 13.5 | 12.3 | 18.6 |
| HighBP High (%) | 38.3 | 38.1 | 44.3 |
| College graduate (%) | 19.9 | 22.9 | 8.7 |
| Household income $75k+ (%) | 28.9 | 32.5 | 18.6 |
| Physically active Yes (%) | 48.9 | 49.1 | 47.5 |
| Smoke now Yes (%) | 46.5 | 42.2 | 60.7 |

## Handoff Notes

- Later modeling should decide whether to retain exact duplicate rows.
- Rows with `BPDiaAve = 0` are removed before blood-pressure modeling.
- Avoid language about national prevalence — survey weights are absent.
- The White/Black sample-size imbalance should be acknowledged in race-specific analyses.
