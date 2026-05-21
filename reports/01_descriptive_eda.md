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

- Rows: 1,508
- Columns: 23
- Missing cells: 0
- Exact duplicate rows across all selected variables: 0
- Unique full profiles after exact-row collapse: 1,508
- `BPDiaAve` values equal to 0: 4
- `HighBP` derivation mismatches: 0

## Main Descriptive Findings

The sample is heavily White: 1,055 White rows and
184 Black rows out of 1,508.

Overall, 13.5% of rows have `Diabetes = Yes` and
38.3% have `HighBP = High`. By race, diabetes is
18.5% for Black rows vs. 12.3% for White rows;
high BP is 44.0% vs. 38.2%.

## Selected Tables

Race-level descriptive rates:

| race | n | diabetes_yes_n | diabetes_yes_pct | highbp_high_n | highbp_high_pct | age_median | bmi_median | poverty_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| White | 1055 | 130 | 12.32 | 403 | 38.2 | 51.0 | 27.46 | 2.75 |
| Black | 184 | 34 | 18.48 | 81 | 44.02 | 48.0 | 28.6 | 1.79 |
| Mexican | 115 | 20 | 17.39 | 45 | 39.13 | 41.0 | 29.63 | 1.35 |
| Hispanic | 71 | 7 | 9.86 | 29 | 40.85 | 40.0 | 28.0 | 1.57 |
| Other | 83 | 13 | 15.66 | 20 | 24.1 | 43.0 | 25.53 | 1.84 |

White/Black profile:

| metric | All | White | Black |
| --- | --- | --- | --- |
| n | 1,508 | 1,055 | 184 |
| Age median [IQR] | 49.00 [34.00, 62.00] | 51.00 [37.00, 64.00] | 48.00 [32.00, 60.00] |
| BMI median [IQR] | 27.70 [24.19, 32.16] | 27.46 [24.00, 31.98] | 28.60 [24.48, 34.62] |
| BPSysAve median [IQR] | 120.00 [111.00, 131.00] | 120.00 [111.00, 131.50] | 123.00 [112.00, 134.00] |
| BPDiaAve median [IQR] | 71.00 [63.00, 78.00] | 71.00 [63.00, 78.00] | 71.50 [64.00, 79.25] |
| TotChol median [IQR] | 4.97 [4.26, 5.72] | 5.04 [4.29, 5.77] | 4.63 [4.03, 5.44] |
| Poverty median [IQR] | 2.43 [1.19, 4.34] | 2.75 [1.31, 4.77] | 1.79 [1.03, 3.44] |
| Diabetes Yes (%) | 13.5 | 12.3 | 18.5 |
| HighBP High (%) | 38.3 | 38.2 | 44.0 |
| College graduate (%) | 19.9 | 22.8 | 8.7 |
| Household income $75k+ (%) | 28.8 | 32.4 | 18.5 |
| Physically active Yes (%) | 48.8 | 49.1 | 47.3 |
| Smoke now Yes (%) | 46.6 | 42.2 | 60.9 |

## Handoff Notes

- Later modeling should decide whether to retain exact duplicate rows.
- The `BPDiaAve = 0` rows should be reviewed before blood-pressure modeling.
- Avoid language about national prevalence — survey weights are absent.
- The White/Black sample-size imbalance should be acknowledged in race-specific analyses.
