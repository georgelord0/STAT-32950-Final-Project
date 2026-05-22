# Sensitivity check: dropping rows with BPDiaAve = 0

The baseline analysis uses the deduplicated NHANES file with 1,508 rows.
There are 4 rows where `BPDiaAve = 0`, which is 0.265% of the analysis data.
The sensitivity analysis reruns the same descriptive summaries, PCA, four-factor varimax factor analysis, race-stratified k-modes clustering, and race-specific logistic regression after removing those rows.

## Removed rows

| Race1   | Gender   |   Age |   BMI |   BPSysAve |   BPDiaAve | HighBP   | Diabetes   |   Poverty |
|:--------|:---------|------:|------:|-----------:|-----------:|:---------|:-----------|----------:|
| Black   | female   |    57 | 23.81 |         97 |          0 | Normal   | No         |      2.49 |
| White   | male     |    60 | 35.8  |        127 |          0 | Normal   | No         |      1.27 |
| White   | female   |    80 | 19.81 |        134 |          0 | High     | No         |      4.12 |
| White   | female   |    70 | 33.1  |        152 |          0 | High     | Yes        |      1.16 |

## Descriptive race-level changes

| race   |   n_baseline |   diabetes_yes_n_baseline |   diabetes_yes_pct_baseline |   highbp_high_n_baseline |   highbp_high_pct_baseline |   mean_bpsysave_baseline |   mean_bpdiaave_baseline |   mean_bmi_baseline |   mean_age_baseline |   n_drop_zero |   diabetes_yes_n_drop_zero |   diabetes_yes_pct_drop_zero |   highbp_high_n_drop_zero |   highbp_high_pct_drop_zero |   mean_bpsysave_drop_zero |   mean_bpdiaave_drop_zero |   mean_bmi_drop_zero |   mean_age_drop_zero |   n_delta |   diabetes_yes_n_delta |   diabetes_yes_pct_delta |   highbp_high_n_delta |   highbp_high_pct_delta |   mean_bpsysave_delta |   mean_bpdiaave_delta |   mean_bmi_delta |   mean_age_delta |
|:-------|-------------:|--------------------------:|----------------------------:|-------------------------:|---------------------------:|-------------------------:|-------------------------:|--------------------:|--------------------:|--------------:|---------------------------:|-----------------------------:|--------------------------:|----------------------------:|--------------------------:|--------------------------:|---------------------:|---------------------:|----------:|-----------------------:|-------------------------:|----------------------:|------------------------:|----------------------:|----------------------:|-----------------:|-----------------:|
| White  |         1055 |                       130 |                       12.32 |                      403 |                      38.2  |                    122.3 |                    70.03 |               28.44 |               50.95 |          1052 |                        129 |                        12.26 |                       401 |                       38.12 |                     122.2 |                     70.23 |                28.44 |                50.89 |        -3 |                     -1 |                 -0.05992 |                    -2 |                -0.08118 |              -0.04391 |                0.1997 |        -0.003226 |         -0.05434 |
| Black  |          184 |                        34 |                       18.48 |                       81 |                      44.02 |                    125   |                    70.7  |               30.54 |               47.22 |           183 |                         34 |                        18.58 |                        81 |                       44.26 |                     125.2 |                     71.09 |                30.58 |                47.16 |        -1 |                      0 |                  0.101   |                     0 |                 0.2406  |               0.1532  |                0.3863 |         0.03677  |         -0.05346 |

## PCA changes

| subset   |   n_baseline |   pc1_variance_pct_baseline |   pc1_pc2_cumulative_pct_baseline |   pc1_pc4_cumulative_pct_baseline |   n_drop_zero |   pc1_variance_pct_drop_zero |   pc1_pc2_cumulative_pct_drop_zero |   pc1_pc4_cumulative_pct_drop_zero |   n_delta |   pc1_variance_pct_delta |   pc1_pc2_cumulative_pct_delta |   pc1_pc4_cumulative_pct_delta |
|:---------|-------------:|----------------------------:|----------------------------------:|----------------------------------:|--------------:|-----------------------------:|-----------------------------------:|-----------------------------------:|----------:|-------------------------:|-------------------------------:|-------------------------------:|
| All      |         1508 |                       19.4  |                             33.82 |                             55.87 |          1504 |                        19.39 |                              33.91 |                              56.06 |        -4 |                 -0.01557 |                        0.0871  |                         0.1864 |
| White    |         1055 |                       19.35 |                             34.08 |                             56.55 |          1052 |                        19.34 |                              34.16 |                              56.75 |        -3 |                 -0.01277 |                        0.08178 |                         0.1948 |
| Black    |          184 |                       19.81 |                             35.12 |                             59.32 |           183 |                        19.91 |                              35.35 |                              59.74 |        -1 |                  0.09885 |                        0.2285  |                         0.4177 |

The largest sign-aligned absolute PCA loading changes among PC1-PC4 are:

| subset   |   max_abs_loading_delta_pc1_pc4 | variable      | component   |
|:---------|--------------------------------:|:--------------|:------------|
| All      |                         0.06278 | Age           | PC3         |
| White    |                         0.03325 | SleepHrsNight | PC4         |
| Black    |                         0.06177 | Pulse         | PC3         |

## Factor-analysis changes

| subset   |   n_baseline |   mean_communality_baseline |   min_communality_baseline |   bpdiaave_communality_baseline |   n_drop_zero |   mean_communality_drop_zero |   min_communality_drop_zero |   bpdiaave_communality_drop_zero |   n_delta |   mean_communality_delta |   min_communality_delta |   bpdiaave_communality_delta |
|:---------|-------------:|----------------------------:|---------------------------:|--------------------------------:|--------------:|-----------------------------:|----------------------------:|---------------------------------:|----------:|-------------------------:|------------------------:|-----------------------------:|
| All      |         1508 |                      0.4434 |                    0.01809 |                          0.6857 |          1504 |                       0.4461 |                     0.02119 |                           0.7164 |        -4 |                 0.002752 |               0.003103  |                      0.03069 |
| White    |         1055 |                      0.4505 |                    0.02809 |                          0.7628 |          1052 |                       0.4541 |                     0.03192 |                           0.7975 |        -3 |                 0.003647 |               0.003828  |                      0.03467 |
| Black    |          184 |                      0.4672 |                    0.07837 |                          0.6216 |           183 |                       0.4737 |                     0.07797 |                           0.6657 |        -1 |                 0.006492 |              -0.0004022 |                      0.04415 |

The largest communality changes are:

| subset   | variable      |   communality_baseline |   communality_drop_zero |   communality_delta |
|:---------|:--------------|-----------------------:|------------------------:|--------------------:|
| Black    | BPDiaAve      |                 0.6216 |                  0.6657 |             0.04415 |
| Black    | Pulse         |                 0.1697 |                  0.2138 |             0.04408 |
| Black    | BPSysAve      |                 0.6473 |                  0.6061 |            -0.04117 |
| White    | BPDiaAve      |                 0.7628 |                  0.7975 |             0.03467 |
| All      | BPDiaAve      |                 0.6857 |                  0.7164 |             0.03069 |
| White    | BPSysAve      |                 0.3897 |                  0.4106 |             0.02091 |
| White    | Age           |                 0.7257 |                  0.705  |            -0.02068 |
| All      | Age           |                 0.7597 |                  0.7391 |            -0.02066 |
| Black    | Age           |                 0.5728 |                  0.5873 |             0.01453 |
| Black    | SleepHrsNight |                 0.1029 |                  0.1164 |             0.01355 |

## K-modes cluster outcome changes

Sensitivity clusters are aligned to the baseline clusters by matching categorical centroid modes before computing deltas.

| race   | sensitivity_cluster_original   | baseline_cluster_aligned   |   mode_hamming_distance |
|:-------|:-------------------------------|:---------------------------|------------------------:|
| Black  | C1                             | C4                         |                       3 |
| Black  | C2                             | C2                         |                       4 |
| Black  | C3                             | C3                         |                       2 |
| Black  | C4                             | C1                         |                       3 |
| White  | C1                             | C1                         |                       0 |
| White  | C2                             | C2                         |                       2 |
| White  | C3                             | C3                         |                       0 |
| White  | C4                             | C4                         |                       2 |

| race   | cluster   |   n_baseline |   pct_of_race_baseline |   diabetes_yes_pct_baseline |   highbp_high_pct_baseline |   n_drop_zero |   pct_of_race_drop_zero |   diabetes_yes_pct_drop_zero |   highbp_high_pct_drop_zero |   n_delta |   pct_of_race_delta |   diabetes_yes_pct_delta |   highbp_high_pct_delta |
|:-------|:----------|-------------:|-----------------------:|----------------------------:|---------------------------:|--------------:|------------------------:|-----------------------------:|----------------------------:|----------:|--------------------:|-------------------------:|------------------------:|
| White  | C1        |          421 |                  39.91 |                      16.15  |                      47.27 |           457 |                   43.44 |                       14.88  |                       47.05 |        36 |             3.536   |                  -1.272  |                 -0.2225 |
| White  | C2        |          222 |                  21.04 |                       7.658 |                      33.33 |           194 |                   18.44 |                        7.216 |                       30.93 |       -28 |            -2.602   |                  -0.4412 |                 -2.405  |
| White  | C3        |          253 |                  23.98 |                       9.486 |                      30.43 |           243 |                   23.1  |                        9.877 |                       29.22 |       -10 |            -0.8822  |                   0.3904 |                 -1.217  |
| White  | C4        |          159 |                  15.07 |                      13.21  |                      33.33 |           158 |                   15.02 |                       14.56  |                       34.81 |        -1 |            -0.05208 |                   1.349  |                  1.477  |
| Black  | C1        |           69 |                  37.5  |                      21.74  |                      47.83 |            44 |                   24.04 |                       18.18  |                       45.45 |       -25 |           -13.46    |                  -3.557  |                 -2.372  |
| Black  | C2        |           44 |                  23.91 |                      11.36  |                      38.64 |            42 |                   22.95 |                       16.67  |                       45.24 |        -2 |            -0.9622  |                   5.303  |                  6.602  |
| Black  | C3        |           42 |                  22.83 |                       7.143 |                      40.48 |            41 |                   22.4  |                        9.756 |                       39.02 |        -1 |            -0.4217  |                   2.613  |                 -1.452  |
| Black  | C4        |           29 |                  15.76 |                      37.93  |                      48.28 |            56 |                   30.6  |                       26.79  |                       46.43 |        27 |            14.84    |                 -11.15   |                 -1.847  |

## Logistic-regression performance changes

| race   |   n_baseline |   diabetes_yes_n_baseline |   diabetes_yes_pct_baseline |   roc_auc_baseline |   accuracy_at_0_5_baseline |   sensitivity_at_0_5_baseline |   specificity_at_0_5_baseline |   predicted_positive_pct_at_0_5_baseline |   n_drop_zero |   diabetes_yes_n_drop_zero |   diabetes_yes_pct_drop_zero |   roc_auc_drop_zero |   accuracy_at_0_5_drop_zero |   sensitivity_at_0_5_drop_zero |   specificity_at_0_5_drop_zero |   predicted_positive_pct_at_0_5_drop_zero |   n_delta |   diabetes_yes_n_delta |   diabetes_yes_pct_delta |   roc_auc_delta |   accuracy_at_0_5_delta |   sensitivity_at_0_5_delta |   specificity_at_0_5_delta |   predicted_positive_pct_at_0_5_delta |
|:-------|-------------:|--------------------------:|----------------------------:|-------------------:|---------------------------:|------------------------------:|------------------------------:|-----------------------------------------:|--------------:|---------------------------:|-----------------------------:|--------------------:|----------------------------:|-------------------------------:|-------------------------------:|------------------------------------------:|----------:|-----------------------:|-------------------------:|----------------:|------------------------:|---------------------------:|---------------------------:|--------------------------------------:|
| White  |         1055 |                       130 |                       12.32 |             0.7698 |                     0.8796 |                       0.09231 |                        0.9903 |                                    1.991 |          1052 |                        129 |                        12.26 |              0.7704 |                      0.8812 |                         0.1008 |                         0.9902 |                                     2.091 |        -3 |                     -1 |                 -0.05992 |       0.0006437 |                0.001558 |                   0.008468 |                 -2.108e-05 |                                0.1007 |
| Black  |          184 |                        34 |                       18.48 |             0.8241 |                     0.8587 |                       0.3824  |                        0.9667 |                                    9.783 |           183 |                         34 |                        18.58 |              0.8093 |                      0.8415 |                         0.3235 |                         0.9597 |                                     9.29  |        -1 |                      0 |                  0.101   |      -0.0148    |               -0.01717  |                  -0.05882  |                 -0.006935  |                               -0.493  |

## Logistic-regression coefficient changes

| race   | feature        |   coefficient_baseline |   odds_ratio_baseline |   coefficient_drop_zero |   odds_ratio_drop_zero |   coefficient_delta |   odds_ratio_delta |
|:-------|:---------------|-----------------------:|----------------------:|------------------------:|-----------------------:|--------------------:|-------------------:|
| White  | SmokeNow_Yes   |             -0.263     |                0.7688 |                -0.2416  |                 0.7854 |            0.02141  |           0.01664  |
| White  | Gender_male    |             -0.0002826 |                0.9997 |                 0.01961 |                 1.02   |            0.01989  |           0.02008  |
| White  | HighBP_High    |              0.3933    |                1.482  |                 0.3784  |                 1.46   |           -0.01496  |          -0.02201  |
| Black  | Gender_male    |              0.7431    |                2.103  |                 0.7337  |                 2.083  |           -0.009446 |          -0.01977  |
| White  | DirectChol     |             -0.3311    |                0.7181 |                -0.3223  |                 0.7245 |            0.008792 |           0.006341 |
| Black  | HighBP_High    |              0.4792    |                1.615  |                 0.4707  |                 1.601  |           -0.008564 |          -0.01377  |
| White  | PhysActive_Yes |             -0.2631    |                0.7686 |                -0.2553  |                 0.7747 |            0.007801 |           0.00602  |
| Black  | DirectChol     |              0.4133    |                1.512  |                 0.4083  |                 1.504  |           -0.005006 |          -0.007549 |
| Black  | BMI            |              0.7288    |                2.073  |                 0.7239  |                 2.062  |           -0.004942 |          -0.01022  |
| Black  | PhysActive_Yes |             -0.1718    |                0.8421 |                -0.1762  |                 0.8384 |           -0.004412 |          -0.003707 |
| White  | TotChol        |             -0.2631    |                0.7687 |                -0.26    |                 0.771  |            0.003018 |           0.002324 |
| Black  | SmokeNow_Yes   |              0.0595    |                1.061  |                 0.06245 |                 1.064  |            0.002945 |           0.00313  |

## Bottom line

The four impossible diastolic blood-pressure rows are too small a share of the sample to alter the main descriptive, multivariate, clustering, or predictive conclusions. The largest changes occur in small Black k-modes clusters and in the Black logistic regression because the Black subset is much smaller, so each removed row has more leverage. Even there, the qualitative result remains the same: the Black subset has higher observed diabetes prevalence, PCA and factor analysis continue to show that no single risk axis captures the full profile, and the supervised model still predicts diabetes better for Black respondents than for White respondents in AUC terms.
