# Rubric verification

Checked against `Multivariate_Stats_Project.tex`, the generated `Multivariate_Stats_Project.pdf`, and a fresh run of `final_analysis.py`.

| Requirement | Status | Evidence |
| --- | --- | --- |
| Clear research question motivated by dataset | Pass | `Research Question and Data` states the White/Black diabetes-risk-profile question and ties it to the available outcome, race label, cardiometabolic variables, and socioeconomic variables. |
| Descriptive data exploration and visualization | Pass | `Data Description` uses sample composition, outcome-rate, cardiometabolic, socioeconomic, and correlation figures generated from `outputs/descriptive_eda/`. |
| At least two course methods | Pass | The report uses PCA, factor analysis, k-modes clustering, and logistic regression. |
| Multivariate structure method | Pass | PCA, factor analysis, and k-modes clustering all address unsupervised multivariate structure. |
| Prediction or supervised comparison method | Pass | Race-specific logistic regression predicts `Diabetes = Yes` with cross-validated AUC, sensitivity, and specificity. |
| Compare methods and explain differences | Pass | `Comparison Across Methods` explains what each method answers and why the conclusions agree or differ. |
| Clear reproducible report with figures/tables | Pass | The PDF is 8 pages, compiles cleanly, and includes 3 tables plus 5 figures with enlarged layouts for readability. |
| Report requirements | Pass | The report includes the research question, dataset description, preprocessing, methods, implementation detail, figures/tables, interpretation, method comparison, limitations, and full group names. Reproducibility is handled by the centralized `final_analysis.py` file rather than a standalone report section. |
| Writing style request | Pass | The report is written in continuous paragraphs without itemized or enumerated sections. |
| Fresh reproducibility check | Pass | `python final_analysis.py` completed successfully and regenerated the analysis outputs used in the report. |
