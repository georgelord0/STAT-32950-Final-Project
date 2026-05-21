# STAT 32950 Final Project

This repository contains the group project work for the NHANES adult health dataset.

## Current Scope

The current completed task is descriptive data exploration and visualization for `data/nhanes_health.csv`. It is intentionally limited to dataset inspection, summary tables, and descriptive plots. Modeling tasks such as factor analysis, logistic regression, and method comparison should be added separately.

## Repository Structure

- `data/`: source data supplied for the project
- `project-instructions/`: project instructions from the course
- `scripts/`: reproducible analysis scripts
- `outputs/descriptive_eda/figures/`: generated EDA figures
- `outputs/descriptive_eda/tables/`: generated EDA summary tables
- `reports/`: short written analysis notes and handoff documents

## Reproduce the Descriptive EDA

Install the Python dependencies, then run:

```bash
python3 scripts/01_descriptive_eda.py
```

The script regenerates all descriptive EDA tables, figures, and the draft EDA report.

