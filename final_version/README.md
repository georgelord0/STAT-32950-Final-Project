# Final version workspace

This folder centralizes the submission-facing analysis without modifying the existing notebook or root outputs.

- `final_analysis.py`: runnable centralized analysis script generated from `notebooks/01_descriptive_eda.ipynb`; writes to `final_version/outputs/`.
- `final_analysis.ipynb`: notebook copy for review.
- `final_report.tex`: LaTeX starter for the final 4--6 page report.
- `verification_summary.md`: comparison between regenerated final-version outputs and the existing root `outputs/` artifacts.

Run from the repository root:

```bash
python3 final_version/final_analysis.py
```

If `kmodes` is not installed in the active Python environment, install it into the local vendor directory:

```bash
python3 -m pip install --target final_version/vendor kmodes
```
