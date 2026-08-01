# Portfolio Upgrade Checklist

Goal: make this repository credible for semiconductor manufacturing AI, FDC, yield/quality analytics, and smart-factory roles by improving code, analysis, reporting, reproducibility, and interview readiness.

## Current Gap Review

| Area | Status | Notes |
|---|---|---|
| Raw data reproducibility | Done | Raw SECOM files are downloaded through `src/fetch_data.py` and ignored by git. |
| Split hygiene | Done | Train/validation/test splits are stratified and saved to `reports/split_class_profile.csv`. |
| Leakage control | Done | Threshold selection is performed on validation scores; test is used for final evaluation only. |
| Class imbalance explanation | Improved | Class profile, class imbalance figure, and all-pass baseline now show why accuracy is misleading. |
| Data quality EDA | Improved | Missingness, zero-variance sensors, and high-correlation sensor pairs are now reported. |
| Baseline comparison | Improved | `all_pass_baseline` is included in `reports/metrics.csv` and README results. |
| Model breadth | Partial | Logistic Regression, RandomForest, and HistGradientBoosting are present; optional XGBoost/LightGBM remain future work. |
| Threshold analysis | Partial | Best-model validation threshold curve is saved; per-model threshold artifacts remain future work. |
| Interpretability | Partial | Built-in feature importance exists; permutation importance and optional SHAP remain future work. |
| Manufacturing deployment story | Partial | FDC framing exists in README; cost-sensitive thresholding and workflow diagrams remain future work. |
| System demo | Not started | FastAPI inference endpoint remains future work. |
| Interview documents | Not started | Dedicated interview notes and semiconductor process notes remain future work. |

## Phase 1 Completed

- Added `all_pass_baseline` to quantify the accuracy trap.
- Added accuracy, false alarm count, and missed fail count to model metrics.
- Added split-level class profile report.
- Added sensor quality report with zero-variance and missingness flags.
- Added high-correlation sensor pair report for redundant sensor analysis.
- Added figures for top missing sensors, sensor quality flags, and accuracy-vs-recall warning.
- Updated README to explain why Fail Recall, F2, and PR-AUC are more appropriate than plain Accuracy.

## Next Implementation Queue

1. Phase 2: add optional model baselines such as `balanced_random_forest` style settings, calibrated Logistic Regression, and optional XGBoost/LightGBM when dependencies are available.
2. Phase 2: save PR curve, threshold trade-off, and confusion matrix per model instead of only for the best model.
3. Phase 3: add permutation importance and compare it with built-in feature importance.
4. Phase 4: add cost-sensitive threshold analysis with assumed false alarm and missed failure costs.
5. Phase 5: add a small FastAPI demo with `/health`, `/model-info`, and `/predict`.
6. Phase 6: add `docs/interview_notes.md`, `docs/semiconductor_process_notes.md`, and `docs/portfolio_onepager.md`.
