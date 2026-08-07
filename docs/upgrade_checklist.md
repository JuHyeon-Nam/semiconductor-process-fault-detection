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
| Model breadth | Improved | Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting, and all-pass baseline are compared. |
| Threshold analysis | Improved | Validation threshold curves, PR curves, confusion matrices, and test predictions are saved per trained model. |
| Interpretability | Improved | Built-in feature importance and validation permutation importance are reported and compared. |
| Manufacturing deployment story | Improved | FDC framing, cost-sensitive thresholding, and operating workflow diagram exist. |
| System demo | Not started | FastAPI inference endpoint remains future work. |
| Interview documents | Partial | `docs/interview_notes.md` exists; semiconductor process notes and one-pager remain future work. |

## Phase 1 Completed

- Added `all_pass_baseline` to quantify the accuracy trap.
- Added accuracy, false alarm count, and missed fail count to model metrics.
- Added split-level class profile report.
- Added sensor quality report with zero-variance and missingness flags.
- Added high-correlation sensor pair report for redundant sensor analysis.
- Added figures for top missing sensors, sensor quality flags, and accuracy-vs-recall warning.
- Updated README to explain why Fail Recall, F2, and PR-AUC are more appropriate than plain Accuracy.

## Phase 2 Completed

- Added weighted/unweighted Logistic Regression comparison.
- Added weighted/unweighted RandomForest comparison.
- Added `extra_trees_balanced` as an additional tree ensemble baseline.
- Saved per-model artifacts under `reports/models/<model_name>/`.
- Added `reports/model_comparison.md` for reviewer-friendly model comparison.
- Updated README result table to reflect the new best model, `extra_trees_balanced`.

## Phase 3 Started

- Added validation-set permutation importance with average precision scoring.
- Added `reports/permutation_importance.csv`.
- Added `reports/importance_comparison.csv`.
- Added `reports/figures/permutation_importance.png`.
- Added `reports/figures/importance_comparison.png`.
- Updated README and run summary to explain built-in importance vs permutation importance.
- Added `docs/interview_notes.md` with a simple project explanation and sensor-anonymization interview answer.

## Phase 4 Started

- Added cost-sensitive threshold scenarios with assumed false alarm and missed fail costs.
- Added `reports/cost_threshold_analysis.csv`.
- Added `reports/cost_threshold_curves.csv`.
- Added `reports/figures/cost_threshold_analysis.png`.
- Updated README and run summary to explain how threshold changes when missed failures are more expensive.
- Added `reports/figures/fdc_operating_workflow.png` to connect sensor data, model scoring, alarm review, PM/recipe checks, and feedback.
- Updated README and interview notes to frame the model as FDC-style manufacturing decision support rather than automatic final judgment.

## Next Implementation Queue

1. Phase 2 residual: consider optional XGBoost/LightGBM only if dependency setup stays lightweight.
2. Phase 3 residual: consider optional SHAP only if dependency setup stays lightweight and stable.
3. Phase 5: add a small FastAPI demo with `/health`, `/model-info`, and `/predict`.
4. Phase 6: add `docs/semiconductor_process_notes.md` and `docs/portfolio_onepager.md`.
