# Portfolio Upgrade Checklist

Goal: make this repository credible for semiconductor manufacturing AI, FDC, yield/quality analytics, and smart-factory roles by improving code, analysis, reporting, reproducibility, and interview readiness.

## Current Gap Review

| Area | Status | Notes |
|---|---|---|
| Raw data reproducibility | Done | Raw SECOM files are downloaded through `src/fetch_data.py` and ignored by git. |
| Output validation | Improved | `src/validate_outputs.py` checks reports, metrics, artifacts, dashboard, Markdown links, and git tracking policy. |
| Split hygiene | Done | Train/validation/test splits are stratified and saved to `reports/split_class_profile.csv`. |
| Leakage control | Done | Threshold selection is performed on validation scores; test is used for final evaluation only. |
| Class imbalance explanation | Improved | Class profile, class imbalance figure, and all-pass baseline now show why accuracy is misleading. |
| Data quality EDA | Improved | Missingness, zero-variance sensors, and high-correlation sensor pairs are now reported. |
| Baseline comparison | Improved | `all_pass_baseline` is included in `reports/metrics.csv` and README results. |
| Model breadth | Improved | Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting, pass-only IsolationForest, and all-pass baseline are compared. |
| Threshold analysis | Improved | Validation threshold curves, PR curves, confusion matrices, and test predictions are saved per trained model. |
| Score interpretation | Improved | Score band analysis checks whether high-risk scores concentrate actual fail samples. |
| Interpretability | Improved | Built-in feature importance and validation permutation importance are reported and compared. |
| Manufacturing deployment story | Improved | FDC framing, cost-sensitive thresholding, and operating workflow diagram exist. |
| System demo | Improved | FastAPI `/health`, `/model-info`, and `/predict` endpoints exist with sample input and smoke test. |
| Result review dashboard | Improved | Static dashboard is generated at `reports/dashboard.html` from metrics and figures. |
| Model card | Improved | Generated `reports/model_card.md` captures intended use, risks, and monitoring needs. |
| Interview documents | Improved | Interview notes, interview Q&A, semiconductor process notes, submission guide, final review, and one-page summary exist. |

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
- Added `isolation_forest_pass_only` as a pass-only anomaly detection baseline.
- Saved per-model artifacts under `reports/models/<model_name>/`.
- Added `reports/model_comparison.md` for reviewer-friendly model comparison.
- Updated README result table to reflect the new best model, `extra_trees_balanced`.

## Phase 3 Completed

- Added validation-set permutation importance with average precision scoring.
- Added `reports/permutation_importance.csv`.
- Added `reports/importance_comparison.csv`.
- Added `reports/figures/permutation_importance.png`.
- Added `reports/figures/importance_comparison.png`.
- Updated README and run summary to explain built-in importance vs permutation importance.
- Added `docs/interview_notes.md` with a simple project explanation and sensor-anonymization interview answer.

## Phase 4 Completed

- Added cost-sensitive threshold scenarios with assumed false alarm and missed fail costs.
- Added `reports/cost_threshold_analysis.csv`.
- Added `reports/cost_threshold_curves.csv`.
- Added `reports/score_band_analysis.csv` and `reports/figures/score_band_analysis.png` to inspect score ranking behavior.
- Added `reports/figures/cost_threshold_analysis.png`.
- Updated README and run summary to explain how threshold changes when missed failures are more expensive.
- Added `reports/figures/fdc_operating_workflow.png` to connect sensor data, model scoring, alarm review, PM/recipe checks, and feedback.
- Updated README and interview notes to frame the model as FDC-style manufacturing decision support rather than automatic final judgment.

## Phase 5 Completed

- Added trained model artifact generation in `src/train.py`.
- Added `src/api.py` with FastAPI `/health`, `/model-info`, and `/predict` endpoints.
- Added `src/make_sample_input.py` to create a reproducible JSON payload from SECOM rows.
- Added `src/smoke_test_api.py` to verify the API without manually running a server.
- Updated README with API run commands and endpoint descriptions.
- Added `src/build_dashboard.py` and generated `reports/dashboard.html` for a compact result review page.
- Added generated `reports/model_card.md` for intended use, non-goals, operating risks, and monitoring conditions.

## Phase 6 Completed

- Added `docs/semiconductor_process_notes.md` to connect Photo, Etch, Diffusion, Thin Film, and CMP/Cleaning to manufacturing data analysis.
- Added `docs/portfolio_onepager.md` as a compact project summary.
- Added `docs/final_review.md` for final readiness, limitations, and interview framing.
- Updated README so these explanation documents are visible from the main page.

## Final Validation Completed

- Added `src/validate_outputs.py` to check required generated files, key metric consistency, per-model artifacts, dashboard content, inference artifact metadata, Markdown links, and git raw-data policy.
- Added `reports/validation_summary.md` as a generated audit trail.
- Updated README with the validation command.
- Added `Makefile` with `make reproduce`, `make validate`, and `make api-smoke`.
- Added `docs/submission_guide.md` and `docs/interview_qa.md`.
- Extended validation to check the final documentation set.

## Remaining Optional Work

1. Phase 2 residual: consider optional XGBoost/LightGBM only if dependency setup stays lightweight.
2. Phase 3 residual: consider optional SHAP only if dependency setup stays lightweight and stable.
3. Optional anomaly residual: compare a lightweight AutoEncoder only if dependency and runtime stay stable.
4. Final polish: review generated dashboard, validation summary, README, and interview Q&A together for consistency.
