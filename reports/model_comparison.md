# Model Comparison Report

## Purpose

This report compares supervised baselines, tree-based models, an all-pass baseline, and a pass-only anomaly detection baseline for SECOM fail detection under the same train/validation/test split. Thresholds are selected on the validation set by F2 score, then evaluated once on the held-out test set.

## Result Table

| model | threshold | accuracy | fail_recall | fail_precision | fail_f2 | pr_auc | missed_fail | false_alarm | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| extra_trees_balanced | 0.10 | 0.7739 | 0.7619 | 0.1951 | 0.4819 | 0.2174 | 5 | 66 | Randomized tree ensemble with class_weight=balanced. |
| random_forest_unweighted | 0.10 | 0.8025 | 0.6667 | 0.2029 | 0.4575 | 0.1701 | 7 | 55 | Tree ensemble without class weighting. |
| random_forest_balanced | 0.08 | 0.7102 | 0.7619 | 0.1569 | 0.4301 | 0.2150 | 5 | 86 | Tree ensemble with class_weight=balanced_subsample. |
| isolation_forest_pass_only | 0.06 | 0.0955 | 0.9524 | 0.0660 | 0.2584 | 0.1606 | 1 | 283 | Unsupervised anomaly baseline trained only on pass samples. |
| logistic_regression_unweighted | 0.06 | 0.8089 | 0.2381 | 0.1020 | 0.1880 | 0.1220 | 16 | 44 | Linear baseline without class weighting. |
| logistic_regression_balanced | 0.62 | 0.8599 | 0.1905 | 0.1290 | 0.1739 | 0.1219 | 17 | 27 | Linear baseline with class_weight=balanced. |
| hist_gradient_boosting | 0.02 | 0.9172 | 0.0952 | 0.2222 | 0.1075 | 0.2137 | 19 | 7 | Histogram gradient boosting baseline. |
| all_pass_baseline | 1.00 | 0.9331 | 0.0000 | 0.0000 | 0.0000 | 0.0669 | 21 | 0 | Naive baseline that predicts every sample as pass. |

## Interpretation

- Best F2 model: `extra_trees_balanced` with fail recall 0.7619, fail precision 0.1951, and F2 0.4819.
- The all-pass baseline reaches 0.9331 accuracy, but it misses all 21 fail cases.
- Compared with `random_forest_balanced`, `extra_trees_balanced` keeps the same test fail recall while reducing false alarms from 86 to 66.
- `isolation_forest_pass_only` is useful as a pass-only anomaly screening reference, but its false alarm count is too high for the selected operating model.
- This is still a decision-support PoC, not a production-ready FDC model. The main value is the explicit metric, threshold, and trade-off analysis.

## Per-Model Artifacts

Each trained model has these generated files under `reports/models/<model_name>/`:

- `summary.csv`
- `validation_threshold_curve.csv`
- `test_predictions.csv`
- `precision_recall_curve.png`
- `threshold_tradeoff.png`
- `confusion_matrix.png`
