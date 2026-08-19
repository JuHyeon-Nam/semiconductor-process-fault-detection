# Model Card

## Model Summary

| Item | Value |
|---|---|
| Selected model | `extra_trees_balanced` |
| Positive class | `fail` |
| Score | estimated fail-risk score |
| Operating threshold | 0.10, selected on validation data by F2 |
| Intended decision | route high-risk samples to engineering review |
| Not intended for | automated recipe control, physical root-cause confirmation, or production release without fab validation |

## Intended Use

This model is a manufacturing decision-support PoC for SECOM semiconductor sensor data. It is designed to demonstrate an FDC-style workflow: sensor snapshot, fail-risk scoring, alarm decision, engineer review, and feedback into maintenance or process investigation.

The useful output is not a final verdict on wafer quality. The useful output is a ranked review signal that helps prioritize samples where missed-fail risk may be more expensive than review effort.

## Data

| Item | Value |
|---|---:|
| Samples | 1567 |
| Sensors | 590 |
| Fail samples | 104 |
| Fail ratio | 0.0664 |
| Zero-variance sensors | 116 |
| Sensors with >=50% missing values | 28 |

## Split

| split | samples | fail count | fail ratio |
|---|---:|---:|---:|
| train | 939 | 62 | 0.0660 |
| validation | 314 | 21 | 0.0669 |
| test | 314 | 21 | 0.0669 |

Train and validation data are used for model fitting and threshold selection. The test split is used only for final evaluation.

## Main Test Metrics

| Metric | Value |
|---|---:|
| Fail recall | 0.7619 |
| Fail precision | 0.1951 |
| Fail F2 | 0.4819 |
| PR-AUC | 0.2174 |
| Accuracy | 0.7739 |
| Missed fail count | 5 |
| False alarm count | 66 |

## Why Accuracy Is Not The Main Metric

The all-pass baseline reaches 0.9331 accuracy but has 0.0000 fail recall. In this dataset, a high accuracy score can still mean the model misses every fail sample. That is why this project uses fail recall, F2, PR-AUC, missed fail count, and false alarm count.

## Operating Trade-Off

At the selected F2 threshold of 0.10, the model catches 16 fail samples and misses 5. A more yield-risk-sensitive cost assumption selects threshold 0.06, reducing missed fails to 1 but increasing false alarms to 207.

The pass-only IsolationForest baseline reaches 0.9524 fail recall, but creates 283 false alarms. It is useful as an anomaly-screening reference, not as the selected operating model.

## Interpretation Policy

Sensor names are anonymized, so important features are not treated as confirmed physical root causes. Built-in feature importance and permutation importance are used as sensor-candidate prioritization signals for engineering review.

## Operational Risks

- Dataset size is small and fail samples are rare.
- Sensor identity is anonymized, limiting physical process interpretation.
- False alarms can overload review capacity if threshold is set too low.
- Missed fails can be more costly than review effort in yield-risk-sensitive settings.
- Any production use would need tool/process-specific validation, drift monitoring, and periodic threshold review.

## Monitoring Before Production Use

Track fail-rate drift, missingness drift, score distribution drift, false alarm review rate, confirmed missed-fail events, and threshold stability. Refit or recalibrate only after checking whether process conditions, maintenance events, or sensor collection logic changed.
