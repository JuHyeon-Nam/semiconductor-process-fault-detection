# SECOM Run Summary

## Data Quality Findings

- Samples: 1,567
- Sensors: 590
- Fail ratio: 6.64%
- Zero-variance sensors: 116
- Sensors with >=50% missing values: 28
- Highly correlated sensor pairs with abs(correlation) >=0.98: 201

## Split Profile

| split | total | pass | fail | fail_ratio |
|---|---:|---:|---:|---:|
| train | 939 | 877 | 62 | 0.0660 |
| validation | 314 | 293 | 21 | 0.0669 |
| test | 314 | 293 | 21 | 0.0669 |

## Top Missing Sensors

| sensor | missing_count | missing_ratio |
|---|---:|---:|
| sensor_157 | 1429 | 0.9119 |
| sensor_292 | 1429 | 0.9119 |
| sensor_293 | 1429 | 0.9119 |
| sensor_158 | 1429 | 0.9119 |
| sensor_492 | 1341 | 0.8558 |
| sensor_358 | 1341 | 0.8558 |
| sensor_085 | 1341 | 0.8558 |
| sensor_220 | 1341 | 0.8558 |
| sensor_246 | 1018 | 0.6496 |
| sensor_109 | 1018 | 0.6496 |

## Best Model

- Model: extra_trees_balanced
- Threshold: 0.10
- Fail Recall: 0.7619
- Fail Precision: 0.1951
- Fail F1: 0.3107
- Fail F2: 0.4819
- PR-AUC: 0.2174
- Accuracy: 0.7739

## Confusion Matrix

| | Pred Pass | Pred Fail |
|---|---:|---:|
| True Pass | 227 | 66 |
| True Fail | 5 | 16 |

## Accuracy Trap Baseline

An all-pass rule reaches 0.9331 accuracy on the test split, but its fail recall is 0.0000. This is why the project reports Fail Recall, F2, and PR-AUC instead of treating accuracy as the primary metric.

## Model Comparison

| model | threshold | accuracy | fail_recall | fail_f2 | pr_auc | missed_fail | false_alarm |
|---|---:|---:|---:|---:|---:|---:|---:|
| extra_trees_balanced | 0.10 | 0.7739 | 0.7619 | 0.4819 | 0.2174 | 5 | 66 |
| random_forest_unweighted | 0.10 | 0.8025 | 0.6667 | 0.4575 | 0.1701 | 7 | 55 |
| random_forest_balanced | 0.08 | 0.7102 | 0.7619 | 0.4301 | 0.2150 | 5 | 86 |
| logistic_regression_unweighted | 0.06 | 0.8089 | 0.2381 | 0.1880 | 0.1220 | 16 | 44 |
| logistic_regression_balanced | 0.62 | 0.8599 | 0.1905 | 0.1739 | 0.1219 | 17 | 27 |
| hist_gradient_boosting | 0.02 | 0.9172 | 0.0952 | 0.1075 | 0.2137 | 19 | 7 |
| all_pass_baseline | 1.00 | 0.9331 | 0.0000 | 0.0000 | 0.0669 | 21 | 0 |

Per-model PR curves, threshold curves, confusion matrices, and test predictions are saved under `reports/models/<model_name>/`.

## Top Sensor Candidates

| feature | importance |
|---|---:|
| sensor_129 | 0.009227 |
| sensor_511 | 0.008027 |
| sensor_064 | 0.007861 |
| sensor_059 | 0.006802 |
| sensor_065 | 0.006011 |
| sensor_103 | 0.005898 |
| sensor_028 | 0.005651 |
| sensor_122 | 0.004884 |
| sensor_130 | 0.004874 |
| sensor_452 | 0.004635 |

## Permutation Importance

Permutation importance is calculated on the validation split with average precision scoring. It answers: if a sensor is randomly shuffled, how much does fail-detection ranking quality drop?

| feature | permutation_importance_mean | std |
|---|---:|---:|
| sensor_064 | 0.023783 | 0.012989 |
| sensor_065 | 0.012138 | 0.009946 |
| sensor_037 | 0.009045 | 0.002907 |
| sensor_028 | 0.005910 | 0.006846 |
| sensor_419 | 0.005770 | 0.003499 |
| sensor_076 | 0.005670 | 0.002932 |
| sensor_210 | 0.005547 | 0.003664 |
| sensor_031 | 0.005145 | 0.002841 |
| sensor_295 | 0.004913 | 0.001205 |
| sensor_125 | 0.004687 | 0.004729 |

## Importance Comparison

Built-in tree importance and permutation importance are complementary. Built-in importance shows how the fitted ensemble used features internally, while permutation importance checks whether validation performance depends on each feature.

| feature | built_in_rank | permutation_rank | built_in_importance | permutation_importance_mean |
|---|---:|---:|---:|---:|
| sensor_064 | 3 | 1 | 0.007861 | 0.023783 |
| sensor_065 | 5 | 2 | 0.006011 | 0.012138 |
| sensor_028 | 7 | 4 | 0.005651 | 0.005910 |
| sensor_419 | 29 | 5 | 0.003657 | 0.005770 |
| sensor_125 | 27 | 10 | 0.003738 | 0.004687 |
| sensor_499 | 24 | 13 | 0.003788 | 0.004310 |
| sensor_129 | 1 | 15 | 0.009227 | 0.003846 |
| sensor_510 | 23 | 18 | 0.003809 | 0.003521 |
| sensor_316 | 14 | 22 | 0.004293 | 0.003171 |

## Interview Message

반도체 제조 데이터는 결측과 불균형이 큰 데이터라고 보고, 정상/불량 정확도보다 불량 미탐을 줄이는 Recall, F2, PR-AUC를 중심으로 평가했습니다. 임계값은 validation set에서 F2 기준으로 결정하고, test set에서 최종 성능을 확인했습니다. 주요 센서 후보는 built-in feature importance와 validation permutation importance를 함께 보며 FDC, 설비 이상탐지, 수율 개선 관점의 엔지니어 검토 후보로 해석했습니다.
