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

- Model: random_forest
- Threshold: 0.08
- Fail Recall: 0.7619
- Fail Precision: 0.1569
- Fail F1: 0.2602
- Fail F2: 0.4301
- PR-AUC: 0.2150
- Accuracy: 0.7102

## Confusion Matrix

| | Pred Pass | Pred Fail |
|---|---:|---:|
| True Pass | 207 | 86 |
| True Fail | 5 | 16 |

## Accuracy Trap Baseline

An all-pass rule reaches 0.9331 accuracy on the test split, but its fail recall is 0.0000. This is why the project reports Fail Recall, F2, and PR-AUC instead of treating accuracy as the primary metric.

## Top Sensor Candidates

| feature | importance |
|---|---:|
| sensor_103 | 0.015516 |
| sensor_059 | 0.015463 |
| sensor_477 | 0.009988 |
| sensor_180 | 0.009166 |
| sensor_129 | 0.008894 |
| sensor_205 | 0.007902 |
| sensor_341 | 0.007729 |
| sensor_039 | 0.007254 |
| sensor_130 | 0.007154 |
| sensor_125 | 0.007047 |

## Interview Message

반도체 제조 데이터는 결측과 불균형이 큰 데이터라고 보고, 정상/불량 정확도보다 불량 미탐을 줄이는 Recall, F2, PR-AUC를 중심으로 평가했습니다. 임계값은 validation set에서 F2 기준으로 결정하고, test set에서 최종 성능을 확인했습니다. 주요 센서 후보는 feature importance로 정리해 FDC, 설비 이상탐지, 수율 개선 관점으로 확장할 수 있게 분석했습니다.
