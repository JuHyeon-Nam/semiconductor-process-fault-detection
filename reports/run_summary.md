# SECOM Run Summary

## Best Model

- Model: random_forest
- Threshold: 0.08
- Fail Recall: 0.7619
- Fail Precision: 0.1569
- Fail F1: 0.2602
- Fail F2: 0.4301
- PR-AUC: 0.2150

## Confusion Matrix

| | Pred Pass | Pred Fail |
|---|---:|---:|
| True Pass | 207 | 86 |
| True Fail | 5 | 16 |

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
