# SECOM Run Summary

## Best Model

- Model: random_forest
- Threshold: 0.10
- Fail Recall: 0.6154
- Fail Precision: 0.1928
- Fail F1: 0.2936
- Fail F2: 0.4278
- PR-AUC: 0.2856

## Top Sensor Candidates

| feature | importance |
|---|---:|
| sensor_059 | 0.012652 |
| sensor_103 | 0.011477 |
| sensor_477 | 0.009378 |
| sensor_033 | 0.008722 |
| sensor_519 | 0.008477 |
| sensor_510 | 0.008331 |
| sensor_213 | 0.007856 |
| sensor_064 | 0.007802 |
| sensor_031 | 0.007391 |
| sensor_130 | 0.007300 |

## Interview Message

반도체 제조 데이터는 결측과 불균형이 큰 데이터라고 보고, 정상/불량 정확도보다 불량 미탐을 줄이는 Recall, F2, PR-AUC를 중심으로 평가했습니다. 특히 기본 임계값 0.5에서는 불량을 놓칠 수 있어, 불량 미탐을 더 크게 벌주는 F2 기준으로 의사결정 임계값을 조정했습니다. 주요 센서 후보를 feature importance로 정리해 FDC, 설비 이상탐지, 수율 개선 관점으로 확장할 수 있게 분석했습니다.
