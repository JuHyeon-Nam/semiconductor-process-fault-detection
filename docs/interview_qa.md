# Interview Q&A

## 1. 이 프로젝트를 한 문장으로 설명하면?

반도체 제조 센서 데이터를 이용해 불량 위험 wafer를 조기에 선별하는 FDC-style decision-support PoC입니다.

## 2. 스펙을 맞추는 프로젝트인가, 수율을 보는 프로젝트인가?

수율 리스크 조기 탐지에 가깝습니다. 모델이 recipe나 spec을 직접 조정하는 것이 아니라, 센서 패턴을 기반으로 fail-risk score를 만들고 threshold를 넘는 샘플을 엔지니어 review 대상으로 올리는 구조입니다.

## 3. 왜 accuracy를 주 지표로 쓰지 않았나?

SECOM test split에는 정상 293개, 불량 21개가 있습니다. 모든 샘플을 정상으로 예측해도 accuracy는 0.9331이지만 fail recall은 0입니다. 제조 문제에서 놓친 불량이 핵심 리스크라서 Fail Recall, F2, PR-AUC, missed fail count를 함께 봤습니다.

## 4. 왜 F2를 썼나?

F2는 precision보다 recall에 더 큰 가중치를 둡니다. FDC-style screening에서는 불량을 놓치는 비용이 추가 review보다 클 수 있으므로, recall을 더 중시하는 지표가 문제 정의에 맞습니다.

## 5. test leakage는 어떻게 피했나?

학습은 train split에서 했고, threshold 선택은 validation split에서만 했습니다. test split은 최종 평가에만 사용했습니다. 모델이나 threshold를 test 결과에 맞춰 다시 고르지 않았습니다.

## 6. 최종 선택 모델과 결과는?

`extra_trees_balanced`입니다. validation에서 선택한 threshold는 0.10이고, test fail recall은 0.7619, F2는 0.4819, missed fail은 5개, false alarm은 66개입니다.

## 7. IsolationForest는 recall이 더 높은데 왜 선택하지 않았나?

IsolationForest는 정상 샘플만 학습한 anomaly baseline입니다. test fail recall은 0.9524로 높지만 false alarm이 283개로 너무 많습니다. 실제 운영에서는 엔지니어 review capacity가 있으므로, 불량을 많이 잡는 것만큼 false alarm 부담도 봐야 합니다.

## 8. threshold를 0.5가 아니라 0.10으로 둔 이유는?

불균형 데이터에서 0.5 threshold는 기본 확률 기준일 뿐 제조 운영 기준이 아닙니다. 이 프로젝트에서는 validation split에서 F2가 가장 좋은 threshold를 선택했습니다. 그리고 cost-sensitive 분석으로 missed fail 비용을 높게 보면 threshold가 더 내려가는 것도 확인했습니다.

## 9. false alarm이 많아도 괜찮은가?

무조건 괜찮다고 보지 않습니다. false alarm은 엔지니어 review 비용입니다. 그래서 false alarm count와 missed fail count를 함께 보고, 비용 가정별 threshold 변화를 별도 리포트로 만들었습니다.

## 10. review capacity 분석은 왜 추가했나?

실제 운영에서는 알람을 무한히 볼 수 없습니다. 그래서 `reports/review_capacity_analysis.csv`에서 상위 score 5%, 10%, 20%, 30%, 50%만 review할 때 불량을 몇 개 잡는지 확인했습니다. 이 분석은 모델 score를 엔지니어 업무량과 연결합니다.

## 11. cost-sensitive threshold 분석은 어떤 의미인가?

가상의 비용 단위로 false alarm과 missed fail의 상대 비용을 바꿔보는 분석입니다. missed fail 비용을 크게 두면 threshold가 0.10에서 0.06으로 내려가고 missed fail은 5개에서 1개로 줄지만 false alarm은 66개에서 207개로 늘어납니다.

## 12. score가 실제 불량 확률이라고 볼 수 있나?

그렇게 과장하지 않습니다. 이 프로젝트에서는 score를 물리적으로 보정된 불량 확률이 아니라 review 우선순위를 정하는 ranking signal로 봅니다. 그래서 `reports/score_band_analysis.csv`에서 상위 score 구간에 실제 fail이 얼마나 몰리는지 확인했습니다.

## 13. 센서명이 익명인데 feature importance가 의미 있나?

물리적 root cause를 단정하는 용도는 아닙니다. 대신 sensor candidate prioritization으로 의미가 있습니다. 실제 fab 데이터라면 이 후보를 FDC tag, chamber, recipe step, PM 이력, metrology 결과와 매핑해서 엔지니어가 검토해야 합니다.

## 14. built-in importance와 permutation importance를 왜 둘 다 봤나?

Built-in importance는 tree ensemble이 학습 중 어떤 feature를 많이 사용했는지 보여줍니다. Permutation importance는 validation 성능이 해당 feature에 실제로 얼마나 의존하는지 확인합니다. correlated sensor가 많은 제조 데이터에서는 두 관점을 같이 보는 편이 더 안전합니다.

## 15. PR-AUC가 낮아 보이는데 문제 아닌가?

불량 비율이 6.64%인 매우 불균형 데이터라 PR-AUC가 높게 나오기 어렵습니다. 그래서 절대값만 보기보다 all-pass baseline, missed fail count, false alarm count, F2, threshold curve와 함께 해석했습니다. 성능을 과장하지 않는 것이 더 중요하다고 봤습니다.

## 16. FastAPI는 왜 넣었나?

보고서에서 끝내지 않고 모델 score를 시스템이 호출할 수 있는 최소 인터페이스를 보여주기 위해 넣었습니다. `/health`, `/model-info`, `/predict`를 제공하지만, production deployment가 아니라 decision-support PoC입니다.

## 17. 실제 현장 데이터라면 다음에 무엇을 하겠나?

센서 후보를 실제 FDC tag와 recipe step에 매핑하고, lot/wafer 이력, chamber, PM 이력, metrology, inspection, E-test 결과를 연결하겠습니다. 그 다음 review capacity와 missed fail 비용을 바탕으로 threshold 운영점을 다시 정하겠습니다.

## 18. 이 프로젝트에서 가장 중요한 배운 점은?

제조 AI에서는 모델 정확도 하나보다 문제 정의가 중요합니다. 결측과 불균형을 먼저 이해하고, test leakage를 피하고, threshold와 비용 trade-off를 설명하고, 모델 결과를 엔지니어 review 흐름으로 연결해야 합니다.

## 19. 이 프로젝트의 한계는?

SECOM은 센서명이 익명화되어 있고 데이터 규모도 크지 않습니다. 그래서 물리적 원인을 확정하거나 production 성능을 주장할 수 없습니다. 대신 이 프로젝트는 제조 센서 데이터 문제를 다루는 분석 절차와 의사결정 구조를 보여주는 PoC입니다.

## 20. 가장 먼저 보여줄 산출물은?

`README.md`, `reports/dashboard.html`, `reports/model_card.md`입니다. README는 프로젝트의 전체 논리를 보여주고, dashboard는 metric, threshold, cost, workflow, sensor candidate를 한 화면에서 보여줍니다. model card는 intended use, 한계, 운영 리스크, monitoring 조건을 정리합니다.

## 21. 재현성은 어떻게 확인하나?

`make reproduce`를 실행하면 의존성 설치, 데이터 다운로드, 학습/리포트 생성, output validation, API smoke test를 순서대로 실행합니다. `src/validate_outputs.py`는 산출물, 핵심 metric, 문서 링크, git policy를 함께 확인합니다.

## 22. 왜 이게 단순 예제보다 낫다고 볼 수 있나?

단순히 모델 성능표만 만든 것이 아니라, all-pass baseline, class imbalance, validation thresholding, cost-sensitive threshold, score band review, review capacity analysis, anomaly baseline, sensor interpretation, FDC workflow, model card, FastAPI, dashboard, validation checks까지 하나의 제조 데이터 문제 해결 흐름으로 연결했기 때문입니다.
