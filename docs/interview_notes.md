# Interview Notes

## 30-Second Project Explanation

이 프로젝트는 반도체 제조 공정에서 나오는 센서 데이터를 보고, 최종 제품이 정상인지 불량 위험이 있는지 미리 판단해보는 FDC 스타일 제조 데이터 분석 프로젝트입니다.

쉽게 말하면, 설비에서 온도, 압력, 전류, 진동 같은 센서값이 많이 쌓인다고 가정하고, 그 신호 패턴을 이용해서 "이 웨이퍼는 불량 가능성이 있으니 엔지니어가 먼저 확인해야 한다"는 알람 후보를 만드는 작업입니다. 스펙을 직접 튜닝하는 프로젝트라기보다는 수율 리스크를 빨리 찾는 프로젝트에 가깝습니다.

## What Problem This Solves

반도체 제조에서는 불량 샘플이 정상 샘플보다 훨씬 적습니다. 그래서 단순히 accuracy만 보면 모델이 모든 샘플을 정상이라고 해도 좋아 보일 수 있습니다. 이 프로젝트에서는 그런 착시를 피하기 위해 all-pass baseline을 만들고, Fail Recall, F2, PR-AUC를 중심으로 평가했습니다.

핵심 목표는 "무조건 높은 정확도"가 아니라 "불량을 놓치지 않기 위한 의사결정 기준을 어떻게 잡을 것인가"입니다.

## My Role and Flow

1. UCI SECOM 반도체 센서 데이터를 다운로드 스크립트로 재현 가능하게 구성했습니다.
2. 결측률, zero-variance 센서, 중복성이 높은 센서쌍, class imbalance를 먼저 분석했습니다.
3. train/validation/test split을 나누고, validation에서 threshold를 결정했습니다.
4. test set은 최종 평가에만 사용해서 leakage를 피했습니다.
5. Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting, all-pass baseline을 비교했습니다.
6. built-in feature importance와 permutation importance를 함께 보고 센서 후보를 정리했습니다.
7. false alarm과 missed fail의 비용 가정을 바꿔가며 threshold가 어떻게 바뀌는지 분석했습니다.

## Best Current Result

| Item | Value |
|---|---:|
| Best model | extra_trees_balanced |
| Threshold | 0.10 |
| Fail recall | 0.7619 |
| Fail precision | 0.1951 |
| Fail F2 | 0.4819 |
| PR-AUC | 0.2174 |
| Missed fail | 5 |
| False alarm | 66 |

All-pass baseline은 test accuracy가 0.9331이지만 fail recall은 0.0000입니다. 그래서 accuracy가 아니라 불량 recall과 F2를 중심으로 봤습니다.

## Cost-Sensitive Threshold

현재 best threshold 0.10은 test에서 missed fail 5개, false alarm 66개입니다. 그런데 missed fail 비용을 false alarm보다 훨씬 크게 보면 validation 기준 최적 threshold가 0.06으로 내려갑니다. 이 경우 test recall은 0.9524로 올라가고 missed fail은 1개로 줄지만, false alarm은 207개로 늘어납니다.

이 결과는 모델 성능 자체보다 현장 운영 정책이 중요하다는 점을 보여줍니다. 불량 미탐을 거의 없애야 하는 상황이면 많은 알람을 감수하고 threshold를 낮출 수 있고, 엔지니어 review capacity가 제한적이면 threshold를 높게 유지해야 합니다.

면접 답변 문장:

"저는 threshold를 단순히 0.5로 고정하지 않고 validation set에서 F2와 비용 가정 기준으로 선택했습니다. missed fail 비용을 크게 보면 threshold가 낮아져 recall은 올라가지만 false alarm도 늘어납니다. 그래서 이 모델은 자동 판정기라기보다, 현장의 비용 구조와 review capacity에 맞춰 운영점을 정해야 하는 decision-support 모델로 해석했습니다."

## Why F2 Instead of Accuracy

제조 현장에서는 불량을 놓치는 비용이 단순 추가 검사보다 클 수 있습니다. Accuracy는 정상 샘플이 많은 데이터에서 과대평가되기 쉽습니다. F2는 precision보다 recall에 더 큰 가중치를 주기 때문에, 불량 미탐을 줄이는 목적에 더 맞습니다.

이 프로젝트에서 all-pass baseline은 accuracy만 보면 93.31%로 좋아 보이지만, 불량 21개 중 21개를 모두 놓칩니다. 그래서 accuracy만 쓰면 제조 AI 문제를 잘못 해석할 수 있습니다.

## How I Avoided Test Leakage

모델 학습은 train set에서 했고, threshold 선택은 validation set에서 F2 기준으로 했습니다. test set은 최종 결과 확인에만 사용했습니다. 모델이나 threshold를 test 결과에 맞춰 다시 고르지 않는 구조로 만들었습니다.

## How to Explain Anonymous Sensors

SECOM 데이터는 센서명이 익명화되어 있어서 `sensor_064`가 실제로 어떤 장비 tag인지, 어떤 공정 변수인지는 알 수 없습니다. 그래서 저는 이 결과를 물리적 root cause 단정이 아니라 sensor candidate prioritization으로 해석했습니다.

실제 fab 데이터라면 다음 단계에서 `sensor_064`, `sensor_065`, `sensor_028` 같은 후보를 MES/FDC tag, chamber, recipe step, PM 이력, 계측 결과와 매핑해서 공정/설비 엔지니어가 검토해야 합니다.

면접 답변 문장:

"센서명이 익명이라 특정 물리 원인을 단정하지는 않았습니다. 대신 모델 내부 중요도와 permutation importance를 함께 봐서, 불량 예측에 영향을 주는 센서 후보를 우선순위화했습니다. 실제 현장 데이터라면 이 후보를 장비 tag와 recipe step에 연결해 엔지니어 검토 대상으로 넘기는 방식이 맞다고 봤습니다."

## Built-in Importance vs Permutation Importance

Built-in importance는 tree ensemble이 학습 과정에서 어떤 센서를 자주, 강하게 사용했는지를 보여줍니다. 하지만 correlated sensor가 많으면 중요도가 나뉘거나 치우칠 수 있습니다.

Permutation importance는 validation set에서 특정 센서 값을 섞었을 때 PR-AUC가 얼마나 떨어지는지를 봅니다. 즉, 모델 성능이 실제로 그 센서에 얼마나 의존하는지 확인하는 방식입니다.

두 결과가 겹치는 센서가 더 강한 후보입니다. 현재는 `sensor_064`, `sensor_065`, `sensor_028`이 두 중요도 관점에서 모두 상위권에 들어와서 우선 검토 후보로 볼 수 있습니다.

## Manufacturing Interpretation

이 모델은 production-ready 자동 판정 모델이 아닙니다. 더 정확한 표현은 manufacturing decision-support PoC입니다.

현장 적용 흐름으로 보면:

1. 공정 센서 데이터 수집
2. 결측/이상치/중복 센서 품질 점검
3. 모델이 fail risk score 계산
4. validation에서 정한 threshold 이상이면 알람 후보 생성
5. 엔지니어가 장비 상태, recipe, PM 이력, 계측 결과와 함께 확인
6. 필요하면 추가 검사, recipe 조정, PM 검토로 연결

## What I Would Improve Next

1. Permutation importance를 반복 실행해 중요 센서 안정성을 확인합니다.
2. FDC 운영 흐름도를 추가해 공정 센서, 알람, 엔지니어 review, PM/recipe 조정까지 연결합니다.
3. FastAPI endpoint를 만들어 `/predict`, `/health`, `/model-info`로 현장 시스템 연결 가능성을 보여줍니다.
4. 실제 tag가 있는 데이터라면 공정 step, chamber, recipe, PM 이력과 연결해 해석력을 높입니다.
