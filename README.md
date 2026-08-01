# Semiconductor Process Fault Detection

반도체 제조 공정 센서 데이터(SECOM)를 활용해 불량 여부를 예측하고, 주요 원인 센서를 해석하는 제조AI 프로젝트입니다.

## 목표

- 반도체 제조 데이터의 결측치, 불균형, 고차원 센서 변수 문제를 다룬다.
- 단순 정확도보다 불량 미탐을 줄이는 Recall, F1, PR-AUC를 중심으로 평가한다.
- feature importance로 주요 센서 변수를 해석해 FDC/양산기술 관점으로 연결한다.

## SK하이닉스 지원 연결

이 프로젝트는 기존 `ServiceRobot_AI`의 로봇 센서 기반 예지보전 경험을 반도체 제조 공정 데이터로 확장한 포트폴리오입니다.

| 기존 경험 | 본 프로젝트 |
|---|---|
| 로봇 센서 기반 고장진단 | 반도체 공정 센서 기반 불량 예측 |
| PdM/PHM | FDC/수율/공정 이상탐지 |
| LightGBM/feature importance | tree model/feature importance |
| 센서 한계 분석 | 결측/불균형/label 한계 분석 |

## 반도체 5개 공정 연결

SK하이닉스 취업 준비 관점에서는 반도체 전공정을 아래 5개 묶음으로 기억합니다.

| 공정 | 데이터 관점 |
|---|---|
| Photo | CD, overlay, 노광 조건, 패턴 불량 |
| Etch | 식각률, 균일도, endpoint, chamber 상태 |
| Diffusion | 온도, 시간, 농도, 열처리 조건 |
| Thin Film | 막 두께, 균일도, gas/pressure |
| CMP/Cleaning | particle, scratch, slurry, 세정 조건 |

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_data.py
python src/train.py
```

## 산출물

- `reports/metrics.csv`: 모델별 성능
- `reports/top_features.csv`: 주요 센서 변수
- `reports/run_summary.md`: 지원서/면접용 요약

## 현재 실행 결과

| 모델 | Threshold | Fail Recall | Fail Precision | Fail F2 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.10 | 0.6154 | 0.1928 | 0.4278 | 0.2856 |

일반 Accuracy만 보면 불량 데이터가 적은 제조 데이터에서 모델이 정상 위주로 예측해도 좋아 보일 수 있습니다. 그래서 이 프로젝트에서는 불량 미탐을 더 크게 벌주는 F2 기준으로 임계값을 조정했습니다.

## 면접용 한 문장

반도체 제조 데이터는 결측과 불균형이 크기 때문에 단순 정확도보다 불량을 놓치지 않는 Recall과 PR-AUC를 중심으로 평가했고, feature importance를 통해 현장 엔지니어가 확인할 수 있는 주요 센서 후보를 정리했습니다.
