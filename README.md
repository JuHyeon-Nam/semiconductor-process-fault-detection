# Semiconductor Manufacturing Fault Detection

> 반도체 제조 공정 센서 데이터 기반 **불량 예측 · 임계값 최적화 · 센서 원인 후보 해석** 프로젝트  
> FDC-style fail risk screening with imbalanced process sensor data

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat-square&logo=numpy&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=flat-square)
![FDC](https://img.shields.io/badge/FDC-Fail%20Risk%20Screening-0B3D91?style=flat-square)

---

## At A Glance

| Item | Summary |
|---|---|
| Dataset | UCI SECOM semiconductor manufacturing sensor data |
| Task | Pass/Fail prediction from 590 anonymized process sensor variables |
| Problem Type | Imbalanced binary classification + fault detection thresholding |
| Best Model | ExtraTrees with validation-based threshold optimization |
| Key Metric | Fail Recall / F2 / PR-AUC, not plain Accuracy |
| Best Test Result | Fail Recall **0.7619**, Fail F2 **0.4819** |
| Baseline Check | All-pass baseline reaches **0.9331 accuracy** but **0.0000 fail recall** |
| Main Output | Static dashboard, data quality profile, model comparison, threshold artifacts, interpretability reports |

![Result dashboard](reports/figures/result_dashboard.png)

---

## Why I Built This

반도체 제조 데이터는 일반적인 예제 데이터처럼 깨끗하지 않습니다. 센서 결측치가 많고, 불량 샘플은 정상 샘플에 비해 매우 적습니다. 이런 상황에서 Accuracy만 보면 모델이 대부분을 정상으로 예측해도 좋아 보일 수 있습니다.

이 프로젝트는 그 문제를 피하기 위해 아래 흐름으로 진행했습니다.

1. 공정 센서 데이터의 결측/불균형/저분산/중복 센서 구조를 먼저 확인한다.
2. `all_pass_baseline`을 추가해 Accuracy가 왜 위험한지 수치로 확인한다.
3. supervised model과 pass-only anomaly baseline을 비교하되, Accuracy보다 불량 미탐을 줄이는 Recall/F2를 본다.
4. 임계값은 test set이 아니라 validation set에서 먼저 결정한다.
5. 최종 test set에서 성능을 확인한다.
6. built-in importance와 permutation importance로 주요 센서 후보를 정리한다.

이 프로젝트의 초점은 모델 성능 과장이 아니라, 제조 센서 데이터에서 결측, 불균형, 임계값 선택, 불량 미탐 비용, 센서 후보 해석을 일관된 분석 흐름으로 다루는 것입니다.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data | pandas, NumPy |
| ML | scikit-learn, Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting, IsolationForest |
| Evaluation | Precision, Recall, F1, F2, PR-AUC, confusion matrix |
| Visualization | matplotlib |
| Serving | FastAPI, Uvicorn |
| Reproducibility | raw-data downloader, deterministic split, generated reports |

---

## Semiconductor Manufacturing Context

이 프로젝트는 익명화된 SECOM 센서 데이터를 사용하므로 특정 공정명을 직접 매핑할 수는 없습니다. 대신 반도체 전공정에서 자주 언급되는 5개 공정 관점으로, 센서 데이터가 어떤 문제와 연결되는지 정리했습니다.

| Process | What Happens | Data / AI View |
|---|---|---|
| Photo | 회로 패턴을 웨이퍼에 전사 | CD, overlay, 노광 조건, 패턴 불량 |
| Etch | 불필요한 막을 제거해 패턴 형성 | 식각률, 균일도, endpoint, chamber 상태 |
| Diffusion | 이온주입/열처리로 전기적 특성 형성 | 온도, 시간, 농도, recipe 안정성 |
| Thin Film | CVD/PVD/ALD로 박막 형성 | 막 두께, 균일도, gas/pressure |
| CMP/Cleaning | 평탄화 및 오염 제거 | particle, scratch, slurry, 세정 조건 |

핵심은 공정명을 외우는 것이 아니라, **공정 조건 - 센서 신호 - 품질/수율 결과**를 데이터 문제로 바꾸는 것입니다.

---

## Dataset

| Property | Value |
|---|---:|
| Samples | 1,567 |
| Sensor Features | 590 |
| Pass Samples | 1,463 |
| Fail Samples | 104 |
| Fail Ratio | 6.64% |
| Zero-Variance Sensors | 116 |
| Sensors with >=50% Missing Values | 28 |
| Highly Correlated Sensor Pairs >=0.98 | 201 |

Raw data is not committed. It is downloaded reproducibly through `src/fetch_data.py`.

![Class imbalance](reports/figures/class_imbalance.png)

![Missingness distribution](reports/figures/missingness_distribution.png)

![Top missing sensors](reports/figures/top_missing_sensors.png)

![Sensor quality summary](reports/figures/sensor_quality_summary.png)

The EDA intentionally separates modeling from data-quality diagnosis:

- `reports/missing_profile.csv` ranks sensors by missing count and missing ratio.
- `reports/sensor_quality_profile.csv` flags zero-variance and all-missing sensors.
- `reports/high_correlation_pairs.csv` lists highly redundant sensor pairs.
- `reports/split_class_profile.csv` confirms the train/validation/test splits keep the fail ratio stable.

---

## Pipeline

```mermaid
flowchart LR
    A["UCI SECOM raw data"] --> B["EDA<br/>imbalance + missingness + redundancy"]
    B --> C["Train / Validation / Test split"]
    C --> D["Baselines + model training<br/>All-pass / LR / RF / ExtraTrees / HGB / IsolationForest"]
    D --> E["Validation threshold search<br/>F2 priority"]
    E --> F["Final test evaluation"]
    F --> G["Sensor candidate ranking"]
    G --> H["Reports + figures"]
```

### Evaluation Design

The decision threshold is selected on the **validation set** using F2 score, then evaluated once on the **test set**. This avoids selecting the threshold directly on the test set.

Why F2?

- In manufacturing fault detection, missed failures can be more expensive than extra inspection.
- F2 gives more weight to Recall than Precision.
- This fits early-warning systems such as FDC, equipment monitoring, and yield risk screening.

Why not Accuracy?

- The test set has 293 pass samples and 21 fail samples.
- A rule that predicts every wafer as pass reaches **0.9331 accuracy**.
- That same rule catches **0 out of 21 fail cases**, so fail recall is **0.0000**.
- For a manufacturing decision-support PoC, this is unacceptable because missed failures are the exact risk the system is supposed to surface.

![Accuracy warning](reports/figures/accuracy_warning.png)

---

## Results

| Model | Threshold | Accuracy | Fail Recall | Fail Precision | Fail F2 | PR-AUC | Missed Fail | False Alarm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtraTrees balanced | 0.10 | 0.7739 | **0.7619** | 0.1951 | **0.4819** | **0.2174** | **5** | 66 |
| RandomForest unweighted | 0.10 | 0.8025 | 0.6667 | 0.2029 | 0.4575 | 0.1701 | 7 | 55 |
| RandomForest balanced | 0.08 | 0.7102 | **0.7619** | 0.1569 | 0.4301 | 0.2150 | **5** | 86 |
| IsolationForest pass-only | 0.06 | 0.0955 | 0.9524 | 0.0660 | 0.2584 | 0.1606 | 1 | 283 |
| Logistic Regression unweighted | 0.06 | 0.8089 | 0.2381 | 0.1020 | 0.1880 | 0.1220 | 16 | 44 |
| Logistic Regression balanced | 0.62 | 0.8599 | 0.1905 | 0.1290 | 0.1739 | 0.1219 | 17 | 27 |
| HistGradientBoosting | 0.02 | 0.9172 | 0.0952 | 0.2222 | 0.1075 | 0.2137 | 19 | 7 |
| All-pass baseline | 1.00 | **0.9331** | 0.0000 | 0.0000 | 0.0000 | 0.0669 | 21 | **0** |

### Confusion Matrix

| | Pred Pass | Pred Fail |
|---|---:|---:|
| True Pass | 227 | 66 |
| True Fail | 5 | 16 |

Interpretation:

- The selected threshold intentionally increases false alarms to reduce missed failures.
- ExtraTrees kept the same test fail recall as the balanced RandomForest while reducing false alarms from 86 to 66.
- IsolationForest trained only on pass samples catches 20 of 21 fail cases, but creates 283 false alarms. It is useful as an anomaly-screening reference, not as the selected operating model.
- This is reasonable for an early-warning manufacturing use case where missed failures are more costly than extra inspection.
- This model is not claimed as production-ready. The value is in the full problem-solving flow: imbalance recognition, threshold design, metric selection, and sensor candidate ranking.

![PR curve](reports/figures/precision_recall_curve.png)

![Threshold tradeoff](reports/figures/threshold_tradeoff.png)

![Confusion matrix](reports/figures/confusion_matrix_best.png)

---

## Cost-Sensitive Thresholding

The table below uses hypothetical cost units. It is not a real fab cost model. The purpose is to show how the operating threshold changes when missed failures are treated as more expensive than extra engineering review.

Thresholds are selected on the validation split and then evaluated once on the held-out test split.

| Scenario | False Alarm Cost | Missed Fail Cost | Threshold | Test Missed Fail | Test False Alarm | Test Recall |
|---|---:|---:|---:|---:|---:|---:|
| balanced_review | 1 | 10 | 0.10 | 5 | 66 | 0.7619 |
| yield_risk_sensitive | 1 | 25 | 0.06 | 1 | 207 | 0.9524 |
| escape_critical | 1 | 50 | 0.06 | 1 | 207 | 0.9524 |

![Cost threshold analysis](reports/figures/cost_threshold_analysis.png)

Interpretation:

- If extra review is acceptable, threshold 0.10 gives fewer false alarms.
- If missed fail risk is much more expensive, threshold 0.06 catches more fail cases but creates many more review alarms.
- This is the core FDC operating trade-off: the model score is only useful when paired with an explicit decision policy.

---

## Manufacturing Decision Workflow

This project treats the model as a decision-support layer, not as an automatic final judgment system. The intended operating flow is:

1. Process/equipment sensors are collected for each wafer or lot.
2. Data quality checks remove obvious sensor issues before scoring.
3. The trained model produces a fail-risk score.
4. A threshold selected on validation data converts the score into an alarm candidate.
5. Engineers review the alarm with equipment state, recipe, PM history, and metrology context.
6. Confirmed outcomes feed back into the next threshold/model validation cycle.

![FDC operating workflow](reports/figures/fdc_operating_workflow.png)

This framing is important because SECOM sensor names are anonymized. The model can prioritize review candidates, but physical root-cause confirmation requires process tags, chamber history, recipe steps, maintenance records, and downstream quality measurements.

---

## Sensor Candidate Interpretation

Top built-in feature importance candidates from the best model:

| Rank | Sensor | Importance |
|---:|---|---:|
| 1 | sensor_129 | 0.009227 |
| 2 | sensor_511 | 0.008027 |
| 3 | sensor_064 | 0.007861 |
| 4 | sensor_059 | 0.006802 |
| 5 | sensor_065 | 0.006011 |
| 6 | sensor_103 | 0.005898 |
| 7 | sensor_028 | 0.005651 |
| 8 | sensor_122 | 0.004884 |
| 9 | sensor_130 | 0.004874 |
| 10 | sensor_452 | 0.004635 |

![Feature importance](reports/figures/feature_importance.png)

Top validation permutation importance candidates:

| Rank | Sensor | Mean AP Drop |
|---:|---|---:|
| 1 | sensor_064 | 0.023783 |
| 2 | sensor_065 | 0.012147 |
| 3 | sensor_037 | 0.009050 |
| 4 | sensor_028 | 0.005910 |
| 5 | sensor_419 | 0.005766 |
| 6 | sensor_076 | 0.005675 |
| 7 | sensor_210 | 0.005539 |
| 8 | sensor_031 | 0.005149 |
| 9 | sensor_295 | 0.004913 |
| 10 | sensor_125 | 0.004687 |

![Permutation importance](reports/figures/permutation_importance.png)

![Importance comparison](reports/figures/importance_comparison.png)

Because SECOM anonymizes sensors, the interpretation is framed as **sensor candidate prioritization** rather than direct physical root-cause naming. In a real fab environment, these candidates would be mapped back to process/equipment tags for process or equipment engineer review.

Built-in tree importance and permutation importance are intentionally reported together:

- Built-in importance shows which sensors the fitted ensemble used often when splitting.
- Permutation importance checks whether validation PR-AUC drops when a sensor is shuffled.
- Overlapping candidates such as `sensor_064`, `sensor_065`, and `sensor_028` are stronger review candidates than a single ranking alone.

---

## How To Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_data.py
python src/train.py
```

`src/train.py` also creates a local inference artifact under `artifacts/`. The artifact is intentionally not committed because it can be regenerated from the raw data and training pipeline.

Open `reports/dashboard.html` in a browser for a compact static review page generated from the latest reports.

### FastAPI Demo

```bash
python src/make_sample_input.py
python src/smoke_test_api.py
uvicorn src.api:app --reload
```

Endpoints:

- `GET /health`: checks whether the trained artifact is available.
- `GET /model-info`: returns model name, threshold, feature count, and held-out test metrics.
- `POST /predict`: accepts 590 ordered sensor values and returns fail-risk score plus alarm decision.

Example local request after starting Uvicorn:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @artifacts/sample_input.json
```

Generated outputs:

- `reports/metrics.csv`
- `reports/dashboard.html`
- `reports/model_comparison.md`
- `reports/run_summary.md`
- `reports/class_profile.csv`
- `reports/missing_profile.csv`
- `reports/sensor_quality_profile.csv`
- `reports/high_correlation_pairs.csv`
- `reports/split_class_profile.csv`
- `reports/threshold_curve_best.csv`
- `reports/top_features.csv`
- `reports/permutation_importance.csv`
- `reports/importance_comparison.csv`
- `reports/cost_threshold_analysis.csv`
- `reports/cost_threshold_curves.csv`
- `reports/models/<model_name>/summary.csv`
- `reports/models/<model_name>/validation_threshold_curve.csv`
- `reports/models/<model_name>/test_predictions.csv`
- `reports/models/<model_name>/*.png`
- `reports/figures/*.png`
- `artifacts/best_model.joblib` local only, regenerated by `python src/train.py`
- `artifacts/sample_input.json` local only, regenerated by `python src/make_sample_input.py`
- `docs/interview_notes.md`
- `docs/semiconductor_process_notes.md`
- `docs/portfolio_onepager.md`
- `docs/upgrade_checklist.md`

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs
│   ├── interview_notes.md
│   ├── portfolio_onepager.md
│   ├── semiconductor_process_notes.md
│   └── upgrade_checklist.md
├── src
│   ├── api.py
│   ├── build_dashboard.py
│   ├── fetch_data.py
│   ├── make_sample_input.py
│   ├── smoke_test_api.py
│   └── train.py
└── reports
    ├── class_profile.csv
    ├── cost_threshold_analysis.csv
    ├── cost_threshold_curves.csv
    ├── dashboard.html
    ├── high_correlation_pairs.csv
    ├── importance_comparison.csv
    ├── metrics.csv
    ├── model_comparison.md
    ├── missing_profile.csv
    ├── models
    │   └── <model_name>
    │       ├── confusion_matrix.png
    │       ├── precision_recall_curve.png
    │       ├── summary.csv
    │       ├── test_predictions.csv
    │       ├── threshold_tradeoff.png
    │       └── validation_threshold_curve.csv
    ├── run_summary.md
    ├── permutation_importance.csv
    ├── sensor_quality_profile.csv
    ├── split_class_profile.csv
    ├── top_features.csv
    ├── threshold_curve_best.csv
    └── figures
        ├── accuracy_warning.png
        ├── result_dashboard.png
        ├── class_imbalance.png
        ├── cost_threshold_analysis.png
        ├── fdc_operating_workflow.png
        ├── missingness_distribution.png
        ├── sensor_quality_summary.png
        ├── top_missing_sensors.png
        ├── precision_recall_curve.png
        ├── threshold_tradeoff.png
        ├── confusion_matrix_best.png
        ├── feature_importance.png
        ├── importance_comparison.png
        └── permutation_importance.png
```

---

## Project Scope

| Focus | Evidence |
|---|---|
| Data quality | imbalance, missingness, zero-variance sensors, correlated sensor pairs |
| Model comparison | all-pass baseline, Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting |
| Anomaly baseline | IsolationForest trained only on pass samples |
| Metric design | Fail Recall, F2, PR-AUC, missed fail count, false alarm count |
| Evaluation hygiene | validation threshold selection, final test evaluation |
| Decision policy | cost-sensitive threshold analysis with explicit false-alarm and missed-fail assumptions |
| Manufacturing workflow | FDC-style sensor, scoring, alarm, engineer review, and feedback loop diagram |
| Sensor interpretation | built-in importance, permutation importance, candidate-prioritization framing |
| System demo | FastAPI `/health`, `/model-info`, and `/predict` endpoints |
| Result review | generated static dashboard at `reports/dashboard.html` |
| Explanation docs | interview notes, semiconductor process notes, portfolio one-pager |
| Reproducibility | raw data download script and generated reports |

---

## Next Improvements

- Add optional SHAP-based local/global explanations
- Compare a lightweight AutoEncoder anomaly baseline only if it stays reproducible
- Add process-tag mapping if a non-anonymized fab dataset is available
- Tighten final README narrative after another end-to-end review
