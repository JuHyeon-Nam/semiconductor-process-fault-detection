# Portfolio One-Pager

## Project

Semiconductor Manufacturing Fault Detection  
UCI SECOM 센서 데이터를 이용한 FDC-style fail risk screening 프로젝트

## Problem

반도체 제조 데이터는 결측 센서가 많고, 불량 샘플이 정상 샘플보다 훨씬 적습니다. 이 조건에서 accuracy만 보면 모든 샘플을 정상으로 예측하는 모델도 좋아 보일 수 있습니다. 실제 제조 관점에서는 불량 미탐, 오경보, 엔지니어 review capacity, threshold 정책을 함께 봐야 합니다.

## What Was Built

- Raw data download script for reproducibility
- Data quality EDA: missingness, zero-variance sensors, highly correlated sensors
- Class imbalance report and all-pass baseline
- Train/validation/test split with validation-only threshold selection
- Model comparison: Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting
- F2-based threshold optimization and final test evaluation
- Cost-sensitive threshold analysis for false alarm vs missed fail trade-off
- Built-in feature importance and validation permutation importance
- FastAPI demo with `/health`, `/model-info`, and `/predict`
- Interview notes, process notes, and generated reports

## Best Current Result

| Item | Value |
|---|---:|
| Best model | ExtraTrees balanced |
| Validation-selected threshold | 0.10 |
| Test fail recall | 0.7619 |
| Test fail precision | 0.1951 |
| Test fail F2 | 0.4819 |
| Test PR-AUC | 0.2174 |
| Test missed fail | 5 |
| Test false alarm | 66 |

All-pass baseline reaches 0.9331 test accuracy but 0.0000 fail recall, so accuracy is not used as the main metric.

## Manufacturing Interpretation

This is not a production-ready automatic judgment model. It is a manufacturing decision-support PoC. The model produces a fail-risk score, and a validation-selected threshold converts that score into an alarm candidate. Engineers should review the alarm with tool state, recipe, PM history, metrology, inspection, and downstream quality data.

Because SECOM sensors are anonymized, top sensors are interpreted as review candidates rather than confirmed physical root causes. In real fab data, these candidates must be mapped to FDC tags, chamber, recipe step, and maintenance history.

## Why It Matters

The project demonstrates the full data problem-solving flow:

1. Recognize manufacturing data quality issues.
2. Avoid misleading accuracy under class imbalance.
3. Select thresholds without test leakage.
4. Explain false alarm vs missed fail trade-offs.
5. Translate model output into an FDC-style engineering review workflow.
6. Provide a small API surface that can connect the model to a dashboard or manufacturing system.

## Main Files

- `README.md`: project overview, results, workflow, run instructions
- `src/train.py`: EDA, model training, threshold selection, reports, inference artifact
- `src/api.py`: FastAPI inference service
- `reports/run_summary.md`: generated analysis summary
- `reports/model_comparison.md`: model comparison table
- `docs/interview_notes.md`: interview-ready explanation
- `docs/semiconductor_process_notes.md`: semiconductor process/data mapping notes
