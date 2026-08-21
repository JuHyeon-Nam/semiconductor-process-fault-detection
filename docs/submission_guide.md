# Submission Guide

## What To Show First

Start with `README.md`. It gives the project context, the reason accuracy is misleading, the selected model result, the cost-sensitive threshold analysis, and the FDC-style operating workflow.

Then show these files in order:

1. `reports/dashboard.html`
2. `reports/model_card.md`
3. `reports/model_comparison.md`
4. `reports/run_summary.md`
5. `docs/final_review.md`
6. `docs/interview_notes.md`
7. `docs/portfolio_onepager.md`

This order moves from high-level result review to detailed technical explanation.

## Fast Reproduction

Use this command for a full local check:

```bash
make reproduce
```

Equivalent manual commands:

```bash
pip install -r requirements.txt
python src/fetch_data.py
python src/train.py
python src/validate_outputs.py
python src/make_sample_input.py
python src/smoke_test_api.py
```

Expected validation result:

- all validation checks passed
- `0 checks failed`
- best model: `extra_trees_balanced`
- threshold: `0.10`
- feature count: `590`

## Short Project Pitch

"This project treats SECOM semiconductor sensor data as an FDC-style fail-risk screening problem. Because failures are rare, I first show why accuracy is misleading with an all-pass baseline. I compare supervised models and a pass-only anomaly baseline, choose the threshold only on validation data, and evaluate once on the held-out test split. The selected ExtraTrees model catches 16 of 21 fail cases at threshold 0.10. I also report false alarms, cost-sensitive threshold behavior, and sensor candidates because manufacturing models must support engineering decisions, not just output a score."

## What To Emphasize

- The project is about yield-risk screening and manufacturing decision support.
- Accuracy is not the main metric because the fail class is rare.
- Threshold selection is performed on validation data, not test data.
- False alarm and missed fail counts are both reported.
- IsolationForest shows why anomaly screening can catch more fails but overwhelm review capacity.
- Sensor importance is framed as candidate prioritization because SECOM sensors are anonymized.
- FastAPI is a PoC interface showing how the model score could be connected to a system.

## What Not To Say

- Do not say the model is production-ready.
- Do not say `sensor_064` or any anonymous sensor is a confirmed physical root cause.
- Do not say the model directly adjusts recipe or process specs.
- Do not claim high accuracy as the achievement.
- Do not hide the false alarm trade-off.

## Files That Matter Most

| File | Why It Matters |
|---|---|
| `README.md` | Main project story and results |
| `reports/dashboard.html` | One-screen review of metrics, trade-offs, and workflow |
| `reports/model_card.md` | Intended use, metrics, risks, and operating conditions |
| `reports/score_band_analysis.csv` | Evidence that high scores are used as ranking signals, not calibrated probabilities |
| `reports/review_capacity_analysis.csv` | Evidence that score ranking is connected to finite engineering review capacity |
| `reports/validation_summary.md` | Evidence that generated outputs are consistent |
| `src/train.py` | Main EDA/model/report pipeline |
| `src/api.py` | Inference endpoint |
| `docs/final_review.md` | Final readiness, limitations, and interview framing |
| `docs/interview_qa.md` | Tough question practice |

## If Asked About Limitations

The strongest answer is direct:

"The dataset is anonymized and small, so I do not claim physical root cause or production performance. The value of the project is the disciplined workflow: data-quality checks, leakage-safe thresholding, class-imbalance-aware metrics, cost trade-off analysis, sensor candidate prioritization, and an API/dashboard surface for decision support."
