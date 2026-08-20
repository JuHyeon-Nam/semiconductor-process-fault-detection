# Final Review

## Current Status

This repository is ready to present as a semiconductor manufacturing AI / FDC-style decision-support project. The project is not positioned as a production fab model. It is positioned as a reproducible PoC that shows how to handle imbalanced manufacturing sensor data, select operating thresholds without test leakage, explain false alarm versus missed fail trade-offs, and connect model output to an engineering review workflow.

## What Is Complete

| Area | Evidence |
|---|---|
| Data reproducibility | `src/fetch_data.py`, raw data ignored by git |
| EDA | missingness, zero-variance sensors, high-correlation pairs, split class profile |
| Baselines | all-pass baseline, supervised models, pass-only IsolationForest |
| Evaluation hygiene | train/validation/test split, validation-only threshold selection |
| Model result | ExtraTrees balanced selected by F2, test recall 0.7619, F2 0.4819 |
| Cost trade-off | cost-sensitive threshold scenarios in `reports/cost_threshold_analysis.csv` |
| Score interpretation | score band analysis in `reports/score_band_analysis.csv` |
| Interpretability | built-in importance, permutation importance, sensor candidate framing |
| Manufacturing story | FDC workflow diagram and decision-support explanation |
| System demo | FastAPI `/health`, `/model-info`, `/predict` |
| Review surface | `reports/dashboard.html` |
| Model governance | `reports/model_card.md` |
| Validation | `src/validate_outputs.py`, generated output and Markdown link checks passed |
| Submission support | `Makefile`, `docs/submission_guide.md`, `docs/interview_qa.md` |

## Main Result To Explain

The selected model is `extra_trees_balanced` with a validation-selected threshold of 0.10.

| Metric | Value |
|---|---:|
| Test fail recall | 0.7619 |
| Test fail precision | 0.1951 |
| Test fail F2 | 0.4819 |
| Test PR-AUC | 0.2174 |
| Missed fail | 5 |
| False alarm | 66 |

The all-pass baseline reaches 0.9331 accuracy but 0.0000 fail recall. This is the central reason the project does not use accuracy as the primary metric.

## Important Trade-Offs

The cost-sensitive analysis shows that when missed fail cost is assumed much higher than false alarm cost, the selected threshold drops from 0.10 to 0.06. Test recall then rises to 0.9524 and missed fails drop to 1, but false alarms rise to 207.

The pass-only IsolationForest baseline also catches 20 of 21 fail cases, but produces 283 false alarms. This is useful as an anomaly-screening reference, but not as the selected operating model.

The score band analysis is a ranking check, not a calibrated probability claim. It shows whether high fail-risk scores concentrate more actual fail samples and can therefore support review prioritization.

## What Not To Overclaim

- Do not claim this is production-ready.
- Do not claim physical root cause from anonymous SECOM sensor names.
- Do not claim the model controls recipe or process specs directly.
- Do not claim high accuracy is the project value.
- Do not claim the fail-risk score is a calibrated physical failure probability.
- Do not claim false alarms are acceptable without review capacity analysis.

## Strong Interview Framing

"I treated the SECOM data as an FDC-style manufacturing decision-support problem. Because the fail class is rare, I first showed why accuracy is misleading with an all-pass baseline. I selected thresholds only on validation data and used the test set for final evaluation. The selected ExtraTrees model catches 16 of 21 fail cases, but I also reported false alarms because an alarm is only useful if engineers can review it. I used cost-sensitive threshold analysis and a pass-only IsolationForest baseline to show how missed-fail risk and review burden change the operating point. Since the sensors are anonymized, I interpret important sensors as review candidates, not confirmed physical root causes."

## How To Reproduce The Full Project

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

## Final Polish Queue

Only do these if there is enough time and the change remains small:

1. Tighten README wording after a final visual review.
2. Add optional SHAP only if dependency setup is stable.
3. Add process-tag mapping only with a non-anonymized dataset.
4. Add a lightweight AutoEncoder only if runtime and reproducibility stay reasonable.
