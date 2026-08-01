from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    x_path = RAW / "secom.data"
    y_path = RAW / "secom_labels.data"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Run `python src/fetch_data.py` first.")

    x = pd.read_csv(x_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(y_path, sep=r"\s+", header=None, names=["label", "timestamp"])

    # UCI SECOM label: -1 means pass, 1 means fail. Convert to 0/1.
    y = labels["label"].map({-1: 0, 1: 1}).astype(int)
    x.columns = [f"sensor_{i:03d}" for i in range(x.shape[1])]
    return x, y


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=300,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def choose_threshold(y_true: pd.Series, positive_score: np.ndarray) -> tuple[float, np.ndarray]:
    best_threshold = 0.5
    best_score = -1.0
    best_pred = (positive_score >= best_threshold).astype(int)

    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (positive_score >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_pred = pred

    return best_threshold, best_pred


def score_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    if hasattr(model[-1], "predict_proba"):
        positive_score = model.predict_proba(x_test)[:, 1]
        threshold, pred = choose_threshold(y_test, positive_score)
    else:
        pred = model.predict(x_test)
        positive_score = pred
        threshold = 0.5

    return {
        "model": name,
        "threshold": threshold,
        "recall_fail": recall_score(y_test, pred, zero_division=0),
        "precision_fail": precision_score(y_test, pred, zero_division=0),
        "f1_fail": f1_score(y_test, pred, zero_division=0),
        "f2_fail": fbeta_score(y_test, pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_test, positive_score),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def feature_importance(model: Pipeline, x_train: pd.DataFrame) -> pd.DataFrame:
    estimator = model[-1]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": x_train.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(30)
    )


def write_summary(metrics: pd.DataFrame, top_features: pd.DataFrame) -> None:
    best = metrics.sort_values(["f2_fail", "pr_auc"], ascending=False).iloc[0]
    feature_lines = ["| feature | importance |", "|---|---:|"]
    for row in top_features.head(10).itertuples(index=False):
        feature_lines.append(f"| {row.feature} | {row.importance:.6f} |")
    feature_table = "\n".join(feature_lines)
    summary = f"""# SECOM Run Summary

## Best Model

- Model: {best["model"]}
- Threshold: {best["threshold"]:.2f}
- Fail Recall: {best["recall_fail"]:.4f}
- Fail Precision: {best["precision_fail"]:.4f}
- Fail F1: {best["f1_fail"]:.4f}
- Fail F2: {best["f2_fail"]:.4f}
- PR-AUC: {best["pr_auc"]:.4f}

## Top Sensor Candidates

{feature_table}

## Interview Message

반도체 제조 데이터는 결측과 불균형이 큰 데이터라고 보고, 정상/불량 정확도보다 불량 미탐을 줄이는 Recall, F2, PR-AUC를 중심으로 평가했습니다. 특히 기본 임계값 0.5에서는 불량을 놓칠 수 있어, 불량 미탐을 더 크게 벌주는 F2 기준으로 의사결정 임계값을 조정했습니다. 주요 센서 후보를 feature importance로 정리해 FDC, 설비 이상탐지, 수율 개선 관점으로 확장할 수 있게 분석했습니다.
"""
    (REPORTS / "run_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    rows: list[dict[str, float | str]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)

    for name, model in build_models().items():
        print(f"train: {name}")
        model.fit(x_train, y_train)
        row = score_model(name, model, x_test, y_test)
        rows.append(row)
        key = (float(row["f2_fail"]), float(row["pr_auc"]))
        if key > best_key:
            best_key = key
            best_model = model

        if hasattr(model[-1], "predict_proba"):
            pred = (model.predict_proba(x_test)[:, 1] >= float(row["threshold"])).astype(int)
        else:
            pred = model.predict(x_test)
        print(classification_report(y_test, pred, target_names=["pass", "fail"], zero_division=0))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)

    top_features = feature_importance(best_model, x_train) if best_model else pd.DataFrame()
    top_features.to_csv(REPORTS / "top_features.csv", index=False)
    write_summary(metrics, top_features)
    print(f"saved reports: {REPORTS}")


if __name__ == "__main__":
    main()
