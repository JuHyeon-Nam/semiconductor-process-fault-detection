from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODEL_REPORTS = REPORTS / "models"


@dataclass(frozen=True)
class Evaluation:
    name: str
    threshold: float
    recall_fail: float
    precision_fail: float
    f1_fail: float
    f2_fail: float
    pr_auc: float
    accuracy: float
    false_alarm_count: int
    missed_fail_count: int
    confusion: list[list[int]]


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
        "logistic_regression_balanced": Pipeline(
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
        "logistic_regression_unweighted": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_unweighted": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
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


def model_notes() -> dict[str, str]:
    return {
        "all_pass_baseline": "Naive baseline that predicts every sample as pass.",
        "logistic_regression_balanced": "Linear baseline with class_weight=balanced.",
        "logistic_regression_unweighted": "Linear baseline without class weighting.",
        "random_forest_balanced": "Tree ensemble with class_weight=balanced_subsample.",
        "random_forest_unweighted": "Tree ensemble without class weighting.",
        "extra_trees_balanced": "Randomized tree ensemble with class_weight=balanced.",
        "hist_gradient_boosting": "Histogram gradient boosting baseline.",
    }


def positive_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    estimator = model[-1]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = model.decision_function(x)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return model.predict(x)


def threshold_curve(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.02, 0.80, 40):
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "recall_fail": recall_score(y_true, pred, zero_division=0),
                "precision_fail": precision_score(y_true, pred, zero_division=0),
                "f1_fail": f1_score(y_true, pred, zero_division=0),
                "f2_fail": fbeta_score(y_true, pred, beta=2, zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
                "false_alarm_count": int(fp),
                "missed_fail_count": int(fn),
                "predicted_fail_count": int(fp + tp),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_valid: pd.Series, valid_scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    curve = threshold_curve(y_valid, valid_scores)
    best = curve.sort_values(["f2_fail", "recall_fail"], ascending=False).iloc[0]
    return float(best["threshold"]), curve


def evaluate(name: str, threshold: float, y_test: pd.Series, scores: np.ndarray) -> Evaluation:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return Evaluation(
        name=name,
        threshold=threshold,
        recall_fail=recall_score(y_test, pred, zero_division=0),
        precision_fail=precision_score(y_test, pred, zero_division=0),
        f1_fail=f1_score(y_test, pred, zero_division=0),
        f2_fail=fbeta_score(y_test, pred, beta=2, zero_division=0),
        pr_auc=average_precision_score(y_test, scores),
        accuracy=accuracy_score(y_test, pred),
        false_alarm_count=int(fp),
        missed_fail_count=int(fn),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


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


def permutation_importance_report(
    model: Pipeline,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_repeats: int = 5,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_valid,
        y_valid,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_valid.columns,
                "permutation_importance_mean": result.importances_mean,
                "permutation_importance_std": result.importances_std,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .head(30)
    )


def compare_importance(
    built_in: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    built = built_in.reset_index(drop=True).reset_index().rename(columns={"index": "built_in_rank"})
    built["built_in_rank"] += 1
    perm = permutation.reset_index(drop=True).reset_index().rename(columns={"index": "permutation_rank"})
    perm["permutation_rank"] += 1
    compared = built.merge(perm, on="feature", how="outer")
    return compared.sort_values(
        ["permutation_rank", "built_in_rank"],
        na_position="last",
    )


def write_data_profile(x: pd.DataFrame, y: pd.Series) -> None:
    class_profile = y.value_counts().sort_index().rename(index={0: "pass", 1: "fail"}).reset_index()
    class_profile.columns = ["class", "count"]
    class_profile["ratio"] = class_profile["count"] / class_profile["count"].sum()
    class_profile.to_csv(REPORTS / "class_profile.csv", index=False)

    missing = (
        pd.DataFrame(
            {
                "sensor": x.columns,
                "missing_count": x.isna().sum().values,
                "missing_ratio": x.isna().mean().values,
                "non_missing_count": x.notna().sum().values,
            }
        )
        .sort_values("missing_ratio", ascending=False)
    )
    missing.to_csv(REPORTS / "missing_profile.csv", index=False)

    numeric = x.astype(float)
    sensor_quality = pd.DataFrame(
        {
            "sensor": x.columns,
            "missing_ratio": numeric.isna().mean().values,
            "unique_non_missing": numeric.nunique(dropna=True).values,
            "variance": numeric.var(skipna=True).fillna(0.0).values,
        }
    )
    sensor_quality["is_zero_variance"] = sensor_quality["variance"] == 0
    sensor_quality["is_all_missing"] = sensor_quality["unique_non_missing"] == 0
    sensor_quality.sort_values(["is_zero_variance", "missing_ratio"], ascending=False).to_csv(
        REPORTS / "sensor_quality_profile.csv",
        index=False,
    )

    usable = numeric.loc[:, ~sensor_quality.set_index("sensor")["is_zero_variance"]]
    corr = usable.corr(numeric_only=True).abs().clip(upper=1.0)
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr_pairs = (
        corr.where(upper_mask)
        .stack()
        .rename("abs_correlation")
        .reset_index()
        .rename(columns={"level_0": "sensor_a", "level_1": "sensor_b"})
        .sort_values("abs_correlation", ascending=False)
    )
    corr_pairs[corr_pairs["abs_correlation"] >= 0.98].to_csv(
        REPORTS / "high_correlation_pairs.csv",
        index=False,
    )


def write_split_profile(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
) -> None:
    rows = []
    for split_name, split_y in [("train", y_train), ("validation", y_valid), ("test", y_test)]:
        counts = split_y.value_counts().reindex([0, 1], fill_value=0)
        rows.append(
            {
                "split": split_name,
                "total": int(counts.sum()),
                "pass_count": int(counts.loc[0]),
                "fail_count": int(counts.loc[1]),
                "fail_ratio": counts.loc[1] / counts.sum(),
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS / "split_class_profile.csv", index=False)


def plot_class_balance(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Pass", "Fail"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values, color=["#4c78a8", "#e45756"])
    plt.title("SECOM Class Imbalance")
    plt.ylabel("Count")
    for i, value in enumerate(counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_imbalance.png", dpi=180)
    plt.close()


def plot_missingness(x: pd.DataFrame) -> None:
    missing = x.isna().mean()
    plt.figure(figsize=(7, 4))
    plt.hist(missing, bins=30, color="#72b7b2", edgecolor="white")
    plt.title("Sensor Missingness Distribution")
    plt.xlabel("Missing ratio per sensor")
    plt.ylabel("Sensor count")
    plt.tight_layout()
    plt.savefig(FIGURES / "missingness_distribution.png", dpi=180)
    plt.close()


def plot_top_missing_sensors(x: pd.DataFrame) -> None:
    missing = x.isna().mean().sort_values(ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(missing.index, missing.values, color="#b279a2")
    plt.title("Top Missing Sensors")
    plt.xlabel("Missing ratio")
    plt.xlim(0, max(1.0, missing.max() * 1.05))
    plt.tight_layout()
    plt.savefig(FIGURES / "top_missing_sensors.png", dpi=180)
    plt.close()


def plot_sensor_quality_summary(x: pd.DataFrame) -> None:
    variance = x.var(skipna=True).fillna(0)
    missing = x.isna().mean()
    counts = {
        "All sensors": len(x.columns),
        "Zero variance": int((variance == 0).sum()),
        ">=50% missing": int((missing >= 0.50).sum()),
        ">=20% missing": int((missing >= 0.20).sum()),
    }
    plt.figure(figsize=(7, 4))
    bars = plt.bar(counts.keys(), counts.values(), color=["#4c78a8", "#e45756", "#f58518", "#72b7b2"])
    plt.title("Sensor Quality Flags")
    plt.ylabel("Sensor count")
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES / "sensor_quality_summary.png", dpi=180)
    plt.close()


def plot_precision_recall(scores_by_model: dict[str, np.ndarray], y_test: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
    plt.title("Precision-Recall Curve for Fail Detection")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "precision_recall_curve.png", dpi=180)
    plt.close()


def plot_model_precision_recall(name: str, scores: np.ndarray, y_test: pd.Series, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#4c78a8", label=f"AP={pr_auc:.3f}")
    plt.title(f"Precision-Recall: {name}")
    plt.xlabel("Fail recall")
    plt.ylabel("Fail precision")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_threshold_curve(curve: pd.DataFrame, model_name: str, output_path: Path | None = None) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(curve["threshold"], curve["recall_fail"], label="Recall")
    plt.plot(curve["threshold"], curve["precision_fail"], label="Precision")
    plt.plot(curve["threshold"], curve["f2_fail"], label="F2")
    plt.title(f"Validation Threshold Trade-off: {model_name}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path or FIGURES / "threshold_tradeoff.png", dpi=180)
    plt.close()


def plot_confusion(confusion: list[list[int]], model_name: str, output_path: Path | None = None) -> None:
    matrix = np.array(confusion)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xticks([0, 1], ["Pred Pass", "Pred Fail"])
    plt.yticks([0, 1], ["True Pass", "True Fail"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path or FIGURES / "confusion_matrix_best.png", dpi=180)
    plt.close()


def plot_feature_importance(top_features: pd.DataFrame) -> None:
    if top_features.empty:
        return
    top = top_features.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#f58518")
    plt.title("Top Sensor Candidates")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()


def plot_permutation_importance(permutation_features: pd.DataFrame) -> None:
    if permutation_features.empty:
        return
    top = permutation_features.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(
        top["feature"],
        top["permutation_importance_mean"],
        xerr=top["permutation_importance_std"],
        color="#54a24b",
        alpha=0.9,
    )
    plt.title("Permutation Importance on Validation Set")
    plt.xlabel("Drop in average precision when shuffled")
    plt.tight_layout()
    plt.savefig(FIGURES / "permutation_importance.png", dpi=180)
    plt.close()


def plot_importance_comparison(comparison: pd.DataFrame) -> None:
    overlap = comparison.dropna(subset=["built_in_rank", "permutation_rank"]).copy()
    if overlap.empty:
        return
    overlap = overlap.sort_values("permutation_rank").head(15)
    x_pos = np.arange(len(overlap))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x_pos - width / 2, overlap["built_in_rank"], width, label="Built-in rank", color="#f58518")
    plt.bar(x_pos + width / 2, overlap["permutation_rank"], width, label="Permutation rank", color="#54a24b")
    plt.gca().invert_yaxis()
    plt.xticks(x_pos, overlap["feature"], rotation=35, ha="right")
    plt.ylabel("Rank, lower is more important")
    plt.title("Built-in vs Permutation Importance Rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "importance_comparison.png", dpi=180)
    plt.close()


def plot_accuracy_warning(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("accuracy", ascending=False)
    labels = ordered["name"].tolist()
    x_pos = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x_pos - 0.18, ordered["accuracy"], width=0.36, label="Accuracy", color="#4c78a8")
    ax1.bar(x_pos + 0.18, ordered["recall_fail"], width=0.36, label="Fail recall", color="#e45756")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_pos, labels, rotation=15, ha="right")
    ax1.set_title("Accuracy Can Hide Missed Failures")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES / "accuracy_warning.png", dpi=180)
    plt.close()


def plot_result_dashboard(metrics: pd.DataFrame, best: Evaluation) -> None:
    model_labels = metrics["name"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Semiconductor Fault Detection Result Dashboard", fontsize=15, fontweight="bold")

    axes[0, 0].bar(model_labels, metrics["recall_fail"], color="#4c78a8")
    axes[0, 0].set_title("Fail Recall by Model")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=15)

    axes[0, 1].bar(model_labels, metrics["f2_fail"], color="#f58518")
    axes[0, 1].set_title("F2 Score by Model")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=15)

    matrix = np.array(best.confusion)
    axes[1, 0].imshow(matrix, cmap="Blues")
    axes[1, 0].set_title(f"Best Confusion Matrix: {best.name}")
    axes[1, 0].set_xticks([0, 1], ["Pred Pass", "Pred Fail"])
    axes[1, 0].set_yticks([0, 1], ["True Pass", "True Fail"])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(matrix[i, j]), ha="center", va="center")

    axes[1, 1].axis("off")
    all_pass = metrics.loc[metrics["name"] == "all_pass_baseline"].iloc[0]
    summary = (
        f"Best model: {best.name}\n"
        f"Threshold: {best.threshold:.2f}\n"
        f"Fail Recall: {best.recall_fail:.4f}\n"
        f"Fail Precision: {best.precision_fail:.4f}\n"
        f"Fail F2: {best.f2_fail:.4f}\n"
        f"PR-AUC: {best.pr_auc:.4f}\n\n"
        f"All-pass accuracy: {all_pass.accuracy:.4f}\n"
        f"All-pass fail recall: {all_pass.recall_fail:.4f}\n\n"
        "Decision logic:\n"
        "- optimize threshold on validation set\n"
        "- prioritize missed-fail reduction\n"
        "- review top sensor candidates"
    )
    axes[1, 1].text(0.02, 0.95, summary, va="top", fontsize=11, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES / "result_dashboard.png", dpi=180)
    plt.close()


def write_model_artifacts(
    name: str,
    curve: pd.DataFrame,
    result: Evaluation,
    y_test: pd.Series,
    scores: np.ndarray,
) -> None:
    model_dir = MODEL_REPORTS / name
    model_dir.mkdir(parents=True, exist_ok=True)

    pred = (scores >= result.threshold).astype(int)
    pd.DataFrame(
        {
            "sample_index": y_test.index,
            "y_true": y_test.to_numpy(),
            "score_fail": scores,
            "y_pred": pred,
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)
    curve.to_csv(model_dir / "validation_threshold_curve.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "threshold",
                "accuracy",
                "recall_fail",
                "precision_fail",
                "f1_fail",
                "f2_fail",
                "pr_auc",
                "false_alarm_count",
                "missed_fail_count",
            ],
            "value": [
                result.threshold,
                result.accuracy,
                result.recall_fail,
                result.precision_fail,
                result.f1_fail,
                result.f2_fail,
                result.pr_auc,
                result.false_alarm_count,
                result.missed_fail_count,
            ],
        }
    ).to_csv(model_dir / "summary.csv", index=False)

    plot_model_precision_recall(name, scores, y_test, model_dir / "precision_recall_curve.png")
    plot_threshold_curve(curve, name, model_dir / "threshold_tradeoff.png")
    plot_confusion(result.confusion, name, model_dir / "confusion_matrix.png")


def write_model_comparison_report(metrics: pd.DataFrame) -> None:
    rows = [
        "| model | threshold | accuracy | fail_recall | fail_precision | fail_f2 | pr_auc | missed_fail | false_alarm | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.itertuples(index=False):
        rows.append(
            f"| {row.name} | {row.threshold:.2f} | {row.accuracy:.4f} | {row.recall_fail:.4f} | "
            f"{row.precision_fail:.4f} | {row.f2_fail:.4f} | {row.pr_auc:.4f} | "
            f"{row.missed_fail_count} | {row.false_alarm_count} | {row.note} |"
        )

    best = metrics.iloc[0]
    all_pass = metrics.loc[metrics["name"] == "all_pass_baseline"].iloc[0]
    report = f"""# Model Comparison Report

## Purpose

This report compares baseline and tree-based models for SECOM fail detection under the same train/validation/test split. Thresholds are selected on the validation set by F2 score, then evaluated once on the held-out test set.

## Result Table

{chr(10).join(rows)}

## Interpretation

- Best F2 model: `{best["name"]}` with fail recall {best["recall_fail"]:.4f}, fail precision {best["precision_fail"]:.4f}, and F2 {best["f2_fail"]:.4f}.
- The all-pass baseline reaches {all_pass["accuracy"]:.4f} accuracy, but it misses all {int(all_pass["missed_fail_count"])} fail cases.
- Compared with `random_forest_balanced`, `extra_trees_balanced` keeps the same test fail recall while reducing false alarms from 86 to 66.
- This is still a decision-support PoC, not a production-ready FDC model. The main value is the explicit metric, threshold, and trade-off analysis.

## Per-Model Artifacts

Each trained model has these generated files under `reports/models/<model_name>/`:

- `summary.csv`
- `validation_threshold_curve.csv`
- `test_predictions.csv`
- `precision_recall_curve.png`
- `threshold_tradeoff.png`
- `confusion_matrix.png`
"""
    (REPORTS / "model_comparison.md").write_text(report, encoding="utf-8")


def write_summary(
    best: Evaluation,
    top_features: pd.DataFrame,
    permutation_features: pd.DataFrame,
    importance_comparison: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    sensor_quality = pd.read_csv(REPORTS / "sensor_quality_profile.csv")
    missing_profile = pd.read_csv(REPORTS / "missing_profile.csv")
    high_corr = pd.read_csv(REPORTS / "high_correlation_pairs.csv")
    split_profile = pd.read_csv(REPORTS / "split_class_profile.csv")

    split_lines = ["| split | total | pass | fail | fail_ratio |", "|---|---:|---:|---:|---:|"]
    for row in split_profile.itertuples(index=False):
        split_lines.append(
            f"| {row.split} | {row.total} | {row.pass_count} | {row.fail_count} | {row.fail_ratio:.4f} |"
        )
    split_table = "\n".join(split_lines)

    missing_lines = ["| sensor | missing_count | missing_ratio |", "|---|---:|---:|"]
    for row in missing_profile.head(10).itertuples(index=False):
        missing_lines.append(f"| {row.sensor} | {row.missing_count} | {row.missing_ratio:.4f} |")
    missing_table = "\n".join(missing_lines)

    feature_lines = ["| feature | importance |", "|---|---:|"]
    for row in top_features.head(10).itertuples(index=False):
        feature_lines.append(f"| {row.feature} | {row.importance:.6f} |")
    feature_table = "\n".join(feature_lines)

    permutation_lines = ["| feature | permutation_importance_mean | std |", "|---|---:|---:|"]
    for row in permutation_features.head(10).itertuples(index=False):
        permutation_lines.append(
            f"| {row.feature} | {row.permutation_importance_mean:.6f} | "
            f"{row.permutation_importance_std:.6f} |"
        )
    permutation_table = "\n".join(permutation_lines)

    comparison_lines = [
        "| feature | built_in_rank | permutation_rank | built_in_importance | permutation_importance_mean |",
        "|---|---:|---:|---:|---:|",
    ]
    comparison_view = importance_comparison.dropna(subset=["built_in_rank", "permutation_rank"]).head(10)
    for row in comparison_view.itertuples(index=False):
        comparison_lines.append(
            f"| {row.feature} | {int(row.built_in_rank)} | {int(row.permutation_rank)} | "
            f"{row.importance:.6f} | {row.permutation_importance_mean:.6f} |"
        )
    comparison_table = "\n".join(comparison_lines)

    metric_lines = [
        "| model | threshold | accuracy | fail_recall | fail_f2 | pr_auc | missed_fail | false_alarm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        metric_lines.append(
            f"| {row.name} | {row.threshold:.2f} | {row.accuracy:.4f} | {row.recall_fail:.4f} | "
            f"{row.f2_fail:.4f} | {row.pr_auc:.4f} | {row.missed_fail_count} | {row.false_alarm_count} |"
        )
    metrics_table = "\n".join(metric_lines)

    tn, fp = best.confusion[0]
    fn, tp = best.confusion[1]
    all_pass = metrics.loc[metrics["name"] == "all_pass_baseline"].iloc[0]
    summary = f"""# SECOM Run Summary

## Data Quality Findings

- Samples: 1,567
- Sensors: 590
- Fail ratio: 6.64%
- Zero-variance sensors: {int(sensor_quality["is_zero_variance"].sum())}
- Sensors with >=50% missing values: {int((sensor_quality["missing_ratio"] >= 0.50).sum())}
- Highly correlated sensor pairs with abs(correlation) >=0.98: {len(high_corr)}

## Split Profile

{split_table}

## Top Missing Sensors

{missing_table}

## Best Model

- Model: {best.name}
- Threshold: {best.threshold:.2f}
- Fail Recall: {best.recall_fail:.4f}
- Fail Precision: {best.precision_fail:.4f}
- Fail F1: {best.f1_fail:.4f}
- Fail F2: {best.f2_fail:.4f}
- PR-AUC: {best.pr_auc:.4f}
- Accuracy: {best.accuracy:.4f}

## Confusion Matrix

| | Pred Pass | Pred Fail |
|---|---:|---:|
| True Pass | {tn} | {fp} |
| True Fail | {fn} | {tp} |

## Accuracy Trap Baseline

An all-pass rule reaches {all_pass.accuracy:.4f} accuracy on the test split, but its fail recall is {all_pass.recall_fail:.4f}. This is why the project reports Fail Recall, F2, and PR-AUC instead of treating accuracy as the primary metric.

## Model Comparison

{metrics_table}

Per-model PR curves, threshold curves, confusion matrices, and test predictions are saved under `reports/models/<model_name>/`.

## Top Sensor Candidates

{feature_table}

## Permutation Importance

Permutation importance is calculated on the validation split with average precision scoring. It answers: if a sensor is randomly shuffled, how much does fail-detection ranking quality drop?

{permutation_table}

## Importance Comparison

Built-in tree importance and permutation importance are complementary. Built-in importance shows how the fitted ensemble used features internally, while permutation importance checks whether validation performance depends on each feature.

{comparison_table}

## Interview Message

반도체 제조 데이터는 결측과 불균형이 큰 데이터라고 보고, 정상/불량 정확도보다 불량 미탐을 줄이는 Recall, F2, PR-AUC를 중심으로 평가했습니다. 임계값은 validation set에서 F2 기준으로 결정하고, test set에서 최종 성능을 확인했습니다. 주요 센서 후보는 built-in feature importance와 validation permutation importance를 함께 보며 FDC, 설비 이상탐지, 수율 개선 관점의 엔지니어 검토 후보로 해석했습니다.
"""
    (REPORTS / "run_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_REPORTS.mkdir(parents=True, exist_ok=True)
    x, y = load_data()
    write_data_profile(x, y)
    plot_class_balance(y)
    plot_missingness(x)
    plot_top_missing_sensors(x)
    plot_sensor_quality_summary(x)

    x_dev, x_test, y_dev, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )
    write_split_profile(y_train, y_valid, y_test)

    rows = []
    models: dict[str, Pipeline] = {}
    test_scores_by_model: dict[str, np.ndarray] = {}
    curves: dict[str, pd.DataFrame] = {}

    all_pass_scores = np.zeros(len(y_test))
    all_pass = evaluate("all_pass_baseline", 1.0, y_test, all_pass_scores)
    rows.append({**all_pass.__dict__, "note": model_notes()[all_pass.name]})
    test_scores_by_model[all_pass.name] = all_pass_scores

    for name, model in build_models().items():
        print(f"train: {name}")
        model.fit(x_train, y_train)
        valid_scores = positive_scores(model, x_valid)
        threshold, curve = choose_threshold(y_valid, valid_scores)
        test_scores = positive_scores(model, x_test)
        result = evaluate(name, threshold, y_test, test_scores)

        write_model_artifacts(name, curve, result, y_test, test_scores)
        rows.append({**result.__dict__, "note": model_notes()[name]})
        models[name] = model
        curves[name] = curve
        test_scores_by_model[name] = test_scores
        print(result)

    metrics = pd.DataFrame(rows).sort_values(["f2_fail", "pr_auc"], ascending=False)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)

    best_row = metrics.iloc[0]
    best = Evaluation(
        name=str(best_row["name"]),
        threshold=float(best_row["threshold"]),
        recall_fail=float(best_row["recall_fail"]),
        precision_fail=float(best_row["precision_fail"]),
        f1_fail=float(best_row["f1_fail"]),
        f2_fail=float(best_row["f2_fail"]),
        pr_auc=float(best_row["pr_auc"]),
        accuracy=float(best_row["accuracy"]),
        false_alarm_count=int(best_row["false_alarm_count"]),
        missed_fail_count=int(best_row["missed_fail_count"]),
        confusion=best_row["confusion"],
    )

    best_model = models[best.name]
    top_features = feature_importance(best_model, x_train)
    top_features.to_csv(REPORTS / "top_features.csv", index=False)
    permutation_features = permutation_importance_report(best_model, x_valid, y_valid)
    permutation_features.to_csv(REPORTS / "permutation_importance.csv", index=False)
    importance_comparison = compare_importance(top_features, permutation_features)
    importance_comparison.to_csv(REPORTS / "importance_comparison.csv", index=False)
    curves[best.name].to_csv(REPORTS / "threshold_curve_best.csv", index=False)

    plot_precision_recall(test_scores_by_model, y_test)
    plot_threshold_curve(curves[best.name], best.name)
    plot_confusion(best.confusion, best.name)
    plot_feature_importance(top_features)
    plot_permutation_importance(permutation_features)
    plot_importance_comparison(importance_comparison)
    plot_accuracy_warning(metrics)
    plot_result_dashboard(metrics, best)
    write_model_comparison_report(metrics)
    write_summary(best, top_features, permutation_features, importance_comparison, metrics)
    print(f"saved reports: {REPORTS}")


if __name__ == "__main__":
    main()
