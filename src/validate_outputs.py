from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = REPORTS / "models"
ARTIFACTS = ROOT / "artifacts"
SUMMARY_PATH = REPORTS / "validation_summary.md"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def require_file(path: Path, checks: list[Check], label: str | None = None) -> None:
    checks.append(
        Check(
            name=f"file: {label or path.relative_to(ROOT)}",
            passed=path.exists() and path.stat().st_size > 0,
            detail="exists and is non-empty" if path.exists() and path.stat().st_size > 0 else "missing or empty",
        )
    )


def add_check(checks: list[Check], name: str, condition: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=bool(condition), detail=detail))


def tracked_files(pathspec: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def validate_required_files(checks: list[Check]) -> None:
    required = [
        ROOT / "Makefile",
        ROOT / "README.md",
        ROOT / "docs" / "final_review.md",
        ROOT / "docs" / "interview_notes.md",
        ROOT / "docs" / "interview_qa.md",
        ROOT / "docs" / "portfolio_onepager.md",
        ROOT / "docs" / "semiconductor_process_notes.md",
        ROOT / "docs" / "submission_guide.md",
        REPORTS / "metrics.csv",
        REPORTS / "model_comparison.md",
        REPORTS / "run_summary.md",
        REPORTS / "dashboard.html",
        REPORTS / "cost_threshold_analysis.csv",
        REPORTS / "threshold_curve_best.csv",
        REPORTS / "permutation_importance.csv",
        REPORTS / "importance_comparison.csv",
        FIGURES / "result_dashboard.png",
        FIGURES / "accuracy_warning.png",
        FIGURES / "cost_threshold_analysis.png",
        FIGURES / "fdc_operating_workflow.png",
        ARTIFACTS / "best_model.joblib",
        ARTIFACTS / "model_metadata.json",
    ]
    for path in required:
        require_file(path, checks)


def validate_metrics(checks: list[Check]) -> pd.DataFrame:
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    expected_models = {
        "all_pass_baseline",
        "logistic_regression_balanced",
        "logistic_regression_unweighted",
        "random_forest_balanced",
        "random_forest_unweighted",
        "extra_trees_balanced",
        "hist_gradient_boosting",
        "isolation_forest_pass_only",
    }
    actual_models = set(metrics["name"])
    add_check(
        checks,
        "metrics: expected model rows",
        expected_models.issubset(actual_models),
        f"{len(actual_models)} model rows found",
    )

    best = metrics.iloc[0]
    add_check(
        checks,
        "metrics: selected best model",
        best["name"] == "extra_trees_balanced",
        f"best row is {best['name']}",
    )
    add_check(
        checks,
        "metrics: best model recall/F2",
        float(best["recall_fail"]) >= 0.75 and float(best["f2_fail"]) >= 0.45,
        f"recall={float(best['recall_fail']):.4f}, f2={float(best['f2_fail']):.4f}",
    )

    all_pass = metrics.loc[metrics["name"] == "all_pass_baseline"].iloc[0]
    add_check(
        checks,
        "metrics: all-pass accuracy trap",
        float(all_pass["accuracy"]) > 0.90 and float(all_pass["recall_fail"]) == 0.0,
        f"accuracy={float(all_pass['accuracy']):.4f}, recall={float(all_pass['recall_fail']):.4f}",
    )

    anomaly = metrics.loc[metrics["name"] == "isolation_forest_pass_only"].iloc[0]
    add_check(
        checks,
        "metrics: anomaly baseline trade-off",
        float(anomaly["recall_fail"]) > float(best["recall_fail"])
        and int(anomaly["false_alarm_count"]) > int(best["false_alarm_count"]),
        (
            f"anomaly recall={float(anomaly['recall_fail']):.4f}, "
            f"anomaly false_alarm={int(anomaly['false_alarm_count'])}"
        ),
    )
    return metrics


def validate_model_artifacts(checks: list[Check], metrics: pd.DataFrame) -> None:
    for model_name in metrics["name"]:
        if model_name == "all_pass_baseline":
            continue
        model_dir = MODELS / str(model_name)
        for filename in [
            "summary.csv",
            "validation_threshold_curve.csv",
            "test_predictions.csv",
            "precision_recall_curve.png",
            "threshold_tradeoff.png",
            "confusion_matrix.png",
        ]:
            require_file(model_dir / filename, checks)


def validate_data_profiles(checks: list[Check]) -> None:
    class_profile = pd.read_csv(REPORTS / "class_profile.csv")
    split_profile = pd.read_csv(REPORTS / "split_class_profile.csv")
    sensor_quality = pd.read_csv(REPORTS / "sensor_quality_profile.csv")

    total_samples = int(class_profile["count"].sum())
    fail_count = int(class_profile.loc[class_profile["class"] == "fail", "count"].iloc[0])
    add_check(
        checks,
        "data profile: sample and fail count",
        total_samples == 1567 and fail_count == 104,
        f"samples={total_samples}, fail={fail_count}",
    )
    add_check(
        checks,
        "data profile: split count",
        int(split_profile["total"].sum()) == 1567,
        f"split total={int(split_profile['total'].sum())}",
    )
    add_check(
        checks,
        "data profile: zero variance sensors",
        int(sensor_quality["is_zero_variance"].sum()) == 116,
        f"zero_variance={int(sensor_quality['is_zero_variance'].sum())}",
    )


def validate_cost_and_dashboard(checks: list[Check]) -> None:
    cost = pd.read_csv(REPORTS / "cost_threshold_analysis.csv")
    add_check(
        checks,
        "cost: scenarios",
        set(cost["scenario"]) == {"balanced_review", "yield_risk_sensitive", "escape_critical"},
        f"scenarios={', '.join(cost['scenario'])}",
    )
    yield_row = cost.loc[cost["scenario"] == "yield_risk_sensitive"].iloc[0]
    add_check(
        checks,
        "cost: yield-risk threshold behavior",
        float(yield_row["selected_threshold_from_validation"]) < 0.10
        and int(yield_row["test_missed_fail_count"]) <= 1,
        (
            f"threshold={float(yield_row['selected_threshold_from_validation']):.2f}, "
            f"missed_fail={int(yield_row['test_missed_fail_count'])}"
        ),
    )

    dashboard = (REPORTS / "dashboard.html").read_text(encoding="utf-8")
    for token in ["extra_trees_balanced", "isolation_forest_pass_only", "All-Pass Accuracy", "Cost-Sensitive"]:
        add_check(checks, f"dashboard: contains {token}", token in dashboard, "token present")


def validate_inference_artifact(checks: list[Check], metrics: pd.DataFrame) -> None:
    artifact_path = ARTIFACTS / "best_model.joblib"
    metadata_path = ARTIFACTS / "model_metadata.json"
    if not artifact_path.exists() or not metadata_path.exists():
        add_check(checks, "artifact: loadable model", False, "missing model artifact or metadata")
        return

    artifact: dict[str, Any] = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best = metrics.iloc[0]
    add_check(
        checks,
        "artifact: model name matches metrics",
        artifact["model_name"] == best["name"] == metadata["model_name"],
        f"artifact={artifact['model_name']}, metrics={best['name']}",
    )
    add_check(
        checks,
        "artifact: threshold matches metrics",
        abs(float(artifact["threshold"]) - float(best["threshold"])) < 1e-12,
        f"artifact={float(artifact['threshold']):.4f}, metrics={float(best['threshold']):.4f}",
    )
    add_check(
        checks,
        "artifact: feature schema",
        len(artifact["feature_names"]) == 590
        and artifact["feature_names"][0] == "sensor_000"
        and artifact["feature_names"][-1] == "sensor_589",
        f"feature_count={len(artifact['feature_names'])}",
    )


def validate_git_policy(checks: list[Check]) -> None:
    raw_files = tracked_files("data/raw")
    artifact_files = tracked_files("artifacts")
    add_check(checks, "git policy: raw data not tracked", len(raw_files) == 0, f"tracked_raw_files={len(raw_files)}")
    add_check(
        checks,
        "git policy: local model artifacts not tracked",
        len(artifact_files) == 0,
        f"tracked_artifact_files={len(artifact_files)}",
    )


def write_summary(checks: list[Check]) -> None:
    passed = sum(check.passed for check in checks)
    failed = len(checks) - passed
    lines = [
        "# Validation Summary",
        "",
        f"- Checks passed: {passed}",
        f"- Checks failed: {failed}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"| {check.name} | {status} | {check.detail} |")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated reports, artifacts, and repository policy.")
    parser.add_argument("--no-write-summary", action="store_true", help="Do not write reports/validation_summary.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[Check] = []
    validate_required_files(checks)
    metrics = validate_metrics(checks)
    validate_model_artifacts(checks, metrics)
    validate_data_profiles(checks)
    validate_cost_and_dashboard(checks)
    validate_inference_artifact(checks, metrics)
    validate_git_policy(checks)

    if not args.no_write_summary:
        write_summary(checks)

    failed = [check for check in checks if not check.passed]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}: {check.name} - {check.detail}")

    if failed:
        raise SystemExit(f"validation failed: {len(failed)} checks failed")

    print(f"validation passed: {len(checks)} checks")
    if not args.no_write_summary:
        print(f"saved validation summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
