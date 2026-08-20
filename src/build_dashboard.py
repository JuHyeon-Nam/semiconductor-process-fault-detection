from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DASHBOARD = REPORTS / "dashboard.html"


def fmt_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def table_html(frame: pd.DataFrame, columns: list[str], numeric: set[str] | None = None, limit: int | None = None) -> str:
    numeric = numeric or set()
    view = frame.loc[:, columns].head(limit) if limit else frame.loc[:, columns]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = []
    for row in view.itertuples(index=False):
        cells = []
        for column, value in zip(columns, row):
            text = fmt_float(value) if column in numeric else escape(str(value))
            cells.append(f"<td>{text}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def metric_card(label: str, value: str, detail: str = "") -> str:
    detail_html = f"<span>{escape(detail)}</span>" if detail else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{escape(label)}</div>
      <div class="metric-value">{escape(value)}</div>
      {detail_html}
    </div>
    """


def build_dashboard(root: Path = ROOT) -> Path:
    reports = root / "reports"
    metrics = pd.read_csv(reports / "metrics.csv")
    class_profile = pd.read_csv(reports / "class_profile.csv")
    missing_profile = pd.read_csv(reports / "missing_profile.csv")
    sensor_quality = pd.read_csv(reports / "sensor_quality_profile.csv")
    cost_summary = pd.read_csv(reports / "cost_threshold_analysis.csv")
    score_bands = pd.read_csv(reports / "score_band_analysis.csv")
    permutation = pd.read_csv(reports / "permutation_importance.csv")

    best = metrics.iloc[0]
    all_pass = metrics.loc[metrics["name"] == "all_pass_baseline"].iloc[0]
    fail_ratio = class_profile.loc[class_profile["class"] == "fail", "ratio"].iloc[0]
    zero_variance = int(sensor_quality["is_zero_variance"].sum())
    high_missing = int((sensor_quality["missing_ratio"] >= 0.50).sum())

    metric_cards = "\n".join(
        [
            metric_card("Best Model", str(best["name"]), f"threshold {float(best['threshold']):.2f}"),
            metric_card("Fail Recall", fmt_float(best["recall_fail"]), "held-out test"),
            metric_card("Fail F2", fmt_float(best["f2_fail"]), "validation threshold"),
            metric_card("PR-AUC", fmt_float(best["pr_auc"]), "fail ranking quality"),
            metric_card("Missed Fail", str(int(best["missed_fail_count"])), "out of 21 test fail cases"),
            metric_card("False Alarm", str(int(best["false_alarm_count"])), "engineering review candidates"),
            metric_card("All-Pass Accuracy", fmt_float(all_pass["accuracy"]), "but zero fail recall"),
            metric_card("Fail Ratio", fmt_float(fail_ratio), "full SECOM dataset"),
        ]
    )

    model_table = table_html(
        metrics,
        [
            "name",
            "threshold",
            "accuracy",
            "recall_fail",
            "precision_fail",
            "f2_fail",
            "pr_auc",
            "missed_fail_count",
            "false_alarm_count",
        ],
        {
            "threshold",
            "accuracy",
            "recall_fail",
            "precision_fail",
            "f2_fail",
            "pr_auc",
        },
    )
    cost_table = table_html(
        cost_summary,
        [
            "scenario",
            "false_alarm_cost",
            "missed_fail_cost",
            "selected_threshold_from_validation",
            "test_missed_fail_count",
            "test_false_alarm_count",
            "test_fail_recall",
        ],
        {
            "false_alarm_cost",
            "missed_fail_cost",
            "selected_threshold_from_validation",
            "test_fail_recall",
        },
    )
    score_band_table = table_html(
        score_bands,
        [
            "band",
            "sample_count",
            "fail_count",
            "fail_rate",
            "cumulative_review_rate",
            "cumulative_fail_capture_rate",
        ],
        {
            "fail_rate",
            "cumulative_review_rate",
            "cumulative_fail_capture_rate",
        },
        limit=5,
    )
    missing_table = table_html(
        missing_profile,
        ["sensor", "missing_count", "missing_ratio", "non_missing_count"],
        {"missing_ratio"},
        limit=10,
    )
    permutation_table = table_html(
        permutation,
        ["feature", "permutation_importance_mean", "permutation_importance_std"],
        {"permutation_importance_mean", "permutation_importance_std"},
        limit=10,
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SECOM FDC Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #5b6677;
      --line: #d8dee8;
      --panel: #ffffff;
      --page: #f6f8fb;
      --blue: #2459a6;
      --red: #cf3f46;
      --green: #317a4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--page);
      color: var(--ink);
      line-height: 1.5;
    }}
    header {{
      padding: 32px 40px 24px;
      background: #111827;
      color: white;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 920px;
      color: #cbd5e1;
      font-size: 15px;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 24px 44px;
    }}
    section {{
      margin: 0 0 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
      min-height: 110px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .metric-card span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #344054;
      background: #f1f5f9;
      font-weight: 700;
    }}
    .note {{
      border-left: 4px solid var(--blue);
      padding: 12px 14px;
      background: #eff6ff;
      color: #25466f;
      margin-top: 14px;
    }}
    @media (max-width: 920px) {{
      header {{ padding: 24px 22px; }}
      .metrics, .grid-two {{ grid-template-columns: 1fr; }}
      main {{ padding: 20px 14px 36px; }}
      section {{ padding: 16px; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SECOM FDC Decision-Support Dashboard</h1>
    <p>Static review page generated from the project reports. It summarizes class imbalance, model performance, threshold trade-offs, sensor candidates, and the manufacturing review workflow.</p>
  </header>
  <main>
    <section>
      <h2>Executive Snapshot</h2>
      <div class="metrics">{metric_cards}</div>
      <div class="note">The all-pass baseline looks strong by accuracy, but catches no fail samples. This dashboard therefore focuses on fail recall, F2, PR-AUC, missed fail, and false alarm counts.</div>
    </section>

    <section>
      <h2>Result Overview</h2>
      <img src="figures/result_dashboard.png" alt="Result dashboard">
    </section>

    <section>
      <h2>Model Comparison</h2>
      {model_table}
    </section>

    <section>
      <h2>Cost-Sensitive Thresholding</h2>
      <p>Thresholds are selected on validation data. The held-out test split is used only for final evaluation.</p>
      <div class="grid-two">
        <div>{cost_table}</div>
        <img src="figures/cost_threshold_analysis.png" alt="Cost threshold analysis">
      </div>
    </section>

    <section>
      <h2>Score Band Review</h2>
      <p>Held-out test samples are sorted by fail-risk score and grouped into equal-sized bands. This is a ranking-quality view, not a calibrated probability claim.</p>
      <div class="grid-two">
        <img src="figures/score_band_analysis.png" alt="Score band analysis">
        <div>{score_band_table}</div>
      </div>
    </section>

    <section>
      <h2>Manufacturing Workflow</h2>
      <img src="figures/fdc_operating_workflow.png" alt="FDC operating workflow">
    </section>

    <section>
      <h2>Data Quality</h2>
      <p>Zero-variance sensors: {zero_variance}. Sensors with at least 50% missing values: {high_missing}.</p>
      <div class="grid-two">
        <img src="figures/sensor_quality_summary.png" alt="Sensor quality summary">
        <div>{missing_table}</div>
      </div>
    </section>

    <section>
      <h2>Sensor Candidate Prioritization</h2>
      <p>SECOM sensors are anonymized, so these are engineering review candidates rather than confirmed physical root causes.</p>
      <div class="grid-two">
        <img src="figures/permutation_importance.png" alt="Permutation importance">
        <div>{permutation_table}</div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output = root / "reports" / "dashboard.html"
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    output = build_dashboard(ROOT)
    print(f"saved dashboard: {output}")


if __name__ == "__main__":
    main()
