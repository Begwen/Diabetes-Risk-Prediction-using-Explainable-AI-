"""
step5_report.py — Generate the HTML dashboard and PDF summary report.

Converted from: notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb  Steps 5 & 8
Phase 2 fix: replaced absolute hardcoded image paths with relative paths in dashboard.html.
             All metric placeholders (was {x:.4f}) are now filled from baseline_metrics.csv.

Run:
    python src/step5_report.py
"""

import os
import sys
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    METRICS_CSV, RESULTS_DIR, PLOTS_DIR, EXPLAIN_DIR,
    DOCS_DIR, REPORTS_DIR,
    make_dirs,
)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------
def generate_dashboard(metrics_df: pd.DataFrame) -> str:
    """Return HTML string for the interactive dashboard."""
    best = metrics_df["F1"].idxmax()

    def img_tag(path: str, width: str = "520px") -> str:
        # Use paths relative to the reports/ directory
        if os.path.exists(path):
            rel = os.path.relpath(path, REPORTS_DIR)
            return f'<div><img src="{rel}" width="{width}" /></div>'
        return ""

    roc_img = img_tag(os.path.join(PLOTS_DIR, f"ROC_{best}.png"))
    pr_img  = img_tag(os.path.join(PLOTS_DIR, f"PR_{best}.png"))
    shap_summary = img_tag(os.path.join(EXPLAIN_DIR, "XGBoost_shap_summary.png"))
    shap_bar     = img_tag(os.path.join(EXPLAIN_DIR, "XGBoost_shap_bar.png"))

    # Build SHAP feature importance table from CSV
    shap_csv = os.path.join(EXPLAIN_DIR, "shap_values_class1.csv")
    shap_imp_html = ""
    if os.path.exists(shap_csv):
        shap_df = pd.read_csv(shap_csv)
        feat_cols = [c for c in shap_df.columns if c not in ("sample_index", "base_value")]
        mean_abs = shap_df[feat_cols].abs().mean().sort_values(ascending=False)
        rows = "".join(
            f"<tr><td>{f}</td><td>{v:.5f}</td></tr>" for f, v in mean_abs.items()
        )
        shap_imp_html = (
            "<h3>Feature Importance (Mean |SHAP|)</h3>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Feature</th><th>Mean |SHAP|</th></tr>"
            f"{rows}</table>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Diabetes Risk — Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
    h1,h2,h3 {{ margin: 0.4rem 0; }}
    .card {{ border:1px solid #ddd; border-radius:8px; padding:16px;
             margin-bottom:16px; background:#fff; }}
    table {{ border-collapse: collapse; }}
    th, td {{ padding: 6px 10px; }}
    .badge {{ display:inline-block; background:#2ecc71; color:#fff;
              padding:4px 10px; border-radius:4px; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>Diabetes Risk Prediction — Dashboard</h1>
  <div class="card">
    <p>Phase 1 Baseline &mdash; Best model: <span class="badge">{best}</span>
    &nbsp; F1={metrics_df.loc[best,"F1"]:.4f} &nbsp; ROC-AUC={metrics_df.loc[best,"ROC_AUC"]:.4f}</p>
    <p>Phase 2 Targets &mdash; F1 &ge; 0.70 &nbsp; Recall &ge; 0.75 &nbsp; ROC-AUC &ge; 0.86</p>
  </div>

  <div class="card">
    <h2>Baseline Metrics</h2>
    {metrics_df.to_html(float_format=lambda x: f"{x:.4f}")}
  </div>

  <div class="card">
    <h2>ROC &amp; PR Curves (Best Model: {best})</h2>
    {roc_img}
    {pr_img}
  </div>

  <div class="card">
    <h2>SHAP Global Explainability (XGBoost — Phase 2 fix)</h2>
    {shap_summary}
    {shap_bar}
  </div>

  <div class="card">
    {shap_imp_html}
  </div>

  <div class="card">
    <small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    Phase 2 | IIT-Guwahati BTP | Priya Sharma 220101081</small>
  </div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
def generate_pdf_report(metrics_df: pd.DataFrame) -> str:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print("reportlab not installed — skipping PDF generation.")
        return ""

    out_path = os.path.join(
        REPORTS_DIR,
        f"Diabetes_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    )

    styles = getSampleStyleSheet()
    story = []

    story += [
        Paragraph("<b>Diabetes Risk Prediction — Final Report</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Automated Modeling & Explainability Summary", styles["Heading2"]),
        Spacer(1, 6),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(
            textwrap.fill(
                "This report summarises data preprocessing, baseline model training, "
                "evaluation, and SHAP-based explainability for the Diabetes Risk "
                "Prediction pipeline. Phase 2 fixes the SHAP explainer to use "
                "XGBoost TreeExplainer with correct base_values (~0.35).",
                120,
            ),
            styles["BodyText"],
        ),
        Spacer(1, 16),
    ]

    # Metrics table
    story.append(Paragraph("<b>Model Performance Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    tbl_data = [["Model"] + cols]
    for m in metrics_df.index:
        tbl_data.append([m] + [f"{metrics_df.loc[m, c]:.4f}" for c in cols])

    tbl = Table(tbl_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONT",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (1, 1), (-1, -1), "CENTER"),
    ]))
    story += [tbl, Spacer(1, 16)]

    # SHAP plots
    for fname, caption in [
        ("XGBoost_shap_summary.png", "SHAP Summary Plot — XGBoost (Phase 2)"),
        ("XGBoost_shap_bar.png",     "Mean |SHAP| Feature Importance"),
        ("patient_5_waterfall.png",  "Local Explanation — Patient 5"),
    ]:
        path = os.path.join(EXPLAIN_DIR, fname)
        if os.path.exists(path):
            story += [
                Paragraph(caption, styles["Heading3"]),
                Image(path, width=14 * cm, height=10 * cm),
                Spacer(1, 12),
            ]

    story.append(Paragraph(
        "<i>Disclaimer: For academic and research use only. "
        "Not a substitute for clinical medical advice.</i>",
        styles["Normal"],
    ))

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    doc.build(story)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    make_dirs()

    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(
            "baseline_metrics.csv not found. Run step3_train.py first."
        )

    metrics_df = pd.read_csv(METRICS_CSV, index_col=0)
    print(f"Loaded metrics for: {metrics_df.index.tolist()}")

    # Dashboard HTML
    html = generate_dashboard(metrics_df)
    html_path = os.path.join(REPORTS_DIR, "dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved dashboard  : {html_path}")

    # PDF report
    pdf_path = generate_pdf_report(metrics_df)
    if pdf_path:
        print(f"Saved PDF report : {pdf_path}")

    report = (
        f"Step 5 — Report & Dashboard\n"
        f"----------------------------\n"
        f"Dashboard : {html_path}\n"
        f"PDF       : {pdf_path or 'skipped (reportlab not installed)'}\n"
    )
    with open(os.path.join(DOCS_DIR, "step_05_report.txt"), "w") as f:
        f.write(report)

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
