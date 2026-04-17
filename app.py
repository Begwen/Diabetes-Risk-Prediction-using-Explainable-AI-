"""
Diabetes Risk Prediction App — Phase 2
Author: Priya Sharma | Roll No. 220101081 | IIT-Guwahati

Phase 2 upgrades:
  - Fixed all hardcoded Colab paths → relative paths via os.path.dirname(__file__)
  - Switched from RandomForest to XGBoost (best model, F1=0.6542)
  - Fixed SHAP: uses TreeExplainer(xgb_model) instead of shap.Explainer(rf)
  - Added input validation (rejects biologically impossible values)
  - Tab 1: Single patient prediction + SHAP waterfall + clinical advice
  - Tab 2: Batch CSV screening with summary stats and download
  - Tab 3: What-if Simulator — adjust sliders and see risk delta live
  - 4-tier clinical risk stratification (Low / Moderate / High / Critical)
  - Personalised advice engine using SHAP-ranked feature importance
  - PDF patient report download
"""

import os
import io
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — always relative to this file, works on any machine or Streamlit Cloud
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
PROC_DIR = os.path.join(BASE, "data", "processed")

MODEL_PATH = os.path.join(RESULTS_DIR, "XGBoost.joblib")
PREPROCESSOR_PATH = os.path.join(PROC_DIR, "preprocessor.joblib")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Optimal clinical threshold (recall-optimised, ~0.38 for this dataset)
OPTIMAL_THRESHOLD = 0.38

# ---------------------------------------------------------------------------
# Load model + preprocessor (cached — loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Clinical risk stratification (ADA Standards of Care 2024)
# ---------------------------------------------------------------------------
def risk_tier(prob: float) -> tuple[str, str, str]:
    """Return (tier_label, colour, clinical_action)."""
    if prob < 0.25:
        return (
            "Low Risk",
            "#2ecc71",
            "Routine monitoring. Maintain healthy lifestyle and annual check-up.",
        )
    elif prob < 0.50:
        return (
            "Moderate Risk",
            "#f39c12",
            "Lifestyle changes recommended. Consider dietary review and HbA1c test within 6 months.",
        )
    elif prob < 0.75:
        return (
            "High Risk",
            "#e67e22",
            "GP referral recommended. Request fasting glucose and HbA1c test within 4 weeks.",
        )
    else:
        return (
            "Critical Risk",
            "#e74c3c",
            "Immediate clinical evaluation recommended. Do not delay.",
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_inputs(d: dict) -> list[str]:
    errors = []
    if d["Glucose"] == 0:
        errors.append("Glucose cannot be 0 — biologically impossible.")
    if d["BMI"] == 0.0:
        errors.append("BMI cannot be 0 — biologically impossible.")
    if d["BloodPressure"] == 0:
        errors.append("Blood Pressure of 0 is not a valid clinical value.")
    if d["Age"] < 10 or d["Age"] > 120:
        errors.append("Age must be between 10 and 120.")
    if d["BMI"] > 70:
        errors.append("BMI > 70 is outside valid clinical range.")
    return errors


# ---------------------------------------------------------------------------
# Feature engineering (same transformations applied at prediction time)
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 6 clinically motivated features for Phase 2 model."""
    df = df.copy()
    df["BMI_Age_interaction"] = df["BMI"] * df["Age"]
    df["Glucose_Insulin_ratio"] = df["Glucose"] / (df["Insulin"] + 1)
    df["Pregnancy_density"] = df["Pregnancies"] / (df["Age"] + 1)
    df["High_Glucose_flag"] = (df["Glucose"] > 140).astype(int)
    df["Obese_flag"] = (df["BMI"] >= 30).astype(int)
    g_norm = (df["Glucose"] - df["Glucose"].mean()) / (df["Glucose"].std() + 1e-9)
    b_norm = (df["BMI"] - df["BMI"].mean()) / (df["BMI"].std() + 1e-9)
    a_norm = (df["Age"] - df["Age"].mean()) / (df["Age"].std() + 1e-9)
    df["Metabolic_risk_score"] = (g_norm + b_norm + a_norm) / 3
    return df


# ---------------------------------------------------------------------------
# Personalised advice engine
# ---------------------------------------------------------------------------
def personalised_advice(feature_names, shap_vals, patient_row: dict) -> list[str]:
    """Return up to 3 targeted clinical recommendations ranked by |SHAP|."""
    advice_map = {
        "Glucose": (
            0.05,
            "Glucose is your highest risk factor. Schedule an HbA1c test with your GP.",
        ),
        "BMI": (
            0.03,
            "BMI is a key driver. Losing 5–7% body weight reduces diabetes risk by up to 58% (ADA 2024).",
        ),
        "Age": (
            0.03,
            "Age increases risk — annual screening is essential for adults over 45.",
        ),
        "DiabetesPedigreeFunction": (
            0.02,
            "Family history is a significant factor. Discuss genetic screening with your doctor.",
        ),
        "Insulin": (
            0.02,
            "Insulin resistance detected. Consider a fasting insulin test.",
        ),
        "Pregnancies": (
            0.01,
            "Multiple pregnancies are linked to gestational diabetes risk — discuss with your GP.",
        ),
    }

    shap_dict = dict(zip(feature_names, shap_vals))
    ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    tips = []
    for feat, val in ranked:
        if feat in advice_map:
            threshold, message = advice_map[feat]
            if abs(val) >= threshold:
                tips.append(message)
        if len(tips) == 3:
            break

    if not tips:
        tips.append(
            "Maintain a balanced diet, regular exercise (150 min/week), and schedule annual check-ups."
        )
    return tips


# ---------------------------------------------------------------------------
# SHAP waterfall figure
# ---------------------------------------------------------------------------
def shap_waterfall_fig(model, preprocessor, df_raw: pd.DataFrame, feature_names):
    X_scaled = preprocessor.transform(df_raw[FEATURE_NAMES])
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_scaled)

    if sv.values.ndim == 3:
        vals = sv.values[0, :, 1]
        base = sv.base_values[0, 1]
    else:
        vals = sv.values[0]
        base = sv.base_values[0]

    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=df_raw[FEATURE_NAMES].values[0],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(exp, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig, vals


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def generate_pdf(patient: dict, prob: float, tier: str, action: str, tips: list[str]) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Diabetes Risk Assessment Report", styles["Title"]))
        story.append(Paragraph("Generated by AI Screening Tool — IIT-Guwahati BTP 2025", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph(f"<b>Risk Tier:</b> {tier}", styles["Normal"]))
        story.append(Paragraph(f"<b>Diabetes Probability:</b> {prob:.1%}", styles["Normal"]))
        story.append(Paragraph(f"<b>Clinical Action:</b> {action}", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("<b>Patient Inputs</b>", styles["Heading2"]))
        table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in patient.items()]
        t = Table(table_data, colWidths=[8*cm, 6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("<b>Personalised Recommendations</b>", styles["Heading2"]))
        for i, tip in enumerate(tips, 1):
            story.append(Paragraph(f"{i}. {tip}", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph(
            "<i>Disclaimer: For clinical decision support only. Not a substitute for medical advice. "
            "Cite: ADA Standards of Medical Care in Diabetes 2024.</i>",
            styles["Normal"],
        ))

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor — Phase 2", layout="wide")
st.title("Diabetes Risk Prediction App — Phase 2")
st.caption("Priya Sharma | 220101081 | IIT-Guwahati BTP 2025")

model, preprocessor = load_artifacts()

tab1, tab2, tab3 = st.tabs(["Single Patient", "Batch Screening", "What-if Simulator"])

# ===========================================================================
# TAB 1 — Single Patient
# ===========================================================================
with tab1:
    st.header("Single Patient Risk Assessment")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
            Glucose = st.number_input("Glucose (mg/dL)", 0, 250, 120)
            BloodPressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
            SkinThickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        with col2:
            Insulin = st.number_input("Insulin (µU/mL)", 0, 900, 80)
            BMI = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
            DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
            Age = st.number_input("Age (years)", 10, 100, 35)
        submitted = st.form_submit_button("Predict Risk", type="primary")

    if submitted:
        patient = {
            "Pregnancies": Pregnancies, "Glucose": Glucose,
            "BloodPressure": BloodPressure, "SkinThickness": SkinThickness,
            "Insulin": Insulin, "BMI": BMI,
            "DiabetesPedigreeFunction": DPF, "Age": Age,
        }
        errors = validate_inputs(patient)
        if errors:
            for e in errors:
                st.error(e)
        else:
            df = pd.DataFrame([patient])
            X_scaled = preprocessor.transform(df[FEATURE_NAMES])
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = int(prob >= OPTIMAL_THRESHOLD)
            tier_label, colour, action = risk_tier(prob)

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Diabetes Probability", f"{prob:.1%}")
            c2.metric("Prediction (threshold={:.2f})".format(OPTIMAL_THRESHOLD),
                      "Diabetic" if pred == 1 else "Non-Diabetic")
            c3.metric("Risk Tier", tier_label)

            st.markdown(
                f"<div style='background:{colour};color:white;padding:12px;"
                f"border-radius:8px;font-size:15px'>"
                f"<b>Clinical Action:</b> {action}</div>",
                unsafe_allow_html=True,
            )

            st.subheader("SHAP Explanation (XGBoost TreeExplainer)")
            with st.spinner("Computing SHAP values…"):
                try:
                    fig, shap_vals = shap_waterfall_fig(model, preprocessor, df, FEATURE_NAMES)
                    st.pyplot(fig)
                    plt.close(fig)

                    tips = personalised_advice(FEATURE_NAMES, shap_vals, patient)
                    st.subheader("Personalised Clinical Recommendations")
                    for i, tip in enumerate(tips, 1):
                        st.info(f"**{i}.** {tip}")

                    pdf_bytes = generate_pdf(patient, prob, tier_label, action, tips)
                    if pdf_bytes:
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_bytes,
                            file_name="diabetes_risk_report.pdf",
                            mime="application/pdf",
                        )
                except Exception as ex:
                    st.warning(f"SHAP explanation unavailable: {ex}")

# ===========================================================================
# TAB 2 — Batch Screening
# ===========================================================================
with tab2:
    st.header("Batch Patient Screening")
    st.write(
        "Upload a CSV with the same 8 columns as the Pima dataset: "
        "`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age`"
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # ── Guard: detect pre-scaled (already z-scored) uploads ──────────────
                # Raw Glucose is 70–200 mg/dL; after z-scoring it becomes roughly -3..+3.
                # If the median Glucose column is in [-5, 5] the user uploaded a
                # processed CSV (e.g. pima_cleaned.csv) instead of the raw dataset.
                # Double-scaling would compress all probabilities to ≈0 → all Low Risk.
                glucose_median = batch_df["Glucose"].median()
                if -5 < glucose_median < 5:
                    st.error(
                        "**Wrong file uploaded.**  \n"
                        "The `Glucose` column in your CSV has values near zero "
                        f"(median = {glucose_median:.2f}), which means it is already "
                        "z-scored / normalised.  \n\n"
                        "Please upload the **raw** dataset: `data/raw/diabetes.csv`  \n"
                        "*(Not `data/processed/pima_cleaned.csv` — that file has already "
                        "been scaled and cannot be fed to the app directly.)*"
                    )
                    st.stop()

                # Replace physiologically impossible zeros with NaN before scaling.
                # The raw Pima CSV encodes missing measurements as 0 in these columns.
                # Without this step the scaler treats 0 as a real value, producing
                # near-zero probabilities for every patient (all show as Low Risk).
                ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                batch_input = batch_df[FEATURE_NAMES].copy()
                batch_input[ZERO_INVALID] = batch_input[ZERO_INVALID].replace(0, np.nan)

                X_batch = preprocessor.transform(batch_input)
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= OPTIMAL_THRESHOLD).astype(int)
                tiers = [risk_tier(p)[0] for p in probs]

                batch_df["Risk_Probability"] = probs.round(4)
                batch_df["Prediction"] = preds
                batch_df["Risk_Tier"] = tiers

                st.success(f"{len(batch_df)} patients screened.")

                tier_counts = pd.Series(tiers).value_counts()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Critical Risk", tier_counts.get("Critical Risk", 0))
                col2.metric("High Risk", tier_counts.get("High Risk", 0))
                col3.metric("Moderate Risk", tier_counts.get("Moderate Risk", 0))
                col4.metric("Low Risk", tier_counts.get("Low Risk", 0))

                st.dataframe(batch_df, use_container_width=True)

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_out,
                    file_name="batch_screening_results.csv",
                    mime="text/csv",
                )
        except Exception as ex:
            st.error(f"Error processing file: {ex}")

# ===========================================================================
# TAB 3 — What-if Simulator
# ===========================================================================
with tab3:
    st.header("What-if Risk Simulator")
    st.write(
        "Set a **baseline** patient profile on the left. "
        "Then adjust sliders on the right to see how risk changes."
    )

    col_base, col_mod = st.columns(2)

    with col_base:
        st.subheader("Baseline Profile")
        b_preg = st.number_input("Pregnancies", 0, 20, 2, key="b_preg")
        b_gluc = st.number_input("Glucose", 0, 250, 148, key="b_gluc")
        b_bp = st.number_input("Blood Pressure", 0, 150, 72, key="b_bp")
        b_skin = st.number_input("Skin Thickness", 0, 99, 35, key="b_skin")
        b_ins = st.number_input("Insulin", 0, 900, 0, key="b_ins")
        b_bmi = st.number_input("BMI", 0.0, 70.0, 33.6, step=0.1, key="b_bmi")
        b_dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.627, step=0.01, key="b_dpf")
        b_age = st.number_input("Age", 10, 100, 50, key="b_age")

    with col_mod:
        st.subheader("Modified Profile (adjust sliders)")
        m_gluc = st.slider("Glucose", 0, 250, b_gluc, key="m_gluc")
        m_bmi = st.slider("BMI", 0.0, 70.0, float(b_bmi), step=0.5, key="m_bmi")
        m_bp = st.slider("Blood Pressure", 0, 150, b_bp, key="m_bp")
        m_ins = st.slider("Insulin", 0, 900, b_ins, key="m_ins")
        m_age = st.slider("Age", 10, 100, b_age, key="m_age")
        m_dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, float(b_dpf), step=0.01, key="m_dpf")

    baseline_input = {
        "Pregnancies": b_preg, "Glucose": b_gluc, "BloodPressure": b_bp,
        "SkinThickness": b_skin, "Insulin": b_ins, "BMI": b_bmi,
        "DiabetesPedigreeFunction": b_dpf, "Age": b_age,
    }
    modified_input = {
        "Pregnancies": b_preg, "Glucose": m_gluc, "BloodPressure": m_bp,
        "SkinThickness": b_skin, "Insulin": m_ins, "BMI": m_bmi,
        "DiabetesPedigreeFunction": m_dpf, "Age": m_age,
    }

    df_base = pd.DataFrame([baseline_input])
    df_mod = pd.DataFrame([modified_input])

    X_base = preprocessor.transform(df_base[FEATURE_NAMES])
    X_mod = preprocessor.transform(df_mod[FEATURE_NAMES])

    prob_base = model.predict_proba(X_base)[0, 1]
    prob_mod = model.predict_proba(X_mod)[0, 1]
    delta = prob_mod - prob_base

    tier_base = risk_tier(prob_base)[0]
    tier_mod = risk_tier(prob_mod)[0]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Risk", f"{prob_base:.1%}", help=tier_base)
    c2.metric("Modified Risk", f"{prob_mod:.1%}",
              delta=f"{delta:+.1%}", delta_color="inverse", help=tier_mod)
    c3.metric("Risk Change", f"{delta:+.1%}",
              delta=f"{'Decreased' if delta < 0 else 'Increased'}",
              delta_color="inverse")

    if abs(delta) > 0.01:
        direction = "DECREASES" if delta < 0 else "INCREASES"
        st.markdown(
            f"**Your risk {direction} from {prob_base:.1%} to {prob_mod:.1%} "
            f"with the modified profile.**"
        )

    # Show top leverage intervention from SHAP
    try:
        explainer = shap.TreeExplainer(model)
        sv_base = explainer(X_base)
        sv_mod = explainer(X_mod)

        def get_vals(sv):
            if sv.values.ndim == 3:
                return sv.values[0, :, 1]
            return sv.values[0]

        vals_base = get_vals(sv_base)
        vals_mod = get_vals(sv_mod)
        impact_diff = vals_base - vals_mod  # positive = intervention reduced risk

        feat_impacts = sorted(
            zip(FEATURE_NAMES, impact_diff), key=lambda x: x[1], reverse=True
        )
        top3 = [(f, v) for f, v in feat_impacts if v > 0.001][:3]
        if top3:
            st.subheader("Top Leverage Interventions")
            for feat, impact in top3:
                st.success(f"**{feat}**: adjusting this feature reduces risk by ~{impact:.3f} SHAP units")
    except Exception:
        pass

st.divider()
st.caption(
    "Disclaimer: For clinical decision support only. Not a substitute for medical advice. "
    "Model: XGBoost (Phase 1 baseline F1=0.6542, ROC-AUC=0.8243). "
    "Cite: ADA Standards of Medical Care in Diabetes 2024."
)
