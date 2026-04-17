"""
step4_explain.py — SHAP global and local explainability for XGBoost.

Converted from: notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb  Step 4
Phase 2 fix: uses shap.TreeExplainer(xgb_model) — the correct explainer.
             Phase 1 notebook used RandomForest as a workaround; that bug is fixed here.

Run:
    python src/step4_explain.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLEANED_CSV, RESULTS_DIR, EXPLAIN_DIR, DOCS_DIR,
    TARGET_COL, RANDOM_STATE, TEST_SIZE,
    make_dirs,
)
from sklearn.model_selection import train_test_split


def main():
    make_dirs()

    # Load data and model
    df = pd.read_csv(CLEANED_CSV)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Phase 2 fix: use XGBoost (best model) with TreeExplainer
    model_path = os.path.join(RESULTS_DIR, "XGBoost.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "XGBoost.joblib not found. Run step3_train.py first."
        )
    model = joblib.load(model_path)
    print("Loaded XGBoost model for SHAP (Phase 2 fix: TreeExplainer, not RF)")

    # Compute SHAP values with TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    print("SHAP values computed. Shape:", shap_values.values.shape)

    # Handle multi-output (TreeExplainer can return 3D for binary classifiers)
    if shap_values.values.ndim == 3:
        sv_class1 = shap_values.values[:, :, 1]
        bv_class1 = shap_values.base_values[:, 1]
    else:
        sv_class1 = shap_values.values
        bv_class1 = shap_values.base_values

    print(f"Base value (should be ~0.35): {bv_class1.mean():.4f}")

    # --- Global: summary beeswarm ---
    plt.figure()
    shap.summary_plot(sv_class1, X_test, show=False)
    plt.title("Global Feature Importance — XGBoost SHAP")
    plt.tight_layout()
    out = os.path.join(EXPLAIN_DIR, "XGBoost_shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- Global: bar chart ---
    plt.figure()
    shap.summary_plot(sv_class1, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP| — XGBoost")
    plt.tight_layout()
    out = os.path.join(EXPLAIN_DIR, "XGBoost_shap_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- Dependency plots — all 8 features ---
    for feature in X_test.columns:
        try:
            plt.figure()
            shap.dependence_plot(feature, sv_class1, X_test, show=False)
            plt.title(f"SHAP Dependence — {feature}")
            plt.tight_layout()
            out = os.path.join(EXPLAIN_DIR, f"shap_dependence_{feature}.png")
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  Skipped {feature}: {e}")
    print("Saved all 8 dependence plots.")

    # --- Waterfall for patient 5 ---
    pos = 5
    exp = shap.Explanation(
        values=sv_class1[pos],
        base_values=bv_class1[pos],
        data=X_test.iloc[pos].values,
        feature_names=X_test.columns.tolist(),
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.title(f"Local Explanation — Patient {pos} (XGBoost)")
    plt.tight_layout()
    out = os.path.join(EXPLAIN_DIR, f"patient_{pos}_waterfall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- Save SHAP value matrix ---
    shap_df = pd.DataFrame(sv_class1, columns=X_test.columns)
    shap_df.insert(0, "base_value", bv_class1)
    shap_df.insert(0, "sample_index", X_test.index.values)
    out = os.path.join(EXPLAIN_DIR, "shap_values_class1.csv")
    shap_df.to_csv(out, index=False)
    print(f"Saved: {out}")

    # Feature importance table
    mean_abs = shap_df.drop(columns=["sample_index", "base_value"]).abs().mean()
    mean_abs = mean_abs.sort_values(ascending=False)
    print("\nMean |SHAP| feature importance:")
    print(mean_abs.to_string())

    summary = (
        f"Step 4 — Explainability Summary\n"
        f"--------------------------------\n"
        f"Model     : XGBoost (Phase 2 fix — TreeExplainer)\n"
        f"Samples   : {len(X_test)}\n"
        f"Base value mean: {bv_class1.mean():.4f} (should be ~0.35)\n"
        f"Top features: {', '.join(mean_abs.index[:5].tolist())}\n\n"
        f"Artefacts:\n"
        f"  explain/XGBoost_shap_summary.png\n"
        f"  explain/XGBoost_shap_bar.png\n"
        f"  explain/shap_dependence_<feature>.png  (8 plots)\n"
        f"  explain/patient_5_waterfall.png\n"
        f"  explain/shap_values_class1.csv\n"
    )
    with open(os.path.join(DOCS_DIR, "step_04_explainability_summary.txt"), "w") as f:
        f.write(summary)

    print("\nStep 4 complete.")
    return shap_df, mean_abs


if __name__ == "__main__":
    main()
