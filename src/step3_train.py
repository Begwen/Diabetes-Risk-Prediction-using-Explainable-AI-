"""
step3_train.py — Train and evaluate 4 baseline ML models.

Converted from: notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb  Step 3
Phase 2 changes: removed Colab paths, uses src/config.py constants,
                 generates ROC/PR plots per model, saves all artefacts.

Run:
    python src/step3_train.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLEANED_CSV, RESULTS_DIR, PLOTS_DIR, DOCS_DIR,
    METRICS_CSV, BASELINE_TXT,
    TARGET_COL, RANDOM_STATE, TEST_SIZE,
    make_dirs,
)


def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False,
        ),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob), 4),
        "PR_AUC":    round(average_precision_score(y_test, y_prob), 4),
    }


def save_roc_pr_plots(models: dict, X_test, y_test) -> None:
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"ROC_{name}.png"), dpi=150)
        plt.close()

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {name}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"PR_{name}.png"), dpi=150)
        plt.close()

    print(f"Saved ROC/PR plots to: {PLOTS_DIR}")


def main():
    make_dirs()

    df = pd.read_csv(CLEANED_CSV)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class distribution (train):\n{y_train.value_counts(normalize=True).to_string()}")

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        joblib.dump(model, os.path.join(RESULTS_DIR, f"{name}.joblib"))
        print(f"  F1={metrics['F1']:.4f}  ROC-AUC={metrics['ROC_AUC']:.4f}  saved.")

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(METRICS_CSV)
    print(f"\nSaved metrics : {METRICS_CSV}")

    best = metrics_df["F1"].idxmax()
    print(f"\nBest model (F1): {best}  F1={metrics_df.loc[best,'F1']:.4f}")
    print(metrics_df.to_string())

    with open(BASELINE_TXT, "w") as f:
        f.write("Baseline Model Performance Summary\n")
        f.write(metrics_df.to_string())

    save_roc_pr_plots(models, X_test, y_test)

    report = (
        f"Step 3 — Model Training & Evaluation\n"
        f"--------------------------------------\n"
        f"Models : {list(models.keys())}\n"
        f"Best   : {best}\n\n"
        f"{metrics_df.to_string()}\n"
    )
    with open(os.path.join(DOCS_DIR, "step_03_report.txt"), "w") as f:
        f.write(report)

    print("\nStep 3 complete.")
    return models, metrics_df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
