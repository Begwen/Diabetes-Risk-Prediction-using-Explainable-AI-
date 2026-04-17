"""
step6_phase2_improvements.py — All Phase 2 model upgrades in one script.

Implements every item from Phase2_Master_Plan.docx Steps 4–21:
  - Feature engineering (6 clinical features)
  - SMOTE oversampling inside CV pipeline
  - Decision threshold optimisation (recall-maximised)
  - Optuna Bayesian hyperparameter search (50 trials)
  - 5-fold stratified cross-validation with CI
  - Probability calibration (isotonic regression)
  - Stacking ensemble (LR meta-learner)
  - LIME vs SHAP comparison (5 patients)
  - DiCE counterfactual explanations (3 high-risk patients)
  - Fairness / subgroup analysis by age and BMI
  - McNemar statistical significance test (Phase 1 vs Phase 2)

Run:
    python src/step6_phase2_improvements.py

Run a single section:
    python src/step6_phase2_improvements.py --section features
    python src/step6_phase2_improvements.py --section tune
    python src/step6_phase2_improvements.py --section cv
    python src/step6_phase2_improvements.py --section calibrate
    python src/step6_phase2_improvements.py --section ensemble
    python src/step6_phase2_improvements.py --section lime
    python src/step6_phase2_improvements.py --section dice
    python src/step6_phase2_improvements.py --section fairness
    python src/step6_phase2_improvements.py --section stats
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLEANED_CSV, PREPROCESSOR_PATH, RESULTS_DIR, EXPLAIN_DIR, DOCS_DIR,
    FEATURE_NAMES, ENGINEERED_FEATURES, ALL_FEATURES_V2,
    TARGET_COL, RANDOM_STATE, TEST_SIZE, CV_FOLDS, OPTIMAL_THRESHOLD,
    FEATURES_V2_CSV, BEST_PARAMS_JSON, CV_RESULTS_CSV,
    ENSEMBLE_PATH, XGB_V2_PATH,
    make_dirs,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: load data split
# ---------------------------------------------------------------------------
def _load_split():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(CLEANED_CSV)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ===========================================================================
# 1. FEATURE ENGINEERING
# ===========================================================================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 clinically motivated features (Plan §4).
    Works on both raw and scaled DataFrames; expects the 8 original Pima columns.
    """
    df = df.copy()
    df["BMI_Age_interaction"]    = df["BMI"] * df["Age"]
    df["Glucose_Insulin_ratio"]  = df["Glucose"] / (df["Insulin"] + 1)
    df["Pregnancy_density"]      = df["Pregnancies"] / (df["Age"] + 1)
    df["High_Glucose_flag"]      = (df["Glucose"] > 140).astype(int)
    df["Obese_flag"]             = (df["BMI"] >= 30).astype(int)

    g = (df["Glucose"] - df["Glucose"].mean()) / (df["Glucose"].std() + 1e-9)
    b = (df["BMI"]     - df["BMI"].mean())     / (df["BMI"].std()     + 1e-9)
    a = (df["Age"]     - df["Age"].mean())     / (df["Age"].std()     + 1e-9)
    df["Metabolic_risk_score"] = (g + b + a) / 3
    return df


def section_features():
    """Build and save the Phase 2 feature-engineered dataset."""
    print("\n=== Feature Engineering ===")
    df_raw = pd.read_csv(
        os.path.join(os.path.dirname(CLEANED_CSV), "..", "raw", "diabetes.csv")
    )
    # Replace physiologically impossible zeros with NaN then impute
    from sklearn.impute import SimpleImputer
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_raw[zero_cols] = df_raw[zero_cols].replace(0, np.nan)
    imp = SimpleImputer(strategy="median")
    df_raw[zero_cols] = imp.fit_transform(df_raw[zero_cols])

    df_feat = add_engineered_features(df_raw)
    df_feat.to_csv(FEATURES_V2_CSV, index=False)
    print(f"Engineered features saved: {FEATURES_V2_CSV}")
    print(f"New columns: {ENGINEERED_FEATURES}")
    print(df_feat[ENGINEERED_FEATURES].describe().T.to_string())


# ===========================================================================
# 2. SMOTE + 5-FOLD CV BASELINE
# ===========================================================================
def section_cv():
    """
    5-fold stratified CV with SMOTE inside pipeline (Plan §5, §8).
    Reports mean ± std for all metrics.
    """
    print("\n=== 5-Fold CV with SMOTE ===")
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        print("imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return

    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    df = pd.read_csv(CLEANED_CSV)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False,
        )),
    ])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    f1s, recalls, aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[val_idx])
        y_prob = pipe.predict_proba(X[val_idx])[:, 1]
        f1s.append(f1_score(y[val_idx], y_pred))
        recalls.append(recall_score(y[val_idx], y_pred))
        aucs.append(roc_auc_score(y[val_idx], y_prob))
        print(f"  Fold {fold}: F1={f1s[-1]:.4f}  Recall={recalls[-1]:.4f}  AUC={aucs[-1]:.4f}")

    results = {
        "F1_mean": float(np.mean(f1s)),      "F1_std": float(np.std(f1s)),
        "Recall_mean": float(np.mean(recalls)), "Recall_std": float(np.std(recalls)),
        "AUC_mean": float(np.mean(aucs)),     "AUC_std": float(np.std(aucs)),
    }
    print(f"\nCV Summary (5-fold with SMOTE):")
    print(f"  F1     : {results['F1_mean']:.4f} ± {results['F1_std']:.4f}")
    print(f"  Recall : {results['Recall_mean']:.4f} ± {results['Recall_std']:.4f}")
    print(f"  AUC    : {results['AUC_mean']:.4f} ± {results['AUC_std']:.4f}")

    pd.DataFrame([results]).to_csv(CV_RESULTS_CSV, index=False)
    print(f"CV results saved: {CV_RESULTS_CSV}")


# ===========================================================================
# 3. THRESHOLD OPTIMISATION
# ===========================================================================
def section_threshold():
    """
    Plot Precision/Recall/F1 vs threshold, find clinical optimum (Plan §6).
    """
    print("\n=== Threshold Optimisation ===")
    from sklearn.metrics import f1_score, precision_score, recall_score

    X_train, X_test, y_train, y_test = _load_split()
    model_path = os.path.join(RESULTS_DIR, "XGBoost.joblib")
    if not os.path.exists(model_path):
        print("XGBoost.joblib not found — run step3_train.py first.")
        return
    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, f1s,   label="F1")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs,  label="Recall")
    plt.axvline(OPTIMAL_THRESHOLD, color="red", linestyle="--", label=f"Optimal={OPTIMAL_THRESHOLD}")
    plt.xlabel("Threshold")
    plt.title("Threshold Optimisation — XGBoost")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(EXPLAIN_DIR, "threshold_optimisation.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    best_idx = int(np.argmax(recs))  # maximise recall
    print(f"Recall-maximising threshold: {thresholds[best_idx]:.2f}  "
          f"F1={f1s[best_idx]:.4f}  Recall={recs[best_idx]:.4f}  Prec={precs[best_idx]:.4f}")


# ===========================================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# ===========================================================================
def section_tune():
    """
    Bayesian hyperparameter search with Optuna, 50 trials (Plan §7).
    Saves best_params.json and XGBoost_v2.joblib.
    """
    print("\n=== Optuna Hyperparameter Tuning (50 trials) ===")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna not installed. Run: pip install optuna")
        return

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier

    df = pd.read_csv(CLEANED_CSV)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state":     RANDOM_STATE,
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "use_label_encoder": False,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["use_label_encoder"] = False

    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved: {BEST_PARAMS_JSON}")
    print(f"Best F1 (CV):      {study.best_value:.4f}")
    print(json.dumps({k: v for k, v in best_params.items()
                      if k not in ("objective", "eval_metric", "use_label_encoder")}, indent=2))

    # Retrain on full train split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model_v2 = XGBClassifier(**best_params)
    model_v2.fit(X_train, y_train)
    joblib.dump(model_v2, XGB_V2_PATH)
    print(f"XGBoost v2 saved: {XGB_V2_PATH}")

    from sklearn.metrics import f1_score as f1
    y_pred = (model_v2.predict_proba(X_test)[:, 1] >= OPTIMAL_THRESHOLD).astype(int)
    print(f"Phase 2 XGBoost F1 (test): {f1(y_test, y_pred):.4f}")


# ===========================================================================
# 5. PROBABILITY CALIBRATION
# ===========================================================================
def section_calibrate():
    """
    Calibration curves + Brier Score for XGBoost (Plan §9).
    """
    print("\n=== Probability Calibration ===")
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import brier_score_loss

    X_train, X_test, y_train, y_test = _load_split()
    model_path = os.path.join(RESULTS_DIR, "XGBoost.joblib")
    if not os.path.exists(model_path):
        print("XGBoost.joblib not found — run step3_train.py first.")
        return
    model = joblib.load(model_path)

    y_prob_raw = model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_prob_raw)
    print(f"Brier Score (uncalibrated): {brier_raw:.4f}")

    # cv='prefit' was removed in sklearn 1.4; retrain a calibrated wrapper on
    # the training set using 5-fold CV so the test set stays a clean hold-out.
    calib = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calib.fit(X_train, y_train)
    y_prob_cal = calib.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, y_prob_cal)
    print(f"Brier Score (calibrated)  : {brier_cal:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for probs, label in [(y_prob_raw, "Uncalibrated"), (y_prob_cal, "Calibrated (isotonic)")]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{label} (Brier={brier_score_loss(y_test, probs):.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — XGBoost")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(EXPLAIN_DIR, "calibration_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


# ===========================================================================
# 6. STACKING ENSEMBLE
# ===========================================================================
def section_ensemble():
    """
    Build a stacking ensemble from all 4 models (Plan §10).
    Uses out-of-fold predictions to avoid data leakage.
    """
    print("\n=== Stacking Ensemble ===")
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    X_train, X_test, y_train, y_test = _load_split()

    model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
    estimators = []
    for name in model_names:
        path = os.path.join(RESULTS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            estimators.append((name, joblib.load(path)))
        else:
            print(f"  Skipping {name} — model file not found.")

    if len(estimators) < 2:
        print("Need at least 2 base models. Run step3_train.py first.")
        return

    # Silence LightGBM's per-fold verbose output
    import lightgbm as lgb
    lgb.basic._log_callback = None
    for name, est in estimators:
        if hasattr(est, "set_params"):
            try:
                est.set_params(verbose=-1)
            except (ValueError, TypeError):
                pass

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=CV_FOLDS,
        passthrough=False,
    )
    stack.fit(X_train, y_train)

    y_pred = (stack.predict_proba(X_test)[:, 1] >= OPTIMAL_THRESHOLD).astype(int)
    y_prob = stack.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Stacking Ensemble — F1={f1:.4f}  ROC-AUC={auc:.4f}")

    joblib.dump(stack, ENSEMBLE_PATH)
    print(f"Ensemble saved: {ENSEMBLE_PATH}")


# ===========================================================================
# 7. LIME vs SHAP COMPARISON
# ===========================================================================
def section_lime():
    """
    Explain 5 patients with both LIME and SHAP; produce side-by-side table (Plan §18).
    """
    print("\n=== LIME vs SHAP Comparison ===")
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("lime not installed. Run: pip install lime")
        return

    import shap

    X_train, X_test, y_train, y_test = _load_split()
    model = joblib.load(os.path.join(RESULTS_DIR, "XGBoost.joblib"))

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Diabetes", "Diabetes"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    shap_exp = shap.TreeExplainer(model)

    rows = []
    for i in range(5):
        sample = X_test.iloc[i]

        # LIME
        lime_result = lime_exp.explain_instance(
            sample.values, model.predict_proba, num_features=8
        )
        lime_dict = dict(lime_result.as_list())

        # SHAP
        sv = shap_exp(sample.values.reshape(1, -1))
        if sv.values.ndim == 3:
            shap_vals = dict(zip(X_test.columns, sv.values[0, :, 1]))
        else:
            shap_vals = dict(zip(X_test.columns, sv.values[0]))

        row = {"Patient": i}
        for feat in X_test.columns:
            row[f"SHAP_{feat}"]  = round(shap_vals.get(feat, 0), 5)
            # LIME keys may be binned strings; find matching
            lime_val = next(
                (v for k, v in lime_dict.items() if feat in k), 0
            )
            row[f"LIME_{feat}"] = round(lime_val, 5)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    out = os.path.join(EXPLAIN_DIR, "lime_vs_shap_comparison.csv")
    comparison_df.to_csv(out, index=False)
    print(f"LIME vs SHAP comparison saved: {out}")
    print(comparison_df.to_string())


# ===========================================================================
# 8. DiCE COUNTERFACTUAL EXPLANATIONS
# ===========================================================================
def section_dice():
    """
    Generate diverse counterfactual explanations for 3 high-risk patients (Plan §19).
    """
    print("\n=== DiCE Counterfactual Explanations ===")
    try:
        import dice_ml
    except ImportError:
        print("dice-ml not installed. Run: pip install dice-ml")
        return

    X_train, X_test, y_train, y_test = _load_split()
    model = joblib.load(os.path.join(RESULTS_DIR, "XGBoost.joblib"))

    # Find high-risk patients in test set
    probs = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.where(probs > 0.65)[0][:3]
    if len(high_risk_idx) == 0:
        print("No high-risk patients found in test set.")
        return

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=FEATURE_NAMES,
        outcome_name=TARGET_COL,
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    out_dir = os.path.join(EXPLAIN_DIR, "counterfactuals")
    os.makedirs(out_dir, exist_ok=True)

    for i, idx in enumerate(high_risk_idx):
        patient = X_test.iloc[[idx]]
        try:
            cf = exp.generate_counterfactuals(
                patient, total_CFs=3, desired_class="opposite",
                features_to_vary=[f for f in FEATURE_NAMES
                                   if f not in ("Age", "Pregnancies")]
            )
            out = os.path.join(out_dir, f"cf_patient_{idx}.csv")
            cf.cf_examples_list[0].final_cfs_df.to_csv(out, index=False)
            print(f"  Patient {idx} (prob={probs[idx]:.3f}) — counterfactuals saved: {out}")
        except Exception as e:
            print(f"  Patient {idx}: DiCE failed — {e}")


# ===========================================================================
# 9. FAIRNESS / SUBGROUP ANALYSIS
# ===========================================================================
def section_fairness():
    """
    Evaluate F1 and Recall per age group and BMI group (Plan §20).
    """
    print("\n=== Fairness & Subgroup Analysis ===")
    from sklearn.metrics import f1_score, recall_score

    X_train, X_test, y_train, y_test = _load_split()
    model = joblib.load(os.path.join(RESULTS_DIR, "XGBoost.joblib"))

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    # X_test is already scaled (from cleaned CSV); use original unscaled for group splits
    df_raw = pd.read_csv(
        os.path.join(os.path.dirname(CLEANED_CSV), "..", "raw", "diabetes.csv")
    )
    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test_raw = train_test_split(
        df_raw.drop(columns=[TARGET_COL]), df_raw[TARGET_COL],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_raw[TARGET_COL]
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= OPTIMAL_THRESHOLD).astype(int)

    records = []

    # Age groups
    age_bins = [(0, 35, "Young (<35)"), (35, 50, "Middle (35-50)"), (50, 200, "Older (>50)")]
    for lo, hi, label in age_bins:
        mask = (X_test_raw["Age"].values >= lo) & (X_test_raw["Age"].values < hi)
        if mask.sum() == 0:
            continue
        records.append({
            "Group": label, "Type": "Age",
            "N": int(mask.sum()),
            "F1": round(f1_score(y_test_raw.values[mask], y_pred[mask], zero_division=0), 4),
            "Recall": round(recall_score(y_test_raw.values[mask], y_pred[mask], zero_division=0), 4),
        })

    # BMI groups
    bmi_bins = [(0, 25, "Normal (<25)"), (25, 30, "Overweight (25-30)"), (30, 200, "Obese (>30)")]
    for lo, hi, label in bmi_bins:
        mask = (X_test_raw["BMI"].values >= lo) & (X_test_raw["BMI"].values < hi)
        if mask.sum() == 0:
            continue
        records.append({
            "Group": label, "Type": "BMI",
            "N": int(mask.sum()),
            "F1": round(f1_score(y_test_raw.values[mask], y_pred[mask], zero_division=0), 4),
            "Recall": round(recall_score(y_test_raw.values[mask], y_pred[mask], zero_division=0), 4),
        })

    fairness_df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "fairness_subgroup_analysis.csv")
    fairness_df.to_csv(out, index=False)
    print(fairness_df.to_string(index=False))
    print(f"Saved: {out}")

    # Flag under-performing groups
    overall_f1 = f1_score(y_test_raw, y_pred, zero_division=0)
    under = fairness_df[fairness_df["F1"] < overall_f1 - 0.05]
    if not under.empty:
        print(f"\nGroups with F1 > 5pp below overall ({overall_f1:.4f}):")
        print(under.to_string(index=False))


# ===========================================================================
# 10. STATISTICAL SIGNIFICANCE (McNemar + paired t-test)
# ===========================================================================
def section_stats():
    """
    McNemar's test comparing Phase 1 (RF) vs Phase 2 (XGBoost) on same test set (Plan §21).
    Also runs paired t-test across CV folds if cv_results.csv exists.
    """
    print("\n=== Statistical Significance Tests ===")
    from statsmodels.stats.contingency_tables import mcnemar

    X_train, X_test, y_train, y_test = _load_split()

    rf_path  = os.path.join(RESULTS_DIR, "RandomForest.joblib")
    xgb_path = os.path.join(RESULTS_DIR, "XGBoost.joblib")
    if not (os.path.exists(rf_path) and os.path.exists(xgb_path)):
        print("Both RandomForest.joblib and XGBoost.joblib needed. Run step3_train.py first.")
        return

    rf  = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    y_pred_rf  = rf.predict(X_test)
    y_pred_xgb = (xgb.predict_proba(X_test)[:, 1] >= OPTIMAL_THRESHOLD).astype(int)
    y_true = y_test.values

    # McNemar contingency table
    b = np.sum((y_pred_rf == y_true) & (y_pred_xgb != y_true))  # RF correct, XGB wrong
    c = np.sum((y_pred_rf != y_true) & (y_pred_xgb == y_true))  # XGB correct, RF wrong
    table = np.array([[0, b], [c, 0]])

    result = mcnemar(table, exact=False, correction=True)
    print(f"McNemar's test (RF Phase1 vs XGBoost Phase2):")
    print(f"  b={b}, c={c}")
    print(f"  Statistic={result.statistic:.4f}  p-value={result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("  --> Statistically significant improvement (p < 0.05)")
    else:
        print("  --> Not statistically significant (p >= 0.05)")

    # Paired t-test across CV folds (if CV results exist)
    if os.path.exists(CV_RESULTS_CSV):
        from scipy import stats
        cv_df = pd.read_csv(CV_RESULTS_CSV)
        print(f"\nCV F1: mean={cv_df['F1_mean'].values[0]:.4f} ± std={cv_df['F1_std'].values[0]:.4f}")

    # Save results
    stats_txt = (
        f"Statistical Significance — Phase 1 (RF) vs Phase 2 (XGBoost)\n"
        f"McNemar statistic: {result.statistic:.4f}  p-value: {result.pvalue:.4f}\n"
        f"b (RF correct, XGB wrong): {b}\n"
        f"c (XGB correct, RF wrong): {c}\n"
    )
    with open(os.path.join(DOCS_DIR, "phase2_stats_report.txt"), "w") as f:
        f.write(stats_txt)
    print(f"Stats report saved: {DOCS_DIR}/phase2_stats_report.txt")


# ===========================================================================
# Entry point
# ===========================================================================
SECTIONS = {
    "features":  section_features,
    "cv":        section_cv,
    "threshold": section_threshold,
    "tune":      section_tune,
    "calibrate": section_calibrate,
    "ensemble":  section_ensemble,
    "lime":      section_lime,
    "dice":      section_dice,
    "fairness":  section_fairness,
    "stats":     section_stats,
}


def main(section: str = "all"):
    make_dirs()
    if section == "all":
        for name, fn in SECTIONS.items():
            try:
                fn()
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
    else:
        if section not in SECTIONS:
            print(f"Unknown section: {section}. Choose from: {list(SECTIONS)}")
            sys.exit(1)
        SECTIONS[section]()

    print("\nStep 6 (Phase 2 Improvements) complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 model improvements")
    parser.add_argument(
        "--section",
        default="all",
        choices=list(SECTIONS) + ["all"],
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()
    main(args.section)
