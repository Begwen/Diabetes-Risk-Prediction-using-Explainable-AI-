"""
config.py — Shared paths, constants, and settings for the entire pipeline.

All other src/ scripts import from here.  Paths are always relative to the
project root so the pipeline runs from any machine or environment.
"""

import os

# ---------------------------------------------------------------------------
# Project root — always the parent of this file's directory
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_RAW_DIR      = os.path.join(ROOT, "data", "raw")
DATA_PROC_DIR     = os.path.join(ROOT, "data", "processed")
RESULTS_DIR       = os.path.join(ROOT, "results")
PLOTS_DIR         = os.path.join(ROOT, "results", "plots")
PREDICTIONS_DIR   = os.path.join(ROOT, "results", "predictions")
EXPLAIN_DIR       = os.path.join(ROOT, "explain")
DOCS_DIR          = os.path.join(ROOT, "docs")
REPORTS_DIR       = os.path.join(ROOT, "reports")

# ---------------------------------------------------------------------------
# Key file paths
# ---------------------------------------------------------------------------
RAW_CSV           = os.path.join(DATA_RAW_DIR,  "diabetes.csv")
CLEANED_CSV       = os.path.join(DATA_PROC_DIR, "pima_cleaned.csv")
PREPROCESSOR_PATH = os.path.join(DATA_PROC_DIR, "preprocessor.joblib")
METRICS_CSV       = os.path.join(RESULTS_DIR,   "baseline_metrics.csv")
BASELINE_TXT      = os.path.join(RESULTS_DIR,   "baseline_summary.txt")

# Phase 2 artefacts
FEATURES_V2_CSV   = os.path.join(DATA_PROC_DIR, "pima_features_v2.csv")
BEST_PARAMS_JSON  = os.path.join(RESULTS_DIR,   "best_params.json")
CV_RESULTS_CSV    = os.path.join(RESULTS_DIR,   "cv_results.csv")
ENSEMBLE_PATH     = os.path.join(RESULTS_DIR,   "Ensemble_v2.joblib")
XGB_V2_PATH       = os.path.join(RESULTS_DIR,   "XGBoost_v2.joblib")

# ---------------------------------------------------------------------------
# Feature names (8 original Pima features)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COL = "Outcome"

# Columns where zero means "missing measurement" in Pima dataset
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5
OPTIMAL_THRESHOLD = 0.38     # recall-optimised for clinical screening

# ---------------------------------------------------------------------------
# Phase 2 feature names (8 original + 6 engineered)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES = [
    "BMI_Age_interaction",
    "Glucose_Insulin_ratio",
    "Pregnancy_density",
    "High_Glucose_flag",
    "Obese_flag",
    "Metabolic_risk_score",
]
ALL_FEATURES_V2 = FEATURE_NAMES + ENGINEERED_FEATURES


# ---------------------------------------------------------------------------
# Utility: ensure all output directories exist
# ---------------------------------------------------------------------------
def make_dirs():
    """Create all output directories if they don't exist yet."""
    for d in [DATA_RAW_DIR, DATA_PROC_DIR, RESULTS_DIR, PLOTS_DIR,
              PREDICTIONS_DIR, EXPLAIN_DIR, DOCS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
