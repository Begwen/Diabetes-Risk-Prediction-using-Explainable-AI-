"""
step2_preprocess.py — Clean, impute, scale, and save the preprocessing pipeline.

Converted from: notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb  Step 2
Phase 2 changes: removed Colab paths, uses src/config.py constants,
                 runs standalone as a script.

Run:
    python src/step2_preprocess.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_CSV, CLEANED_CSV, PREPROCESSOR_PATH,
    FEATURE_NAMES, TARGET_COL, ZERO_INVALID_COLS,
    DOCS_DIR, make_dirs,
)


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df


def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible zeros with NaN in suspect columns."""
    df = df.copy()
    df[ZERO_INVALID_COLS] = df[ZERO_INVALID_COLS].replace(0, np.nan)
    print("Replaced zeros with NaN in:", ZERO_INVALID_COLS)
    print("Missing values after replacement:")
    print(df.isna().sum().to_string())
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features    = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [("num", numeric_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features


def main():
    make_dirs()

    df = load_raw()
    df = replace_zero_with_nan(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    preprocessor, num_feats, cat_feats = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # Reconstruct feature names
    cat_names = []
    if cat_feats:
        cat_names = list(
            preprocessor.named_transformers_["cat"]["encoder"]
            .get_feature_names_out(cat_feats)
        )
    col_names = num_feats + cat_names

    X_clean = pd.DataFrame(X_processed, columns=col_names)
    X_clean[TARGET_COL] = y.values

    print(f"\nCleaned dataset shape: {X_clean.shape}")
    print(f"Missing values remaining: {X_clean.isna().sum().sum()}")

    # Save artefacts
    X_clean.to_csv(CLEANED_CSV, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved cleaned dataset : {CLEANED_CSV}")
    print(f"Saved preprocessor    : {PREPROCESSOR_PATH}")

    summary = (
        f"Preprocessing Summary — Diabetes BTP Step 2\n"
        f"--------------------------------------------\n"
        f"Rows              : {X_clean.shape[0]}\n"
        f"Columns           : {X_clean.shape[1]}\n"
        f"Numeric features  : {len(num_feats)}\n"
        f"Categorical feat. : {len(cat_feats)}\n"
        f"Imputation        : median (numeric), mode (categorical)\n"
        f"Scaling           : StandardScaler (z-score)\n"
        f"Missing post-proc : {X_clean.isna().sum().sum()}\n"
        f"Outputs:\n"
        f"  - {CLEANED_CSV}\n"
        f"  - {PREPROCESSOR_PATH}\n"
    )
    out = os.path.join(DOCS_DIR, "step_02_preprocessing_summary.txt")
    with open(out, "w") as f:
        f.write(summary)
    print(f"Saved: {out}")
    print("\nStep 2 complete.")
    return X_clean, preprocessor


if __name__ == "__main__":
    main()
