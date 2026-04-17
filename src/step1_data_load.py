"""
step1_data_load.py — Load the Pima diabetes dataset and run initial EDA.

Converted from: notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb  Step 1
Phase 2 changes: removed all Colab-specific code (google.colab, !pip, !ls),
                 replaced hardcoded paths with src/config.py constants.

Run:
    python src/step1_data_load.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_RAW_DIR, DATA_PROC_DIR, DOCS_DIR,
    RAW_CSV, FEATURE_NAMES, TARGET_COL, ZERO_INVALID_COLS,
    make_dirs,
)


def load_dataset(path: str = RAW_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please place diabetes.csv in the data/raw/ directory."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {path}")
    print(f"  Shape      : {df.shape}")
    print(f"  Columns    : {df.columns.tolist()}")
    return df


def run_eda(df: pd.DataFrame) -> None:
    print("\n--- Dataset Info ---")
    df.info()
    print("\n--- Describe ---")
    print(df.describe().T.to_string())

    # Check class balance
    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts()
        prevalence = counts.get(1, 0) / counts.sum()
        print(f"\nTarget class counts:\n{counts.to_string()}")
        print(f"Prevalence (positive class): {prevalence:.3f} ({prevalence*100:.1f}%)")

    # Check zero counts in suspect columns
    print("\n--- Zero / missing counts ---")
    for c in ZERO_INVALID_COLS:
        if c in df.columns:
            zeros = (df[c] == 0).sum()
            print(f"  {c}: {zeros} zeros ({zeros/len(df)*100:.1f}%)")


def plot_distributions(df: pd.DataFrame, out_dir: str = DOCS_DIR) -> None:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for i, c in enumerate(num_cols):
        ax = axes[i // 3][i % 3]
        ax.hist(df[c].dropna(), bins=20, edgecolor="black")
        ax.set_title(c)
    for j in range(i + 1, 9):
        axes[j // 3][j % 3].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "eda_distributions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # Correlation matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    out = os.path.join(out_dir, "eda_correlation.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


def save_readme(df: pd.DataFrame) -> None:
    readme = (
        f"Pima dataset loaded.\n"
        f"Columns : {df.columns.tolist()}\n"
        f"Rows    : {df.shape[0]}\n"
        f"Source  : data/raw/diabetes.csv\n"
    )
    out = os.path.join(DOCS_DIR, "step_01_data_readme.txt")
    with open(out, "w") as f:
        f.write(readme)
    print(f"Saved: {out}")


def main():
    make_dirs()
    df = load_dataset()
    run_eda(df)
    plot_distributions(df)
    save_readme(df)
    print("\nStep 1 complete.")
    return df


if __name__ == "__main__":
    main()
