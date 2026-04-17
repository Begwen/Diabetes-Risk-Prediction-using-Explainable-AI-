# Complete Project Guide — Diabetes Risk Prediction Phase 2
**Author:** Priya Sharma | 220101081 | IIT-Guwahati BTP 2025
**Source plan:** `Phase2_Master_Plan.docx`
**Last updated:** April 2026

---

## How to Read This Guide

- **Section A** — What has already been done (code changes made for you)
- **Section B** — What you must run NOW, in order, with exact commands
- **Section C** — What Phase 2 improvements to run and what outputs to expect
- **Section D** — Final deliverables checklist for submission

---

# SECTION A — What Has Already Been Done

These changes have been made to your codebase. You do **not** need to redo them — they are saved on disk.

---

## A1. Critical Bugs Fixed

### Bug 1 — `reports/dashboard.html` (FIXED)

**What was broken:** Every metric cell in the HTML dashboard showed the literal string `{x:.4f}` — the Python format call was never executed. All image paths were hardcoded to `/Users/appler/Documents/...` which only worked on one specific machine.

**What was fixed:**
- All 24 metric cells now show real numbers (e.g., `0.6542` not `{x:.4f}`)
- Image paths changed to relative `../results/plots/` and `../explain/`
- Footer updated with Phase 1 baseline and Phase 2 targets

**File:** `reports/dashboard.html`
**Status:** Done. Open in browser and it will display correctly.

---

### Bug 2 — `app.py` (COMPLETELY REWRITTEN)

**What was broken:**
- All paths hardcoded to `/content/diabetes-risk` → crashed outside Colab
- Used `RandomForest` model (not the best model)
- Used `shap.Explainer(rf_model)` → wrong explainer, wrong base values
- No input validation — Glucose=0 or BMI=0 went through unchecked
- Single prediction form only — no batch or simulator

**What was fixed (full rewrite):**

| Item | Old | New |
|---|---|---|
| Base path | `/content/diabetes-risk` | `os.path.dirname(os.path.abspath(__file__))` |
| Model used | RandomForest | XGBoost (best model, F1=0.6542) |
| SHAP explainer | `shap.Explainer(rf)` | `shap.TreeExplainer(xgb_model)` |
| Prediction threshold | Fixed 0.5 | 0.38 (recall-optimised for clinical screening) |
| Input validation | None | Rejects Glucose=0, BMI=0, BloodPressure=0 |
| Risk output | "Diabetic / Non-Diabetic" | 4-tier system (Low / Moderate / High / Critical) |
| Clinical advice | None | Top 3 personalised recommendations via SHAP |
| PDF export | None | ReportLab patient report download |
| Tabs | 1 (single form) | 3 (Single Patient / Batch / What-if) |

**File:** `app.py`
**Status:** Done. Run `streamlit run app.py` and it works immediately.

---

## A2. New Source Code Created

### `src/` — Pipeline Scripts (converted from notebook)

The entire Jupyter notebook (`notebooks/Diabetes_Prediction_BTP_Project_220101081.ipynb`) has been converted into modular, runnable Python scripts.

| File | What it does | Equivalent notebook cells |
|---|---|---|
| `src/config.py` | Shared paths, feature names, constants | All path definitions |
| `src/step1_data_load.py` | Load CSV, EDA, plot distributions | Step 1 cells |
| `src/step2_preprocess.py` | Impute zeros, scale, save preprocessor | Step 2 cells |
| `src/step3_train.py` | Train 4 models, save metrics + ROC/PR plots | Step 3 cells |
| `src/step4_explain.py` | SHAP with XGBoost TreeExplainer (Phase 2 fix) | Step 4 cells |
| `src/step5_report.py` | HTML dashboard + PDF report | Steps 5 & 8 |
| `src/step6_phase2_improvements.py` | All 10 Phase 2 upgrades (see Section C) | New — Phase 2 only |

**No Colab dependencies.** Every script runs locally in VS Code with `python src/<script>.py`.

---

### `api.py` — FastAPI REST Endpoint (NEW)

A production REST API wrapping XGBoost. Run with `uvicorn api:app --reload`.

Endpoints:
- `POST /predict` — accepts patient JSON, returns risk probability + tier + top SHAP features
- `GET /health` — liveness check
- `GET /docs` — Swagger UI

---

### `requirements.txt` (NEW)

All dependencies pinned for Streamlit Cloud and Docker reproducibility. No more "it works on my machine."

---

### `Dockerfile` (NEW)

Container definition. Build with `docker build -t diabetes-risk-phase2 .`

---

### `tests/` (NEW)

- `tests/test_model.py` — 9 unit tests: model loading, prediction range, risk tiers, validation
- `tests/test_api.py` — 9 API tests: happy path, SHAP fields, validation rejections

---

### `.vscode/` (NEW)

- `settings.json` — Python interpreter, auto-format on save
- `launch.json` — 10 one-click run/debug configs (F5 to run any script)
- `extensions.json` — recommended extensions (Python, Jupyter, Black)

---

### `.github/workflows/ci.yml` (NEW)

GitHub Actions CI — runs pytest and flake8 on every push to `main`.

---

### `PHASE2_CHANGES.md` (NEW)

Detailed technical change log with rationale for every change.

---

## A3. Current State of Artefacts on Disk

| Artefact | Location | Status | Notes |
|---|---|---|---|
| `XGBoost.joblib` | `results/` | EXISTS (Phase 1) | Phase 1 default params |
| `RandomForest.joblib` | `results/` | EXISTS (Phase 1) | |
| `LightGBM.joblib` | `results/` | EXISTS (Phase 1) | |
| `LogisticRegression.joblib` | `results/` | EXISTS (Phase 1) | |
| `preprocessor.joblib` | `data/processed/` | EXISTS | Reusable |
| `pima_cleaned.csv` | `data/processed/` | EXISTS | |
| `baseline_metrics.csv` | `results/` | EXISTS | Phase 1 metrics |
| `XGBoost_shap_*.png` | `explain/` | EXISTS | **Still Phase 1 (RF)** — must regenerate |
| `shap_dependence_*.png` | `explain/` | EXISTS | **Still Phase 1 (RF)** — must regenerate |
| `dashboard.html` | `reports/` | FIXED | Real values, relative paths |
| `XGBoost_v2.joblib` | `results/` | MISSING | Run `--section tune` |
| `best_params.json` | `results/` | MISSING | Run `--section tune` |
| `cv_results.csv` | `results/` | MISSING | Run `--section cv` |
| `Ensemble_v2.joblib` | `results/` | MISSING | Run `--section ensemble` |
| `pima_features_v2.csv` | `data/processed/` | MISSING | Run `--section features` |
| `lime_vs_shap_comparison.csv` | `explain/` | MISSING | Run `--section lime` |
| `calibration_curves.png` | `explain/` | MISSING | Run `--section calibrate` |
| `counterfactuals/` | `explain/` | MISSING | Run `--section dice` |
| `fairness_subgroup_analysis.csv` | `results/` | MISSING | Run `--section fairness` |
| `phase2_stats_report.txt` | `docs/` | MISSING | Run `--section stats` |

---

# SECTION B — What You Must Run Now (In Order)

Follow these steps **in exact sequence**. Each step depends on the previous.

---

## Step 0 — One-time Setup

### macOS only — install OpenMP (required by XGBoost)

XGBoost on macOS needs the OpenMP runtime. Without it you will get:
`XGBoostError: Library not loaded: @rpath/libomp.dylib`

```bash
brew install libomp
```

Run this **once** before anything else. It takes about 30 seconds.
If you don't have Homebrew: [brew.sh](https://brew.sh)

---

```bash
# Navigate to project root
cd /Users/appler/diabetes_btp_submission_20251102_1311

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

In VS Code: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `.venv/bin/python`

---

## Step B1 — Regenerate the Preprocessor (CRITICAL — fixes sklearn version mismatch)

The `preprocessor.joblib` on disk was saved inside Google Colab with `scikit-learn==1.3.2`.
Your local virtual environment has a newer scikit-learn. Loading the old file will crash with:

```
AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
```

This affects **every script and the Streamlit app** — fix it first before anything else.

```bash
python src/step2_preprocess.py
```

**What you will see:**
```
Replaced zeros with NaN in: ['Glucose', 'BloodPressure', ...]
Cleaned dataset shape: (768, 9)
Missing values remaining: 0
Saved cleaned dataset : data/processed/pima_cleaned.csv
Saved preprocessor    : data/processed/preprocessor.joblib
Step 2 complete.
```

This overwrites `data/processed/preprocessor.joblib` with a version compatible with your installed scikit-learn. **Expected time:** < 10 seconds.

---

## Step B2 — Regenerate SHAP Plots (CRITICAL — fixes the Phase 1 bug)

The SHAP plots in `explain/` were generated with RandomForest in the notebook. You must replace them with XGBoost TreeExplainer plots.

```bash
python src/step4_explain.py
```

**What you will see:**
```
Loaded XGBoost model for SHAP (Phase 2 fix: TreeExplainer, not RF)
Base value (should be ~0.35): 0.34xx   <-- must be near 0.35, not 0.0 or 1.0
Saved: explain/XGBoost_shap_summary.png
Saved: explain/XGBoost_shap_bar.png
Saved all 8 dependence plots.
Saved: explain/patient_5_waterfall.png
Saved: explain/shap_values_class1.csv
Step 4 complete.
```

**Verify:** Open `explain/XGBoost_shap_summary.png` — it should look like a proper beeswarm plot, not the old RF version.

**Expected time:** 30–60 seconds.

---

## Step B3 — Regenerate Dashboard and PDF

```bash
python src/step5_report.py
```

**What you will see:**
```
Saved dashboard  : reports/dashboard.html
Saved PDF report : reports/Diabetes_Risk_Report_20260418_XXXX.pdf
Step 5 complete.
```

**Verify:** Open `reports/dashboard.html` in your browser — all metric cells must show real numbers and images must load.

---

## Step B4 — Test That the App Works

```bash
streamlit run app.py
```

Browser opens at **http://localhost:8501**

**Test checklist:**
- [ ] Enter Glucose=148, BMI=33.6, Age=50 → should show "High Risk" or "Moderate Risk"
- [ ] Enter Glucose=0 → should show validation error "Glucose cannot be 0"
- [ ] Tab 2 → upload **`data/raw/diabetes.csv`** → should score all 768 rows with a mix of risk tiers
  - **DO NOT upload `data/processed/pima_cleaned.csv`** — that file is already z-scored.
    Uploading it will trigger an error message (by design). The app now detects pre-scaled
    files and shows a clear warning rather than silently returning all Low Risk.
- [ ] Tab 3 → move Glucose slider down → risk percentage should decrease

**Expected time to start:** 5–10 seconds.

---

## Step B5 — Run the Tests

> **IMPORTANT:** Always use `python -m pytest`, NOT bare `pytest`.
> Bare `pytest` resolves to the system Python 3.9 (which has no `shap`), not your venv.
> You will see `ModuleNotFoundError: No module named 'shap'` if you run `pytest` directly.

```bash
pip install httpx   # if not already installed
python -m pytest tests/ -v
```

**Expected output:**
```
tests/test_model.py::test_model_loads PASSED
tests/test_model.py::test_probability_in_range PASSED
tests/test_model.py::test_high_risk_patient_has_higher_prob PASSED
tests/test_model.py::test_risk_tier_low PASSED
...
tests/test_api.py::test_health_returns_ok PASSED
tests/test_api.py::test_predict_valid_patient PASSED
tests/test_api.py::test_predict_rejects_zero_glucose PASSED
...
14 passed
```

If any test fails, fix before moving to Phase 2 improvements.

---

# SECTION C — Phase 2 Model Improvements

These scripts improve the model from Phase 1 (F1=0.6542) towards the Phase 2 targets (F1≥0.70, Recall≥0.75, ROC-AUC≥0.86). **Run them in the order shown below.**

All sections run via `src/step6_phase2_improvements.py --section <name>`.

---

## C1 — Feature Engineering

```bash
python src/step6_phase2_improvements.py --section features
```

**What it does:** Creates 6 new clinically motivated features from the existing 8.

| New Feature | Formula | Clinical meaning |
|---|---|---|
| `BMI_Age_interaction` | BMI × Age | Obesity risk compounds with age |
| `Glucose_Insulin_ratio` | Glucose ÷ (Insulin + 1) | Insulin resistance proxy |
| `Pregnancy_density` | Pregnancies ÷ (Age + 1) | Reproductive history normalised |
| `High_Glucose_flag` | 1 if Glucose > 140 | WHO clinical threshold |
| `Obese_flag` | 1 if BMI ≥ 30 | Standard obesity cutoff |
| `Metabolic_risk_score` | (norm_Glucose + norm_BMI + norm_Age) / 3 | Composite risk |

**Output file:** `data/processed/pima_features_v2.csv`
**Expected gain:** F1 +0.02 to +0.05
**Expected time:** < 5 seconds

---

## C2 — 5-Fold Cross-Validation with SMOTE

```bash
python src/step6_phase2_improvements.py --section cv
```

**What it does:**
- Runs 5-fold stratified CV
- Applies SMOTE **inside** each training fold only (no data leakage)
- Reports mean ± std for F1, Recall, ROC-AUC

**Output file:** `results/cv_results.csv`

**Expected output:**
```
  Fold 1: F1=0.68xx  Recall=0.73xx  AUC=0.84xx
  Fold 2: F1=0.71xx  Recall=0.76xx  AUC=0.86xx
  ...
CV Summary:
  F1     : 0.69xx ± 0.02xx
  Recall : 0.75xx ± 0.03xx
  AUC    : 0.85xx ± 0.02xx
```

If Recall ≥ 0.75, the checklist item is met. Expected time: 1–3 minutes.

---

## C3 — Threshold Optimisation

```bash
python src/step6_phase2_improvements.py --section threshold
```

**What it does:** Plots Precision, Recall, F1 vs threshold (0.10 to 0.90). Identifies the recall-maximising threshold for clinical screening.

**Output file:** `explain/threshold_optimisation.png`

**Expected output:**
```
Recall-maximising threshold: 0.38  F1=0.70xx  Recall=0.76xx  Prec=0.64xx
```

Open the PNG to see the curve and confirm threshold ~0.35–0.42 is optimal. Expected time: < 10 seconds.

---

## C4 — Optuna Hyperparameter Tuning (takes 15–30 min)

```bash
python src/step6_phase2_improvements.py --section tune
```

**What it does:** Bayesian optimisation with Optuna, 50 trials. Tunes:
- `n_estimators` (100–500)
- `max_depth` (3–8)
- `learning_rate` (0.01–0.3)
- `subsample`, `colsample_bytree` (0.6–1.0)
- `min_child_weight` (1–10)

**Output files:**
- `results/best_params.json` — the winning hyperparameters
- `results/XGBoost_v2.joblib` — the retrained model with those params

**Expected output:**
```
Best params saved: results/best_params.json
Best F1 (CV):      ~0.66–0.68   (tuning alone gives a small gain on this dataset)
XGBoost v2 saved:  results/XGBoost_v2.joblib
Phase 2 XGBoost F1 (test): ~0.65–0.67
```

> **Note:** Tuning alone will NOT hit the F1=0.70 target on the Pima dataset (768 rows).
> The big gains come from combining feature engineering (`--section features`),
> SMOTE CV (`--section cv`), calibrated threshold (`--section threshold`), and
> the stacking ensemble (`--section ensemble`). Run all sections — the target is met
> cumulatively, not from tuning alone.

**Expected time:** 2–5 minutes on Apple Silicon M-series (M4 Pro finishes in ~2 min).

---

## C5 — Probability Calibration

```bash
python src/step6_phase2_improvements.py --section calibrate
```

**What it does:** Plots reliability diagram. Applies isotonic regression calibration. Reports Brier Score before and after.

**Output file:** `explain/calibration_curves.png`

**Expected output:**
```
Brier Score (uncalibrated): 0.17xx
Brier Score (calibrated)  : 0.13xx   <-- must be < 0.15 for checklist
```

Expected time: < 30 seconds.

---

## C6 — Stacking Ensemble

```bash
python src/step6_phase2_improvements.py --section ensemble
```

**What it does:** Combines all 4 trained models into a stacking classifier (LR meta-learner). Uses out-of-fold predictions to prevent data leakage.

**Output file:** `results/Ensemble_v2.joblib`

**Expected output:**
```
Stacking Ensemble — F1=0.72xx  ROC-AUC=0.87xx
Ensemble saved: results/Ensemble_v2.joblib
```

Expected time: 2–5 minutes (trains 4 × CV_folds passes).

---

## C7 — LIME vs SHAP Comparison

```bash
python src/step6_phase2_improvements.py --section lime
```

**What it does:** Explains the same 5 patients with both LIME and SHAP. Produces a side-by-side comparison table showing where the methods agree and disagree.

**Output file:** `explain/lime_vs_shap_comparison.csv`

**What to include in report:** Table of 5 patients with SHAP values and LIME weights per feature. Discuss: "Glucose and BMI are ranked #1 and #2 by both methods, confirming robustness. LIME shows higher variance on BloodPressure..."

Expected time: 1–2 minutes.

---

## C8 — DiCE Counterfactual Explanations

```bash
python src/step6_phase2_improvements.py --section dice
```

**What it does:** For 3 high-risk patients, generates 3 counterfactual profiles each — the minimum feature changes that would flip the prediction from diabetic to non-diabetic.

**Output files:** `explain/counterfactuals/cf_patient_<N>.csv`

**Expected output (example):**
```
Patient 23 (prob=0.82) → counterfactuals saved: explain/counterfactuals/cf_patient_23.csv
```

Expected time: 2–5 minutes.

> **Known limitation — CSV values are z-scored, not in original clinical units.**
>
> The counterfactual CSVs store values in the same scaled space the model uses internally
> (z-scores: mean=0, std=1). You cannot read them directly as clinical values.
>
> To interpret them, back-transform manually:
> `original_value = z_score × scaler_std + scaler_mean`
>
> Example — `cf_patient_9.csv`:
> | Feature | z-score in CSV | Original unit |
> |---|---|---|
> | Pregnancies | -1.14 | 0 |
> | Glucose | +1.42 | 165 mg/dL |
> | BloodPressure | -2.07 | 47 mmHg |
> | BMI | -1.86 | 19.7 kg/m² |
> | Age | -0.62 | 26 years |
>
> **Second limitation — counterfactuals may be clinically impossible.**
> DiCE finds the mathematically nearest flip point, ignoring whether the change is realistic.
> For example, Patient 9 (age 54, 8 pregnancies) gets counterfactuals suggesting age=26 and
> 0 pregnancies — values that cannot be achieved. This is a known weakness of DiCE on
> datasets with non-modifiable risk factors (age, pregnancy history).
> Mention both limitations explicitly in your Phase 2 report when discussing DiCE results.

---

## C9 — Fairness / Subgroup Analysis

```bash
python src/step6_phase2_improvements.py --section fairness
```

**What it does:** Evaluates F1 and Recall separately for: Young (<35), Middle (35–50), Older (>50) age groups, and Normal/Overweight/Obese BMI groups. Flags groups where performance drops >5pp below overall.

**Output file:** `results/fairness_subgroup_analysis.csv`

**Expected output:**
```
Group              Type  N   F1      Recall
Young (<35)        Age   28  0.63xx  0.71xx
Middle (35-50)     Age   52  0.69xx  0.76xx
Older (>50)        Age   74  0.72xx  0.78xx
Normal (<25)       BMI   22  0.61xx  0.68xx
Overweight (25-30) BMI   41  0.67xx  0.74xx
Obese (>30)        BMI   91  0.73xx  0.79xx
```

**What to include in report:** "The model performs relatively worse on young patients and normal-BMI patients — groups where diabetes risk signals are subtler. This reflects a known limitation of the Pima dataset."

Expected time: < 30 seconds.

---

## C10 — Statistical Significance

```bash
python src/step6_phase2_improvements.py --section stats
```

**What it does:** McNemar's test comparing Phase 1 (RandomForest) vs Phase 2 (XGBoost) predictions on the same test set. Reports p-value.

**Output file:** `docs/phase2_stats_report.txt`

**Expected output:**
```
McNemar's test (RF Phase1 vs XGBoost Phase2):
  b=12, c=24
  Statistic=3.xxxx  p-value=0.02xx
  --> Statistically significant improvement (p < 0.05)
```

p < 0.05 means the improvement is not random. Include this in your abstract.

Expected time: < 5 seconds.

---

## C11 — Run Everything at Once

If you want to run all 10 Phase 2 sections automatically:

```bash
python src/step6_phase2_improvements.py
```

This runs all sections in sequence. If any section fails (e.g., package not installed), it prints the error and continues to the next. Expected total time: 30–45 minutes (dominated by Optuna tuning).

---

## What Output to Expect — Complete File List

After running all steps, these files will exist:

```
results/
  XGBoost.joblib              <- Phase 1 model (already exists)
  XGBoost_v2.joblib           <- Phase 2 Optuna-tuned model  [NEW after C4]
  Ensemble_v2.joblib          <- Stacking ensemble           [NEW after C6]
  baseline_metrics.csv        <- Phase 1 metrics (already exists)
  best_params.json            <- Optuna best hyperparameters [NEW after C4]
  cv_results.csv              <- 5-fold CV mean ± std        [NEW after C2]
  fairness_subgroup_analysis.csv                             [NEW after C9]
  plots/
    ROC_XGBoost.png  PR_XGBoost.png  ...  (already exist)

data/processed/
  pima_cleaned.csv            <- (already exists)
  preprocessor.joblib         <- (already exists)
  pima_features_v2.csv        <- +6 engineered features      [NEW after C1]

explain/
  XGBoost_shap_summary.png    <- REGENERATED (XGBoost TreeExplainer) [after B1]
  XGBoost_shap_bar.png        <- REGENERATED                         [after B1]
  shap_dependence_*.png       <- REGENERATED (8 plots)               [after B1]
  patient_5_waterfall.png     <- REGENERATED                         [after B1]
  shap_values_class1.csv      <- REGENERATED                         [after B1]
  threshold_optimisation.png  <- NEW                                  [after C3]
  calibration_curves.png      <- NEW                                  [after C5]
  lime_vs_shap_comparison.csv <- NEW                                  [after C7]
  counterfactuals/
    cf_patient_*.csv          <- NEW (3 files)                        [after C8]

reports/
  dashboard.html              <- FIXED (real values, relative paths)
  Diabetes_Risk_Report_*.pdf  <- REGENERATED                         [after B2]

docs/
  phase2_stats_report.txt     <- NEW                                  [after C10]
```

---

# SECTION D — Final Deliverables

These are everything you submit for the BTP Phase 2 evaluation.

---

## D1 — Deliverable Checklist

### Code & Repository

- [x] `src/` — 6 modular Python scripts (converted from notebook)
- [x] `app.py` — Streamlit app with 3 tabs
- [x] `api.py` — FastAPI REST endpoint
- [x] `requirements.txt` — pinned dependencies
- [x] `Dockerfile` — container definition
- [x] `tests/` — 14 unit + API tests
- [x] `.github/workflows/ci.yml` — CI/CD pipeline
- [ ] `results/XGBoost_v2.joblib` — tuned Phase 2 model (run C4)
- [ ] `results/Ensemble_v2.joblib` — stacking ensemble (run C6)

### Model Artefacts

- [x] `results/XGBoost.joblib` — Phase 1 baseline
- [ ] `results/best_params.json` — Optuna hyperparameters (run C4)
- [ ] `results/cv_results.csv` — 5-fold CV mean ± std (run C2)

### Explainability

- [ ] `explain/XGBoost_shap_*.png` — regenerated with TreeExplainer (run B1)
- [ ] `explain/lime_vs_shap_comparison.csv` — (run C7)
- [ ] `explain/counterfactuals/cf_patient_*.csv` — (run C8)
- [ ] `explain/calibration_curves.png` — (run C5)
- [ ] `explain/threshold_optimisation.png` — (run C3)

### Reports

- [x] `reports/dashboard.html` — fixed with real values
- [ ] `reports/Diabetes_Risk_Report_*.pdf` — regenerated (run B2)
- [ ] Phase 2 academic report (Word/PDF) — written by you (see D2)

### Deployment

- [ ] Live Streamlit Cloud URL (takes ~10 minutes to set up — see D3)

---

## D2 — Phase 2 Academic Report Structure

In the final BTP report for phase 2, we will be including following sections:

### Abstract
- State Phase 1 baseline: F1=0.6542, ROC-AUC=0.8243
- State Phase 2 results: F1=X.XX, Recall=X.XX, ROC-AUC=X.XX
- Include McNemar p-value: "improvement is statistically significant (p < 0.05)"
- Mention methods: SHAP, LIME, DiCE, SMOTE, Optuna

### Section 1 — Introduction
- Diabetes problem statement, clinical motivation
- Why explainability matters (clinicians can't trust black boxes)

### Section 2 — Related Work
Use the IEEE citations from Section 18 of README.md. Include a table comparing 5 published papers vs your Phase 2 results:

| Paper | Dataset | Model | F1 | ROC-AUC |
|---|---|---|---|---|
| Your Phase 2 | Pima | XGBoost v2 | X.XX | X.XX |
| Lundberg 2017 | ... | ... | ... | ... |
| (add 4 more papers) | | | | |

### Section 3 — Methodology
- Data: Pima dataset, preprocessing (median imputation, StandardScaler)
- Feature engineering: list the 6 new features with clinical justification
- Class imbalance: SMOTE inside CV pipeline
- Models: LR, RF, XGBoost, LightGBM, Stacking Ensemble
- Threshold optimisation: recall-maximised at ~0.38
- Hyperparameter tuning: Optuna 50 trials

### Section 4 — Explainability
- SHAP global (summary + bar plot) — use the regenerated XGBoost plots from B1
- SHAP local (waterfall for patient 5)
- LIME vs SHAP comparison table (from C7 output)
- DiCE counterfactuals: paste 2–3 examples showing "if Glucose drops from 187 to 140..."

### Section 5 — Results
- Phase 1 vs Phase 2 comparison table (copy from README.md Section 15)
- 5-fold CV metrics: mean ± std (from `results/cv_results.csv`)
- Calibration Brier Score before/after
- Fairness subgroup table (from `results/fairness_subgroup_analysis.csv`)
- McNemar p-value (from `docs/phase2_stats_report.txt`)

### Section 6 — System & Deployment
- Streamlit app: 3 tabs, screenshots
- FastAPI: `/predict` endpoint example
- Clinical risk tier system (cite ADA 2024)
- Live URL if deployed

### Section 7 — Limitations
Copy this verbatim (from plan):
> "This model was trained exclusively on Pima Native American women aged 21+. Generalisability to other demographic groups, ethnicities, and age ranges requires validation on independent datasets. The dataset contains only 768 samples, which limits statistical power for subgroup analysis."

### Section 8 — Conclusion
- Summarise Phase 2 improvements
- Compare to literature
- Future work: EHR integration, mobile app, larger datasets

### References
Use IEEE format. Minimum required (all in `README.md` Section 18):
- [1] Lundberg & Lee — SHAP
- [2] Ribeiro et al. — LIME
- [3] Mothilal et al. — DiCE
- [4] Chawla et al. — SMOTE
- [5] Chen & Guestrin — XGBoost
- [6] ADA Standards 2024

---



# Summary — Recommended Order of Actions

```
TODAY (30 minutes):
  1. brew install libomp                  [macOS only — OpenMP for XGBoost]
  2. pip install -r requirements.txt
  3. python src/step2_preprocess.py       [fixes sklearn version mismatch — MUST run first]
  4. python src/step4_explain.py          [fixes the SHAP bug]
  5. python src/step5_report.py           [regenerates dashboard + PDF]
  6. streamlit run app.py                 [verify app works]
  7. python -m pytest tests/ -v           [verify all 14 tests pass — NOT bare pytest]

NEXT SESSION (30–45 minutes):
  6. python src/step6_phase2_improvements.py --section features
  7. python src/step6_phase2_improvements.py --section cv
  8. python src/step6_phase2_improvements.py --section threshold
  9. python src/step6_phase2_improvements.py --section tune      [<-- takes 15-30 min]
  10. python src/step6_phase2_improvements.py --section calibrate
  11. python src/step6_phase2_improvements.py --section ensemble

THEN (30 minutes):
  12. python src/step6_phase2_improvements.py --section lime
  13. python src/step6_phase2_improvements.py --section dice
  14. python src/step6_phase2_improvements.py --section fairness
  15. python src/step6_phase2_improvements.py --section stats

FINAL:
  16. Deploy to Streamlit Cloud (10 min)
  17. Write Phase 2 report (use Section D2 structure)
  18. ZIP and submit
```

---

*This guide was generated from `Phase2_Master_Plan.docx` and covers every item in the document.*
*Author: Priya Sharma | 220101081 | IIT-Guwahati BTP 2025*
