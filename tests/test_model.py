"""
Unit tests for the diabetes risk prediction model and preprocessing pipeline.
Phase 2 | Author: Priya Sharma | IIT-Guwahati BTP 2025

Run: pytest tests/test_model.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "results", "XGBoost.joblib")
PREPROCESSOR_PATH = os.path.join(BASE, "data", "processed", "preprocessor.joblib")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

SAMPLE_LOW_RISK = {
    "Pregnancies": 0, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
    "Insulin": 0, "BMI": 22.0, "DiabetesPedigreeFunction": 0.35, "Age": 25,
}

SAMPLE_HIGH_RISK = {
    "Pregnancies": 8, "Glucose": 196, "BloodPressure": 90, "SkinThickness": 0,
    "Insulin": 0, "BMI": 39.8, "DiabetesPedigreeFunction": 0.451, "Age": 58,
}


@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def preprocessor():
    import joblib
    return joblib.load(PREPROCESSOR_PATH)


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------
def test_model_loads(model):
    assert model is not None


def test_preprocessor_loads(preprocessor):
    assert preprocessor is not None


# ---------------------------------------------------------------------------
# Prediction output shape and range
# ---------------------------------------------------------------------------
def test_predict_proba_shape(model, preprocessor):
    df = pd.DataFrame([SAMPLE_LOW_RISK])[FEATURE_NAMES]
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2), "predict_proba should return (n_samples, 2)"


def test_probability_in_range(model, preprocessor):
    df = pd.DataFrame([SAMPLE_LOW_RISK])[FEATURE_NAMES]
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1]"


def test_high_risk_patient_has_higher_prob(model, preprocessor):
    df_low = pd.DataFrame([SAMPLE_LOW_RISK])[FEATURE_NAMES]
    df_high = pd.DataFrame([SAMPLE_HIGH_RISK])[FEATURE_NAMES]
    p_low = model.predict_proba(preprocessor.transform(df_low))[0, 1]
    p_high = model.predict_proba(preprocessor.transform(df_high))[0, 1]
    assert p_high > p_low, (
        f"High-risk patient ({p_high:.3f}) should have higher probability "
        f"than low-risk patient ({p_low:.3f})"
    )


# ---------------------------------------------------------------------------
# Risk tier logic
# ---------------------------------------------------------------------------
def test_risk_tier_low():
    from app import risk_tier
    label, _, _ = risk_tier(0.10)
    assert label == "Low Risk"


def test_risk_tier_moderate():
    from app import risk_tier
    label, _, _ = risk_tier(0.35)
    assert label == "Moderate Risk"


def test_risk_tier_high():
    from app import risk_tier
    label, _, _ = risk_tier(0.60)
    assert label == "High Risk"


def test_risk_tier_critical():
    from app import risk_tier
    label, _, _ = risk_tier(0.85)
    assert label == "Critical Risk"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_validation_rejects_zero_glucose():
    from app import validate_inputs
    bad = {**SAMPLE_LOW_RISK, "Glucose": 0}
    errors = validate_inputs(bad)
    assert any("Glucose" in e for e in errors)


def test_validation_rejects_zero_bmi():
    from app import validate_inputs
    bad = {**SAMPLE_LOW_RISK, "BMI": 0.0}
    errors = validate_inputs(bad)
    assert any("BMI" in e for e in errors)


def test_validation_passes_valid_patient():
    from app import validate_inputs
    errors = validate_inputs(SAMPLE_LOW_RISK)
    assert errors == []


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def test_batch_prediction(model, preprocessor):
    batch = pd.DataFrame([SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK])[FEATURE_NAMES]
    X = preprocessor.transform(batch)
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
