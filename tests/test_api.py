"""
Unit tests for the FastAPI REST endpoint.
Phase 2 | Author: Priya Sharma | IIT-Guwahati BTP 2025

Run: pytest tests/test_api.py -v
Requires: pip install httpx
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

VALID_PATIENT = {
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47,
}

HIGH_RISK_PATIENT = {
    "Pregnancies": 8,
    "Glucose": 196,
    "BloodPressure": 90,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 39.8,
    "DiabetesPedigreeFunction": 0.451,
    "Age": 58,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "XGBoost"


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------
def test_predict_valid_patient():
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["risk_tier"] in ("Low Risk", "Moderate Risk", "High Risk", "Critical Risk")
    assert len(data["top_shap_features"]) == 3


def test_predict_returns_correct_shap_keys():
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    for feat in r.json()["top_shap_features"]:
        assert "feature" in feat
        assert "shap_value" in feat


def test_predict_high_risk_patient():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert data["probability"] >= 0.5, "High-risk patient should have high probability"


# ---------------------------------------------------------------------------
# /predict — validation rejections
# ---------------------------------------------------------------------------
def test_predict_rejects_zero_glucose():
    bad = {**VALID_PATIENT, "Glucose": 0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_zero_bmi():
    bad = {**VALID_PATIENT, "BMI": 0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_pregnancies():
    bad = {**VALID_PATIENT, "Pregnancies": -1}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_impossible_glucose():
    bad = {**VALID_PATIENT, "Glucose": 9999}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "Age"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422
