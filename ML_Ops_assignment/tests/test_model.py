"""
test_model.py

Smoke tests for the persisted heart_model.pkl artifact.
"""

import os

import joblib
import pandas as pd
import pytest

from src.data.preprocess import FEATURE_COLUMNS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_model.pkl")


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model artifact missing at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_pickle_loads_as_pipeline(model):
    """Loaded artifact must be a sklearn Pipeline with preprocessor + classifier steps."""
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)
    step_names = [name for name, _ in model.steps]
    assert "preprocess" in step_names
    assert "classifier" in step_names


def test_predict_returns_binary_label(model, sample_high_risk):
    df = pd.DataFrame([sample_high_risk], columns=FEATURE_COLUMNS)
    pred = model.predict(df)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)


def test_predict_proba_sums_to_one(model, sample_high_risk):
    df = pd.DataFrame([sample_high_risk], columns=FEATURE_COLUMNS)
    proba = model.predict_proba(df)
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-6


def test_high_risk_patient_flagged_positive(model, sample_high_risk):
    """Sanity check: an obviously high-risk patient gets prediction=1 with high prob."""
    df = pd.DataFrame([sample_high_risk], columns=FEATURE_COLUMNS)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    assert pred == 1, f"Expected disease prediction, got {pred}"
    assert proba > 0.8, f"Expected disease probability > 0.8, got {proba:.3f}"


def test_low_risk_patient_flagged_negative(model, sample_low_risk):
    """Sanity check: an obviously low-risk patient gets prediction=0 with low prob."""
    df = pd.DataFrame([sample_low_risk], columns=FEATURE_COLUMNS)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    assert pred == 0, f"Expected no_disease prediction, got {pred}"
    assert proba < 0.2, f"Expected disease probability < 0.2, got {proba:.3f}"


def test_batch_prediction(model, sample_high_risk, sample_low_risk):
    """Pipeline accepts a batch of records."""
    df = pd.DataFrame([sample_high_risk, sample_low_risk], columns=FEATURE_COLUMNS)
    preds = model.predict(df)
    probas = model.predict_proba(df)
    assert preds.shape == (2,)
    assert probas.shape == (2, 2)
    assert preds[0] != preds[1], "High- and low-risk patients should differ"
