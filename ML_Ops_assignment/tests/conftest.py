"""
conftest.py

Shared pytest fixtures for the Heart Disease test suite.
"""

import pytest


@pytest.fixture(scope="session")
def sample_high_risk():
    """Patient profile expected to trigger a positive prediction."""
    return {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5,
        "slope": 2, "ca": 3, "thal": 7,
    }


@pytest.fixture(scope="session")
def sample_low_risk():
    """Patient profile expected to trigger a negative prediction."""
    return {
        "age": 40, "sex": 0, "cp": 3, "trestbps": 120, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 3,
    }


@pytest.fixture(scope="session")
def client():
    """Flask test client (in-process; no network required)."""
    from src.serving.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
