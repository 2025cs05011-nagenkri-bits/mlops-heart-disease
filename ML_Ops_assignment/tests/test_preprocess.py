"""
test_preprocess.py

Unit tests for the preprocessing ColumnTransformer.
"""

import numpy as np
import pandas as pd

from src.data.preprocess import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_WITH_MISSING,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    NUMERIC_WITH_MISSING,
    TARGET,
    build_preprocessor,
    split_features_target,
)


def _toy_dataframe():
    """Small in-memory DataFrame mimicking the Cleveland schema."""
    return pd.DataFrame({
        "age":      [63, 40, 57, 50],
        "sex":      [1, 0, 1, 0],
        "cp":       [1, 3, 4, 2],
        "trestbps": [145, 120, 140, 130],
        "chol":     [233, 180, 250, 210],
        "fbs":      [1, 0, 0, 1],
        "restecg":  [2, 0, 1, 0],
        "thalach":  [150, 180, 130, 165],
        "exang":    [0, 0, 1, 0],
        "oldpeak":  [2.3, 0.0, 1.5, 0.8],
        "slope":    [3, 1, 2, 1],
        "ca":       [0.0, np.nan, 1.0, 2.0],
        "thal":     [6.0, 3.0, np.nan, 7.0],
        "target":   [1, 0, 1, 0],
    })


def test_feature_columns_size():
    """Heart Disease has exactly 13 input features."""
    assert len(FEATURE_COLUMNS) == 13


def test_feature_groups_partition_features():
    """Numeric + categorical + binary groups equal FEATURE_COLUMNS (no overlap, no gap)."""
    union = (
        set(NUMERIC_FEATURES)
        | set(NUMERIC_WITH_MISSING)
        | set(CATEGORICAL_FEATURES)
        | set(CATEGORICAL_WITH_MISSING)
        | set(BINARY_FEATURES)
    )
    assert union == set(FEATURE_COLUMNS)


def test_split_features_target_separates_x_and_y():
    df = _toy_dataframe()
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET
    assert len(X) == len(y) == 4


def test_preprocessor_fits_and_produces_2d_array():
    """fit_transform should return a dense 2D numpy array."""
    df = _toy_dataframe()
    X, _ = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert isinstance(Xt, np.ndarray)
    assert Xt.ndim == 2
    assert Xt.shape[0] == 4


def test_preprocessor_handles_missing_values():
    """ca and thal contain NaNs; transform must still produce a finite matrix."""
    df = _toy_dataframe()
    X, _ = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any(), "Output contains NaN — imputers did not fire"


def test_preprocessor_numeric_columns_are_scaled():
    """StandardScaler output should have ~zero mean per numeric column."""
    df = _toy_dataframe()
    X, _ = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    n_numeric = len(NUMERIC_FEATURES)
    means = Xt[:, :n_numeric].mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-7), f"means not centered: {means}"
