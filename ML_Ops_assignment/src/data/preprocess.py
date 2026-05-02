"""
preprocess.py

Build the preprocessing ColumnTransformer for the Heart Disease dataset.

Strategy (from EDA findings):
  - Numeric continuous (no missing) : StandardScaler
  - Numeric ordinal with missing    : median impute + StandardScaler  (ca)
  - Nominal categorical (no missing): OneHotEncoder                   (cp, restecg, slope)
  - Nominal categorical with missing: most-frequent impute + OneHot   (thal)
  - Binary (0/1) features           : passthrough                     (sex, fbs, exang)

The transformer is wrapped inside an sklearn Pipeline at training time
so it is fitted on the training set only and serialized with the model.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NUMERIC_WITH_MISSING = ["ca"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope"]
CATEGORICAL_WITH_MISSING = ["thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + NUMERIC_WITH_MISSING
    + CATEGORICAL_FEATURES
    + CATEGORICAL_WITH_MISSING
    + BINARY_FEATURES
)


def build_preprocessor():
    """Return a ColumnTransformer that handles imputation, scaling, encoding."""
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])

    numeric_missing_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    categorical_missing_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("num_missing", numeric_missing_pipe, NUMERIC_WITH_MISSING),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("cat_missing", categorical_missing_pipe, CATEGORICAL_WITH_MISSING),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def split_features_target(df):
    """Return (X, y) from a DataFrame containing the target column."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return X, y
