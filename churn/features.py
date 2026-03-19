"""
Feature pipeline: preprocessing, encoding, scaling.

Returns sklearn Pipeline objects so the same transforms apply at
train time and inference time without leakage.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Columns that the pipeline knows about (derived at fit time in practice,
# but listed here as the canonical feature set).
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SupportCalls",
    "SeniorCitizen",
]
CATEGORICAL_FEATURES = [
    "gender",
    "PhoneService",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "StreamingTV",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return (X, y)."""
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    y_raw = df.pop(TARGET_COL)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name=TARGET_COL)
    return df, y


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer."""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, feature_name_combiner="concat"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Wrap preprocessor + model into a single sklearn Pipeline."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
