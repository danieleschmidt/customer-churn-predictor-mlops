"""Tests for the feature pipeline."""
import numpy as np
import pytest
from churn.generate_data import generate
from churn.features import load_data, build_preprocessor, build_pipeline
from sklearn.linear_model import LogisticRegression


def _get_Xy(n=300):
    import tempfile, os
    df = generate(n_samples=n, random_state=42)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        path = f.name
    X, y = load_data(path)
    os.unlink(path)
    return X, y


def test_load_data_shape():
    X, y = _get_Xy(200)
    assert len(X) == 200
    assert len(y) == 200
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns


def test_load_data_binary_target():
    _, y = _get_Xy(200)
    assert set(y.unique()).issubset({0, 1})


def test_preprocessor_output_shape():
    X, y = _get_Xy(200)
    prep = build_preprocessor()
    X_t = prep.fit_transform(X)
    assert X_t.shape[0] == 200
    assert X_t.shape[1] > 5  # should expand categoricals


def test_preprocessor_no_nan():
    X, y = _get_Xy(200)
    prep = build_preprocessor()
    X_t = prep.fit_transform(X)
    assert not np.isnan(X_t).any()


def test_pipeline_fit_predict():
    X, y = _get_Xy(300)
    pipeline = build_pipeline(LogisticRegression(max_iter=200, random_state=0))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert set(preds).issubset({0, 1})
    probs = pipeline.predict_proba(X)
    assert probs.shape == (300, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
