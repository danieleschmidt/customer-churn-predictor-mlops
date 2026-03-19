"""Tests for evaluation helpers."""
import numpy as np
import pytest
from churn.evaluate import compute_metrics, get_feature_importance
from churn.generate_data import generate
from churn.features import load_data, build_pipeline
from sklearn.linear_model import LogisticRegression
import tempfile, os


def _trained_pipeline():
    df = generate(n_samples=400, random_state=42)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        path = f.name
    X, y = load_data(path)
    os.unlink(path)
    pipeline = build_pipeline(LogisticRegression(max_iter=300, random_state=0))
    pipeline.fit(X, y)
    return pipeline, X, y


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in m
    assert "f1" in m
    assert "avg_precision" in m


def test_compute_metrics_range():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.1, 0.9, 0.85, 0.15, 0.2, 0.8])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_feature_importance_returns_dataframe():
    pipeline, X, y = _trained_pipeline()
    fi = get_feature_importance(pipeline, top_n=10)
    assert len(fi) > 0
    assert "feature" in fi.columns
    assert "importance" in fi.columns


def test_feature_importance_sorted():
    pipeline, X, y = _trained_pipeline()
    fi = get_feature_importance(pipeline)
    assert fi["importance"].is_monotonic_decreasing or len(fi) <= 1
