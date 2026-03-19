"""Tests for synthetic data generation."""
import pytest
import pandas as pd
from churn.generate_data import generate


def test_generate_shape():
    df = generate(n_samples=200, random_state=1)
    assert len(df) == 200
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_generate_churn_balance():
    """Churn rate should be roughly 20-40% with default params."""
    df = generate(n_samples=2000, random_state=42)
    rate = (df["Churn"] == "Yes").mean()
    assert 0.15 <= rate <= 0.50, f"Unexpected churn rate: {rate:.2%}"


def test_generate_no_nulls():
    df = generate(n_samples=500)
    assert df.isnull().sum().sum() == 0


def test_generate_reproducible():
    df1 = generate(n_samples=100, random_state=7)
    df2 = generate(n_samples=100, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_required_columns():
    df = generate(n_samples=50)
    required = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "SupportCalls"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_numeric_ranges():
    df = generate(n_samples=1000)
    assert df["tenure"].between(1, 72).all(), "tenure out of expected range"
    assert df["MonthlyCharges"].between(10, 130).all(), "MonthlyCharges out of range"
    assert (df["SeniorCitizen"].isin([0, 1])).all()
