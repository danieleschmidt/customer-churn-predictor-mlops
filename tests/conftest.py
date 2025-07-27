"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_customer_data() -> pd.DataFrame:
    """Create sample customer data for testing."""
    return pd.DataFrame({
        'customerID': ['001', '002', '003', '004', '005'],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [12, 24, 36, 6, 48],
        'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'OnlineSecurity': ['No', 'No', 'Yes', 'No internet service', 'No'],
        'OnlineBackup': ['Yes', 'No', 'Yes', 'No internet service', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No', 'Yes', 'No internet service', 'No'],
        'StreamingTV': ['No', 'No', 'Yes', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                         'Credit card (automatic)', 'Electronic check'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
        'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', '151.65'],
        'Churn': ['No', 'No', 'No', 'Yes', 'Yes']
    })


@pytest.fixture
def sample_processed_features() -> pd.DataFrame:
    """Create sample processed features for testing."""
    return pd.DataFrame({
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'tenure': [12, 24, 36, 6, 48],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
        'TotalCharges': [29.85, 1889.5, 108.15, 1840.75, 151.65],
        'gender_Female': [1, 0, 1, 0, 1],
        'gender_Male': [0, 1, 0, 1, 0],
        'Partner_No': [0, 1, 0, 1, 0],
        'Partner_Yes': [1, 0, 1, 0, 1],
        'Dependents_No': [1, 1, 0, 1, 0],
        'Dependents_Yes': [0, 0, 1, 0, 1],
    })


@pytest.fixture
def sample_target() -> pd.Series:
    """Create sample target values for testing."""
    return pd.Series([0, 0, 0, 1, 1], name='Churn')


@pytest.fixture
def api_client() -> TestClient:
    """Create a test client for API testing."""
    return TestClient(app)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        'API_KEY': 'test-api-key-for-testing',
        'ENVIRONMENT': 'testing',
        'LOG_LEVEL': 'DEBUG',
        'MLFLOW_TRACKING_URI': 'sqlite:///test.db',
        'CHURN_THRESHOLD': '0.8'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create a temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create a temporary directory for data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    return data_dir


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    # Clean up any test files that might have been created
    test_files = [
        'test_model.joblib',
        'test_preprocessor.joblib',
        'test_features.json',
        'test_data.csv',
        'test_predictions.csv'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Return the path to the docker-compose file for integration tests."""
    return Path(__file__).parent.parent / "docker-compose.yml"


class MockMLflowClient:
    """Mock MLflow client for testing."""
    
    def __init__(self):
        self.run_id = "test-run-id-123"
        self.artifacts = {}
    
    def get_run(self, run_id):
        """Mock get_run method."""
        return type('Run', (), {
            'info': type('Info', (), {'run_id': run_id})(),
            'data': type('Data', (), {'metrics': {'accuracy': 0.85}})()
        })()
    
    def download_artifacts(self, run_id, path, dst_path=None):
        """Mock download_artifacts method."""
        if dst_path:
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            Path(dst_path).write_text("mock artifact content")
        return dst_path or "mock_artifact_path"


@pytest.fixture
def mock_mlflow_client():
    """Provide a mock MLflow client."""
    return MockMLflowClient()


# Performance testing fixtures
@pytest.fixture
def performance_test_data() -> pd.DataFrame:
    """Generate larger dataset for performance testing."""
    size = 10000
    return pd.DataFrame({
        'SeniorCitizen': [0, 1] * (size // 2),
        'tenure': list(range(1, size + 1)),
        'MonthlyCharges': [50.0 + i * 0.1 for i in range(size)],
        'TotalCharges': [500.0 + i * 10 for i in range(size)],
        'gender_Female': [1, 0] * (size // 2),
        'gender_Male': [0, 1] * (size // 2),
    })


# Security testing fixtures
@pytest.fixture
def invalid_api_keys():
    """Provide invalid API keys for security testing."""
    return [
        '',
        'short',
        'invalid-key',
        'too-long-' * 20,
        '12345',
        'null',
        None,
    ]


# Pytest markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )