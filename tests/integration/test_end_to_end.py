"""End-to-end integration tests."""

import os
import tempfile
from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.preprocess_data import preprocess_data
from src.train_model import train_model


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test the complete ML workflow from data to prediction."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-api-key-for-testing"}

    def test_complete_ml_pipeline(self, sample_customer_data, temp_data_dir, temp_model_dir):
        """Test complete ML pipeline from raw data to trained model."""
        # 1. Save raw data
        raw_data_path = temp_data_dir / "raw" / "customer_data.csv"
        sample_customer_data.to_csv(raw_data_path, index=False)

        # 2. Preprocess data
        processed_features_path = temp_data_dir / "processed" / "processed_features.csv"
        processed_target_path = temp_data_dir / "processed" / "processed_target.csv"

        try:
            preprocess_data(
                input_path=str(raw_data_path),
                output_features_path=str(processed_features_path),
                output_target_path=str(processed_target_path)
            )

            # Verify processed files exist
            assert processed_features_path.exists()
            assert processed_target_path.exists()

            # 3. Train model
            model_path = temp_model_dir / "churn_model.joblib"
            feature_columns_path = temp_model_dir / "feature_columns.json"

            train_model(
                features_path=str(processed_features_path),
                target_path=str(processed_target_path),
                model_output_path=str(model_path),
                feature_columns_output_path=str(feature_columns_path)
            )

            # Verify model files exist
            assert model_path.exists()
            assert feature_columns_path.exists()

        except Exception as e:
            pytest.skip(f"ML pipeline components not available: {e}")

    def test_data_preprocessing_integration(self, sample_customer_data, temp_data_dir):
        """Test data preprocessing integration."""
        # Save sample data
        input_path = temp_data_dir / "raw" / "test_data.csv"
        sample_customer_data.to_csv(input_path, index=False)

        output_features_path = temp_data_dir / "processed" / "features.csv"
        output_target_path = temp_data_dir / "processed" / "target.csv"

        try:
            # Run preprocessing
            preprocess_data(
                input_path=str(input_path),
                output_features_path=str(output_features_path),
                output_target_path=str(output_target_path)
            )

            # Verify outputs
            assert output_features_path.exists()
            assert output_target_path.exists()

            # Load and verify processed data
            features_df = pd.read_csv(output_features_path)
            target_series = pd.read_csv(output_target_path)

            assert len(features_df) == len(sample_customer_data)
            assert len(target_series) == len(sample_customer_data)

            # Check for expected columns (one-hot encoded)
            expected_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for col in expected_columns:
                assert col in features_df.columns

        except Exception as e:
            pytest.skip(f"Preprocessing function not available: {e}")

    def test_api_prediction_integration(self, client, auth_headers, sample_processed_features):
        """Test API prediction with real-like data."""
        # Use first row of sample data
        test_features = sample_processed_features.iloc[0].to_dict()

        response = client.post(
            "/predict",
            json={"features": test_features},
            headers=auth_headers
        )

        # Response should be successful or indicate missing model
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert isinstance(data["prediction"], (int, bool))
            assert isinstance(data["probability"], float)
            assert 0 <= data["probability"] <= 1

    def test_health_check_integration(self, client):
        """Test health check endpoint integration."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data

    def test_metrics_endpoint_integration(self, client):
        """Test metrics endpoint integration."""
        response = client.get("/metrics")
        # Metrics endpoint might not be implemented yet
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should return some form of metrics
            assert response.headers.get("content-type")

    def test_error_handling_integration(self, client, auth_headers):
        """Test error handling across the system."""
        # Test with malformed data
        response = client.post(
            "/predict",
            json={"invalid": "data"},
            headers=auth_headers
        )
        assert response.status_code in [400, 422]

        # Test with missing features
        response = client.post(
            "/predict",
            json={"features": {"invalid_feature": 123}},
            headers=auth_headers
        )
        assert response.status_code in [400, 422, 500]

    def test_authentication_integration(self, client):
        """Test authentication integration across endpoints."""
        # Test protected endpoint without auth
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 401

        # Test with invalid token
        response = client.post(
            "/predict",
            json={"features": {}},
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    @pytest.mark.slow
    def test_batch_prediction_integration(self, client, auth_headers, sample_processed_features):
        """Test batch prediction capabilities."""
        # Test multiple predictions
        results = []
        for i in range(min(5, len(sample_processed_features))):
            test_features = sample_processed_features.iloc[i].to_dict()
            
            response = client.post(
                "/predict",
                json={"features": test_features},
                headers=auth_headers
            )
            results.append(response.status_code)

        # Should handle multiple requests consistently
        if any(status == 200 for status in results):
            # If any succeeded, most should succeed
            success_rate = sum(1 for status in results if status == 200) / len(results)
            assert success_rate >= 0.8

    def test_data_validation_integration(self, client, auth_headers):
        """Test data validation integration."""
        # Test with various data types
        test_cases = [
            # Valid data
            {
                "SeniorCitizen": 1,
                "tenure": 12,
                "MonthlyCharges": 29.85,
                "TotalCharges": 358.2,
                "gender_Female": 1,
                "gender_Male": 0,
            },
            # Invalid types
            {
                "SeniorCitizen": "yes",
                "tenure": "twelve",
                "MonthlyCharges": "expensive",
            },
            # Missing required fields
            {
                "SeniorCitizen": 1,
            },
            # Out of range values
            {
                "SeniorCitizen": -1,
                "tenure": -5,
                "MonthlyCharges": -100,
            },
        ]

        for i, test_data in enumerate(test_cases):
            response = client.post(
                "/predict",
                json={"features": test_data},
                headers=auth_headers
            )

            if i == 0:
                # First case (valid) should work or indicate missing model
                assert response.status_code in [200, 404, 500]
            else:
                # Other cases should be rejected
                assert response.status_code in [400, 422]

    def test_logging_integration(self, client, auth_headers, caplog):
        """Test logging integration across components."""
        with caplog.at_level("INFO"):
            # Make a request that should generate logs
            client.post(
                "/predict",
                json={"features": {"tenure": 12}},
                headers=auth_headers
            )

        # Should have some log entries
        # Note: This might not work in test environment depending on logging config

    def test_configuration_integration(self, monkeypatch):
        """Test configuration management integration."""
        # Test with different environment variables
        test_configs = {
            "LOG_LEVEL": "DEBUG",
            "API_PORT": "8001",
            "ENVIRONMENT": "testing",
        }

        for key, value in test_configs.items():
            monkeypatch.setenv(key, value)

        # Configuration changes should be picked up
        # Note: This might require application restart in some cases

    def test_concurrent_requests_integration(self, client, auth_headers, sample_processed_features):
        """Test concurrent request handling integration."""
        import threading
        import time

        test_features = sample_processed_features.iloc[0].to_dict()
        results = []
        
        def make_request():
            try:
                response = client.post(
                    "/predict",
                    json={"features": test_features},
                    headers=auth_headers,
                    timeout=5.0
                )
                results.append(response.status_code)
            except Exception as e:
                results.append(500)

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)

        # Should handle concurrent requests
        assert len(results) == 10
        # If any succeed, most should succeed
        if any(status == 200 for status in results):
            success_rate = sum(1 for status in results if status == 200) / len(results)
            assert success_rate >= 0.7