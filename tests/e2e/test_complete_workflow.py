"""End-to-end tests for the complete ML workflow."""

import os
import tempfile
import pandas as pd
import pytest
from pathlib import Path

from tests.fixtures import create_sample_customer_data
from src.preprocess_data import preprocess_raw_data
from src.train_model import train_churn_model
from src.predict_churn import ChurnPredictor


@pytest.mark.e2e
class TestCompleteWorkflow:
    """Test the complete ML workflow from raw data to predictions."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_raw_data(self, temp_data_dir):
        """Create sample raw data for testing."""
        data = create_sample_customer_data(n_samples=200, include_target=True)
        raw_data_path = temp_data_dir / "raw_data.csv"
        data.to_csv(raw_data_path, index=False)
        return raw_data_path
    
    def test_complete_ml_pipeline(self, temp_data_dir, sample_raw_data):
        """Test the complete ML pipeline from data preprocessing to prediction."""
        # Step 1: Preprocessing
        processed_features_path = temp_data_dir / "processed_features.csv"
        processed_target_path = temp_data_dir / "processed_target.csv"
        preprocessor_path = temp_data_dir / "preprocessor.joblib"
        
        preprocess_raw_data(
            raw_data_path=str(sample_raw_data),
            processed_features_path=str(processed_features_path),
            processed_target_path=str(processed_target_path),
            preprocessor_path=str(preprocessor_path)
        )
        
        # Verify preprocessing outputs
        assert processed_features_path.exists()
        assert processed_target_path.exists()
        assert preprocessor_path.exists()
        
        # Load processed data
        X = pd.read_csv(processed_features_path)
        y = pd.read_csv(processed_target_path).iloc[:, 0]
        
        # Verify data shape and content
        assert len(X) == 200
        assert len(y) == 200
        assert X.shape[1] > 10  # Should have multiple features after encoding
        assert set(y.unique()).issubset({0, 1})  # Binary target
        
        # Step 2: Model Training
        model_path = temp_data_dir / "model.joblib"
        feature_columns_path = temp_data_dir / "feature_columns.json"
        
        model, accuracy = train_churn_model(
            X=X,
            y=y,
            model_path=str(model_path),
            feature_columns_path=str(feature_columns_path),
            test_size=0.2,
            random_state=42
        )
        
        # Verify training outputs
        assert model_path.exists()
        assert feature_columns_path.exists()
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.5  # Should be better than random
        
        # Step 3: Prediction
        predictor = ChurnPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            feature_columns_path=str(feature_columns_path)
        )
        
        # Test single prediction
        sample_customer = {
            "tenure": 12,
            "MonthlyCharges": 65.50,
            "TotalCharges": 786.00,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes"
        }
        
        prediction, probability = predictor.predict_single(sample_customer)
        
        # Verify prediction outputs
        assert prediction in [0, 1]
        assert 0.0 <= probability <= 1.0
        
        # Test batch prediction
        test_data = create_sample_customer_data(n_samples=10, include_target=False)
        predictions, probabilities = predictor.predict_batch(test_data)
        
        # Verify batch prediction outputs
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0.0 <= prob <= 1.0 for prob in probabilities)
    
    def test_workflow_with_missing_data(self, temp_data_dir):
        """Test workflow handles missing data appropriately."""
        # Create data with missing values
        data = create_sample_customer_data(n_samples=100, include_target=True)
        
        # Introduce missing values
        data.loc[data.index[:10], 'TotalCharges'] = ''
        data.loc[data.index[10:20], 'tenure'] = None
        
        raw_data_path = temp_data_dir / "raw_data_missing.csv"
        data.to_csv(raw_data_path, index=False)
        
        # Test preprocessing handles missing data
        processed_features_path = temp_data_dir / "processed_features.csv"
        processed_target_path = temp_data_dir / "processed_target.csv"
        preprocessor_path = temp_data_dir / "preprocessor.joblib"
        
        preprocess_raw_data(
            raw_data_path=str(raw_data_path),
            processed_features_path=str(processed_features_path),
            processed_target_path=str(processed_target_path),
            preprocessor_path=str(preprocessor_path)
        )
        
        # Verify preprocessing completed successfully
        assert processed_features_path.exists()
        assert processed_target_path.exists()
        
        # Load and verify processed data
        X = pd.read_csv(processed_features_path)
        y = pd.read_csv(processed_target_path).iloc[:, 0]
        
        # Should have handled missing values (no NaN in final data)
        assert not X.isnull().any().any()
        assert not y.isnull().any()
    
    def test_model_persistence_and_loading(self, temp_data_dir, sample_raw_data):
        """Test that models can be saved and loaded correctly."""
        # Process data and train model
        processed_features_path = temp_data_dir / "processed_features.csv"
        processed_target_path = temp_data_dir / "processed_target.csv"
        preprocessor_path = temp_data_dir / "preprocessor.joblib"
        
        preprocess_raw_data(
            raw_data_path=str(sample_raw_data),
            processed_features_path=str(processed_features_path),
            processed_target_path=str(processed_target_path),
            preprocessor_path=str(preprocessor_path)
        )
        
        X = pd.read_csv(processed_features_path)
        y = pd.read_csv(processed_target_path).iloc[:, 0]
        
        model_path = temp_data_dir / "model.joblib"
        feature_columns_path = temp_data_dir / "feature_columns.json"
        
        model, accuracy = train_churn_model(
            X=X,
            y=y,
            model_path=str(model_path),
            feature_columns_path=str(feature_columns_path),
            test_size=0.2,
            random_state=42
        )
        
        # Create predictor and make prediction
        predictor1 = ChurnPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            feature_columns_path=str(feature_columns_path)
        )
        
        sample_customer = {
            "tenure": 24,
            "MonthlyCharges": 75.50,
            "TotalCharges": 1812.00,
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "Yes",
            "PhoneService": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No"
        }
        
        pred1, prob1 = predictor1.predict_single(sample_customer)
        
        # Create new predictor instance (simulating reload)
        predictor2 = ChurnPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            feature_columns_path=str(feature_columns_path)
        )
        
        pred2, prob2 = predictor2.predict_single(sample_customer)
        
        # Predictions should be identical
        assert pred1 == pred2
        assert abs(prob1 - prob2) < 1e-10  # Should be exactly the same
    
    def test_workflow_performance_benchmarks(self, temp_data_dir, sample_raw_data):
        """Test that workflow meets basic performance benchmarks."""
        import time
        
        # Benchmark preprocessing
        start_time = time.time()
        
        processed_features_path = temp_data_dir / "processed_features.csv"
        processed_target_path = temp_data_dir / "processed_target.csv"
        preprocessor_path = temp_data_dir / "preprocessor.joblib"
        
        preprocess_raw_data(
            raw_data_path=str(sample_raw_data),
            processed_features_path=str(processed_features_path),
            processed_target_path=str(processed_target_path),
            preprocessor_path=str(preprocessor_path)
        )
        
        preprocessing_time = time.time() - start_time
        assert preprocessing_time < 10.0  # Should complete within 10 seconds
        
        # Benchmark training
        X = pd.read_csv(processed_features_path)
        y = pd.read_csv(processed_target_path).iloc[:, 0]
        
        start_time = time.time()
        
        model_path = temp_data_dir / "model.joblib"
        feature_columns_path = temp_data_dir / "feature_columns.json"
        
        model, accuracy = train_churn_model(
            X=X,
            y=y,
            model_path=str(model_path),
            feature_columns_path=str(feature_columns_path),
            test_size=0.2,
            random_state=42
        )
        
        training_time = time.time() - start_time
        assert training_time < 30.0  # Should complete within 30 seconds
        
        # Benchmark prediction
        predictor = ChurnPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            feature_columns_path=str(feature_columns_path)
        )
        
        sample_customer = {
            "tenure": 12,
            "MonthlyCharges": 65.50,
            "TotalCharges": 786.00,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes"
        }
        
        # Benchmark single prediction
        start_time = time.time()
        for _ in range(100):  # 100 predictions
            predictor.predict_single(sample_customer)
        single_prediction_time = (time.time() - start_time) / 100
        
        # Should be fast for single predictions
        assert single_prediction_time < 0.1  # Less than 100ms per prediction
        
        # Benchmark batch prediction
        test_data = create_sample_customer_data(n_samples=1000, include_target=False)
        
        start_time = time.time()
        predictions, probabilities = predictor.predict_batch(test_data)
        batch_prediction_time = time.time() - start_time
        
        # Should handle large batches efficiently
        assert batch_prediction_time < 5.0  # Less than 5 seconds for 1000 predictions
        assert len(predictions) == 1000
        assert len(probabilities) == 1000