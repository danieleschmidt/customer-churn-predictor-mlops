import os
import shutil
import tempfile
import unittest
import pandas as pd
import mlflow
from unittest.mock import patch, call

from src.train_model import train_churn_model
from src.predict_churn import make_prediction, MODEL_PATH, FEATURE_COLUMNS_PATH, RUN_ID_PATH

class TestPredictChurn(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        self.mlflow_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.mlflow_dir}")
        mlflow.set_experiment("Default")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            os.remove(FEATURE_COLUMNS_PATH)
        if os.path.exists(RUN_ID_PATH):
            os.remove(RUN_ID_PATH)

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            os.remove(FEATURE_COLUMNS_PATH)
        if os.path.exists(RUN_ID_PATH):
            os.remove(RUN_ID_PATH)
        shutil.rmtree(self.mlflow_dir, ignore_errors=True)

    def test_make_prediction_returns_probabilities(self):
        train_churn_model(self.X_path, self.y_path)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict)
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_make_prediction_handles_missing_feature(self):
        train_churn_model(self.X_path, self.y_path)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        # remove one feature
        removed_key = list(input_dict.keys())[0]
        input_dict.pop(removed_key)
        pred, prob = make_prediction(input_dict)
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_prediction_downloads_columns_if_missing(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(FEATURE_COLUMNS_PATH)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict)
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_prediction_downloads_model_if_missing(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict)
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_prediction_uses_env_run_id(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        os.remove(RUN_ID_PATH)
        os.remove(FEATURE_COLUMNS_PATH)
        os.environ["MLFLOW_RUN_ID"] = run_id
        try:
            X = pd.read_csv(self.X_path)
            input_dict = X.iloc[0].to_dict()
            pred, prob = make_prediction(input_dict)
            self.assertTrue(os.path.exists(MODEL_PATH))
            self.assertIn(pred, [0, 1])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            os.environ.pop("MLFLOW_RUN_ID", None)

    def test_prediction_run_id_argument(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        os.remove(RUN_ID_PATH)
        os.remove(FEATURE_COLUMNS_PATH)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict, run_id=run_id)
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @patch('src.predict_churn.safe_write_json')
    def test_prediction_uses_secure_file_operations(self, mock_safe_write):
        """Test that predict_churn uses safe_write_json instead of direct file operations"""
        # Train model first to set up MLflow run
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        
        # Remove feature columns to trigger download and writing
        if os.path.exists(FEATURE_COLUMNS_PATH):
            os.remove(FEATURE_COLUMNS_PATH)
        
        # Make a prediction which should trigger secure file writing
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict)
        
        # Verify that safe_write_json was called instead of direct file operations
        mock_safe_write.assert_called()
        # Check that the call was made with the feature columns path
        call_args = mock_safe_write.call_args_list
        self.assertTrue(any(FEATURE_COLUMNS_PATH in str(call) for call in call_args))
        
        # Verify prediction still works correctly
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

if __name__ == '__main__':
    unittest.main()
