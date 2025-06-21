import os
import shutil
import tempfile
import unittest
import mlflow

from src.monitor_performance import evaluate_model, monitor_and_retrain, MODEL_PATH
from src.train_model import train_churn_model

class TestMonitorPerformance(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        # ensure temporary mlflow tracking to avoid pollution
        self.mlflow_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.mlflow_dir}")
        mlflow.set_experiment("Default")
        # remove existing model if any
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.rmtree(self.mlflow_dir, ignore_errors=True)

    def test_evaluate_model_after_training(self):
        # train a model
        train_churn_model(self.X_path, self.y_path)
        accuracy, f1 = evaluate_model()
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(f1, float)

    def test_monitor_and_retrain_without_model(self):
        # ensure model doesn't exist
        self.assertFalse(os.path.exists(MODEL_PATH))
        monitor_and_retrain()
        self.assertTrue(os.path.exists(MODEL_PATH))

if __name__ == '__main__':
    unittest.main()
