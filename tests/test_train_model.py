import os
import tempfile
import unittest
import shutil
import joblib
import mlflow

from src.train_model import train_churn_model

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        self.model_path = 'models/churn_model.joblib'
        # Ensure a clean environment
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        self.mlflow_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.mlflow_dir}")
        mlflow.set_experiment("Default")

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        shutil.rmtree(self.mlflow_dir, ignore_errors=True)

    def test_train_model_outputs_file(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(run_id, str)
        model = joblib.load(model_path)
        self.assertTrue(hasattr(model, "predict"))

if __name__ == '__main__':
    unittest.main()
