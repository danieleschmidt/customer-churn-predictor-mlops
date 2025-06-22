import os
import tempfile
import unittest
import shutil
import joblib
import mlflow
import json

from src.train_model import train_churn_model

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        self.model_path = 'models/churn_model.joblib'
        self.feature_path = 'models/feature_columns.json'
        self.run_id_path = 'models/mlflow_run_id.txt'
        # Ensure a clean environment
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.feature_path):
            os.remove(self.feature_path)
        if os.path.exists(self.run_id_path):
            os.remove(self.run_id_path)
        self.mlflow_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.mlflow_dir}")
        mlflow.set_experiment("Default")

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.feature_path):
            os.remove(self.feature_path)
        if os.path.exists(self.run_id_path):
            os.remove(self.run_id_path)
        shutil.rmtree(self.mlflow_dir, ignore_errors=True)

    def test_train_model_outputs_file(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(run_id, str)
        model = joblib.load(model_path)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(os.path.exists(self.feature_path))
        with open(self.feature_path) as f:
            cols = json.load(f)
        self.assertIsInstance(cols, list)
        self.assertGreater(len(cols), 0)
        self.assertTrue(os.path.exists(self.run_id_path))
        with open(self.run_id_path) as f:
            saved_run = f.read().strip()
        self.assertEqual(saved_run, run_id)
        exp = mlflow.get_experiment_by_name("Default")
        artifact_path = os.path.join(self.mlflow_dir, exp.experiment_id, run_id, 'artifacts', 'feature_columns.json')
        self.assertTrue(os.path.exists(artifact_path))

if __name__ == '__main__':
    unittest.main()
