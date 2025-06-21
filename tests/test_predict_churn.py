import os
import shutil
import tempfile
import unittest
import pandas as pd
import mlflow

from src.train_model import train_churn_model
from src.predict_churn import make_prediction, MODEL_PATH

class TestPredictChurn(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        self.mlflow_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.mlflow_dir}")
        mlflow.set_experiment("Default")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.rmtree(self.mlflow_dir, ignore_errors=True)

    def test_make_prediction_returns_probabilities(self):
        train_churn_model(self.X_path, self.y_path)
        X = pd.read_csv(self.X_path)
        input_dict = X.iloc[0].to_dict()
        pred, prob = make_prediction(input_dict)
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

if __name__ == '__main__':
    unittest.main()
