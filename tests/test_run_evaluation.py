import os
import tempfile
import json
import unittest
import mlflow
import shutil

from scripts.run_evaluation import run_evaluation
from src.train_model import train_churn_model
from src.monitor_performance import MODEL_PATH


class TestRunEvaluation(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.tmpdir}")
        mlflow.set_experiment("Default")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_evaluation_creates_output(self):
        train_churn_model(self.X_path, self.y_path)
        output_file = os.path.join(self.tmpdir, 'metrics.json')
        acc, f1 = run_evaluation(MODEL_PATH, self.X_path, self.y_path, output_file)
        self.assertTrue(os.path.exists(output_file))
        with open(output_file) as f:
            data = json.load(f)
        self.assertAlmostEqual(data['accuracy'], acc)
        self.assertAlmostEqual(data['f1_score'], f1)


if __name__ == '__main__':
    unittest.main()
