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

    def test_run_evaluation_downloads_model(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        output_file = os.path.join(self.tmpdir, 'metrics_missing.json')
        acc, f1 = run_evaluation(MODEL_PATH, self.X_path, self.y_path, output_file)
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertTrue(os.path.exists(output_file))

    def test_run_evaluation_uses_env_run_id(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')
        os.environ['MLFLOW_RUN_ID'] = run_id
        try:
            output_file = os.path.join(self.tmpdir, 'metrics_env.json')
            acc, f1 = run_evaluation(MODEL_PATH, self.X_path, self.y_path, output_file)
            self.assertTrue(os.path.exists(MODEL_PATH))
            self.assertTrue(os.path.exists(output_file))
        finally:
            os.environ.pop('MLFLOW_RUN_ID', None)

    def test_run_evaluation_run_id_argument(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')
        output_file = os.path.join(self.tmpdir, 'metrics_arg.json')
        acc, f1 = run_evaluation(MODEL_PATH, self.X_path, self.y_path, output_file, run_id=run_id)
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertTrue(os.path.exists(output_file))

    def test_run_evaluation_detailed(self):
        train_churn_model(self.X_path, self.y_path)
        output_file = os.path.join(self.tmpdir, 'metrics_detailed.json')
        acc, f1, report = run_evaluation(
            MODEL_PATH,
            self.X_path,
            self.y_path,
            output_file,
            detailed=True,
        )
        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(report, dict)
        with open(output_file) as f:
            data = json.load(f)
        self.assertIn('classification_report', data)


if __name__ == '__main__':
    unittest.main()
