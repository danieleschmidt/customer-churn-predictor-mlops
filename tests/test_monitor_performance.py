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

    def test_evaluate_model_downloads_model(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        acc, f1 = evaluate_model()
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)

    def test_evaluate_model_uses_env_run_id(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')
        os.environ['MLFLOW_RUN_ID'] = run_id
        try:
            acc, f1 = evaluate_model()
            self.assertTrue(os.path.exists(MODEL_PATH))
            self.assertIsInstance(acc, float)
            self.assertIsInstance(f1, float)
        finally:
            os.environ.pop('MLFLOW_RUN_ID', None)

    def test_evaluate_model_run_id_argument(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')
        acc, f1 = evaluate_model(run_id=run_id)
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)

    def test_evaluate_model_logs_metrics(self):
        train_churn_model(self.X_path, self.y_path)
        evaluate_model()
        exp = mlflow.get_experiment_by_name("Default")
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"], max_results=1)
        latest_run = runs[0]
        self.assertIn("accuracy", latest_run.data.metrics)
        self.assertIn("f1_score", latest_run.data.metrics)

    def test_monitor_retrain_threshold_env(self):
        train_churn_model(self.X_path, self.y_path)
        os.environ['CHURN_THRESHOLD'] = '1.1'  # force retraining since accuracy < 1.1
        try:
            monitor_and_retrain()
            self.assertTrue(os.path.exists(MODEL_PATH))
        finally:
            os.environ.pop('CHURN_THRESHOLD', None)

    def test_monitor_custom_paths(self):
        from unittest.mock import patch

        with patch('src.monitor_performance.evaluate_model', return_value=(0.5, 0.5)) as mock_eval, patch(
            'src.monitor_performance.train_churn_model'
        ) as mock_train:
            monitor_and_retrain(
                threshold=0.6,
                X_path='custom_X.csv',
                y_path='custom_y.csv',
            )
            mock_eval.assert_called_once_with(
                MODEL_PATH, 'custom_X.csv', 'custom_y.csv'
            )
            mock_train.assert_called_once()
            args = mock_train.call_args.args
            self.assertEqual(args[0], 'custom_X.csv')
            self.assertEqual(args[1], 'custom_y.csv')

if __name__ == '__main__':
    unittest.main()
