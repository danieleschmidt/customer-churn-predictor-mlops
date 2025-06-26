import os
import tempfile
import shutil
import unittest
import mlflow

from scripts.run_training import run_training
from src.train_model import MODEL_PATH


class TestRunTraining(unittest.TestCase):
    def setUp(self):
        self.X_path = "data/processed/processed_features.csv"
        self.y_path = "data/processed/processed_target.csv"
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.tmpdir}")
        mlflow.set_experiment("Default")

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_training_creates_model(self):
        model_path, run_id = run_training(self.X_path, self.y_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(run_id, str)

    def test_run_training_custom_params(self):
        model_path, run_id = run_training(
            self.X_path,
            self.y_path,
            solver="liblinear",
            C=0.7,
            penalty="l1",
            random_state=123,
            max_iter=150,
            test_size=0.3,
        )
        self.assertTrue(os.path.exists(model_path))
        import joblib

        model = joblib.load(model_path)
        params = model.get_params()
        self.assertAlmostEqual(params["C"], 0.7)
        self.assertEqual(params["penalty"], "l1")
        self.assertEqual(params["random_state"], 123)
        self.assertEqual(params["max_iter"], 150)

    def test_test_size_forwarded(self):
        from unittest.mock import patch

        with patch(
            "scripts.run_training.train_churn_model",
            return_value=(MODEL_PATH, "run"),
        ) as mock_train:
            run_training(self.X_path, self.y_path, test_size=0.3, penalty="l1")
            mock_train.assert_called_once()
            self.assertAlmostEqual(mock_train.call_args.kwargs["test_size"], 0.3)
            self.assertEqual(mock_train.call_args.kwargs["penalty"], "l1")


if __name__ == "__main__":
    unittest.main()
