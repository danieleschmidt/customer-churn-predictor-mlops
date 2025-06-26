import os
import tempfile
import shutil
import unittest
import mlflow

from scripts.run_pipeline import run_pipeline
from src.train_model import MODEL_PATH
import joblib


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{self.tmpdir}")
        mlflow.set_experiment("Default")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    def tearDown(self):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipeline_runs(self):
        acc, f1 = run_pipeline()
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)

    def test_pipeline_custom_params(self):
        acc, f1 = run_pipeline(
            solver="liblinear",
            C=0.6,
            penalty="l1",
            random_state=123,
            max_iter=150,
            test_size=0.3,
        )
        self.assertTrue(os.path.exists(MODEL_PATH))
        model = joblib.load(MODEL_PATH)
        params = model.get_params()
        self.assertEqual(params["solver"], "liblinear")
        self.assertAlmostEqual(params["C"], 0.6)
        self.assertEqual(params["penalty"], "l1")
        self.assertEqual(params["random_state"], 123)
        self.assertEqual(params["max_iter"], 150)

    def test_test_size_forwarded(self):
        from unittest.mock import patch

        with patch(
            "scripts.run_pipeline.run_training",
            return_value=(MODEL_PATH, "run"),
        ) as mock_training, patch(
            "scripts.run_pipeline.run_evaluation", return_value=(0.0, 0.0)
        ):
            run_pipeline(test_size=0.25, penalty="l1")
            mock_training.assert_called_once()
            self.assertAlmostEqual(mock_training.call_args.kwargs["test_size"], 0.25)
            self.assertEqual(mock_training.call_args.kwargs["penalty"], "l1")


if __name__ == "__main__":
    unittest.main()
