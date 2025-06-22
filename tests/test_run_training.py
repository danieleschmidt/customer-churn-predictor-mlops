import os
import tempfile
import shutil
import unittest
import mlflow

from scripts.run_training import run_training
from src.train_model import MODEL_PATH


class TestRunTraining(unittest.TestCase):
    def setUp(self):
        self.X_path = 'data/processed/processed_features.csv'
        self.y_path = 'data/processed/processed_target.csv'
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


if __name__ == '__main__':
    unittest.main()
