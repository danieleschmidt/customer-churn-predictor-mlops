import os
import tempfile
import shutil
import unittest
import mlflow

from scripts.run_pipeline import run_pipeline
from src.train_model import MODEL_PATH

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

if __name__ == '__main__':
    unittest.main()
