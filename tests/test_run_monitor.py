import os
import tempfile
import shutil
import unittest
import mlflow
from scripts.run_monitor import main
from src.monitor_performance import MODEL_PATH
from src.train_model import train_churn_model


class TestRunMonitor(unittest.TestCase):
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

    def test_run_monitor_trains_if_missing(self):
        self.assertFalse(os.path.exists(MODEL_PATH))
        import sys
        old_argv = sys.argv
        sys.argv = ['run_monitor.py']
        try:
            main()
        finally:
            sys.argv = old_argv
        self.assertTrue(os.path.exists(MODEL_PATH))

    def test_run_monitor_evaluates_existing_model(self):
        train_churn_model(self.X_path, self.y_path)
        import sys
        old_argv = sys.argv
        sys.argv = ['run_monitor.py']
        try:
            main()
        finally:
            sys.argv = old_argv
        # ensure model still exists and evaluation didn't remove it
        self.assertTrue(os.path.exists(MODEL_PATH))

    def test_run_monitor_threshold_argument(self):
        train_churn_model(self.X_path, self.y_path)
        os.remove(MODEL_PATH)
        main_args = ['--threshold', '1.1']
        # Temporarily patch sys.argv
        import sys
        old_argv = sys.argv
        sys.argv = ['run_monitor.py'] + main_args
        try:
            main()
            self.assertTrue(os.path.exists(MODEL_PATH))
        finally:
            sys.argv = old_argv


if __name__ == '__main__':
    unittest.main()
