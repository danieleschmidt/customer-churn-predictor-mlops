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

    def test_monitor_custom_params_forwarded(self):
        from unittest.mock import patch

        with patch('src.monitor_performance.train_churn_model') as mock_train:
            import sys
            old_argv = sys.argv
            sys.argv = [
                'run_monitor.py',
                '--threshold', '0.0',
                '--solver', 'saga',
                '--C', '0.9',
                '--penalty', 'l1',
                '--random_state', '99',
                '--max_iter', '150',
                '--test_size', '0.3',
            ]
            try:
                main()
                mock_train.assert_called()
                kwargs = mock_train.call_args.kwargs
                self.assertEqual(kwargs['solver'], 'saga')
                self.assertAlmostEqual(kwargs['C'], 0.9)
                self.assertEqual(kwargs['penalty'], 'l1')
                self.assertEqual(kwargs['random_state'], 99)
                self.assertEqual(kwargs['max_iter'], 150)
                self.assertAlmostEqual(kwargs['test_size'], 0.3)
            finally:
                sys.argv = old_argv

    def test_monitor_custom_paths(self):
        from unittest.mock import patch

        with patch('src.monitor_performance.train_churn_model') as mock_train, patch(
            'src.monitor_performance.evaluate_model', return_value=(0.5, 0.5)
        ):
            import sys
            old_argv = sys.argv
            sys.argv = [
                'run_monitor.py',
                '--threshold', '0.6',
                '--X_path', 'custom_X.csv',
                '--y_path', 'custom_y.csv',
            ]
            try:
                main()
                mock_train.assert_called()
                args = mock_train.call_args.args
                self.assertEqual(args[0], 'custom_X.csv')
                self.assertEqual(args[1], 'custom_y.csv')
            finally:
                sys.argv = old_argv


if __name__ == '__main__':
    unittest.main()
