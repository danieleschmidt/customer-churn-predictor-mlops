import os
import tempfile
import unittest
import pandas as pd
import mlflow
import shutil

from scripts.run_prediction import run_predictions
from src.train_model import train_churn_model
from src.predict_churn import MODEL_PATH, FEATURE_COLUMNS_PATH


class TestRunPrediction(unittest.TestCase):
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

    def test_run_predictions_creates_output(self):
        # Train model first
        train_churn_model(self.X_path, self.y_path)
        input_df = pd.read_csv(self.X_path).head(3)
        input_csv = os.path.join(self.tmpdir, 'input.csv')
        output_csv = os.path.join(self.tmpdir, 'preds.csv')
        input_df.to_csv(input_csv, index=False)

        run_predictions(input_csv, output_csv)

        self.assertTrue(os.path.exists(output_csv))
        preds_df = pd.read_csv(output_csv)
        self.assertIn('prediction', preds_df.columns)
        self.assertIn('probability', preds_df.columns)
        self.assertEqual(len(preds_df), len(input_df))

    def test_run_predictions_downloads_model_if_missing(self):
        # Train and then remove the model file to simulate missing artifact
        train_churn_model(self.X_path, self.y_path)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        input_df = pd.read_csv(self.X_path).head(2)
        input_csv = os.path.join(self.tmpdir, 'input_missing.csv')
        output_csv = os.path.join(self.tmpdir, 'preds_missing.csv')
        input_df.to_csv(input_csv, index=False)

        run_predictions(input_csv, output_csv)

        # The prediction script should have downloaded the model again
        self.assertTrue(os.path.exists(MODEL_PATH))
        preds_df = pd.read_csv(output_csv)
        self.assertEqual(len(preds_df), len(input_df))

    def test_run_predictions_downloads_columns_if_missing(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            os.remove(FEATURE_COLUMNS_PATH)

        input_df = pd.read_csv(self.X_path).head(2)
        input_csv = os.path.join(self.tmpdir, 'input_cols.csv')
        output_csv = os.path.join(self.tmpdir, 'preds_cols.csv')
        input_df.to_csv(input_csv, index=False)

        run_predictions(input_csv, output_csv)

        self.assertTrue(os.path.exists(FEATURE_COLUMNS_PATH))
        preds_df = pd.read_csv(output_csv)
        self.assertEqual(len(preds_df), len(input_df))

    def test_run_predictions_uses_env_run_id(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')

        os.environ['MLFLOW_RUN_ID'] = run_id
        try:
            input_df = pd.read_csv(self.X_path).head(2)
            input_csv = os.path.join(self.tmpdir, 'input_env.csv')
            output_csv = os.path.join(self.tmpdir, 'preds_env.csv')
            input_df.to_csv(input_csv, index=False)

            run_predictions(input_csv, output_csv)

            self.assertTrue(os.path.exists(MODEL_PATH))
            preds_df = pd.read_csv(output_csv)
            self.assertEqual(len(preds_df), len(input_df))
        finally:
            os.environ.pop('MLFLOW_RUN_ID', None)

    def test_run_predictions_run_id_argument(self):
        model_path, run_id = train_churn_model(self.X_path, self.y_path)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists('models/mlflow_run_id.txt'):
            os.remove('models/mlflow_run_id.txt')

        input_df = pd.read_csv(self.X_path).head(2)
        input_csv = os.path.join(self.tmpdir, 'input_arg.csv')
        output_csv = os.path.join(self.tmpdir, 'preds_arg.csv')
        input_df.to_csv(input_csv, index=False)

        run_predictions(input_csv, output_csv, run_id=run_id)

        self.assertTrue(os.path.exists(MODEL_PATH))
        preds_df = pd.read_csv(output_csv)
        self.assertEqual(len(preds_df), len(input_df))


if __name__ == '__main__':
    unittest.main()
