import os
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score

from src.train_model import train_churn_model
from src.predict_churn import _get_run_id
from .constants import (
    MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    PROCESSED_FEATURES_PATH,
    PROCESSED_TARGET_PATH,
    DEFAULT_THRESHOLD,
    THRESHOLD_ENV_VAR,
)

THRESHOLD_ACCURACY = DEFAULT_THRESHOLD


def evaluate_model(
    model_path=MODEL_PATH,
    X_path=PROCESSED_FEATURES_PATH,
    y_path=PROCESSED_TARGET_PATH,
    run_id=None,
):
    """Evaluate the existing model on the processed dataset."""
    if run_id is None:
        run_id = _get_run_id()
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    elif run_id:
        try:
            print(f"Downloading model from MLflow run {run_id}...")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"Saved downloaded model to {model_path}")
        except Exception as e:
            print(f"Error downloading model from MLflow: {e}")
            raise FileNotFoundError(f"Model not found at {model_path}") from e
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Log evaluation metrics to MLflow for tracking
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

    return accuracy, f1


def monitor_and_retrain(threshold: float | None = None):
    """Monitor performance and retrain the model if accuracy falls below threshold."""
    if threshold is None:
        # Check environment variable first
        env_val = os.environ.get(THRESHOLD_ENV_VAR)
        if env_val is not None:
            try:
                threshold = float(env_val)
            except ValueError:
                print(f"Invalid {THRESHOLD_ENV_VAR} value: {env_val}. Using default {THRESHOLD_ACCURACY}.")
                threshold = THRESHOLD_ACCURACY
        else:
            threshold = THRESHOLD_ACCURACY

    try:
        accuracy, f1 = evaluate_model()
        print(f"Current model accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    except FileNotFoundError as e:
        print(e)
        print("Training model from scratch...")
        train_churn_model(PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH)
        return

    if accuracy < threshold:
        print(
            f"Accuracy {accuracy:.4f} is below threshold {threshold}. Retraining model..."
        )
        train_churn_model(PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH)
    else:
        print("Model performance is acceptable. No retraining required.")


if __name__ == '__main__':
    monitor_and_retrain()
