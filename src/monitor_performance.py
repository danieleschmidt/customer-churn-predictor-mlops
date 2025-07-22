import os
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from typing import Union, Tuple, Dict, Any, Optional
from .validation import safe_read_csv, ValidationError

from src.train_model import train_churn_model
from src.predict_churn import _get_run_id
from .constants import (
    MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    PROCESSED_FEATURES_PATH,
    PROCESSED_TARGET_PATH,
)
from src.logging_config import get_logger
from src.env_config import env_config

logger = get_logger(__name__)


def evaluate_model(
    model_path: str = MODEL_PATH,
    X_path: str = PROCESSED_FEATURES_PATH,
    y_path: str = PROCESSED_TARGET_PATH,
    run_id: Optional[str] = None,
    *,
    detailed: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """Evaluate the existing model on the processed dataset.

    Parameters
    ----------
    model_path, X_path, y_path : str
        Locations of the model and processed datasets.
    run_id : str, optional
        If provided, artifacts will be downloaded from MLflow when the model is
        missing locally.
    detailed : bool, optional
        If ``True`` return a classification report in addition to accuracy and
        F1-score. The report is also logged to MLflow as an artifact.
        Defaults to ``False``.
    """
    if run_id is None:
        run_id = _get_run_id()
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    elif run_id:
        try:
            logger.info(f"Downloading model from MLflow run {run_id}...")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            logger.info(f"Saved downloaded model to {model_path}")
        except (ImportError, OSError, RuntimeError) as e:
            logger.error(f"Error downloading model from MLflow: {e}")
            raise FileNotFoundError(f"Model not found at {model_path}") from e
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        X = safe_read_csv(X_path)
        y = safe_read_csv(y_path).squeeze()
    except ValidationError as e:
        raise ValueError(f"Failed to read input data safely: {e}") from e

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    report = None
    if detailed:
        report = classification_report(y, y_pred, output_dict=True)

    # Log evaluation metrics to MLflow for tracking
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        if detailed and report is not None:
            tmp_path = "classification_report.json"
            with open(tmp_path, "w") as f:
                json.dump(report, f)
            mlflow.log_artifact(tmp_path)
            os.remove(tmp_path)

    if detailed:
        return accuracy, f1, report
    return accuracy, f1


def monitor_and_retrain(
    threshold: Optional[float] = None,
    *,
    X_path: str = PROCESSED_FEATURES_PATH,
    y_path: str = PROCESSED_TARGET_PATH,
    solver: str = "liblinear",
    C: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
) -> None:
    """Monitor performance and retrain the model if accuracy falls below threshold.

    Parameters
    ----------
    threshold : float, optional
        Accuracy threshold below which the model will be retrained.
        Defaults to ``DEFAULT_THRESHOLD`` or ``CHURN_THRESHOLD`` environment
        variable.
    X_path, y_path : str, optional
        Locations of the processed feature and target datasets.
        Defaults to :data:`PROCESSED_FEATURES_PATH` and
        :data:`PROCESSED_TARGET_PATH`.
    solver, C, penalty, random_state, max_iter, test_size :
        Same as in :func:`~src.train_model.train_churn_model` and forwarded when
        retraining is triggered.
    """
    if threshold is None:
        # Use validated environment configuration
        threshold = env_config.churn_threshold
        logger.info(f"Using validated churn threshold: {threshold}")

    try:
        accuracy, f1 = evaluate_model(
            MODEL_PATH,
            X_path,
            y_path,
        )
        logger.info(f"Current model accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Training model from scratch...")
        train_churn_model(
            X_path,
            y_path,
            solver=solver,
            C=C,
            penalty=penalty,
            random_state=random_state,
            max_iter=max_iter,
            test_size=test_size,
        )
        return

    if accuracy < threshold:
        logger.warning(
            f"Accuracy {accuracy:.4f} is below threshold {threshold}. Retraining model..."
        )
        train_churn_model(
            X_path,
            y_path,
            solver=solver,
            C=C,
            penalty=penalty,
            random_state=random_state,
            max_iter=max_iter,
            test_size=test_size,
        )
    else:
        logger.info("Model performance is acceptable. No retraining required.")


if __name__ == '__main__':
    monitor_and_retrain()
