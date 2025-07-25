import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
import joblib
import os
import json
from typing import Tuple

from .constants import MODEL_PATH, FEATURE_COLUMNS_PATH, RUN_ID_PATH, MODEL_ARTIFACT_PATH
from src.logging_config import get_logger
from .validation import safe_read_csv, safe_write_json, safe_write_text, DEFAULT_PATH_VALIDATOR, DataValidator, ValidationError

logger = get_logger(__name__)

def train_churn_model(
    X_path: str,
    y_path: str,
    *,
    solver: str = "liblinear",
    C: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
) -> Tuple[str, str]:
    """Train a churn prediction model and log it with MLflow.

    Parameters
    ----------
    X_path : str
        Path to the processed features CSV file.
    y_path : str
        Path to the processed target CSV file.
    solver : str, optional
        Solver to use for :class:`~sklearn.linear_model.LogisticRegression`,
        by default ``"liblinear"``.
    C : float, optional
        Inverse of regularization strength, by default ``1.0``.
    penalty : str, optional
        Penalty (regularization) to use. Defaults to ``"l2"``.
    random_state : int, optional
        Random seed used for both the train/test split and the model, by
        default ``42``.
    max_iter : int, optional
        Maximum number of iterations for optimization, by default ``100``.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default
        ``0.2``.

    Returns
    -------
    tuple[str, str]
        The path to the saved model and the MLflow run ID.
    """
    logger.info(f"Loading data from {X_path} and {y_path}...")
    
    try:
        # Use safe CSV reading with validation
        X: pd.DataFrame = safe_read_csv(X_path, validator=DEFAULT_PATH_VALIDATOR)
        y_df: pd.DataFrame = safe_read_csv(y_path, validator=DEFAULT_PATH_VALIDATOR)
        y: pd.Series = y_df.squeeze()  # Convert DataFrame column to Series
        
        # Validate data structure
        DataValidator.validate_dataframe(X, min_rows=10)  # Ensure minimum training data
        if len(y) != len(X):
            raise ValidationError(f"Feature and target data length mismatch: {len(X)} vs {len(y)}")
            
    except ValidationError as e:
        logger.error(f"Data validation failed: {e}")
        raise

    # Split data
    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Initialize and train Logistic Regression model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(
        solver=solver,
        C=C,
        penalty=penalty,
        random_state=random_state,
        max_iter=max_iter,
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    logger.info(f"Test Set Accuracy: {accuracy:.4f}")
    logger.info(f"Test Set F1-score: {f1:.4f}")

    # Save the trained model
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    # Save feature column order for prediction
    safe_write_json(X.columns.tolist(), FEATURE_COLUMNS_PATH)

    run_id = None
    if MLFLOW_AVAILABLE:
        # Initialize MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            # Log parameters
            logger.info("Logging model parameters to MLflow...")
            mlflow.log_param("solver", model.get_params()['solver'])
            mlflow.log_param("C", model.get_params()['C'])
            mlflow.log_param("penalty", model.get_params()['penalty'])
            mlflow.log_param("random_state", model.get_params()['random_state'])
            mlflow.log_param("max_iter", model.get_params()['max_iter'])
            mlflow.log_param("test_size", test_size)

            # Log metrics
            logger.info("Logging model metrics to MLflow...")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # Log model
            logger.info("Logging model to MLflow...")
            mlflow.sklearn.log_model(model, MODEL_ARTIFACT_PATH)

            # Log feature columns artifact
            mlflow.log_artifact(FEATURE_COLUMNS_PATH)

            # Persist the MLflow run ID so prediction utilities can retrieve
            # artifacts later if needed.
            safe_write_text(run_id, RUN_ID_PATH)

            logger.info("Model training and MLflow logging complete.")
    else:
        logger.info("MLflow not available, skipping MLflow logging")
        run_id = "local_training"
        safe_write_text(run_id, RUN_ID_PATH)

    return MODEL_PATH, run_id

if __name__ == '__main__':
    # This part is for direct script execution testing, if needed.
    # Typically, this would be called by run_training.py
    logger.info("Starting model training directly (for testing purposes)...")
    # Define default paths for X and y, assuming they are in data/processed
    default_X_path = 'data/processed/processed_features.csv'
    default_y_path = 'data/processed/processed_target.csv'

    if not (os.path.exists(default_X_path) and os.path.exists(default_y_path)):
        logger.error(f"Error: Processed data not found at {default_X_path} or {default_y_path}.")
        logger.info("Please run the preprocessing script first (e.g., scripts/run_preprocessing.py).")
    else:
        train_churn_model(default_X_path, default_y_path)
