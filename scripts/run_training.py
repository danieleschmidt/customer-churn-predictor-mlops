import argparse
import os

from src.train_model import train_churn_model  # Ensure this import works
from src.config import load_config
from src.logging_config import get_logger
from src.validation import DEFAULT_PATH_VALIDATOR, DEFAULT_ML_VALIDATOR, ValidationError

logger = get_logger(__name__)


def run_training(
    X_path: str | None = None,
    y_path: str | None = None,
    *,
    solver: str = "liblinear",
    C: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
):
    """Train the churn model using the provided processed datasets.

    Parameters
    ----------
    X_path, y_path : str
        Paths to the processed feature and target CSV files.
    solver : str, optional
        Logistic regression solver, by default ``"liblinear"``.
    C : float, optional
        Inverse of regularization strength, by default ``1.0``.
    penalty : str, optional
        Penalty (regularization) to use. Defaults to ``"l2"``.
    random_state : int, optional
        Random seed for reproducibility, by default ``42``.
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
    cfg = load_config()
    X_path = X_path or cfg["data"]["processed_features"]
    y_path = y_path or cfg["data"]["processed_target"]

    logger.info("Starting model training script...")
    
    try:
        # Validate input file paths
        DEFAULT_PATH_VALIDATOR.validate_path(X_path, must_exist=True)
        DEFAULT_PATH_VALIDATOR.validate_path(y_path, must_exist=True)
        
        # Validate hyperparameters
        hyperparams = {
            'solver': solver,
            'C': C,
            'penalty': penalty,
            'random_state': random_state,
            'max_iter': max_iter,
            'test_size': test_size
        }
        validated_params = DEFAULT_ML_VALIDATOR.validate_model_hyperparameters(hyperparams)
        logger.info(f"Validated hyperparameters: {validated_params}")
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        return None, None

    model_path, run_id = train_churn_model(
        X_path,
        y_path,
        solver=solver,
        C=C,
        penalty=penalty,
        random_state=random_state,
        max_iter=max_iter,
        test_size=test_size,
    )
    logger.info(f"Training complete. Model saved to: {model_path}")
    logger.info(f"MLflow Run ID: {run_id}")
    return model_path, run_id


def main():
    """
    Command-line interface for training the customer churn prediction model.
    
    This script provides a command-line interface for training a logistic
    regression model to predict customer churn. It accepts various hyperparameters
    and data paths as command-line arguments, with sensible defaults from the
    configuration file.
    
    The training process includes:
    - Loading processed feature and target datasets
    - Training a logistic regression model with specified parameters
    - Evaluating model performance on test data
    - Saving the trained model and logging to MLflow
    
    Command-line Arguments
    ----------------------
    --X_path : str
        Path to processed features CSV file (default from config)
    --y_path : str
        Path to processed target CSV file (default from config)
    --solver : str
        Logistic regression solver algorithm (default: 'liblinear')
    --C : float
        Inverse of regularization strength (default: 1.0)
    --penalty : str
        Regularization penalty type (default: 'l2')
    --random_state : int
        Random seed for reproducibility (default: 42)
    --max_iter : int
        Maximum number of training iterations (default: 100)
    --test_size : float
        Proportion of data for testing (default: 0.2)
    
    Examples
    --------
    Train with default parameters:
    $ python scripts/run_training.py
    
    Train with custom regularization:
    $ python scripts/run_training.py --C 0.5 --penalty l1
    
    Train with specific data paths:
    $ python scripts/run_training.py --X_path data/features.csv --y_path data/target.csv
    """
    parser = argparse.ArgumentParser(description="Train churn model")
    cfg = load_config()
    parser.add_argument(
        "--X_path",
        default=cfg["data"]["processed_features"],
        help="Path to processed features CSV",
    )
    parser.add_argument(
        "--y_path",
        default=cfg["data"]["processed_target"],
        help="Path to processed target CSV",
    )
    parser.add_argument(
        "--solver",
        default="liblinear",
        help="Logistic regression solver",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength",
    )
    parser.add_argument(
        "--penalty",
        default="l2",
        help="Penalty (regularization) type",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing",
    )
    args = parser.parse_args()

    run_training(
        args.X_path,
        args.y_path,
        solver=args.solver,
        C=args.C,
        penalty=args.penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
