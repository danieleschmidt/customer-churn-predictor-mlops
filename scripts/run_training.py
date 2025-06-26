import argparse
import sys
import os

# Add src directory to Python path to allow direct import of train_model
# This is useful if running the script directly from the 'scripts' directory or root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import train_churn_model  # Ensure this import works
from src.constants import PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH


def run_training(
    X_path: str = PROCESSED_FEATURES_PATH,
    y_path: str = PROCESSED_TARGET_PATH,
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
    print("Starting model training script...")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"Error: Processed data not found at {X_path} or {y_path}.")
        print("Please ensure you have run the preprocessing script successfully.")
        print(
            "Expected files: processed_features.csv and processed_target.csv in data/processed/"
        )
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
    print(f"Training complete. Model saved to: {model_path}")
    print(f"MLflow Run ID: {run_id}")
    return model_path, run_id

def main():
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument(
        "--X_path",
        default=PROCESSED_FEATURES_PATH,
        help="Path to processed features CSV",
    )
    parser.add_argument(
        "--y_path",
        default=PROCESSED_TARGET_PATH,
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

if __name__ == '__main__':
    main()
