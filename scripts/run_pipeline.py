import os
import argparse
import pandas as pd

from src.preprocess_data import preprocess
from src.constants import PREPROCESSOR_PATH
from src.config import load_config
from scripts.run_training import run_training
from scripts.run_evaluation import run_evaluation
from src.logging_config import get_logger

logger = get_logger(__name__)


def run_pipeline(
    raw_path: str | None = None,
    *,
    solver: str = "liblinear",
    C: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
):
    """Run preprocessing, training, and evaluation as a single pipeline.

    Parameters
    ----------
    raw_path : str, optional
        Path to the raw customer CSV file. Defaults to ``'data/raw/customer_data.csv'``.
    solver : str, optional
        Solver for :class:`~sklearn.linear_model.LogisticRegression`. Defaults to ``"liblinear"``.
        C : float, optional
            Inverse of regularization strength. Defaults to ``1.0``.
        penalty : str, optional
            Penalty (regularization) type. Defaults to ``"l2"``.
        random_state : int, optional
            Random seed for reproducibility. Defaults to ``42``.
    max_iter : int, optional
        Maximum optimization iterations. Defaults to ``100``.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to ``0.2``.

    Returns
    -------
        tuple[float, float]
            The accuracy and F1-score of the trained model.
    """
    cfg = load_config()
    raw_path = raw_path or cfg["data"]["raw"]
    processed_features = cfg["data"]["processed_features"]
    processed_target = cfg["data"]["processed_target"]

    os.makedirs(os.path.dirname(processed_features), exist_ok=True)
    X_proc, y_proc, _ = preprocess(
        raw_path,
        return_preprocessor=True,
        save_preprocessor=True,
    )
    logger.info(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")
    X_proc.to_csv(processed_features, index=False)
    pd.DataFrame(y_proc, columns=["Churn"]).to_csv(processed_target, index=False)

    model_path, run_id = run_training(
        processed_features,
        processed_target,
        solver=solver,
        C=C,
        penalty=penalty,
        random_state=random_state,
        max_iter=max_iter,
        test_size=test_size,
    )
    accuracy, f1 = run_evaluation(
        model_path, processed_features, processed_target, run_id=run_id
    )
    logger.info(f"Pipeline completed. Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return accuracy, f1


def main():
    """
    Command-line interface for running the complete churn prediction pipeline.
    
    This script executes the entire machine learning pipeline from raw data
    to trained model in a single command. It sequentially runs preprocessing,
    model training, and evaluation steps with consistent parameters and
    data splits.
    
    This is equivalent to running the preprocess, train, and evaluate scripts
    in sequence, but ensures parameter consistency and eliminates the need
    for intermediate manual steps.
    
    Command-line Arguments
    ----------------------
    --raw_path : str
        Path to raw customer data CSV file (default from config)
    --solver : str
        Logistic regression solver algorithm (default: 'liblinear')
    --C : float
        Inverse of regularization strength (default: 1.0)
    --penalty : str
        Regularization penalty type (default: 'l2')
    --random_state : int
        Random seed for reproducible results (default: 42)
    --max_iter : int
        Maximum training iterations (default: 100)
    --test_size : float
        Proportion of data for testing (default: 0.2)
    
    Examples
    --------
    Run full pipeline with default settings:
    $ python scripts/run_pipeline.py
    
    Run with custom hyperparameters:
    $ python scripts/run_pipeline.py --C 0.5 --penalty l1 --max_iter 200
    
    Use custom data source:
    $ python scripts/run_pipeline.py --raw_path /path/to/custom_data.csv
    """
    parser = argparse.ArgumentParser(description="Run full churn prediction pipeline")
    cfg = load_config()
    parser.add_argument(
        "--raw_path", default=cfg["data"]["raw"], help="Path to raw customer CSV"
    )
    parser.add_argument(
        "--solver", default="liblinear", help="Logistic regression solver"
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="Inverse of regularization strength"
    )
    parser.add_argument("--penalty", default="l2", help="Penalty (regularization) type")
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for training"
    )
    parser.add_argument(
        "--max_iter", type=int, default=100, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of dataset for testing"
    )
    args = parser.parse_args()
    run_pipeline(
        args.raw_path,
        solver=args.solver,
        C=args.C,
        penalty=args.penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
