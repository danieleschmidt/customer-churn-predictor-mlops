import os
import sys
import argparse
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess_data import preprocess
from scripts.run_training import run_training
from scripts.run_evaluation import run_evaluation
from src.constants import PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH


def run_pipeline(
    raw_path: str = 'data/raw/customer_data.csv',
    *,
    solver: str = 'liblinear',
    C: float = 1.0,
    penalty: str = 'l2',
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
    processed_features = PROCESSED_FEATURES_PATH
    processed_target = PROCESSED_TARGET_PATH

    os.makedirs(os.path.dirname(processed_features), exist_ok=True)
    X_proc, y_proc = preprocess(raw_path)
    X_proc.to_csv(processed_features, index=False)
    pd.DataFrame(y_proc, columns=['Churn']).to_csv(processed_target, index=False)

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
    accuracy, f1 = run_evaluation(model_path, processed_features, processed_target, run_id=run_id)
    print(f"Pipeline completed. Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Run full churn prediction pipeline")
    parser.add_argument('--raw_path', default='data/raw/customer_data.csv', help='Path to raw customer CSV')
    parser.add_argument('--solver', default='liblinear', help='Logistic regression solver')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
    parser.add_argument('--penalty', default='l2', help='Penalty (regularization) type')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for training')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset for testing')
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


if __name__ == '__main__':
    main()
