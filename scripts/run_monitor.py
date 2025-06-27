import argparse

from src.monitor_performance import monitor_and_retrain
from src.config import load_config


def main():
    """Evaluate current model performance and retrain if necessary."""
    parser = argparse.ArgumentParser(
        description="Monitor model and retrain if accuracy drops"
    )
    cfg = load_config()
    parser.add_argument(
        "--threshold",
        type=float,
        help="Accuracy threshold below which to retrain. Overrides CHURN_THRESHOLD env var.",
    )
    parser.add_argument(
        "--X_path",
        default=cfg["data"]["processed_features"],
        help="Path to processed feature CSV",
    )
    parser.add_argument(
        "--y_path",
        default=cfg["data"]["processed_target"],
        help="Path to processed target CSV",
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

    monitor_and_retrain(
        threshold=args.threshold,
        X_path=args.X_path,
        y_path=args.y_path,
        solver=args.solver,
        C=args.C,
        penalty=args.penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
