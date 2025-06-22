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
):
    """Train the churn model using the provided processed datasets."""
    print("Starting model training script...")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"Error: Processed data not found at {X_path} or {y_path}.")
        print("Please ensure you have run the preprocessing script successfully.")
        print(
            "Expected files: processed_features.csv and processed_target.csv in data/processed/"
        )
        return None, None

    model_path, run_id = train_churn_model(X_path, y_path)
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
    args = parser.parse_args()

    run_training(args.X_path, args.y_path)

if __name__ == '__main__':
    main()
