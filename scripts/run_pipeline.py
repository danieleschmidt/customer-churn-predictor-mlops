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


def run_pipeline(raw_path: str = 'data/raw/customer_data.csv'):
    """Run preprocessing, training, and evaluation as a single pipeline."""
    processed_features = PROCESSED_FEATURES_PATH
    processed_target = PROCESSED_TARGET_PATH

    os.makedirs(os.path.dirname(processed_features), exist_ok=True)
    X_proc, y_proc = preprocess(raw_path)
    X_proc.to_csv(processed_features, index=False)
    pd.DataFrame(y_proc, columns=['Churn']).to_csv(processed_target, index=False)

    model_path, run_id = run_training(processed_features, processed_target)
    accuracy, f1 = run_evaluation(model_path, processed_features, processed_target, run_id=run_id)
    print(f"Pipeline completed. Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Run full churn prediction pipeline")
    parser.add_argument('--raw_path', default='data/raw/customer_data.csv', help='Path to raw customer CSV')
    args = parser.parse_args()
    run_pipeline(args.raw_path)


if __name__ == '__main__':
    main()
