import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitor_performance import evaluate_model
from src.constants import PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH, MODEL_PATH


def run_evaluation(
    model_path=MODEL_PATH,
    X_path=PROCESSED_FEATURES_PATH,
    y_path=PROCESSED_TARGET_PATH,
    output=None,
    run_id=None,
):
    """Evaluate the model and optionally save metrics to a JSON file."""
    accuracy, f1 = evaluate_model(model_path, X_path, y_path, run_id=run_id)
    if output:
        metrics = {'accuracy': accuracy, 'f1_score': f1}
        with open(output, 'w') as f:
            json.dump(metrics, f)
        print(f"Saved metrics to {output}")
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument('--model_path', default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--X_path', default=PROCESSED_FEATURES_PATH, help='Processed features CSV')
    parser.add_argument('--y_path', default=PROCESSED_TARGET_PATH, help='Processed target CSV')
    parser.add_argument('--output', help='Optional JSON file to store metrics')
    parser.add_argument('--run_id', help='MLflow run ID to download artifacts')
    args = parser.parse_args()

    accuracy, f1 = run_evaluation(
        args.model_path, args.X_path, args.y_path, args.output, run_id=args.run_id
    )
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == '__main__':
    main()
