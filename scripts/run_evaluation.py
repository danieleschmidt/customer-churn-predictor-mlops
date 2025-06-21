import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitor_performance import evaluate_model


def run_evaluation(model_path='models/churn_model.joblib',
                   X_path='data/processed/processed_features.csv',
                   y_path='data/processed/processed_target.csv',
                   output=None):
    """Evaluate the model and optionally save metrics to a JSON file."""
    accuracy, f1 = evaluate_model(model_path, X_path, y_path)
    if output:
        metrics = {'accuracy': accuracy, 'f1_score': f1}
        with open(output, 'w') as f:
            json.dump(metrics, f)
        print(f"Saved metrics to {output}")
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument('--model_path', default='models/churn_model.joblib', help='Path to model file')
    parser.add_argument('--X_path', default='data/processed/processed_features.csv', help='Processed features CSV')
    parser.add_argument('--y_path', default='data/processed/processed_target.csv', help='Processed target CSV')
    parser.add_argument('--output', help='Optional JSON file to store metrics')
    args = parser.parse_args()

    accuracy, f1 = run_evaluation(args.model_path, args.X_path, args.y_path, args.output)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == '__main__':
    main()
