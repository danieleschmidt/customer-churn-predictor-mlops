import argparse
import os
import sys
import pandas as pd

# Ensure src directory is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict_churn import make_prediction


def run_predictions(input_csv: str, output_csv: str):
    """Run churn predictions on a CSV of processed features."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    predictions = []
    probabilities = []

    for _, row in df.iterrows():
        pred, prob = make_prediction(row.to_dict())
        predictions.append(pred)
        probabilities.append(prob)

    df['prediction'] = predictions
    df['probability'] = probabilities
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run churn predictions on a CSV file of processed features.")
    parser.add_argument('input_csv', help='Path to processed features CSV')
    parser.add_argument('--output_csv', default='predictions.csv', help='Where to save predictions')
    args = parser.parse_args()
    run_predictions(args.input_csv, args.output_csv)


if __name__ == '__main__':
    main()
