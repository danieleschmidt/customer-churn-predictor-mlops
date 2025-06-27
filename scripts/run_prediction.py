import argparse
import os
import pandas as pd

from src.predict_churn import make_prediction
from src.config import load_config


def run_predictions(input_csv: str, output_csv: str, run_id: str | None = None):
    """Run churn predictions on a CSV of processed features."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    predictions = []
    probabilities = []

    for _, row in df.iterrows():
        pred, prob = make_prediction(row.to_dict(), run_id=run_id)
        predictions.append(pred)
        probabilities.append(prob)

    df["prediction"] = predictions
    df["probability"] = probabilities
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run churn predictions on a CSV file of processed features."
    )
    cfg = load_config()
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=cfg["data"]["processed_features"],
        help="Path to processed features CSV",
    )
    parser.add_argument(
        "--output_csv",
        default="predictions.csv",
        help="Where to save predictions",
    )
    parser.add_argument("--run_id", help="MLflow run ID to download artifacts")
    args = parser.parse_args()
    run_predictions(args.input_csv, args.output_csv, run_id=args.run_id)


if __name__ == "__main__":
    main()
