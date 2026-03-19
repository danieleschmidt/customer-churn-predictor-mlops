#!/usr/bin/env python3
"""
Batch prediction script.

Loads a saved pipeline and produces churn probability scores for new
customer records. Writes a CSV with columns:
  customerID, churn_prob, churn_pred

Usage:
    python predict.py --input data/new_customers.csv [--model models/gbm_pipeline.pkl]
    python predict.py --demo   # generate 10 synthetic records and score them
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from churn.features import DROP_COLS, TARGET_COL


def predict(input_path: str, model_path: str, output_path: str | None = None) -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_path)

    # Preserve customerID for output
    customer_ids = df.get("customerID", pd.Series(range(len(df)), name="customerID"))

    # Drop columns the pipeline doesn't need
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")

    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)

    results = pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_prob": probs.round(4),
            "churn_pred": ["Yes" if p == 1 else "No" for p in preds],
        }
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"Predictions written to {out}")

    return results


def _demo() -> None:
    """Score 10 synthetic records using the GBM model (if available)."""
    model_path = "models/gbm_pipeline.pkl"
    if not Path(model_path).exists():
        print("No trained model found. Run  python train.py  first.")
        sys.exit(1)

    from churn.generate_data import generate

    df = generate(n_samples=10, random_state=99)
    tmp = Path("data/demo_input.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)

    results = predict(str(tmp), model_path)
    print("\nDemo predictions (10 synthetic customers):\n")
    print(results.to_string(index=False))
    actual = df["Churn"].values
    print("\nActual labels:", actual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--input", help="Path to input CSV")
    parser.add_argument("--model", default="models/gbm_pipeline.pkl")
    parser.add_argument("--output", help="Path to output CSV (optional)")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        _demo()
    elif args.input:
        results = predict(args.input, args.model, args.output)
        print(results.to_string(index=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
