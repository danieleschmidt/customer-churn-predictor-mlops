import argparse
import os
import pandas as pd
from typing import Optional

from src.predict_churn import make_batch_predictions, make_prediction
from src.config import load_config
from src.logging_config import get_logger
from src.validation import safe_read_csv, safe_write_csv, ValidationError, DEFAULT_PATH_VALIDATOR

logger = get_logger(__name__)


def run_predictions(input_csv: str, output_csv: str, run_id: Optional[str] = None, batch_mode: bool = True):
    """
    Run churn predictions on a CSV of processed features.
    
    Args:
        input_csv: Path to input CSV file with features
        output_csv: Path to save predictions
        run_id: Optional MLflow run ID for model artifacts
        batch_mode: If True, use optimized batch predictions (default: True)
        
    Raises:
        ValidationError: If file paths or data validation fails
    """
    try:
        # Validate and read input CSV with security checks
        df = safe_read_csv(input_csv, validator=DEFAULT_PATH_VALIDATOR)
        
        # Validate output path
        DEFAULT_PATH_VALIDATOR.validate_path(output_csv, allow_create=True)
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
    
    if df.empty:
        # Handle empty DataFrame
        df["prediction"] = []
        df["probability"] = []
        safe_write_csv(df, output_csv, validator=DEFAULT_PATH_VALIDATOR)
        return
    
    if batch_mode:
        # Use optimized batch prediction (vectorized operations)
        logger.info(f"Processing {len(df)} predictions in batch mode...")
        predictions, probabilities = make_batch_predictions(df, run_id=run_id)
        
        if predictions is None or probabilities is None:
            logger.error("Batch prediction failed, falling back to row-by-row processing...")
            batch_mode = False
        else:
            df["prediction"] = predictions
            df["probability"] = probabilities
    
    if not batch_mode:
        # Fallback to row-by-row prediction (for error recovery or debugging)
        logger.info(f"Processing {len(df)} predictions row-by-row...")
        predictions = []
        probabilities = []

        for _, row in df.iterrows():
            pred, prob = make_prediction(row.to_dict(), run_id=run_id)
            predictions.append(pred)
            probabilities.append(prob)

        df["prediction"] = predictions
        df["probability"] = probabilities

    # Use safe CSV writing with validation
    safe_write_csv(df, output_csv, validator=DEFAULT_PATH_VALIDATOR)


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
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch mode (use row-by-row processing for debugging)"
    )
    args = parser.parse_args()
    run_predictions(
        args.input_csv, 
        args.output_csv, 
        run_id=args.run_id,
        batch_mode=not args.no_batch
    )


if __name__ == "__main__":
    main()
