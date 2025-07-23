import argparse
import os
import pandas as pd
from typing import Optional

from src.predict_churn import make_batch_predictions, make_prediction
from src.config import load_config
from src.logging_config import get_logger
from src.validation import safe_read_csv, safe_write_csv, ValidationError, DEFAULT_PATH_VALIDATOR

logger = get_logger(__name__)


def run_predictions(input_csv: str, output_csv: str, run_id: Optional[str] = None):
    """
    Run churn predictions on a CSV of processed features using optimized batch processing.
    
    Args:
        input_csv: Path to input CSV file with features
        output_csv: Path to save predictions
        run_id: Optional MLflow run ID for model artifacts
        
    Raises:
        ValidationError: If file paths or data validation fails
        RuntimeError: If batch prediction fails due to model or data issues
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
    
    # Use optimized batch prediction (vectorized operations only)
    logger.info(f"Processing {len(df)} predictions in batch mode...")
    predictions, probabilities = make_batch_predictions(df, run_id=run_id)
    
    if predictions is None or probabilities is None:
        logger.error("Batch prediction failed - model or data issues detected")
        raise RuntimeError("Batch prediction failed. Check model availability and data format. Use --run_id if model files are missing.")
    
    df["prediction"] = predictions
    df["probability"] = probabilities

    # Use safe CSV writing with validation
    safe_write_csv(df, output_csv, validator=DEFAULT_PATH_VALIDATOR)


def main():
    """
    Command-line interface for generating customer churn predictions.
    
    This script provides a command-line interface for making high-performance
    batch predictions on customer data using a trained churn prediction model.
    All predictions use optimized vectorized operations for maximum performance.
    
    The script loads a trained model and generates predictions for customers
    provided in a CSV file. Output includes both binary predictions (0/1)
    and churn probabilities for each customer.
    
    Command-line Arguments
    ----------------------
    input_csv : str
        Path to CSV file containing customer features for prediction.
        Defaults to processed features from config if not specified.
    --output_csv : str
        Path where prediction results will be saved (default: 'predictions.csv')
    --run_id : str
        MLflow run ID to download model artifacts from if model not found locally
    
    Examples
    --------
    Generate predictions with default settings:
    $ python scripts/run_prediction.py
    
    Predict on custom input file:
    $ python scripts/run_prediction.py customer_data.csv --output_csv results.csv
    
    Use specific MLflow model:
    $ python scripts/run_prediction.py --run_id abc123 --output_csv predictions.csv
    """
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
    run_predictions(
        args.input_csv, 
        args.output_csv, 
        run_id=args.run_id
    )


if __name__ == "__main__":
    main()
