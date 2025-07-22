import os
from typing import Optional
import pandas as pd

from src.preprocess_data import preprocess
from src.constants import get_preprocessor_path_constant
from src.config import load_config
from src.logging_config import get_logger
from src.path_config import PathConfig

logger = get_logger(__name__)


def run_preprocessing(path_config: Optional[PathConfig] = None) -> tuple[str, str]:
    """Run preprocessing and save processed datasets.

    Parameters
    ----------
    path_config : PathConfig, optional
        PathConfig instance for managing file paths. If None, creates
        a default configuration from environment variables if available.

    Returns
    -------
    tuple[str, str]
        Paths to the processed features and target CSV files.
    """
    # Create path config if not provided
    if path_config is None:
        path_config = PathConfig.from_environment()
    
    # Load configuration using the path config
    cfg = load_config(path_config=path_config)
    raw_data_path = cfg["data"]["raw"]
    processed_features_path = cfg["data"]["processed_features"]
    processed_target_path = cfg["data"]["processed_target"]
    
    # Get preprocessor path using the same config
    preprocessor_path = get_preprocessor_path_constant(config=path_config)

    # Ensure all required directories exist
    path_config.ensure_directories()

    # Preprocess data
    logger.info(f"Loading raw data from {raw_data_path}...")
    X_processed, y_processed, preprocessor = preprocess(
        raw_data_path,
        return_preprocessor=True,
        save_preprocessor=True,
    )

    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")

    # Save processed data
    logger.info(f"Saving processed features to {processed_features_path}...")
    X_processed.to_csv(processed_features_path, index=False)

    logger.info(f"Saving processed target to {processed_target_path}...")
    pd.DataFrame(y_processed, columns=["Churn"]).to_csv(
        processed_target_path, index=False
    )

    logger.info("Preprocessing complete.")
    return processed_features_path, processed_target_path


def main():
    """
    Command-line interface for preprocessing raw customer data.
    
    This script serves as the entry point for the data preprocessing pipeline.
    It loads raw customer data from the configured source, performs comprehensive
    data cleaning and feature engineering, and saves the processed datasets
    for model training.
    
    The preprocessing pipeline includes:
    - Loading raw customer data from CSV
    - Data cleaning and missing value handling
    - Feature engineering and encoding
    - Data splitting into training and test sets
    - Saving processed datasets to the configured output location
    
    No command-line arguments are required; all configuration is loaded
    from the application configuration file.
    
    Examples
    --------
    Run the preprocessing pipeline:
    $ python scripts/run_preprocessing.py
    
    See Also
    --------
    src.preprocess_data.preprocess : The core preprocessing function
    """
    run_preprocessing()


if __name__ == "__main__":
    main()
