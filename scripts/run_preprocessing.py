import os
import pandas as pd

from src.preprocess_data import preprocess
from src.constants import PREPROCESSOR_PATH
from src.config import load_config
from src.logging_config import get_logger
from src.path_config import get_path_config

logger = get_logger(__name__)


def run_preprocessing() -> tuple[str, str]:
    """Run preprocessing and save processed datasets.

    Returns
    -------
    tuple[str, str]
        Paths to the processed features and target CSV files.
    """
    cfg = load_config()
    raw_data_path = cfg["data"]["raw"]
    processed_features_path = cfg["data"]["processed_features"]
    processed_target_path = cfg["data"]["processed_target"]

    # Ensure all required directories exist
    path_config = get_path_config()
    path_config.ensure_directories()

    # Preprocess data
    logger.info(f"Loading raw data from {raw_data_path}...")
    X_processed, y_processed, preprocessor = preprocess(
        raw_data_path,
        return_preprocessor=True,
        save_preprocessor=True,
    )

    logger.info(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")

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
    run_preprocessing()


if __name__ == "__main__":
    main()
