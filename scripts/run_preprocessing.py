import os
import pandas as pd

from src.preprocess_data import preprocess
from src.constants import PREPROCESSOR_PATH
from src.config import load_config


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

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_target_path), exist_ok=True)

    # Preprocess data
    print(f"Loading raw data from {raw_data_path}...")
    X_processed, y_processed, preprocessor = preprocess(
        raw_data_path,
        return_preprocessor=True,
        save_preprocessor=True,
    )

    print(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")

    # Save processed data
    print(f"Saving processed features to {processed_features_path}...")
    X_processed.to_csv(processed_features_path, index=False)

    print(f"Saving processed target to {processed_target_path}...")
    pd.DataFrame(y_processed, columns=["Churn"]).to_csv(
        processed_target_path, index=False
    )

    print("Preprocessing complete.")
    return processed_features_path, processed_target_path


def main():
    run_preprocessing()


if __name__ == "__main__":
    main()
