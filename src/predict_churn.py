import joblib
import pandas as pd
import os
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from .validation import safe_read_csv, safe_write_json, safe_read_json, safe_write_text, safe_read_text, ValidationError
from .mlflow_utils import (
    download_model_from_mlflow,
    download_preprocessor_from_mlflow,
    download_feature_columns_from_mlflow,
    MLflowError,
    is_mlflow_available
)

from .constants import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    RUN_ID_PATH,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
)
from .logging_config import get_logger
from .env_config import env_config
from .model_cache import cached_load_model, cached_load_preprocessor, cached_load_metadata, get_cache_stats
from .metrics import record_prediction_latency, record_prediction_count, record_model_accuracy, request_tracker

logger = get_logger(__name__)

# Assume the model and preprocessor/column info might be needed.
# For this prompt, we only load the model.
# However, a robust prediction script would need to ensure the input_data_dict
# is transformed into the exact same format as the data used for training.
# This includes one-hot encoding, scaling, column order, etc.
# For simplicity, this version assumes input_data_dict is *already*
# in the post-processed, numerical format expected by the model.
# A future improvement would be to save and load the preprocessor pipeline.


# It's good practice to also save the columns used during training
# to ensure the input for prediction matches.
# This could be saved as a list in a .json file or similar.
# For now, we'll assume this step was done during training or is handled
# by the caller providing data in the correct one-hot encoded format.


def _get_run_id() -> Optional[str]:
    """Return the MLflow run ID from validated env config or file if available."""
    run_id = env_config.mlflow_run_id
    if run_id:
        return run_id
    if os.path.exists(RUN_ID_PATH):
        try:
            return safe_read_text(RUN_ID_PATH).strip()
        except ValidationError as e:
            logger.warning(f"Failed to read run ID safely: {e}")
            return None
    return None


def make_batch_predictions(input_df: pd.DataFrame, run_id: Optional[str] = None) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """
    Loads the trained model and makes churn predictions for a batch of customers.
    
    This function is optimized for performance when processing multiple predictions
    by loading the model once and applying vectorized operations.
    
    Args:
        input_df (pd.DataFrame): A DataFrame where each row represents a customer's
            features. If a fitted preprocessor is available it will be applied;
            otherwise the DataFrame must already be in the processed format.
        run_id (str, optional): MLflow run ID to download artifacts from.
    
    Returns:
        tuple: (predictions, probabilities) where each is a list of the same length
               as the input DataFrame. Returns (None, None) if model not found.
    """
    if input_df.empty:
        return [], []
    
    # Track this as a batch request
    with request_tracker("batch_prediction"):
        prediction_start_time = time.time()
        
        model: Optional[Any] = None
        preprocessor: Optional[Any] = None
        if run_id is None:
            run_id = _get_run_id()
        
        # Load model with caching
        try:
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading model from {MODEL_PATH} (with caching)...")
                model = cached_load_model(Path(MODEL_PATH), joblib.load, run_id=run_id)
            elif run_id and is_mlflow_available():
                # Download and cache model from MLflow
                downloaded_model = download_model_from_mlflow(run_id)
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                joblib.dump(downloaded_model, MODEL_PATH)
                logger.info(f"Downloaded and saved model to {MODEL_PATH}")
                model = cached_load_model(Path(MODEL_PATH), joblib.load, run_id=run_id)
            else:
                logger.error(f"Error: Model not found at {MODEL_PATH} and no run ID available")
                record_prediction_count(len(input_df), "error", "batch")
                return None, None
        except (FileNotFoundError, EOFError, OSError, RuntimeError, MLflowError) as e:
            logger.error(f"Error loading model: {e}")
            record_prediction_count(len(input_df), "error", "batch")
            return None, None
        
        # Load preprocessor with caching if available
        try:
            if os.path.exists(PREPROCESSOR_PATH):
                preprocessor = cached_load_preprocessor(Path(PREPROCESSOR_PATH), joblib.load, run_id=run_id)
                logger.debug("Loaded preprocessor with caching")
            elif run_id and is_mlflow_available():
                preprocessor_path = download_preprocessor_from_mlflow(run_id)
                preprocessor = joblib.load(preprocessor_path)
                os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
                joblib.dump(preprocessor, PREPROCESSOR_PATH)
                # Now cache the saved preprocessor
                preprocessor = cached_load_preprocessor(Path(PREPROCESSOR_PATH), joblib.load, run_id=run_id)
                logger.info("Downloaded, saved, and cached preprocessor")
        except (MLflowError, OSError, FileNotFoundError, RuntimeError) as e:
            logger.error(f"Error loading preprocessor: {e}")
            preprocessor = None
        
        # Load feature columns with caching
        columns: Optional[List[str]] = None
        try:
            if os.path.exists(FEATURE_COLUMNS_PATH):
                columns = cached_load_metadata(Path(FEATURE_COLUMNS_PATH), safe_read_json, run_id=run_id)
                logger.debug("Loaded feature columns with caching")
            elif run_id and is_mlflow_available():
                columns = download_feature_columns_from_mlflow(run_id)
                safe_write_json(columns, FEATURE_COLUMNS_PATH)
                # Cache the saved columns
                columns = cached_load_metadata(Path(FEATURE_COLUMNS_PATH), safe_read_json, run_id=run_id)
                logger.info("Downloaded, saved, and cached feature columns")
        except (MLflowError, ValidationError) as e:
            logger.error(f"Error loading feature columns: {e}")
            columns = None
        
        try:
            # Prepare the DataFrame for prediction
            if preprocessor is not None:
                if columns is None:
                    columns = list(preprocessor.get_feature_names_out())
                    safe_write_json(columns, FEATURE_COLUMNS_PATH)
                
                # Check if input is raw or processed data
                raw_features = set(getattr(preprocessor, "feature_names_in_", []))
                if raw_features and set(input_df.columns).issubset(raw_features):
                    # Apply preprocessing to the entire DataFrame at once
                    X_proc = preprocessor.transform(input_df)
                    processed_df = pd.DataFrame(X_proc, columns=columns, index=input_df.index)
                else:
                    # Data is already processed, just align columns
                    processed_df = input_df.reindex(columns=columns, fill_value=0)
            elif columns:
                # Align DataFrame columns with expected feature order
                processed_df = input_df.reindex(columns=columns, fill_value=0)
            else:
                # Use DataFrame as-is
                processed_df = input_df
            
            logger.info(f"Making batch predictions for {len(processed_df)} samples...")
            
            # Vectorized prediction - much faster than iterating
            predictions = model.predict(processed_df)
            probabilities = model.predict_proba(processed_df)
            
            # Extract probabilities for positive class (churn=1)
            churn_probabilities = probabilities[:, 1]
            
            logger.info(f"Completed batch predictions for {len(predictions)} samples")
            
            # Record successful prediction metrics
            record_prediction_latency(prediction_start_time, "batch")
            record_prediction_count(len(predictions), "success", "batch")
            
            return predictions.tolist(), churn_probabilities.tolist()
            
        except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
            logger.error(f"Error during batch prediction: {e}")
            # Record error metrics
            record_prediction_count(len(input_df) if not input_df.empty else 0, "error", "batch")
            return None, None


def make_prediction(input_data_dict: Dict[str, Any], run_id: Optional[str] = None) -> Tuple[Optional[int], Optional[float]]:
    """
    Loads the trained model and makes a churn prediction.

    Args:
        input_data_dict (dict): A dictionary representing a single customer's
            features. If a fitted preprocessor is available it will be applied
            to this raw dictionary; otherwise the dict must already be in the
            processed (one-hot encoded) format expected by the model.
        run_id (str, optional): MLflow run ID to download artifacts from.
            Defaults to None, in which case the value is loaded from
            ``MLFLOW_RUN_ID`` or ``models/mlflow_run_id.txt``.

    Returns:
        tuple: (prediction (0 or 1), probability (float))
               Returns (None, None) if model not found or error occurs.

    Notes:
        If ``models/churn_model.joblib`` or ``models/feature_columns.json`` are
        missing, this function will attempt to download them from MLflow using
        the provided ``run_id``. If ``run_id`` is ``None``, the value is
        resolved from the ``MLFLOW_RUN_ID`` environment variable or the
        ``models/mlflow_run_id.txt`` file.
    """
    model: Optional[Any] = None
    preprocessor: Optional[Any] = None
    if run_id is None:
        run_id = _get_run_id()
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
        except (FileNotFoundError, EOFError, OSError, RuntimeError) as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    elif run_id and is_mlflow_available():
        try:
            model = download_model_from_mlflow(run_id)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            logger.info(f"Saved downloaded model to {MODEL_PATH}")
        except MLflowError as e:
            logger.error(f"Error downloading model from MLflow: {e}")
            return None, None
    else:
        logger.error(
            f"Model not found at {MODEL_PATH} and no run ID available via {RUN_ID_PATH} or environment"
        )
        return None, None

    # Load preprocessing pipeline if available
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        except (FileNotFoundError, EOFError, OSError, RuntimeError) as e:
            logger.error(f"Error loading preprocessor: {e}")
    elif run_id and is_mlflow_available():
        try:
            preprocessor_path = download_preprocessor_from_mlflow(run_id)
            preprocessor = joblib.load(preprocessor_path)
            os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
        except (MLflowError, OSError, FileNotFoundError, RuntimeError) as e:
            logger.error(f"Error downloading preprocessor from MLflow: {e}")

    columns: Optional[List[str]] = None
    if os.path.exists(FEATURE_COLUMNS_PATH):
        try:
            with open(FEATURE_COLUMNS_PATH) as f:
                columns = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading feature columns: {e}")
            columns = None

    if columns is None and run_id and is_mlflow_available():
        try:
            columns = download_feature_columns_from_mlflow(run_id)
            # Persist for future predictions
            safe_write_json(columns, FEATURE_COLUMNS_PATH)
        except (MLflowError, ValidationError) as e:
            logger.error(f"Error downloading feature columns from MLflow: {e}")
            columns = None

    try:
        # Convert the input dictionary to a DataFrame.
        # The model expects a 2D array-like structure (e.g., a DataFrame or NumPy array).
        # The order of columns in this DataFrame MUST match the order of features
        # the model was trained on.
        # For this example, we'll rely on the dictionary being ordered correctly,
        # or more robustly, we'd use a list of training columns.

        # Create a single-row DataFrame.
        # Important: The column order must match the training data.
        # A more robust solution would involve saving and loading the column order
        # from the training phase. For this example, we'll assume the dict provides this.
        # If input_data_dict keys don't match model's expected features, it will fail.

        # To make this more robust, we should ideally have the list of feature names
        # that the model was trained on. Let's assume for now that the input_data_dict
        # will have all necessary features. A production system would need to load
        # the `ColumnTransformer` used in preprocessing or at least the list of
        # `X_processed_df.columns` from `preprocess_data.py`.

        if preprocessor is not None:
            if columns is None:
                columns = list(preprocessor.get_feature_names_out())
                safe_write_json(columns, FEATURE_COLUMNS_PATH)

            # Determine whether input_dict is raw or already processed
            raw_features = set(getattr(preprocessor, "feature_names_in_", []))
            if set(input_data_dict.keys()).issubset(raw_features):
                raw_df = pd.DataFrame([input_data_dict])
                X_proc = preprocessor.transform(raw_df)
                input_df = pd.DataFrame(X_proc, columns=columns)
            else:
                input_df = pd.DataFrame([input_data_dict], columns=columns).fillna(0)
        elif columns:
            input_df = pd.DataFrame([input_data_dict], columns=columns).fillna(0)
        else:
            input_df = pd.DataFrame([input_data_dict])

        # A more robust way, if you saved `X_processed_df.columns` during/after preprocessing:
        # Load `X_train_columns.json` (example name)
        # X_train_cols = load_json('X_train_columns.json')
        # input_df = pd.DataFrame(columns=X_train_cols)
        # input_df = input_df.append(input_data_dict, ignore_index=True).fillna(0) # Fill missing one-hot encoded features with 0

        logger.info(f"Making prediction for input: {input_data_dict}")
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # probability is an array of shape (n_samples, n_classes).
        # For binary classification, this is typically [[prob_0, prob_1]].
        # We want the probability of the positive class (churn=1).
        churn_probability = probability[0, 1]

        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Raw probability: {probability}")
        logger.info(
            f"Predicted class: {prediction[0]}, Churn probability: {churn_probability:.4f}"
        )

        return int(prediction[0]), float(churn_probability)

    except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
        logger.error(f"Error during prediction: {e}")
        # This could be due to mismatched feature names/order, or unexpected data types.
        # Example: if input_data_dict doesn't have all columns model expects based on training.
        return None, None


if __name__ == "__main__":
    # Example usage:
    # This example assumes you know the *exact* feature names after one-hot encoding.
    # In a real scenario, these feature names would come from your preprocessing step.
    # You would need to run preprocessing, get the column names from X_processed_df,
    # and then construct a similar dictionary.

    logger.info("Starting prediction script directly (for testing purposes)...")
    if not os.path.exists(MODEL_PATH):
        logger.error(
            f"Model {MODEL_PATH} not found. Please train the model first using scripts/run_training.py."
        )
    else:
        # Create a sample input dictionary.
        # IMPORTANT: This is a placeholder. The actual features and their names
        # depend on the one-hot encoding of your *specific* training data.
        # You would need to inspect the columns of 'processed_features.csv'
        # from the preprocessing step to create a valid example.

        # Let's try to load the columns from the processed_features.csv to make this example more realistic
        processed_features_path = "data/processed/processed_features.csv"
        if os.path.exists(processed_features_path):
            try:
                try:
                    X_processed_df = safe_read_csv(processed_features_path)
                except ValidationError as e:
                    logger.warning(f"Failed to read processed features safely: {e}")
                    X_processed_df = pd.DataFrame()  # Empty DataFrame as fallback
                sample_feature_names = X_processed_df.columns.tolist()

                # Create a dummy input dict using the first row of processed data
                # or by creating a dict with all zeros.
                if not X_processed_df.empty:
                    example_input_dict = X_processed_df.iloc[0].to_dict()
                    logger.info(
                        f"Using features from the first row of {processed_features_path} for example prediction."
                    )
                else:
                    logger.info(
                        f"{processed_features_path} is empty. Creating a zeroed-out example dict."
                    )
                    example_input_dict = {col: 0 for col in sample_feature_names}

                # Modify a few values for demonstration if needed
                # Example: if 'TotalCharges' is a feature and you want to set a specific value
                if (
                    "TotalCharges" in example_input_dict
                ):  # This name might be different if it was scaled/transformed
                    example_input_dict["TotalCharges"] = 2000
                # Example: if 'tenure' is a feature
                if "tenure" in example_input_dict:
                    example_input_dict["tenure"] = 24

                logger.info(
                    f"Example input data (first 5 items): {dict(list(example_input_dict.items())[:5])}..."
                )
                prediction, probability = make_prediction(example_input_dict)

                if prediction is not None:
                    logger.info("\nExample Prediction successful:")
                    logger.info(
                        f"Predicted Churn: {'Yes' if prediction == 1 else 'No'} (Class: {prediction})"
                    )
                    logger.info(f"Probability of Churn: {probability:.4f}")
                else:
                    logger.error("\nExample Prediction failed. Check logs for errors.")
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IndexError) as e:
                logger.error(
                    f"Could not create example input for testing predict_churn.py: {e}"
                )
                logger.error(
                    "This might be because processed_features.csv is not available or is not in the expected format."
                )
                logger.info("Please run scripts/run_preprocessing.py first.")

        else:
            logger.error(
                f"Cannot run example: Processed features file not found at {processed_features_path}."
            )
            logger.info("Please run scripts/run_preprocessing.py to generate it.")
