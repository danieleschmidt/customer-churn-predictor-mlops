import joblib
import pandas as pd
import os
import json
import mlflow

from .constants import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    RUN_ID_PATH,
    RUN_ID_ENV_VAR,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
)

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


def _get_run_id():
    """Return the MLflow run ID from env var or file if available."""
    run_id = os.environ.get(RUN_ID_ENV_VAR)
    if run_id:
        return run_id
    if os.path.exists(RUN_ID_PATH):
        with open(RUN_ID_PATH) as f:
            return f.read().strip()
    return None


def make_prediction(input_data_dict, run_id=None):
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
    model = None
    preprocessor = None
    if run_id is None:
        run_id = _get_run_id()
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    elif run_id:
        try:
            print(f"Downloading model from MLflow run {run_id}...")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            print(f"Saved downloaded model to {MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading model from MLflow: {e}")
            return None, None
    else:
        print(
            f"Error: Model not found at {MODEL_PATH} and no run ID available via {RUN_ID_PATH} or {RUN_ID_ENV_VAR}"
        )
        return None, None

    # Load preprocessing pipeline if available
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
    elif run_id:
        try:
            print(f"Downloading preprocessor from MLflow run {run_id}...")
            dl_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="preprocessor.joblib"
            )
            path = (
                os.path.join(dl_path, "preprocessor.joblib")
                if os.path.isdir(dl_path)
                else dl_path
            )
            preprocessor = joblib.load(path)
            os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
        except Exception as e:
            print(f"Error downloading preprocessor from MLflow: {e}")

    columns = None
    if os.path.exists(FEATURE_COLUMNS_PATH):
        try:
            with open(FEATURE_COLUMNS_PATH) as f:
                columns = json.load(f)
        except Exception as e:
            print(f"Error loading feature columns: {e}")
            columns = None

    if columns is None and run_id:
        try:
            dl_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_columns.json"
            )
            if os.path.isdir(dl_path):
                path = os.path.join(dl_path, "feature_columns.json")
            else:
                path = dl_path
            with open(path) as f:
                columns = json.load(f)
            # Persist for future predictions
            os.makedirs(os.path.dirname(FEATURE_COLUMNS_PATH), exist_ok=True)
            with open(FEATURE_COLUMNS_PATH, "w") as out_f:
                json.dump(columns, out_f)
        except Exception as e:
            print(f"Error downloading feature columns from MLflow: {e}")
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
                os.makedirs(os.path.dirname(FEATURE_COLUMNS_PATH), exist_ok=True)
                with open(FEATURE_COLUMNS_PATH, "w") as f:
                    json.dump(columns, f)

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

        print(f"Making prediction for input: {input_data_dict}")
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # probability is an array of shape (n_samples, n_classes).
        # For binary classification, this is typically [[prob_0, prob_1]].
        # We want the probability of the positive class (churn=1).
        churn_probability = probability[0, 1]

        print(f"Raw prediction: {prediction}")
        print(f"Raw probability: {probability}")
        print(
            f"Predicted class: {prediction[0]}, Churn probability: {churn_probability:.4f}"
        )

        return int(prediction[0]), float(churn_probability)

    except Exception as e:
        print(f"Error during prediction: {e}")
        # This could be due to mismatched feature names/order, or unexpected data types.
        # Example: if input_data_dict doesn't have all columns model expects based on training.
        return None, None


if __name__ == "__main__":
    # Example usage:
    # This example assumes you know the *exact* feature names after one-hot encoding.
    # In a real scenario, these feature names would come from your preprocessing step.
    # You would need to run preprocessing, get the column names from X_processed_df,
    # and then construct a similar dictionary.

    print("Starting prediction script directly (for testing purposes)...")
    if not os.path.exists(MODEL_PATH):
        print(
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
                X_processed_df = pd.read_csv(processed_features_path)
                sample_feature_names = X_processed_df.columns.tolist()

                # Create a dummy input dict using the first row of processed data
                # or by creating a dict with all zeros.
                if not X_processed_df.empty:
                    example_input_dict = X_processed_df.iloc[0].to_dict()
                    print(
                        f"Using features from the first row of {processed_features_path} for example prediction."
                    )
                else:
                    print(
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

                print(
                    f"Example input data (first 5 items): {dict(list(example_input_dict.items())[:5])}..."
                )
                prediction, probability = make_prediction(example_input_dict)

                if prediction is not None:
                    print("\nExample Prediction successful:")
                    print(
                        f"Predicted Churn: {'Yes' if prediction == 1 else 'No'} (Class: {prediction})"
                    )
                    print(f"Probability of Churn: {probability:.4f}")
                else:
                    print("\nExample Prediction failed. Check logs for errors.")
            except Exception as e:
                print(
                    f"Could not create example input for testing predict_churn.py: {e}"
                )
                print(
                    "This might be because processed_features.csv is not available or is not in the expected format."
                )
                print("Please run scripts/run_preprocessing.py first.")

        else:
            print(
                f"Cannot run example: Processed features file not found at {processed_features_path}."
            )
            print("Please run scripts/run_preprocessing.py to generate it.")
