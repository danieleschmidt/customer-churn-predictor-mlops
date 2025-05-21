import joblib
import pandas as pd
import numpy as np
import os

# Assume the model and preprocessor/column info might be needed.
# For this prompt, we only load the model.
# However, a robust prediction script would need to ensure the input_data_dict
# is transformed into the exact same format as the data used for training.
# This includes one-hot encoding, scaling, column order, etc.
# For simplicity, this version assumes input_data_dict is *already*
# in the post-processed, numerical format expected by the model.
# A future improvement would be to save and load the preprocessor pipeline.

MODEL_PATH = 'models/churn_model.joblib'

# It's good practice to also save the columns used during training
# to ensure the input for prediction matches.
# This could be saved as a list in a .json file or similar.
# For now, we'll assume this step was done during training or is handled
# by the caller providing data in the correct one-hot encoded format.

def make_prediction(input_data_dict):
    """
    Loads the trained model and makes a churn prediction.

    Args:
        input_data_dict (dict): A dictionary representing a single customer's
                                features, in the pre-processed format expected
                                by the model (i.e., after one-hot encoding).
                                Example: {'gender_Female': 0, 'gender_Male': 1, ... , 'TotalCharges': 500}

    Returns:
        tuple: (prediction (0 or 1), probability (float))
               Returns (None, None) if model not found or error occurs.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None, None

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

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

        # For now, let's create a DataFrame directly from the dict.
        # This assumes input_data_dict has the correct features in a flat structure.
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
        print(f"Predicted class: {prediction[0]}, Churn probability: {churn_probability:.4f}")
        
        return int(prediction[0]), float(churn_probability)

    except Exception as e:
        print(f"Error during prediction: {e}")
        # This could be due to mismatched feature names/order, or unexpected data types.
        # Example: if input_data_dict doesn't have all columns model expects based on training.
        return None, None

if __name__ == '__main__':
    # Example usage:
    # This example assumes you know the *exact* feature names after one-hot encoding.
    # In a real scenario, these feature names would come from your preprocessing step.
    # You would need to run preprocessing, get the column names from X_processed_df,
    # and then construct a similar dictionary.
    
    print("Starting prediction script directly (for testing purposes)...")
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Please train the model first using scripts/run_training.py.")
    else:
        # Create a sample input dictionary.
        # IMPORTANT: This is a placeholder. The actual features and their names
        # depend on the one-hot encoding of your *specific* training data.
        # You would need to inspect the columns of 'processed_features.csv'
        # from the preprocessing step to create a valid example.
        
        # Let's try to load the columns from the processed_features.csv to make this example more realistic
        processed_features_path = 'data/processed/processed_features.csv'
        if os.path.exists(processed_features_path):
            try:
                X_processed_df = pd.read_csv(processed_features_path)
                sample_feature_names = X_processed_df.columns.tolist()
                
                # Create a dummy input dict using the first row of processed data
                # or by creating a dict with all zeros.
                if not X_processed_df.empty:
                    example_input_dict = X_processed_df.iloc[0].to_dict()
                    print(f"Using features from the first row of {processed_features_path} for example prediction.")
                else:
                    print(f"{processed_features_path} is empty. Creating a zeroed-out example dict.")
                    example_input_dict = {col: 0 for col in sample_feature_names}
                
                # Modify a few values for demonstration if needed
                # Example: if 'TotalCharges' is a feature and you want to set a specific value
                if 'TotalCharges' in example_input_dict: # This name might be different if it was scaled/transformed
                    example_input_dict['TotalCharges'] = 2000 
                # Example: if 'tenure' is a feature
                if 'tenure' in example_input_dict:
                     example_input_dict['tenure'] = 24


                print(f"Example input data (first 5 items): {dict(list(example_input_dict.items())[:5])}...")
                prediction, probability = make_prediction(example_input_dict)

                if prediction is not None:
                    print(f"\nExample Prediction successful:")
                    print(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'} (Class: {prediction})")
                    print(f"Probability of Churn: {probability:.4f}")
                else:
                    print("\nExample Prediction failed. Check logs for errors.")
            except Exception as e:
                print(f"Could not create example input for testing predict_churn.py: {e}")
                print("This might be because processed_features.csv is not available or is not in the expected format.")
                print("Please run scripts/run_preprocessing.py first.")

        else:
            print(f"Cannot run example: Processed features file not found at {processed_features_path}.")
            print("Please run scripts/run_preprocessing.py to generate it.")
            print("Falling back to a very basic example dict (likely to fail if features don't match model):")
            # This basic example is highly unlikely to match the actual model features
            # after one-hot encoding in a real scenario.
            basic_example_input = {
                'gender_Female': 1, 'gender_Male': 0, 'SeniorCitizen': 0, 'Partner_Yes': 1, 'Partner_No': 0,
                'Dependents_No': 1, 'Dependents_Yes': 0, 'tenure': 10, 'PhoneService_Yes': 1, 'PhoneService_No': 0,
                # ... many more one-hot encoded features would be here ...
                'MonthlyCharges': 70.0, 'TotalCharges': 700.0
            }
            # It's better to inform the user that this basic example is insufficient.
            print("The basic_example_input is a placeholder and likely does not match your model's required features.")
            print("For a meaningful test, ensure processed_features.csv exists or provide a dict with correct one-hot encoded column names.")

```
