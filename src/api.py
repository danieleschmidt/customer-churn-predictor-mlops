from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys

# Add src to sys.path to allow direct import of preprocess_data if needed for preprocessing new data.
# This assumes the API will receive data that might need *some* preprocessing
# similar to what the model was trained on.
# For this iteration, the API will expect data that's ALMOST ready for the model,
# specifically, data that can be directly processed by the *ColumnTransformer*
# saved within the model pipeline, or data that is already fully preprocessed.

# A more robust API would involve loading the *exact* ColumnTransformer used during training
# and applying it to the raw input data. For now, we assume `churn_model.joblib`
# might be a Pipeline containing the preprocessor and the model, OR that the
# input data is already appropriately transformed. The `predict_churn.py` script
# assumed pre-transformed input. This API will also start with that assumption for simplicity.

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.joblib') 
# To load preprocess_data if the API were to handle raw feature transformation:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src') # Go up to project root, then src
# if SRC_DIR not in sys.path:
#    sys.path.append(os.path.dirname(SRC_DIR)) # Add project root to path
# try:
#    from preprocess_data import preprocess # This would require df_path or df
# except ImportError:
#    print("Warning: preprocess_data module not found. API will expect preprocessed data.")
#    preprocess = None


app = Flask(__name__)

# Load the model when the application starts
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. API will not work.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}. API will not work.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Assuming data is a dictionary where keys are feature names
        # matching the columns expected by the model (i.e., after one-hot encoding etc.)
        # This is similar to the assumption in src/predict_churn.py's make_prediction
        
        # It's crucial that the order of columns in the DataFrame matches training.
        # A robust way is to have the training script save the column list,
        # and the API load it to ensure correct DataFrame construction.
        # For now, we rely on the client sending data with correct feature names.
        # If the model is a scikit-learn Pipeline, it handles preprocessing internally.
        # If it's just the classifier, input must be fully preprocessed.
        # Our 'churn_model.joblib' is the best estimator from GridSearchCV, which is just the classifier.
        # Therefore, the input to this API endpoint MUST be preprocessed.
        
        # Example: a single instance prediction
        # The keys in 'data' should be the column names after all preprocessing.
        
        # To make this more robust, we would need to know the exact column order
        # from the training data (X_processed_df.columns).
        # Let's assume for now the client sends a dictionary that can be directly
        # converted to a DataFrame row.
        
        # A simple approach: create a DataFrame from the input dictionary.
        # This assumes the dictionary keys are the feature names in the correct order
        # or that pandas handles it if they are not. More safely, one should
        # have a list of expected feature names.
        
        # For now, this will likely fail if the input data is not *exactly*
        # what `model.predict()` expects (i.e., a DataFrame with specific columns
        # in a specific order, all numerically encoded).
        
        # A better approach for a production API:
        # 1. Define expected raw feature schema.
        # 2. Apply the *exact same preprocessing pipeline* used in training.
        # This means saving the `ColumnTransformer` (preprocessor) from `preprocess_data.py`
        # and loading it here.
        # For this iteration, we'll stick to the simpler (but less robust) assumption
        # that the input 'data' is already preprocessed.

        input_df = pd.DataFrame([data]) # Convert single dict to DataFrame row
        
        # Ensure all necessary columns are present, fill missing with 0 or default
        # This step is critical if the model is not a pipeline that handles preprocessing.
        # For now, we assume 'input_df' is correctly structured by the client.
        # If a saved list of training columns exists:
        # try:
        #     with open('models/training_columns.json', 'r') as f:
        #         training_columns = json.load(f)
        #     input_df = input_df.reindex(columns=training_columns, fill_value=0)
        # except FileNotFoundError:
        #     return jsonify({"error": "Training columns file not found. Cannot ensure input structure."}), 500


        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # Output formatting
        churn_prediction = int(prediction[0])
        churn_probability = float(probability[0, 1]) # Probability of Churn=Yes

        return jsonify({
            "prediction": churn_prediction, # 0 or 1
            "probability_churn": churn_probability 
        })

    except Exception as e:
        # Log the exception e
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({"status": "ok", "message": "Model loaded."}), 200
    else:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500

if __name__ == '__main__':
    # Determine port and host
    port = int(os.environ.get("PORT", 5000)) # Default to 5000 if PORT not set
    # Run the app
    # Accessible externally if Docker maps port or if run on a host with 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=True) # Set debug=False for production
