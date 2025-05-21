import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

def train_churn_model(X_path, y_path):
    """
    Trains a churn prediction model, logs it with MLflow, and saves it.

    Args:
        X_path (str): Path to the processed features CSV file.
        y_path (str): Path to the processed target CSV file.
    """
    print(f"Loading data from {X_path} and {y_path}...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze() # Use squeeze() to convert DataFrame column to Series

    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Initialize and train Logistic Regression model
        print("Training Logistic Regression model...")
        model = LogisticRegression(solver='liblinear', C=1.0, random_state=42) # Example parameters
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        print(f"Test Set Accuracy: {accuracy:.4f}")
        print(f"Test Set F1-score: {f1:.4f}")

        # Log parameters
        print("Logging model parameters to MLflow...")
        mlflow.log_param("solver", model.get_params()['solver'])
        mlflow.log_param("C", model.get_params()['C'])
        mlflow.log_param("random_state", model.get_params()['random_state'])

        # Log metrics
        print("Logging model metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "churn_model")

        # Save the trained model
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'churn_model.joblib')
        print(f"Saving trained model to {model_path}...")
        joblib.dump(model, model_path)
        
        print("Model training and MLflow logging complete.")
        return model_path, run_id

if __name__ == '__main__':
    # This part is for direct script execution testing, if needed.
    # Typically, this would be called by run_training.py
    print("Starting model training directly (for testing purposes)...")
    # Define default paths for X and y, assuming they are in data/processed
    default_X_path = 'data/processed/processed_features.csv'
    default_y_path = 'data/processed/processed_target.csv'

    if not (os.path.exists(default_X_path) and os.path.exists(default_y_path)):
        print(f"Error: Processed data not found at {default_X_path} or {default_y_path}.")
        print("Please run the preprocessing script first (e.g., scripts/run_preprocessing.py).")
    else:
        train_churn_model(default_X_path, default_y_path)
