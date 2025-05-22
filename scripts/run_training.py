import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import train_churn_model # Ensure this import works

def main():
    processed_features_path = 'data/processed/processed_features.csv'
    processed_target_path = 'data/processed/processed_target.csv'
    mlflow_experiment_name = "ChurnPredictionExperiment" # Can be configurable if needed

    print("Starting model training script (for multiple models)...")
    if not (os.path.exists(processed_features_path) and os.path.exists(processed_target_path)):
        print(f"Error: Processed data not found at {processed_features_path} or {processed_target_path}.")
        print("Please ensure you have run the preprocessing script successfully.")
        return

    # Call the updated training function
    # It now handles multiple models and returns the path of the best one.
    best_model_path = train_churn_model(
        X_path=processed_features_path,
        y_path=processed_target_path,
        experiment_name=mlflow_experiment_name
    )

    if best_model_path:
        print(f"Training script complete. Best overall model saved to: {best_model_path}")
        print(f"Individual models and the overall best (as churn_model.joblib) are in the 'models' directory.")
        print(f"MLflow experiment '{mlflow_experiment_name}' has been updated with runs for each model type.")
    else:
        print("Training script completed, but no best model path was returned. Check logs.")

if __name__ == '__main__':
    main()
