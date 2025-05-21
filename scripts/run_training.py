import sys
import os

# Add src directory to Python path to allow direct import of train_model
# This is useful if running the script directly from the 'scripts' directory or root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import train_churn_model # Ensure this import works

def main():
    processed_features_path = 'data/processed/processed_features.csv'
    processed_target_path = 'data/processed/processed_target.csv'

    print("Starting model training script...")
    if not (os.path.exists(processed_features_path) and os.path.exists(processed_target_path)):
        print(f"Error: Processed data not found at {processed_features_path} or {processed_target_path}.")
        print("Please ensure you have run the preprocessing script successfully.")
        print("Expected files: processed_features.csv and processed_target.csv in data/processed/")
        return

    model_path, run_id = train_churn_model(processed_features_path, processed_target_path)
    print(f"Training complete. Model saved to: {model_path}")
    print(f"MLflow Run ID: {run_id}")

if __name__ == '__main__':
    main()
