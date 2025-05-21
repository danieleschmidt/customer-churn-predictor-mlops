import os
import pandas as pd
from src.preprocess_data import preprocess # Ensure this import works based on your project structure

def main():
    # Define paths
    raw_data_path = 'data/raw/customer_data.csv'
    processed_features_path = 'data/processed/processed_features.csv'
    processed_target_path = 'data/processed/processed_target.csv'

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_target_path), exist_ok=True)

    # Preprocess data
    print(f"Loading raw data from {raw_data_path}...")
    X_processed, y_processed = preprocess(raw_data_path)

    # Save processed data
    print(f"Saving processed features to {processed_features_path}...")
    X_processed.to_csv(processed_features_path, index=False)
    
    print(f"Saving processed target to {processed_target_path}...")
    pd.DataFrame(y_processed, columns=['Churn']).to_csv(processed_target_path, index=False)
    
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
