import os
import pandas as pd
import joblib # Add joblib import
import sys # Add sys import
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root to path
from src.preprocess_data import preprocess 

def main():
    raw_data_path = 'data/raw/customer_data.csv'
    processed_features_path = 'data/processed/processed_features.csv'
    processed_target_path = 'data/processed/processed_target.csv'
    preprocessor_path = 'models/preprocessor.joblib' # Path to save preprocessor

    os.makedirs(os.path.dirname(processed_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True) # Ensure models dir exists

    print(f"Loading raw data from {raw_data_path}...")
    # Modify preprocess to return the fitted preprocessor object
    X_processed, y_processed, fitted_preprocessor = preprocess(raw_data_path) # Expecting 3 return values now

    print(f"Saving processed features to {processed_features_path}...")
    X_processed.to_csv(processed_features_path, index=False)
    
    print(f"Saving processed target to {processed_target_path}...")
    pd.DataFrame(y_processed, columns=['Churn']).to_csv(processed_target_path, index=False)
    
    print(f"Saving fitted preprocessor to {preprocessor_path}...")
    joblib.dump(fitted_preprocessor, preprocessor_path) # Save the preprocessor
    
    print("Preprocessing complete. Features, target, and preprocessor saved.")

if __name__ == '__main__':
    main()
