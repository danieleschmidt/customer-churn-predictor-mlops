import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def preprocess(df_path):
    """
    Preprocesses the raw customer data.

    Args:
        df_path (str): Path to the raw CSV data file.

    Returns:
        tuple: A tuple containing processed features (X) and target (y).
    """
    df = pd.read_csv(df_path)

    # Convert 'TotalCharges' to numeric, coercing errors, and fill NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Encode target variable 'Churn'
    label_encoder_churn = LabelEncoder()
    df['Churn'] = label_encoder_churn.fit_transform(df['Churn'])
    y = df['Churn']

    # Separate features (X)
    X = df.drop('Churn', axis=1)
    X = X.drop('customerID', axis=1) # Drop customerID as it's an identifier

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for categorical and numerical features
    # One-hot encode categorical features
    # For numerical features, we'll just pass them through for now, but scaling could be added here.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep numerical columns not specified in transformers
    )

    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    # Get feature names from OneHotEncoder
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # Concatenate OHE feature names with remaining numerical feature names
    # Ensure numerical features are correctly identified and ordered as in X after categorical processing
    # The remainder='passthrough' puts numerical columns at the end, in their original relative order.
    processed_feature_names = list(ohe_feature_names) + list(numerical_features)

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X.index)

    return X_processed_df, y
