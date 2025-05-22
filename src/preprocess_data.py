import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# Function to validate raw data
def validate_raw_data(df: pd.DataFrame):
    """Performs basic validation on the raw customer DataFrame."""
    print("Starting raw data validation...")
    errors = []

    # 1. Column Presence
    expected_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing expected columns: {', '.join(missing_columns)}")

    extra_columns = [col for col in df.columns if col not in expected_columns]
    if extra_columns: # This might be too strict, a warning might be better. For now, an error.
        errors.append(f"Found unexpected columns: {', '.join(extra_columns)}")

    # Perform further checks only if all expected columns are present
    if not missing_columns:
        # 2. Critical Column Non-Null
        if df['customerID'].isnull().any():
            errors.append("'customerID' column contains null values.")
        if df['Churn'].isnull().any():
            errors.append("'Churn' column contains null values.")

        # 3. Data Type Checks (on original dtypes where appropriate)
        if not pd.api.types.is_numeric_dtype(df['SeniorCitizen']):
            errors.append(f"'SeniorCitizen' is not numeric. Found dtype: {df['SeniorCitizen'].dtype}")
        
        # For tenure, MonthlyCharges, TotalCharges, initial dtypes can be object if they need cleaning.
        # A more robust check for these might be to try pd.to_numeric on a sample and see if it fails,
        # but for now, checking if they are not purely numeric initially is less useful than post-cleaning checks.
        # We'll rely on the main preprocessing step to handle their conversion.
        # TotalCharges is specifically expected to be object due to spaces.
        if df['TotalCharges'].dtype != 'object': 
             errors.append(f"'TotalCharges' is not 'object' initially, which might mean spaces are not present as expected. Dtype: {df['TotalCharges'].dtype}")


        # 4. Categorical Value Sets (handling potential NaNs if they were allowed)
        expected_gender_values = ['Male', 'Female']
        # Check only non-null values against the expected set
        # Ensure that all non-NA values are in the expected set
        if not df['gender'].dropna().isin(expected_gender_values).all() and df['gender'].notna().any() : 
            unique_genders = df['gender'].dropna().unique()
            # Check if there are any values in unique_genders that are not in expected_gender_values
            if not set(unique_genders).issubset(set(expected_gender_values)):
                errors.append(f"Unexpected values in 'gender'. Found: {unique_genders}. Expected subset of: {expected_gender_values}")

        expected_churn_values = ['Yes', 'No']
        if not df['Churn'].dropna().isin(expected_churn_values).all() and df['Churn'].notna().any():
            unique_churns = df['Churn'].dropna().unique()
            if not set(unique_churns).issubset(set(expected_churn_values)):
                errors.append(f"Unexpected values in 'Churn'. Found: {unique_churns}. Expected subset of: {expected_churn_values}")
            
    if errors:
        error_message = "Raw data validation failed with the following errors:\n" + "\n".join(errors)
        raise ValueError(error_message)
    else:
        print("Raw data validation passed successfully.")


def preprocess(df_path):
    df = pd.read_csv(df_path)

    # Call validation function
    validate_raw_data(df.copy()) # Pass a copy to avoid unintended modifications by validator

    # Convert 'TotalCharges' to numeric, coercing errors, and fill NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0) # Fill NaNs *after* coercion

    # Encode target variable 'Churn'
    label_encoder_churn = LabelEncoder()
    # Fit LabelEncoder only on non-null values if Churn could have NaNs (already checked in validate_raw_data)
    df['Churn'] = label_encoder_churn.fit_transform(df['Churn']) 
    y = df['Churn']

    # Separate features (X)
    X = df.drop('Churn', axis=1)
    X = X.drop('customerID', axis=1) # Drop customerID

    # ADVANCED FEATURE ENGINEERING
    X['tenure_MonthlyCharges_interaction'] = X['tenure'] * X['MonthlyCharges']
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    # Ensure MonthlyCharges is float for PolynomialFeatures
    # MonthlyCharges should already be numeric from the CSV or initial load if no strange values.
    # If it could be an object string needing conversion (e.g. "$50.00"), it needs pd.to_numeric first.
    # Assuming MonthlyCharges is loaded as numeric or cleanable to numeric by this point.
    # If it was an object that pd.read_csv couldn't convert, it would fail here or earlier.
    # The raw data CSV has it as numeric.
    X['MonthlyCharges'] = X['MonthlyCharges'].astype(float) 
    monthly_charges_poly = poly.fit_transform(X[['MonthlyCharges']])
    X['MonthlyCharges_poly2'] = monthly_charges_poly[:, 1]
    
    tenure_bins = [0, 12, 36, 60, 100] # Bins must cover min/max of data or handle outliers
    tenure_labels = ['0-1yr', '1-3yrs', '3-5yrs', '5+yrs']
    # Ensure tenure is numeric before binning
    X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0) # Example: fill NaNs with 0 if any
    X['tenure_binned'] = pd.cut(X['tenure'], bins=tenure_bins, labels=tenure_labels, right=True, include_lowest=True)
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Explicitly ensure 'tenure_binned' is in categorical_features list if not already
    if 'tenure_binned' in X.columns and X['tenure_binned'].dtype.name == 'category':
        if 'tenure_binned' not in categorical_features:
             categorical_features.append('tenure_binned')
    # Sort to ensure consistent column order for OHE if that matters downstream (good practice)
    categorical_features = sorted(list(set(categorical_features)))


    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Ensure original 'tenure' (if kept and numeric) is in numerical_features
    # If 'tenure' was binned and we don't want the original scaled value, remove from numerical_features
    # Current logic keeps 'tenure' in numerical_features and scales it, which is fine.

    # Sort for consistent order if needed (ColumnTransformer order is defined by list order)
    numerical_features = sorted(list(set(numerical_features)))


    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough' 
    )

    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformations
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    # Numerical feature names are preserved by StandardScaler, ensure they are taken from the list used in ColumnTransformer
    processed_feature_names = list(ohe_feature_names) + numerical_features # Order matters!
    
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X.index)

    return X_processed_df, y, preprocessor
