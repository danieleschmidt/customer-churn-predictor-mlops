"""Sample data fixtures for testing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def create_sample_customer_data(n_samples: int = 100, include_target: bool = True) -> pd.DataFrame:
    """Create sample customer data for testing.
    
    Args:
        n_samples: Number of sample records to generate
        include_target: Whether to include the churn target column
        
    Returns:
        DataFrame with sample customer data
    """
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
        'TotalCharges': ''  # Will be calculated
    }
    
    # Calculate TotalCharges based on tenure and MonthlyCharges
    data['TotalCharges'] = [
        str(round(tenure * monthly + np.random.uniform(-50, 50), 2))
        if tenure > 0 else ''
        for tenure, monthly in zip(data['tenure'], data['MonthlyCharges'])
    ]
    
    df = pd.DataFrame(data)
    
    if include_target:
        # Create realistic churn patterns
        churn_prob = 0.1 + 0.3 * (df['tenure'] < 12).astype(int) + \
                    0.2 * (df['MonthlyCharges'] > 80).astype(int) + \
                    0.15 * (df['Contract'] == 'Month-to-month').astype(int)
        churn_prob = np.clip(churn_prob, 0, 1)
        df['Churn'] = np.random.binomial(1, churn_prob, n_samples).astype(str)
        df['Churn'] = df['Churn'].map({'0': 'No', '1': 'Yes'})
    
    return df


def create_processed_features(n_samples: int = 100) -> pd.DataFrame:
    """Create sample processed features for testing.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        DataFrame with processed features matching expected format
    """
    np.random.seed(42)
    
    # Features after preprocessing (one-hot encoded)
    data = {
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'tenure': np.random.randint(1, 73, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(18.25, 8500.0, n_samples), 2),
        'gender_Female': np.random.choice([0, 1], n_samples),
        'gender_Male': np.random.choice([0, 1], n_samples),
        'Partner_No': np.random.choice([0, 1], n_samples),
        'Partner_Yes': np.random.choice([0, 1], n_samples),
        'Dependents_No': np.random.choice([0, 1], n_samples),
        'Dependents_Yes': np.random.choice([0, 1], n_samples),
        'PhoneService_No': np.random.choice([0, 1], n_samples),
        'PhoneService_Yes': np.random.choice([0, 1], n_samples),
        'Contract_Month-to-month': np.random.choice([0, 1], n_samples),
        'Contract_One year': np.random.choice([0, 1], n_samples),
        'Contract_Two year': np.random.choice([0, 1], n_samples),
        'PaperlessBilling_No': np.random.choice([0, 1], n_samples),
        'PaperlessBilling_Yes': np.random.choice([0, 1], n_samples),
    }
    
    return pd.DataFrame(data)


def create_sample_target(n_samples: int = 100) -> pd.Series:
    """Create sample target variable for testing.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        Series with binary target values
    """
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], n_samples, p=[0.73, 0.27]))


def create_api_request_data() -> Dict:
    """Create sample API request data for testing.
    
    Returns:
        Dictionary with sample customer data for API requests
    """
    return {
        "customer_data": {
            "tenure": 12,
            "MonthlyCharges": 65.50,
            "TotalCharges": 786.00,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes"
        }
    }


def create_batch_request_data(n_customers: int = 5) -> Dict:
    """Create sample batch API request data for testing.
    
    Args:
        n_customers: Number of customers in the batch
        
    Returns:
        Dictionary with batch customer data for API requests
    """
    customers = []
    for i in range(n_customers):
        customer = {
            "tenure": np.random.randint(1, 73),
            "MonthlyCharges": round(np.random.uniform(18.25, 118.75), 2),
            "TotalCharges": round(np.random.uniform(18.25, 8500.0), 2),
            "gender": np.random.choice(["Male", "Female"]),
            "SeniorCitizen": np.random.choice([0, 1]),
            "Partner": np.random.choice(["Yes", "No"]),
            "Dependents": np.random.choice(["Yes", "No"]),
            "PhoneService": np.random.choice(["Yes", "No"]),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": np.random.choice(["Yes", "No"])
        }
        customers.append(customer)
    
    return {"customers": customers}


def create_invalid_data_samples() -> List[Dict]:
    """Create invalid data samples for testing error handling.
    
    Returns:
        List of dictionaries with invalid data for testing
    """
    return [
        # Missing required fields
        {
            "customer_data": {
                "tenure": 12,
                "MonthlyCharges": 65.50
                # Missing other required fields
            }
        },
        # Invalid data types
        {
            "customer_data": {
                "tenure": "twelve",  # Should be int
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Female",
                "SeniorCitizen": 0
            }
        },
        # Out of range values
        {
            "customer_data": {
                "tenure": -5,  # Should be positive
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Female",
                "SeniorCitizen": 0
            }
        },
        # Invalid categorical values
        {
            "customer_data": {
                "tenure": 12,
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Other",  # Should be Male or Female
                "SeniorCitizen": 0
            }
        }
    ]


# Test data constants
TEST_DATA_CONSTANTS = {
    "MIN_TENURE": 1,
    "MAX_TENURE": 72,
    "MIN_MONTHLY_CHARGES": 18.25,
    "MAX_MONTHLY_CHARGES": 118.75,
    "EXPECTED_FEATURE_COUNT": 17,  # After preprocessing
    "CHURN_CLASSES": [0, 1],
    "CATEGORICAL_FEATURES": [
        "gender", "Partner", "Dependents", "PhoneService", 
        "Contract", "PaperlessBilling"
    ],
    "NUMERICAL_FEATURES": [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]
}