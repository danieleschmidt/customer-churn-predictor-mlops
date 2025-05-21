import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import sys
import os

# Add src directory to Python path to import preprocess_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess_data import preprocess # Adjust import if necessary

class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # Create a small sample DataFrame for testing
        self.sample_data = {
            'customerID': ['123-ABC', '456-DEF', '789-GHI', '012-JKL'],
            'gender': ['Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 0],
            'Partner': ['Yes', 'No', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes', 'No'],
            'tenure': [1, 10, 24, 2],
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No', 'No internet service'],
            'TechSupport': ['No', 'No', 'Yes', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check'],
            'MonthlyCharges': [29.85, 90.00, 50.00, 20.00],
            'TotalCharges': ['29.85', '900', '1200.00', ' '], # Includes a space to test coercion
            'Churn': ['No', 'Yes', 'No', 'Yes']
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Create a dummy CSV file for the preprocess function to read
        self.test_csv_path = 'test_sample_data.csv'
        self.df.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        # Clean up the dummy CSV file
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_total_charges_conversion_and_nan_fill(self):
        # Call preprocess once to get X_processed
        X_processed, _ = preprocess(self.test_csv_path) 
        
        # Ensure NaN handling was effective for TotalCharges.
        self.assertNotIn('TotalCharges_nan', X_processed.columns) 
        
        # Infer the name of the TotalCharges column after transformation
        total_charges_col_name = [col for col in X_processed.columns if 'TotalCharges' in col][-1]
        
        self.assertTrue(pd.api.types.is_float_dtype(X_processed[total_charges_col_name]))
        
        # Check that the space in 'TotalCharges' for '012-JKL' (index 3) was converted to 0.0
        # This customer is at index 3 in the original self.df and X_processed (assuming index preservation)
        self.assertEqual(X_processed.loc[3, total_charges_col_name], 0.0)
        
        # General check for no NaNs in the column
        self.assertFalse(X_processed[total_charges_col_name].isnull().any(), "TotalCharges column should not have NaNs after processing.")


    def test_categorical_encoding(self):
        X_processed, _ = preprocess(self.test_csv_path)
        
        original_categorical_cols = self.df.select_dtypes(include=['object']).drop(columns=['customerID', 'TotalCharges', 'Churn']).columns
        for col in original_categorical_cols:
            self.assertNotIn(col, X_processed.columns)

        self.assertTrue(any(col.startswith('gender_') for col in X_processed.columns))
        self.assertTrue(any(col.startswith('Contract_') for col in X_processed.columns))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(X_processed[col]) for col in X_processed.columns))


    def test_target_encoding(self):
        _, y_processed = preprocess(self.test_csv_path)
        self.assertTrue(pd.api.types.is_integer_dtype(y_processed)) 
        self.assertEqual(y_processed.nunique(), 2) 
        expected_y = pd.Series([0, 1, 0, 1], name='Churn')
        assert_series_equal(y_processed, expected_y, check_dtype=False)


    def test_output_shape(self):
        X_processed, y_processed = preprocess(self.test_csv_path)
        self.assertEqual(X_processed.shape[0], self.df.shape[0]) 
        self.assertEqual(y_processed.shape[0], self.df.shape[0]) 
        self.assertTrue(X_processed.shape[1] > self.df.shape[1] - 2) 

    def test_id_column_dropped(self):
        X_processed, _ = preprocess(self.test_csv_path)
        self.assertNotIn('customerID', X_processed.columns)

if __name__ == '__main__':
    unittest.main()
