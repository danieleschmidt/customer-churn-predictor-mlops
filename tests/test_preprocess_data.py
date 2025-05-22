import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Import both preprocess and validate_raw_data if direct testing of validate_raw_data is desired
from src.preprocess_data import preprocess, validate_raw_data 

class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # This sample data should pass the new validation by default
        self.valid_sample_data = {
            'customerID': ['123-ABC', '456-DEF', '789-GHI', '012-JKL', '345-MNO'],
            'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 0, 1], # Numeric
            'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
            'tenure': [1, 10, 24, 40, 65], 
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'No', 'No internet service', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No', 'No internet service', 'No'],
            'TechSupport': ['No', 'No', 'Yes', 'No internet service', 'No'],
            'StreamingTV': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'No', 'Yes', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check', 'Credit card (automatic)'],
            'MonthlyCharges': [29.85, 90.00, 50.00, 20.00, 105.50], # Numeric
            'TotalCharges': ['29.85', '900', '1200.00', ' ', '6800.0'], # Object, contains space
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        }
        self.df_valid = pd.DataFrame(self.valid_sample_data)
        self.test_csv_path_valid = 'test_valid_sample_data.csv'
        self.df_valid.to_csv(self.test_csv_path_valid, index=False)
        
        # Pre-run preprocess on valid data for other tests
        self.X_processed, self.y_processed = preprocess(self.test_csv_path_valid)


    def tearDown(self):
        if os.path.exists(self.test_csv_path_valid):
            os.remove(self.test_csv_path_valid)
        # Clean up any other test CSVs created by specific test methods
        for f in ['test_missing_col.csv', 'test_extra_col.csv', 'test_null_id.csv', 
                  'test_null_churn.csv', 'test_bad_senior.csv', 'test_bad_gender.csv',
                  'test_bad_churn_val.csv']:
            if os.path.exists(f):
                os.remove(f)

    # --- Tests for validate_raw_data (mostly via calling preprocess) ---

    def test_successful_validation(self):
        # preprocess calls validate_raw_data. If this doesn't raise error, validation passed.
        try:
            # Re-create the valid CSV for this specific test to ensure independence
            # though setUp already runs preprocess on it.
            self.df_valid.to_csv(self.test_csv_path_valid, index=False) # Ensure it exists
            preprocess(self.test_csv_path_valid)
        except ValueError as e:
            self.fail(f"Validation failed on presumably valid data: {e}")

    def test_validation_missing_column(self):
        data_missing_col = self.df_valid.copy().drop(columns=['tenure'])
        path = 'test_missing_col.csv'
        data_missing_col.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Missing expected columns: tenure"):
            preprocess(path)

    def test_validation_extra_column(self):
        data_extra_col = self.df_valid.copy()
        data_extra_col['ExtraUnexpectedColumn'] = "test"
        path = 'test_extra_col.csv'
        data_extra_col.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Found unexpected columns: ExtraUnexpectedColumn"):
            preprocess(path)

    def test_validation_null_customerid(self):
        data_null_id = self.df_valid.copy()
        data_null_id.loc[0, 'customerID'] = np.nan
        path = 'test_null_id.csv'
        data_null_id.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "'customerID' column contains null values"):
            preprocess(path)
            
    def test_validation_null_churn(self):
        data_null_churn = self.df_valid.copy()
        data_null_churn.loc[0, 'Churn'] = np.nan
        path = 'test_null_churn.csv'
        data_null_churn.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "'Churn' column contains null values"):
            preprocess(path)

    def test_validation_bad_seniorcitizen_type(self):
        data_bad_senior = self.df_valid.copy()
        data_bad_senior['SeniorCitizen'] = data_bad_senior['SeniorCitizen'].astype(str) # Make it string
        path = 'test_bad_senior.csv'
        data_bad_senior.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "'SeniorCitizen' is not numeric"):
            preprocess(path)

    def test_validation_bad_gender_value(self):
        data_bad_gender = self.df_valid.copy()
        data_bad_gender.loc[0, 'gender'] = 'Other'
        path = 'test_bad_gender.csv'
        data_bad_gender.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected values in 'gender'"):
            preprocess(path)
            
    def test_validation_bad_churn_value(self):
        data_bad_churn = self.df_valid.copy()
        data_bad_churn.loc[0, 'Churn'] = 'Maybe'
        path = 'test_bad_churn_val.csv'
        data_bad_churn.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected values in 'Churn'"):
            preprocess(path)

    # --- Existing tests for preprocess functionality (should still pass) ---
    # (These tests use self.X_processed, self.y_processed from setUp which used valid data)

    def test_output_shapes(self): 
        self.assertEqual(self.X_processed.shape[0], self.df_valid.shape[0])
        self.assertEqual(self.y_processed.shape[0], self.df_valid.shape[0])
        self.assertTrue(self.X_processed.shape[1] > self.df_valid.shape[1] - 2 + 3)

    def test_total_charges_handling(self): 
        self.assertIn('TotalCharges', self.X_processed.columns)
        self.assertFalse(self.X_processed['TotalCharges'].isnull().any())
        self.assertTrue(pd.api.types.is_float_dtype(self.X_processed['TotalCharges']))


    def test_target_encoding(self): 
        self.assertTrue(pd.api.types.is_integer_dtype(self.y_processed))
        self.assertEqual(self.y_processed.nunique(), 2)
        expected_y = pd.Series([0, 1, 0, 1, 0], name='Churn')
        assert_series_equal(self.y_processed, expected_y, check_dtype=False)

    def test_id_column_dropped(self): 
        self.assertNotIn('customerID', self.X_processed.columns)

    def test_interaction_feature(self): 
        self.assertIn('tenure_MonthlyCharges_interaction', self.X_processed.columns)
        self.assertTrue(pd.api.types.is_float_dtype(self.X_processed['tenure_MonthlyCharges_interaction']))


    def test_polynomial_feature(self): 
        self.assertIn('MonthlyCharges_poly2', self.X_processed.columns)
        self.assertTrue(pd.api.types.is_float_dtype(self.X_processed['MonthlyCharges_poly2']))


    def test_tenure_binning_and_ohe(self): 
        self.assertTrue(any(col.startswith('tenure_binned_') for col in self.X_processed.columns))
        self.assertIn('tenure_binned_0-1yr', self.X_processed.columns)
        self.assertEqual(self.X_processed.loc[0, 'tenure_binned_0-1yr'], 1) 
        self.assertEqual(self.X_processed.loc[1, 'tenure_binned_0-1yr'], 1) 
        
        self.assertIn('tenure_binned_1-3yrs', self.X_processed.columns) 
        self.assertEqual(self.X_processed.loc[2, 'tenure_binned_1-3yrs'], 1) 

        self.assertIn('tenure_binned_3-5yrs', self.X_processed.columns)
        self.assertEqual(self.X_processed.loc[3, 'tenure_binned_3-5yrs'], 1)

        self.assertIn('tenure_binned_5+yrs', self.X_processed.columns)
        self.assertEqual(self.X_processed.loc[4, 'tenure_binned_5+yrs'], 1) 

        self.assertIn('tenure', self.X_processed.columns) 
        self.assertTrue(pd.api.types.is_float_dtype(self.X_processed['tenure']))


    def test_numerical_feature_scaling(self): 
        numerical_cols_in_output = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'tenure_MonthlyCharges_interaction', 'MonthlyCharges_poly2'
        ]
        for col in numerical_cols_in_output:
            self.assertIn(col, self.X_processed.columns, f"Scaled column {col} not found.")
            self.assertTrue(pd.api.types.is_float_dtype(self.X_processed[col]))
            self.assertTrue(np.isclose(self.X_processed[col].mean(), 0, atol=0.2), f"Mean of {col} ({self.X_processed[col].mean()}) is not close to 0.")
            # Check std is positive (not zero), exact value of 1 is hard with small N and ddof=1
            self.assertTrue(self.X_processed[col].std() > 0.1, f"Std dev of {col} ({self.X_processed[col].std()}) is too low or zero.")


    def test_categorical_features_are_one_hot_encoded(self): 
        original_categorical_cols = self.df_valid.select_dtypes(include=['object']).drop(columns=['customerID', 'TotalCharges', 'Churn']).columns
        for col in original_categorical_cols:
            self.assertFalse(any(c == col for c in self.X_processed.columns))
        self.assertTrue(any(col.startswith('gender_') for col in self.X_processed.columns))
        self.assertTrue(any(col.startswith('Contract_') for col in self.X_processed.columns))
        self.assertTrue(any(col.startswith('tenure_binned_') for col in self.X_processed.columns))


if __name__ == '__main__':
    unittest.main()
