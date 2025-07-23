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

class TestPreprocessDataEdgeCases(unittest.TestCase):
    """Test suite for edge cases in data preprocessing."""
    
    def setUp(self):
        """Set up edge case test data."""
        # Minimal valid data for edge case testing
        self.minimal_data = {
            'customerID': ['EDGE-001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [1],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [50.0],
            'Churn': ['No']
        }
    
    def test_single_row_preprocessing(self):
        """Test preprocessing with single row dataset."""
        df = pd.DataFrame(self.minimal_data)
        temp_file = 'temp_single_row.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should successfully process single row
            self.assertEqual(len(X), 1)
            self.assertEqual(len(y), 1)
            self.assertTrue(isinstance(X, pd.DataFrame))
            self.assertTrue(isinstance(y, pd.Series))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_all_missing_total_charges(self):
        """Test preprocessing when all TotalCharges are missing/invalid."""
        data = self.minimal_data.copy()
        # Create multiple rows with various missing TotalCharges patterns
        for i in range(3):
            for key in self.minimal_data:
                if key == 'customerID':
                    data[key].append(f'EDGE-00{i+2}')
                elif key == 'TotalCharges':
                    data[key].append(['', 'invalid', np.nan][i])  # Different missing patterns
                else:
                    data[key].append(self.minimal_data[key][0])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_missing_charges.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should handle missing TotalCharges by filling with 0
            self.assertEqual(len(X), 4)
            # Check that TotalCharges column exists and has no NaN values
            total_charges_col = [col for col in X.columns if 'TotalCharges' in str(col)]
            if total_charges_col:
                self.assertFalse(X[total_charges_col[0]].isna().any())
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_extreme_categorical_values(self):
        """Test preprocessing with unusual but valid categorical values."""
        data = self.minimal_data.copy()
        
        # Test all combinations of service types
        extreme_cases = [
            {'InternetService': 'No', 'OnlineSecurity': 'No internet service'},
            {'PhoneService': 'No', 'MultipleLines': 'No phone service'},
            {'Contract': 'Two year', 'PaymentMethod': 'Credit card (automatic)'}
        ]
        
        for i, case in enumerate(extreme_cases):
            for key in self.minimal_data:
                if key == 'customerID':
                    data[key].append(f'EXTREME-00{i+1}')
                elif key in case:
                    data[key].append(case[key])
                else:
                    data[key].append(self.minimal_data[key][0])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_extreme_categorical.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should successfully encode all valid categorical combinations
            self.assertEqual(len(X), 4)  # Original + 3 extreme cases
            self.assertTrue(X.dtypes.apply(lambda x: x.kind in 'biufc').all())  # All numeric types
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_boundary_numeric_values(self):
        """Test preprocessing with boundary numeric values."""
        data = self.minimal_data.copy()
        
        # Test boundary cases
        boundary_cases = [
            {'tenure': 0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0},
            {'tenure': 72, 'MonthlyCharges': 200.0, 'TotalCharges': 14400.0},  # High but reasonable
            {'tenure': 1, 'MonthlyCharges': 0.01, 'TotalCharges': 0.01}  # Very low but valid
        ]
        
        for i, case in enumerate(boundary_cases):
            for key in self.minimal_data:
                if key == 'customerID':
                    data[key].append(f'BOUNDARY-00{i+1}')
                elif key in case:
                    data[key].append(case[key])
                else:
                    data[key].append(self.minimal_data[key][0])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_boundary_values.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should handle boundary values without errors
            self.assertEqual(len(X), 4)
            # Check that all numeric columns are finite
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            self.assertTrue(np.isfinite(X[numeric_cols]).all().all())
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_mixed_data_types_in_totalcharges(self):
        """Test preprocessing with mixed data types in TotalCharges column."""
        data = self.minimal_data.copy()
        
        # Add rows with different TotalCharges formats
        totalcharges_variants = ['100.5', '200', '', 'invalid', np.nan]
        
        for i, tc_value in enumerate(totalcharges_variants):
            for key in self.minimal_data:
                if key == 'customerID':
                    data[key].append(f'MIXED-00{i+1}')
                elif key == 'TotalCharges':
                    data[key].append(tc_value)
                else:
                    data[key].append(self.minimal_data[key][0])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_mixed_types.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should convert all to numeric, filling invalid values with 0
            self.assertEqual(len(X), 6)  # Original + 5 variants
            # TotalCharges should be numeric in output
            total_charges_cols = [col for col in X.columns if 'TotalCharges' in str(col)]
            if total_charges_cols:
                tc_col = total_charges_cols[0]
                self.assertTrue(pd.api.types.is_numeric_dtype(X[tc_col]))
                self.assertFalse(X[tc_col].isna().any())
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_highly_imbalanced_churn_data(self):
        """Test preprocessing with highly imbalanced target variable."""
        data = {}
        
        # Create 100 rows with 99% 'No' and 1% 'Yes' churn
        size = 100
        for key in self.minimal_data:
            if key == 'customerID':
                data[key] = [f'IMBAL-{i:03d}' for i in range(size)]
            elif key == 'Churn':
                # 99 'No' and 1 'Yes'
                data[key] = ['No'] * 99 + ['Yes']
            else:
                data[key] = [self.minimal_data[key][0]] * size
        
        df = pd.DataFrame(data)
        temp_file = 'temp_imbalanced.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should handle imbalanced data correctly
            self.assertEqual(len(X), 100)
            self.assertEqual(len(y), 100)
            
            # Check that both classes are represented in target
            unique_targets = y.unique()
            self.assertEqual(len(unique_targets), 2)
            
            # Check class distribution
            class_counts = y.value_counts()
            self.assertEqual(class_counts.max(), 99)  # Majority class
            self.assertEqual(class_counts.min(), 1)   # Minority class
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_all_services_disabled(self):
        """Test preprocessing with customer having no services."""
        data = self.minimal_data.copy()
        
        # Create customer with minimal services
        no_services_data = {
            'customerID': 'NO-SERVICES-001',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'No',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 20.0,  # Minimal charge
            'TotalCharges': 20.0,
            'Churn': 'Yes'  # Likely to churn with no services
        }
        
        for key in self.minimal_data:
            data[key].append(no_services_data[key])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_no_services.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should handle "no services" customer correctly
            self.assertEqual(len(X), 2)
            self.assertEqual(len(y), 2)
            
            # Check that one-hot encoding creates appropriate columns
            self.assertGreater(len(X.columns), 10)  # Should have multiple encoded columns
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_duplicate_customer_ids(self):
        """Test preprocessing behavior with duplicate customer IDs."""
        data = self.minimal_data.copy()
        
        # Add duplicate customer ID with different data
        duplicate_data = self.minimal_data.copy()
        for key in duplicate_data:
            if key == 'customerID':
                duplicate_data[key] = ['EDGE-001']  # Same ID as original
            elif key == 'Churn':
                duplicate_data[key] = ['Yes']  # Different churn value
            else:
                duplicate_data[key] = [duplicate_data[key][0]]
        
        for key in self.minimal_data:
            data[key].extend(duplicate_data[key])
        
        df = pd.DataFrame(data)
        temp_file = 'temp_duplicates.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should process both rows (preprocessing doesn't deduplicate)
            self.assertEqual(len(X), 2)
            self.assertEqual(len(y), 2)
            
            # CustomerID should be dropped from features
            self.assertNotIn('customerID', X.columns)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_column_order_variation(self):
        """Test preprocessing with different column orders."""
        # Create data with shuffled column order
        original_columns = list(self.minimal_data.keys())
        shuffled_columns = original_columns.copy()
        np.random.seed(42)  # For reproducible shuffling
        np.random.shuffle(shuffled_columns)
        
        # Create DataFrame with shuffled columns
        shuffled_data = {col: self.minimal_data[col] for col in shuffled_columns}
        df = pd.DataFrame(shuffled_data)
        
        temp_file = 'temp_shuffled_columns.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            X, y = preprocess(temp_file)
            
            # Should handle different column orders correctly
            self.assertEqual(len(X), 1)
            self.assertEqual(len(y), 1)
            
            # Output should be consistent regardless of input column order
            self.assertTrue(isinstance(X, pd.DataFrame))
            self.assertTrue(isinstance(y, pd.Series))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_memory_efficiency_simulation(self):
        """Test preprocessing memory efficiency with larger dataset simulation."""
        # Create moderately large dataset (1000 rows) to test memory usage
        size = 1000
        large_data = {}
        
        for key in self.minimal_data:
            if key == 'customerID':
                large_data[key] = [f'LARGE-{i:04d}' for i in range(size)]
            elif key == 'Churn':
                # Balanced dataset
                large_data[key] = ['No', 'Yes'] * (size // 2)
            elif key in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                # Add some variation to numeric fields
                base_val = self.minimal_data[key][0]
                if isinstance(base_val, (int, float)):
                    large_data[key] = [base_val + (i % 50) for i in range(size)]
                else:
                    large_data[key] = [base_val] * size
            else:
                large_data[key] = [self.minimal_data[key][0]] * size
        
        df = pd.DataFrame(large_data)
        temp_file = 'temp_large_dataset.csv'
        
        try:
            df.to_csv(temp_file, index=False)
            
            # Measure memory usage and processing time
            import time
            import tracemalloc
            
            tracemalloc.start()
            start_time = time.time()
            
            X, y = preprocess(temp_file)
            
            processing_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Should complete within reasonable time and memory
            self.assertEqual(len(X), size)
            self.assertEqual(len(y), size)
            self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
            self.assertLess(peak / 1024 / 1024, 500)  # Should use less than 500MB peak memory
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
