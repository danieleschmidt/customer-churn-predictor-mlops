"""
Tests for comprehensive data validation framework.

This module tests the enhanced data validation capabilities including:
- Schema validation for customer churn data
- Business rule validation
- Data quality checks
- ML-specific validation
"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_validation import (
    ChurnDataValidator,
    ValidationError,
    ValidationReport,
    validate_customer_data,
    create_data_schema
)


class TestChurnDataValidator:
    """Test suite for ChurnDataValidator class."""
    
    @pytest.fixture
    def valid_data(self):
        """Create valid customer data for testing."""
        return pd.DataFrame({
            'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK'],
            'gender': ['Female', 'Male', 'Male'],
            'SeniorCitizen': [0, 0, 0],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'No', 'No'],
            'tenure': [1, 34, 2],
            'PhoneService': ['No', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'No'],
            'InternetService': ['DSL', 'DSL', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'TechSupport': ['No', 'No', 'No'],
            'StreamingTV': ['No', 'No', 'No'],
            'StreamingMovies': ['No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check'],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': [29.85, 1889.5, 108.15],
            'Churn': ['No', 'No', 'Yes']
        })
    
    @pytest.fixture
    def validator(self):
        """Create ChurnDataValidator instance."""
        return ChurnDataValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator is not None
        assert hasattr(validator, 'schema')
        assert hasattr(validator, 'validate')
    
    def test_valid_data_passes_validation(self, validator, valid_data):
        """Test that valid data passes all validation checks."""
        report = validator.validate(valid_data)
        assert isinstance(report, ValidationReport)
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
    
    def test_missing_required_columns(self, validator):
        """Test validation fails when required columns are missing."""
        data = pd.DataFrame({'customerID': ['test']})
        report = validator.validate(data)
        assert not report.is_valid
        assert len(report.errors) > 0
        assert any('missing' in error.lower() for error in report.errors)
    
    def test_invalid_data_types(self, validator, valid_data):
        """Test validation fails with invalid data types."""
        # Test invalid SeniorCitizen (should be 0 or 1)
        invalid_data = valid_data.copy()
        invalid_data['SeniorCitizen'] = ['invalid', 'also_invalid', 2]
        report = validator.validate(invalid_data)
        assert not report.is_valid
        assert any('SeniorCitizen' in error for error in report.errors)
    
    def test_invalid_categorical_values(self, validator, valid_data):
        """Test validation fails with invalid categorical values."""
        # Test invalid gender
        invalid_data = valid_data.copy()
        invalid_data['gender'] = ['InvalidGender', 'Male', 'Female']
        report = validator.validate(invalid_data)
        assert not report.is_valid
        assert any('gender' in error for error in report.errors)
    
    def test_business_rule_validation(self, validator, valid_data):
        """Test business rule validation."""
        # Test invalid TotalCharges (should be >= MonthlyCharges for tenure >= 1)
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'TotalCharges'] = 10.0  # Less than MonthlyCharges
        invalid_data.loc[0, 'tenure'] = 5  # Non-zero tenure
        report = validator.validate(invalid_data)
        assert not report.is_valid
        assert any('TotalCharges' in error or 'business rule' in error.lower() 
                  for error in report.errors)
    
    def test_outlier_detection(self, validator, valid_data):
        """Test outlier detection for numeric fields."""
        # Test extreme MonthlyCharges
        outlier_data = valid_data.copy()
        outlier_data.loc[0, 'MonthlyCharges'] = 99999.99
        report = validator.validate(outlier_data)
        # Should generate warnings for outliers
        assert len(report.warnings) > 0
        assert any('outlier' in warning.lower() or 'extreme' in warning.lower() 
                  for warning in report.warnings)
    
    def test_duplicate_customer_ids(self, validator, valid_data):
        """Test detection of duplicate customer IDs."""
        duplicate_data = valid_data.copy()
        duplicate_data.loc[1, 'customerID'] = duplicate_data.loc[0, 'customerID']
        report = validator.validate(duplicate_data)
        assert not report.is_valid
        assert any('duplicate' in error.lower() for error in report.errors)
    
    def test_internet_service_consistency(self, validator, valid_data):
        """Test consistency between InternetService and related fields."""
        # If InternetService is 'No', online services should be 'No internet service'
        inconsistent_data = valid_data.copy()
        inconsistent_data.loc[0, 'InternetService'] = 'No'
        inconsistent_data.loc[0, 'OnlineSecurity'] = 'Yes'  # Should be 'No internet service'
        report = validator.validate(inconsistent_data)
        assert not report.is_valid
        assert any('internet service' in error.lower() or 'consistency' in error.lower() 
                  for error in report.errors)


class TestValidationReport:
    """Test suite for ValidationReport class."""
    
    def test_validation_report_creation(self):
        """Test ValidationReport creation and properties."""
        errors = ['Error 1', 'Error 2']
        warnings = ['Warning 1']
        report = ValidationReport(errors, warnings)
        
        assert report.errors == errors
        assert report.warnings == warnings
        assert not report.is_valid  # Has errors
        assert report.error_count == 2
        assert report.warning_count == 1
    
    def test_valid_report(self):
        """Test ValidationReport with no errors."""
        report = ValidationReport([], ['Warning 1'])
        assert report.is_valid  # No errors
        assert report.error_count == 0
        assert report.warning_count == 1
    
    def test_report_summary(self):
        """Test ValidationReport summary generation."""
        errors = ['Critical error']
        warnings = ['Minor warning']
        report = ValidationReport(errors, warnings)
        summary = report.get_summary()
        
        assert 'Validation failed' in summary
        assert '1 error' in summary
        assert '1 warning' in summary


class TestDataSchema:
    """Test suite for data schema creation and validation."""
    
    def test_create_data_schema(self):
        """Test data schema creation."""
        schema = create_data_schema()
        assert schema is not None
        assert 'customerID' in schema
        assert 'gender' in schema
        assert 'MonthlyCharges' in schema
    
    def test_schema_contains_required_fields(self):
        """Test schema contains all required customer data fields."""
        schema = create_data_schema()
        required_fields = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        for field in required_fields:
            assert field in schema, f"Required field {field} missing from schema"


class TestValidateCustomerData:
    """Test suite for validate_customer_data function."""
    
    def test_validate_customer_data_function(self):
        """Test the main validation function."""
        valid_data = pd.DataFrame({
            'customerID': ['test'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],
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
            'TotalCharges': [600.0]
        })
        
        report = validate_customer_data(valid_data)
        assert isinstance(report, ValidationReport)
    
    def test_validate_with_invalid_file_path(self):
        """Test validation with invalid file path raises appropriate error."""
        with pytest.raises((FileNotFoundError, ValidationError)):
            validate_customer_data('/nonexistent/file.csv')


class TestMLValidation:
    """Test suite for ML-specific validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for ML testing."""
        return ChurnDataValidator()
    
    def test_feature_distribution_validation(self, validator, valid_data):
        """Test feature distribution validation."""
        # This would typically compare against reference distributions
        # For now, test that the method exists and runs
        report = validator.validate(valid_data, check_distribution=True)
        assert isinstance(report, ValidationReport)
    
    def test_prediction_input_validation(self, validator):
        """Test validation of prediction inputs."""
        # Test minimal prediction input (no Churn column)
        prediction_data = pd.DataFrame({
            'customerID': ['test'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],
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
            'TotalCharges': [600.0]
        })
        
        report = validator.validate_for_prediction(prediction_data)
        assert isinstance(report, ValidationReport)


class TestEdgeCaseDataValidation:
    """Test suite for edge cases in data validation."""
    
    @pytest.fixture
    def validator(self):
        """Create ChurnDataValidator instance."""
        return ChurnDataValidator()
    
    @pytest.fixture
    def minimal_valid_data(self):
        """Create minimal valid dataset for edge case testing."""
        return pd.DataFrame({
            'customerID': ['TEST-001'],
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
            'TotalCharges': [50.0]
        })
    
    def test_extreme_outliers_within_bounds(self, validator, minimal_valid_data):
        """Test handling of extreme but technically valid values."""
        # Test extreme low values
        extreme_data = minimal_valid_data.copy()
        extreme_data.loc[0, 'MonthlyCharges'] = 0.01
        extreme_data.loc[0, 'TotalCharges'] = 0.01
        extreme_data.loc[0, 'tenure'] = 0
        
        report = validator.validate(extreme_data)
        # Should pass validation but may generate warnings
        assert report.is_valid
        assert len(report.warnings) > 0  # Should warn about extreme values
    
    def test_extreme_outliers_above_bounds(self, validator, minimal_valid_data):
        """Test handling of extreme values beyond reasonable bounds."""
        extreme_data = minimal_valid_data.copy()
        extreme_data.loc[0, 'MonthlyCharges'] = 999.99
        extreme_data.loc[0, 'TotalCharges'] = 99999.99
        extreme_data.loc[0, 'tenure'] = 150  # Beyond max of 100
        
        report = validator.validate(extreme_data)
        assert not report.is_valid
        assert any('tenure' in error for error in report.errors)
    
    def test_boundary_values_exact(self, validator, minimal_valid_data):
        """Test exact boundary values for numeric fields."""
        # Test tenure boundaries
        boundary_cases = [
            ({'tenure': 0}, True),  # Minimum tenure
            ({'tenure': 100}, True),  # Maximum tenure  
            ({'tenure': 101}, False),  # Just over maximum
            ({'MonthlyCharges': 0.0}, True),  # Minimum charge
            ({'MonthlyCharges': 200.0}, True),  # Maximum charge
            ({'MonthlyCharges': 200.01}, False),  # Just over maximum
        ]
        
        for field_update, should_be_valid in boundary_cases:
            test_data = minimal_valid_data.copy()
            for field, value in field_update.items():
                test_data.loc[0, field] = value
            
            report = validator.validate(test_data)
            if should_be_valid:
                assert report.is_valid, f"Expected {field_update} to be valid"
            else:
                assert not report.is_valid, f"Expected {field_update} to be invalid"
    
    def test_unicode_customer_ids(self, validator, minimal_valid_data):
        """Test customer IDs with unicode and special characters."""
        unicode_test_cases = [
            ('CUST-001', True),  # Normal ASCII
            ('客户-001', False),  # Chinese characters (should fail pattern)
            ('CUST@001', False),  # Special character (should fail pattern)
            ('CUST-ñáéí', False),  # Accented characters (should fail pattern)
            ('', False),  # Empty string
            ('CUST_001', True),  # Underscore (allowed by pattern)
        ]
        
        for customer_id, should_be_valid in unicode_test_cases:
            test_data = minimal_valid_data.copy()
            test_data.loc[0, 'customerID'] = customer_id
            
            report = validator.validate(test_data)
            if should_be_valid:
                assert report.is_valid, f"Expected customerID '{customer_id}' to be valid"
            else:
                assert not report.is_valid, f"Expected customerID '{customer_id}' to be invalid"
    
    def test_mixed_data_types_in_numeric_fields(self, validator, minimal_valid_data):
        """Test mixed data types in numeric fields."""
        # Create data with mixed types
        mixed_data = pd.DataFrame([
            {**minimal_valid_data.iloc[0].to_dict(), 'MonthlyCharges': 50.0, 'tenure': 12},
            {**minimal_valid_data.iloc[0].to_dict(), 'customerID': 'TEST-002', 'MonthlyCharges': '60.5', 'tenure': '24'},  # String numbers
            {**minimal_valid_data.iloc[0].to_dict(), 'customerID': 'TEST-003', 'MonthlyCharges': 'invalid', 'tenure': 36}  # Invalid string
        ])
        
        report = validator.validate(mixed_data)
        assert not report.is_valid
        assert any('MonthlyCharges' in error for error in report.errors)
    
    def test_null_vs_empty_string_handling(self, validator, minimal_valid_data):
        """Test different representations of missing data."""
        # Test various null representations
        null_test_data = minimal_valid_data.copy()
        
        # Add rows with different null representations
        additional_rows = []
        for i, null_value in enumerate([np.nan, None, '', 'NULL', 'N/A'], start=1):
            row = minimal_valid_data.iloc[0].to_dict()
            row['customerID'] = f'TEST-00{i+1}'
            row['TotalCharges'] = null_value  # TotalCharges can be nullable
            additional_rows.append(row)
        
        test_data = pd.concat([null_test_data, pd.DataFrame(additional_rows)], ignore_index=True)
        
        report = validator.validate(test_data)
        # Should handle NaN/None gracefully, but reject string representations
        assert not report.is_valid or len(report.warnings) > 0
    
    def test_negative_values_validation(self, validator, minimal_valid_data):
        """Test negative values in fields that should be positive."""
        negative_test_cases = [
            ('tenure', -1),
            ('MonthlyCharges', -50.0),
            ('TotalCharges', -100.0),
        ]
        
        for field, negative_value in negative_test_cases:
            test_data = minimal_valid_data.copy()
            test_data.loc[0, field] = negative_value
            
            report = validator.validate(test_data)
            assert not report.is_valid, f"Expected negative {field} to be invalid"
            assert any(field in error for error in report.errors)
    
    def test_floating_point_precision_issues(self, validator, minimal_valid_data):
        """Test floating point precision edge cases."""
        # Test values that might cause precision issues
        precision_data = minimal_valid_data.copy()
        precision_data.loc[0, 'MonthlyCharges'] = 0.1 + 0.2  # != 0.3 due to float precision
        precision_data.loc[0, 'TotalCharges'] = 0.3
        
        report = validator.validate(precision_data)
        # Should handle precision issues gracefully
        assert report.is_valid
    
    def test_data_corruption_scenarios(self, validator):
        """Test various data corruption patterns."""
        # Test partially corrupted data
        corrupted_data = pd.DataFrame({
            'customerID': ['VALID-001', None, 'VALID-003', ''],
            'gender': ['Male', 'Female', 'Invalid', 'Male'],
            'SeniorCitizen': [0, 1, 2, 'invalid'],  # Mix of valid and invalid
            'Partner': ['Yes', 'No', 'Maybe', 'Yes'],  # Invalid value 'Maybe'
            'tenure': [12, 24, -5, 1000],  # Negative and too large values
            'MonthlyCharges': [50.0, 75.5, 'corrupted', np.inf],  # String and infinity
        })
        
        # Add minimal required columns to avoid structure errors
        for col in ['Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                   'PaymentMethod', 'TotalCharges']:
            corrupted_data[col] = ['No', 'No', 'No', 'No']
        
        report = validator.validate(corrupted_data)
        assert not report.is_valid
        assert len(report.errors) >= 5  # Multiple types of corruption
    
    def test_extremely_imbalanced_categorical_data(self, validator):
        """Test highly imbalanced categorical distributions."""
        # Create data where 99% are one category
        size = 1000
        imbalanced_data = pd.DataFrame({
            'customerID': [f'CUST-{i:04d}' for i in range(size)],
            'gender': ['Male'] * 999 + ['Female'],  # 99.9% Male
            'SeniorCitizen': [0] * size,
            'Partner': ['No'] * size,
            'Dependents': ['No'] * size,
            'tenure': [12] * size,
            'PhoneService': ['Yes'] * size,
            'MultipleLines': ['No'] * size,
            'InternetService': ['DSL'] * size,
            'OnlineSecurity': ['No'] * size,
            'OnlineBackup': ['No'] * size,
            'DeviceProtection': ['No'] * size,
            'TechSupport': ['No'] * size,
            'StreamingTV': ['No'] * size,
            'StreamingMovies': ['No'] * size,
            'Contract': ['Month-to-month'] * size,
            'PaperlessBilling': ['Yes'] * size,
            'PaymentMethod': ['Electronic check'] * size,
            'MonthlyCharges': [50.0] * size,
            'TotalCharges': [600.0] * size,
        })
        
        report = validator.validate(imbalanced_data)
        assert report.is_valid  # Should pass validation
        assert len(report.warnings) > 0  # But generate warnings about imbalance
        assert any('imbalanced' in warning.lower() for warning in report.warnings)
    
    def test_single_row_edge_cases(self, validator):
        """Test validation with single row datasets."""
        # Single valid row
        single_row = pd.DataFrame({
            'customerID': ['SINGLE-001'],
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
        })
        
        report = validator.validate(single_row)
        assert report.is_valid
        
        # Single invalid row
        single_invalid = single_row.copy()
        single_invalid.loc[0, 'gender'] = 'Invalid'
        
        report = validator.validate(single_invalid)
        assert not report.is_valid
    
    def test_empty_dataframe_validation(self, validator):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        report = validator.validate(empty_df)
        assert not report.is_valid
        assert any('empty' in error.lower() for error in report.errors)
    
    def test_very_large_dataset_sample(self, validator, minimal_valid_data):
        """Test validation performance with larger dataset (simulation)."""
        # Create a moderately large dataset for testing
        base_row = minimal_valid_data.iloc[0].to_dict()
        
        # Generate 1000 rows with slight variations
        large_data_rows = []
        for i in range(1000):
            row = base_row.copy()
            row['customerID'] = f'LARGE-{i:04d}'
            row['tenure'] = (i % 72) + 1  # Vary tenure 1-72
            row['MonthlyCharges'] = 50.0 + (i % 100)  # Vary charges
            large_data_rows.append(row)
        
        large_data = pd.DataFrame(large_data_rows)
        
        # Should complete validation in reasonable time
        import time
        start_time = time.time()
        report = validator.validate(large_data)
        validation_time = time.time() - start_time
        
        assert report.is_valid
        assert validation_time < 5.0  # Should complete within 5 seconds
        assert report.statistics['total_rows'] == 1000


class TestModelTrainingEdgeCases:
    """Test suite for edge cases in model training."""
    
    def test_insufficient_training_data_scenarios(self):
        """Test model training with insufficient data."""
        # This would be implemented when we have access to the training module
        # For now, create a placeholder test
        pytest.skip("Requires model training module access")
    
    def test_single_class_training_data(self):
        """Test training with data containing only one class."""
        pytest.skip("Requires model training module access")
    
    def test_highly_imbalanced_training_data(self):
        """Test training with 99:1 class imbalance."""
        pytest.skip("Requires model training module access")


class TestPredictionEdgeCases:
    """Test suite for edge cases in predictions."""
    
    def test_batch_prediction_memory_limits(self):
        """Test batch predictions with very large datasets."""
        pytest.skip("Requires prediction module access")
    
    def test_concurrent_prediction_safety(self):
        """Test thread safety of prediction operations."""
        pytest.skip("Requires prediction module access")


if __name__ == '__main__':
    pytest.main([__file__])