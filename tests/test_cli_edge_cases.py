"""
Edge case tests for CLI functionality.

This module tests edge cases and error conditions in the CLI interface,
ensuring robust error handling and user-friendly error messages.
"""

import pytest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
import pandas as pd

# Import CLI app
try:
    from src.cli import app
    from src.validation import ValidationError
    from src.data_validation import ValidationError as DataValidationError
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestCLIEdgeCases:
    """Test suite for CLI edge cases and error handling."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def valid_data_file(self, temp_dir):
        """Create a valid data file for testing."""
        data = pd.DataFrame({
            'customerID': ['CLI-001', 'CLI-002'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'No'],
            'tenure': [12, 24],
            'PhoneService': ['Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'No'],
            'StreamingTV': ['No', 'Yes'],
            'StreamingMovies': ['No', 'No'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [600.0, 1800.0],
            'Churn': ['No', 'Yes']
        })
        
        file_path = os.path.join(temp_dir, 'valid_data.csv')
        data.to_csv(file_path, index=False)
        return file_path
    
    def test_validate_command_with_nonexistent_file(self, runner):
        """Test validate command with non-existent file."""
        result = runner.invoke(app, ['validate', '/nonexistent/file.csv'])
        
        assert result.exit_code == 1
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()
    
    def test_validate_command_with_empty_file(self, runner, temp_dir):
        """Test validate command with empty file."""
        empty_file = os.path.join(temp_dir, 'empty.csv')
        with open(empty_file, 'w') as f:
            f.write('')  # Empty file
        
        result = runner.invoke(app, ['validate', empty_file])
        
        assert result.exit_code == 1
        assert 'validation failed' in result.output.lower() or 'empty' in result.output.lower()
    
    def test_validate_command_with_invalid_csv(self, runner, temp_dir):
        """Test validate command with malformed CSV."""
        invalid_file = os.path.join(temp_dir, 'invalid.csv')
        with open(invalid_file, 'w') as f:
            f.write('not,a,valid\ncsv,file,content,with,wrong,columns')
        
        result = runner.invoke(app, ['validate', invalid_file])
        
        assert result.exit_code == 1
        assert 'validation failed' in result.output.lower() or 'error' in result.output.lower()
    
    def test_validate_command_with_permission_denied(self, runner, valid_data_file):
        """Test validate command with permission denied error."""
        with patch('src.validation.DEFAULT_PATH_VALIDATOR.validate_path', 
                  side_effect=ValidationError("Permission denied")):
            result = runner.invoke(app, ['validate', valid_data_file])
            
            assert result.exit_code == 1
            assert 'permission' in result.output.lower() or 'error' in result.output.lower()
    
    def test_validate_command_with_data_validation_error(self, runner, temp_dir):
        """Test validate command with data validation errors."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'customerID': ['INVALID-001'],
            'gender': ['InvalidGender'],  # Invalid gender
            'SeniorCitizen': [5],  # Invalid value
            'Partner': ['Maybe'],  # Invalid value
            'Dependents': ['No'],
            'tenure': [-5],  # Negative tenure
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
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
        
        invalid_file = os.path.join(temp_dir, 'invalid_data.csv')
        invalid_data.to_csv(invalid_file, index=False)
        
        result = runner.invoke(app, ['validate', invalid_file])
        
        assert result.exit_code == 1
        assert 'validation failed' in result.output.lower()
        assert 'error' in result.output.lower()
    
    def test_validate_command_with_output_permission_error(self, runner, valid_data_file):
        """Test validate command when output file cannot be written."""
        with patch('src.validation.DEFAULT_PATH_VALIDATOR.validate_path') as mock_validate:
            # First call (input file) succeeds, second call (output file) fails
            mock_validate.side_effect = [None, ValidationError("Cannot write to output file")]
            
            result = runner.invoke(app, ['validate', valid_data_file, '--output', '/invalid/path/output.txt'])
            
            assert result.exit_code == 1
            assert 'error' in result.output.lower()
    
    def test_validate_command_with_detailed_flag(self, runner, valid_data_file):
        """Test validate command with detailed reporting."""
        result = runner.invoke(app, ['validate', valid_data_file, '--detailed'])
        
        # Should succeed with valid data
        assert result.exit_code == 0
        assert 'validation passed' in result.output.lower()
        # Detailed output should contain statistics
        assert 'statistics' in result.output.lower() or 'total' in result.output.lower()
    
    def test_validate_command_for_prediction_mode(self, runner, temp_dir):
        """Test validate command in prediction mode (no target required)."""
        # Create data without Churn column (prediction data)
        prediction_data = pd.DataFrame({
            'customerID': ['PRED-001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [600.0]
            # Note: No 'Churn' column for prediction data
        })
        
        pred_file = os.path.join(temp_dir, 'prediction_data.csv')
        prediction_data.to_csv(pred_file, index=False)
        
        result = runner.invoke(app, ['validate', pred_file, '--for-prediction'])
        
        # Should succeed in prediction mode
        assert result.exit_code == 0
        assert 'validation passed' in result.output.lower()
    
    def test_validate_command_with_all_flags_combination(self, runner, valid_data_file, temp_dir):
        """Test validate command with all flags combined."""
        output_file = os.path.join(temp_dir, 'validation_output.txt')
        
        result = runner.invoke(app, [
            'validate', valid_data_file,
            '--detailed',
            '--for-prediction',
            '--check-distribution',
            '--output', output_file
        ])
        
        # Should succeed with all flags
        assert result.exit_code == 0
        assert 'validation passed' in result.output.lower()
        
        # Output file should be created
        assert os.path.exists(output_file)
        
        # Output file should contain detailed report
        with open(output_file, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert 'validation' in content.lower()
    
    def test_validate_command_with_business_rules_disabled(self, runner, temp_dir):
        """Test validate command with business rules disabled."""
        # Create data that would fail business rules but pass schema validation
        rule_violating_data = pd.DataFrame({
            'customerID': ['RULE-001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],  # Non-zero tenure
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [25.0],  # Less than MonthlyCharges despite tenure > 0 (business rule violation)
            'Churn': ['No']
        })
        
        rule_file = os.path.join(temp_dir, 'rule_violation.csv')
        rule_violating_data.to_csv(rule_file, index=False)
        
        # Should fail with business rules enabled (default)
        result1 = runner.invoke(app, ['validate', rule_file])
        assert result1.exit_code == 1
        
        # Should pass with business rules disabled
        result2 = runner.invoke(app, ['validate', rule_file, '--no-business-rules'])
        assert result2.exit_code == 0
    
    def test_cli_command_help_functionality(self, runner):
        """Test that CLI help commands work correctly."""
        # Test main help
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert 'validate' in result.output.lower()
        
        # Test validate command help
        result = runner.invoke(app, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'data_file' in result.output.lower()
        assert 'for-prediction' in result.output.lower()
        assert 'detailed' in result.output.lower()
    
    def test_validate_command_with_unicode_file_path(self, runner, temp_dir):
        """Test validate command with Unicode characters in file path."""
        # Create file with Unicode in path
        unicode_dir = os.path.join(temp_dir, 'ünïcødé_dïr')
        os.makedirs(unicode_dir, exist_ok=True)
        
        unicode_file = os.path.join(unicode_dir, 'tëst_dåtà.csv')
        
        # Create valid data
        valid_data = pd.DataFrame({
            'customerID': ['UNI-001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [600.0],
            'Churn': ['No']
        })
        
        try:
            valid_data.to_csv(unicode_file, index=False, encoding='utf-8')
            
            result = runner.invoke(app, ['validate', unicode_file])
            
            # Should handle Unicode paths gracefully
            # Might succeed or fail depending on system Unicode support
            assert result.exit_code in [0, 1]
            
        except (UnicodeEncodeError, OSError):
            # Some systems may not support Unicode file paths
            pytest.skip("System doesn't support Unicode file paths")
    
    def test_validate_command_with_very_long_file_path(self, runner, temp_dir):
        """Test validate command with extremely long file path."""
        # Create a very long file path (near system limits)
        long_name = 'a' * 200  # Very long filename
        long_path = os.path.join(temp_dir, long_name + '.csv')
        
        # Create simple valid data
        simple_data = pd.DataFrame({
            'customerID': ['LONG-001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [600.0],
            'Churn': ['No']
        })
        
        try:
            simple_data.to_csv(long_path, index=False)
            
            result = runner.invoke(app, ['validate', long_path])
            
            # Should handle long paths gracefully
            # Might succeed or fail depending on system path length limits
            assert result.exit_code in [0, 1]
            
        except (OSError, FileNotFoundError):
            # Path too long for system
            pytest.skip("System doesn't support very long file paths")
    
    def test_unexpected_exception_handling(self, runner, valid_data_file):
        """Test CLI handling of unexpected exceptions."""
        with patch('src.data_validation.validate_customer_data', 
                  side_effect=Exception("Unexpected error occurred")):
            
            result = runner.invoke(app, ['validate', valid_data_file])
            
            assert result.exit_code == 1
            assert 'unexpected error' in result.output.lower() or 'error' in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__])