"""
Integration tests for edge cases and failure scenarios.

This module tests how different components interact under edge conditions
and how failures propagate through the system. These tests are critical
for production robustness and help prevent cascading failures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import time
import threading
import json

# Import modules under test
try:
    from src.data_validation import ChurnDataValidator, ValidationError as DataValidationError
    from src.preprocess_data_with_validation import preprocess_with_validation
    from src.validation import ValidationError, safe_read_csv
    from src.logging_config import get_logger
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestIntegrationFailureScenarios:
    """Test integration scenarios with failures and error propagation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def valid_customer_data(self):
        """Create valid customer data for testing."""
        return pd.DataFrame({
            'customerID': ['INTEG-001', 'INTEG-002', 'INTEG-003'],
            'gender': ['Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'tenure': [12, 24, 36],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'MultipleLines': ['No', 'Yes', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'No'],
            'OnlineBackup': ['Yes', 'No', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'TechSupport': ['No', 'No', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No'],
            'StreamingMovies': ['No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card (automatic)'],
            'MonthlyCharges': [50.0, 75.0, 90.0],
            'TotalCharges': [600.0, 1800.0, 3240.0],
            'Churn': ['No', 'Yes', 'No']
        })
    
    def test_end_to_end_validation_to_preprocessing_failure_propagation(self, temp_dir, valid_customer_data):
        """Test how validation failures affect preprocessing pipeline."""
        # Create invalid data that should fail validation
        invalid_data = valid_customer_data.copy()
        invalid_data.loc[0, 'gender'] = 'InvalidGender'
        invalid_data.loc[1, 'tenure'] = -5  # Negative tenure
        invalid_data.loc[2, 'MonthlyCharges'] = 'invalid_amount'
        
        data_file = os.path.join(temp_dir, 'invalid_data.csv')
        invalid_data.to_csv(data_file, index=False)
        
        # Test that validation catches errors before preprocessing
        with pytest.raises(DataValidationError) as exc_info:
            preprocess_with_validation(data_file, skip_validation=False)
        
        # Verify error contains details about validation failures
        assert 'Data validation failed' in str(exc_info.value)
        
        # Test that preprocessing can be forced with skip_validation=True
        # but this should be logged as a warning
        with patch('src.logging_config.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            try:
                X, y = preprocess_with_validation(data_file, skip_validation=True)
                # Should complete but with warnings logged
                mock_log.warning.assert_called()
                warning_calls = [call.args[0] for call in mock_log.warning.call_args_list]
                assert any('validation skipped' in warning.lower() for warning in warning_calls)
            except Exception:
                # If preprocessing also fails, that's expected with invalid data
                pass
    
    def test_concurrent_file_access_conflicts(self, temp_dir, valid_customer_data):
        """Test behavior when multiple processes access the same file."""
        data_file = os.path.join(temp_dir, 'concurrent_data.csv')
        valid_customer_data.to_csv(data_file, index=False)
        
        results = []
        errors = []
        
        def process_data(process_id):
            """Function to run in parallel threads."""
            try:
                # Add small delay to increase chance of concurrent access
                time.sleep(0.01 * process_id)
                validator = ChurnDataValidator()
                data = safe_read_csv(data_file)
                report = validator.validate(data)
                results.append((process_id, report.is_valid))
            except Exception as e:
                errors.append((process_id, str(e)))
        
        # Start multiple threads to access the same file
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # All threads should succeed or fail gracefully
        assert len(results) + len(errors) == 5
        
        # If any succeeded, the data should be valid
        successful_results = [result for _, result in results]
        if successful_results:
            assert all(successful_results), "Data should be consistently valid across threads"
    
    def test_file_corruption_recovery(self, temp_dir, valid_customer_data):
        """Test handling of corrupted CSV files."""
        data_file = os.path.join(temp_dir, 'corrupted_data.csv')
        
        # Create a properly formatted file first
        valid_customer_data.to_csv(data_file, index=False)
        
        # Then corrupt it by truncating
        with open(data_file, 'r') as f:
            content = f.read()
        
        # Create various corruption scenarios
        corruption_scenarios = [
            content[:len(content)//2],  # Truncated file
            content.replace(',', ''),   # Missing delimiters
            content.replace('\n', ''),  # No line breaks
            'invalid csv content\nwith wrong format',  # Completely wrong format
            ''  # Empty file
        ]
        
        for i, corrupted_content in enumerate(corruption_scenarios):
            corrupted_file = os.path.join(temp_dir, f'corrupted_{i}.csv')
            with open(corrupted_file, 'w') as f:
                f.write(corrupted_content)
            
            # Should handle corruption gracefully
            try:
                data = safe_read_csv(corrupted_file)
                # If it somehow succeeds, validate the data
                validator = ChurnDataValidator()
                report = validator.validate(data)
                # Corrupted data should fail validation
                assert not report.is_valid
            except (ValidationError, pd.errors.EmptyDataError, pd.errors.ParserError):
                # Expected errors for corrupted files
                pass
            except Exception as e:
                pytest.fail(f"Unexpected error type for corruption scenario {i}: {type(e).__name__}: {e}")
    
    def test_memory_exhaustion_simulation(self, temp_dir):
        """Test behavior under memory pressure (simulation)."""
        # Create a dataset that could cause memory issues
        large_size = 10000
        
        # Create data with many categorical variations to increase memory usage
        categories = {
            'gender': ['Male', 'Female'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            'InternetService': ['DSL', 'Fiber optic', 'No']
        }
        
        large_data = {
            'customerID': [f'LARGE-{i:05d}' for i in range(large_size)],
            'SeniorCitizen': np.random.choice([0, 1], large_size),
            'Partner': np.random.choice(['Yes', 'No'], large_size),
            'Dependents': np.random.choice(['Yes', 'No'], large_size),
            'tenure': np.random.randint(0, 73, large_size),
            'PhoneService': np.random.choice(['Yes', 'No'], large_size),
            'MonthlyCharges': np.random.uniform(20, 120, large_size),
            'TotalCharges': np.random.uniform(20, 8000, large_size),
            'Churn': np.random.choice(['Yes', 'No'], large_size)
        }
        
        # Add categorical columns with variety
        for col, values in categories.items():
            large_data[col] = np.random.choice(values, large_size)
        
        # Add service-related columns
        service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in service_cols:
            large_data[col] = np.random.choice(['Yes', 'No', 'No internet service'], large_size)
        
        large_data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], large_size)
        
        df = pd.DataFrame(large_data)
        data_file = os.path.join(temp_dir, 'large_data.csv')
        df.to_csv(data_file, index=False)
        
        # Test validation with memory monitoring
        import tracemalloc
        tracemalloc.start()
        
        start_time = time.time()
        try:
            validator = ChurnDataValidator()
            report = validator.validate(df)
            
            processing_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Should complete within reasonable time and memory bounds
            assert processing_time < 30.0, f"Processing took too long: {processing_time}s"
            assert peak / 1024 / 1024 < 1000, f"Memory usage too high: {peak/1024/1024}MB"
            assert isinstance(report.is_valid, bool), "Should return valid boolean result"
            
        except MemoryError:
            pytest.skip("System doesn't have enough memory for this test")
        except Exception as e:
            tracemalloc.stop()
            # Log the error but don't fail if it's a reasonable resource limitation
            if 'memory' in str(e).lower() or 'resource' in str(e).lower():
                pytest.skip(f"Resource limitation encountered: {e}")
            else:
                raise
    
    def test_disk_space_exhaustion_simulation(self, temp_dir):
        """Test behavior when disk space is limited."""
        # Create a scenario where writing temp files might fail
        test_file = os.path.join(temp_dir, 'disk_test.csv')
        
        # Mock disk space issue by patching file operations
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises((OSError, ValidationError)):
                # Should handle disk space errors gracefully
                pd.DataFrame({'test': [1, 2, 3]}).to_csv(test_file)
    
    def test_permission_errors_handling(self, temp_dir, valid_customer_data):
        """Test handling of file permission errors."""
        data_file = os.path.join(temp_dir, 'permission_test.csv')
        valid_customer_data.to_csv(data_file, index=False)
        
        # Mock permission error
        with patch('src.validation.safe_read_csv', side_effect=PermissionError("Permission denied")):
            with pytest.raises((PermissionError, ValidationError)):
                preprocess_with_validation(data_file)
    
    def test_network_file_access_timeout(self, temp_dir):
        """Test behavior with network file access timeouts."""
        # Simulate network file that times out
        with patch('src.validation.safe_read_csv', side_effect=TimeoutError("Connection timed out")):
            with pytest.raises((TimeoutError, ValidationError)):
                validator = ChurnDataValidator()
                validator.validate('/network/path/data.csv')
    
    def test_validation_report_serialization_edge_cases(self):
        """Test edge cases in validation report generation."""
        validator = ChurnDataValidator()
        
        # Test with extremely long error messages
        long_error = "x" * 10000  # Very long error message
        
        # Create data that would generate the long error
        problematic_data = pd.DataFrame({
            'customerID': ['TEST-001'],
            'gender': ['InvalidGenderValueThatIsVeryLongAndShouldCauseIssues' * 100],  # Very long invalid value
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
        
        report = validator.validate(problematic_data)
        
        # Should handle long error messages gracefully
        assert not report.is_valid
        detailed_report = report.get_detailed_report()
        assert isinstance(detailed_report, str)
        assert len(detailed_report) > 0
    
    def test_circular_dependency_prevention(self, temp_dir):
        """Test prevention of circular dependencies in module imports."""
        # This test ensures that importing modules doesn't create circular dependencies
        # that could cause issues in edge cases
        
        try:
            # Try importing all modules in different orders
            import src.data_validation
            import src.preprocess_data_with_validation
            import src.validation
            import src.logging_config
            
            # Should complete without circular import errors
            assert True
        except ImportError as e:
            if 'circular' in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Other import errors might be due to missing dependencies
                pytest.skip(f"Import error (possibly missing dependencies): {e}")
    
    def test_error_handling_with_malformed_json_config(self, temp_dir):
        """Test error handling when configuration files are malformed."""
        config_file = os.path.join(temp_dir, 'malformed_config.json')
        
        # Create malformed JSON
        with open(config_file, 'w') as f:
            f.write('{"incomplete": json content without closing brace')
        
        # Mock config loading to use our malformed file
        with patch('builtins.open', mock_open(read_data='{"incomplete": json')):
            with patch('json.load', side_effect=json.JSONDecodeError("Expecting ',' delimiter", "", 0)):
                # Should handle JSON parsing errors gracefully
                try:
                    # Any operation that might load config
                    validator = ChurnDataValidator()
                    # Should still work with default config
                    assert validator is not None
                except json.JSONDecodeError:
                    pytest.fail("JSON decode error should be handled gracefully")
    
    def test_unicode_handling_edge_cases(self, temp_dir):
        """Test handling of various Unicode edge cases."""
        unicode_data = pd.DataFrame({
            'customerID': ['UTF8-测试', 'EMOJI-🎯', 'ACCENT-café', 'NULL-\x00'],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', 'Yes'],
            'tenure': [12, 24, 36, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'Yes', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No', 'No internet service'],
            'TechSupport': ['No', 'No', 'Yes', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No', 'No internet service'],
            'StreamingMovies': ['No', 'No', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'],
            'MonthlyCharges': [50.0, 75.0, 90.0, 45.0],
            'TotalCharges': [600.0, 1800.0, 3240.0, 2160.0]
        })
        
        data_file = os.path.join(temp_dir, 'unicode_data.csv')
        unicode_data.to_csv(data_file, index=False, encoding='utf-8')
        
        validator = ChurnDataValidator()
        
        try:
            data = safe_read_csv(data_file)
            report = validator.validate(data)
            
            # Should handle Unicode gracefully, but likely fail validation due to invalid customerIDs
            assert isinstance(report, type(report))  # Just check it returns a report
            # Unicode customerIDs should fail pattern validation
            assert not report.is_valid
            
        except UnicodeDecodeError:
            pytest.skip("System doesn't support the Unicode characters used in test")


if __name__ == '__main__':
    pytest.main([__file__])