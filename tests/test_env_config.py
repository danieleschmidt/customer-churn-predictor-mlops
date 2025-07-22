"""
Tests for environment configuration module.

This module tests environment variable validation and configuration
management for the churn prediction system.
"""

import os
import tempfile
import logging
import pytest
from unittest.mock import patch, Mock

from src.env_config import (
    EnvConfig,
    EnvironmentValidationError,
    validate_environment,
    env_config
)


class TestEnvConfigDataclass:
    """Test cases for EnvConfig dataclass."""
    
    def test_default_initialization(self):
        """Test EnvConfig with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = EnvConfig()
            
            assert config.mlflow_run_id is None
            assert config.churn_threshold == 0.8
            assert config.log_level == "INFO"
            assert config.log_file is None
    
    def test_environment_variable_loading(self):
        """Test loading values from environment variables."""
        test_env = {
            'MLFLOW_RUN_ID': '1234567890abcdef1234567890abcdef',
            'CHURN_THRESHOLD': '0.75',
            'LOG_LEVEL': 'DEBUG',
            'LOG_FILE': '/tmp/test.log'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            
            assert config.mlflow_run_id == '1234567890abcdef1234567890abcdef'
            assert config.churn_threshold == 0.75
            assert config.log_level == 'DEBUG'
            assert config.log_file == '/tmp/test.log'
    
    def test_partial_environment_loading(self):
        """Test loading with only some environment variables set."""
        test_env = {
            'CHURN_THRESHOLD': '0.9',
            'LOG_LEVEL': 'WARNING'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            
            assert config.mlflow_run_id is None  # Not set, should be None
            assert config.churn_threshold == 0.9  # From environment
            assert config.log_level == 'WARNING'  # From environment  
            assert config.log_file is None  # Not set, should be None


class TestMLflowRunIDValidation:
    """Test MLflow run ID validation."""
    
    def test_valid_mlflow_run_id(self):
        """Test valid MLflow run ID."""
        valid_run_id = '1234567890abcdef1234567890abcdef'
        test_env = {'MLFLOW_RUN_ID': valid_run_id}
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            assert config.mlflow_run_id == valid_run_id
    
    def test_valid_mlflow_run_id_uppercase(self):
        """Test valid MLflow run ID with uppercase letters."""
        valid_run_id = '1234567890ABCDEF1234567890ABCDEF'
        test_env = {'MLFLOW_RUN_ID': valid_run_id}
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            assert config.mlflow_run_id == valid_run_id
    
    def test_invalid_mlflow_run_id_length(self):
        """Test invalid MLflow run ID (wrong length)."""
        test_env = {'MLFLOW_RUN_ID': '1234567890abcdef'}  # Too short
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "32 characters long" in str(excinfo.value)
    
    def test_invalid_mlflow_run_id_characters(self):
        """Test invalid MLflow run ID (invalid characters)."""
        test_env = {'MLFLOW_RUN_ID': '1234567890abcdef1234567890abcdeg'}  # 'g' is invalid
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "hexadecimal characters" in str(excinfo.value)
    
    def test_none_mlflow_run_id_passes_validation(self):
        """Test that None MLflow run ID passes validation."""
        with patch.dict(os.environ, {}, clear=True):
            config = EnvConfig()
            assert config.mlflow_run_id is None  # Should not raise


class TestChurnThresholdValidation:
    """Test churn threshold validation."""
    
    def test_valid_churn_threshold_float(self):
        """Test valid churn threshold as float."""
        test_env = {'CHURN_THRESHOLD': '0.75'}
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            assert config.churn_threshold == 0.75
            assert isinstance(config.churn_threshold, float)
    
    def test_valid_churn_threshold_boundaries(self):
        """Test valid churn threshold at boundaries."""
        # Test 0.0
        test_env = {'CHURN_THRESHOLD': '0.0'}
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            assert config.churn_threshold == 0.0
        
        # Test 1.0
        test_env = {'CHURN_THRESHOLD': '1.0'}
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            assert config.churn_threshold == 1.0
    
    def test_invalid_churn_threshold_string(self):
        """Test invalid churn threshold (not a number)."""
        test_env = {'CHURN_THRESHOLD': 'not_a_number'}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "valid float" in str(excinfo.value)
    
    def test_invalid_churn_threshold_out_of_range_high(self):
        """Test invalid churn threshold (too high)."""
        test_env = {'CHURN_THRESHOLD': '1.5'}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "between 0.0 and 1.0" in str(excinfo.value)
    
    def test_invalid_churn_threshold_out_of_range_low(self):
        """Test invalid churn threshold (too low)."""
        test_env = {'CHURN_THRESHOLD': '-0.1'}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "between 0.0 and 1.0" in str(excinfo.value)


class TestLogLevelValidation:
    """Test log level validation."""
    
    def test_valid_log_levels(self):
        """Test all valid log levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            test_env = {'LOG_LEVEL': level}
            with patch.dict(os.environ, test_env, clear=True):
                config = EnvConfig()
                assert config.log_level == level
    
    def test_log_level_case_normalization(self):
        """Test that log level is normalized to uppercase."""
        test_cases = ['debug', 'info', 'Debug', 'INFO', 'WaRnInG']
        expected = ['DEBUG', 'INFO', 'DEBUG', 'INFO', 'WARNING']
        
        for test_level, expected_level in zip(test_cases, expected):
            test_env = {'LOG_LEVEL': test_level}
            with patch.dict(os.environ, test_env, clear=True):
                config = EnvConfig()
                assert config.log_level == expected_level
    
    def test_invalid_log_level(self):
        """Test invalid log level."""
        test_env = {'LOG_LEVEL': 'INVALID_LEVEL'}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "must be one of" in str(excinfo.value)
            assert "DEBUG" in str(excinfo.value)


class TestLogFileValidation:
    """Test log file validation."""
    
    def test_valid_log_file_in_existing_directory(self):
        """Test valid log file path in existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, 'test.log')
            test_env = {'LOG_FILE': log_file_path}
            
            with patch.dict(os.environ, test_env, clear=True):
                config = EnvConfig()
                assert config.log_file == log_file_path
    
    def test_valid_log_file_existing_file(self):
        """Test valid log file that already exists and is writable."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                test_env = {'LOG_FILE': temp_file.name}
                
                with patch.dict(os.environ, test_env, clear=True):
                    config = EnvConfig()
                    assert config.log_file == temp_file.name
            finally:
                os.unlink(temp_file.name)
    
    def test_invalid_log_file_nonexistent_directory(self):
        """Test invalid log file with non-existent parent directory."""
        log_file_path = '/nonexistent/directory/test.log'
        test_env = {'LOG_FILE': log_file_path}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError) as excinfo:
                EnvConfig()
            assert "parent directory does not exist" in str(excinfo.value)
    
    def test_log_file_none_passes_validation(self):
        """Test that None log file passes validation."""
        with patch.dict(os.environ, {}, clear=True):
            config = EnvConfig()
            assert config.log_file is None
    
    @patch('os.access')
    def test_invalid_log_file_not_writable_existing_file(self, mock_access):
        """Test invalid log file that exists but is not writable."""
        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock os.access to return False for write permission
            mock_access.return_value = False
            
            test_env = {'LOG_FILE': temp_file.name}
            
            with patch.dict(os.environ, test_env, clear=True):
                with pytest.raises(EnvironmentValidationError) as excinfo:
                    EnvConfig()
                assert "not writable" in str(excinfo.value)
    
    @patch('os.access')
    def test_invalid_log_file_directory_not_writable(self, mock_access):
        """Test invalid log file in non-writable directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, 'test.log')
            
            # Mock os.access to return False for directory write permission
            def access_side_effect(path, mode):
                if path == temp_dir and mode == os.W_OK:
                    return False
                return True
            
            mock_access.side_effect = access_side_effect
            
            test_env = {'LOG_FILE': log_file_path}
            
            with patch.dict(os.environ, test_env, clear=True):
                with pytest.raises(EnvironmentValidationError) as excinfo:
                    EnvConfig()
                assert "Cannot write to LOG_FILE directory" in str(excinfo.value)


class TestEnvConfigMethods:
    """Test EnvConfig utility methods."""
    
    def test_get_log_level_numeric(self):
        """Test getting numeric log level."""
        test_cases = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO), 
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
            ('CRITICAL', logging.CRITICAL)
        ]
        
        for level_name, expected_numeric in test_cases:
            test_env = {'LOG_LEVEL': level_name}
            with patch.dict(os.environ, test_env, clear=True):
                config = EnvConfig()
                assert config.get_log_level_numeric() == expected_numeric
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        test_env = {
            'MLFLOW_RUN_ID': '1234567890abcdef1234567890abcdef',
            'CHURN_THRESHOLD': '0.9',
            'LOG_LEVEL': 'DEBUG',
            'LOG_FILE': '/tmp/test.log'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = EnvConfig()
            config_dict = config.to_dict()
            
            expected = {
                'mlflow_run_id': '1234567890abcdef1234567890abcdef',
                'churn_threshold': 0.9,
                'log_level': 'DEBUG',
                'log_file': '/tmp/test.log'
            }
            
            assert config_dict == expected
            assert isinstance(config_dict, dict)
    
    def test_to_dict_with_none_values(self):
        """Test to_dict with None values."""
        with patch.dict(os.environ, {}, clear=True):
            config = EnvConfig()
            config_dict = config.to_dict()
            
            assert config_dict['mlflow_run_id'] is None
            assert config_dict['log_file'] is None
            assert config_dict['churn_threshold'] == 0.8
            assert config_dict['log_level'] == 'INFO'


class TestValidateEnvironmentFunction:
    """Test the validate_environment function."""
    
    @patch('src.env_config.logger')
    def test_validate_environment_success(self, mock_logger):
        """Test successful environment validation."""
        with patch.dict(os.environ, {}, clear=True):
            config = validate_environment()
            
            assert isinstance(config, EnvConfig)
            mock_logger.info.assert_called_with("Environment variable validation successful")
            mock_logger.debug.assert_called_once()
    
    @patch('src.env_config.logger')
    def test_validate_environment_failure(self, mock_logger):
        """Test environment validation failure."""
        test_env = {'CHURN_THRESHOLD': 'invalid'}
        
        with patch.dict(os.environ, test_env, clear=True):
            with pytest.raises(EnvironmentValidationError):
                validate_environment()
            
            mock_logger.error.assert_called_with("Environment variable validation failed")


class TestGlobalEnvConfigInstance:
    """Test the global env_config instance."""
    
    def test_env_config_exists(self):
        """Test that env_config global instance exists."""
        # The global instance should exist
        assert env_config is not None
        assert isinstance(env_config, EnvConfig)
    
    def test_env_config_has_expected_attributes(self):
        """Test that env_config has expected attributes."""
        assert hasattr(env_config, 'mlflow_run_id')
        assert hasattr(env_config, 'churn_threshold')
        assert hasattr(env_config, 'log_level')
        assert hasattr(env_config, 'log_file')
        assert hasattr(env_config, 'get_log_level_numeric')
        assert hasattr(env_config, 'to_dict')


class TestEnvironmentValidationError:
    """Test EnvironmentValidationError exception."""
    
    def test_environment_validation_error_is_exception(self):
        """Test that EnvironmentValidationError is an Exception."""
        error = EnvironmentValidationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_environment_validation_error_inheritance(self):
        """Test EnvironmentValidationError inheritance."""
        assert issubclass(EnvironmentValidationError, Exception)


class TestComplexScenarios:
    """Test complex validation scenarios."""
    
    def test_mixed_valid_and_invalid_environment(self):
        """Test environment with mix of valid and invalid variables."""
        test_env = {
            'MLFLOW_RUN_ID': '1234567890abcdef1234567890abcdef',  # Valid
            'CHURN_THRESHOLD': '2.0',  # Invalid (out of range)
            'LOG_LEVEL': 'INFO'  # Valid
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # Should fail due to invalid threshold
            with pytest.raises(EnvironmentValidationError):
                EnvConfig()
    
    def test_empty_string_environment_variables(self):
        """Test handling of empty string environment variables."""
        test_env = {
            'MLFLOW_RUN_ID': '',
            'CHURN_THRESHOLD': '',  # Empty strings are falsy, so defaults are used
            'LOG_LEVEL': '',
            'LOG_FILE': ''
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # Empty strings should be ignored and defaults used
            config = EnvConfig()
            
            assert config.mlflow_run_id is None  # Default
            assert config.churn_threshold == 0.8  # Default
            assert config.log_level == 'INFO'  # Default
            assert config.log_file is None  # Default
    
    def test_whitespace_environment_variables(self):
        """Test handling of whitespace-only environment variables."""
        test_env = {
            'LOG_LEVEL': '  INFO  '  # Whitespace around valid value
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # This should fail because whitespace is not handled
            with pytest.raises(EnvironmentValidationError):
                EnvConfig()


if __name__ == "__main__":
    pytest.main([__file__])