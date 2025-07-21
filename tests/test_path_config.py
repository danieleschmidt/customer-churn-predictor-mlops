"""
Tests for configurable path system.

This module tests the ability to configure all file paths via environment variables
and configuration files, supporting different deployment environments.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.path_config import (
    PathConfig,
    get_model_path,
    get_data_path,
    get_log_path,
    get_feature_columns_path,
    get_preprocessor_path,
    get_processed_features_path,
    get_processed_target_path,
    get_mlflow_run_id_path,
    configure_paths_from_env
)


class TestPathConfig:
    """Test cases for PathConfig class."""
    
    def test_default_paths(self):
        """Test that default paths are set correctly."""
        config = PathConfig()
        
        assert config.base_dir == "."
        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.logs_dir == "logs"
        assert config.processed_dir == "data/processed"
    
    def test_custom_base_directory(self):
        """Test setting custom base directory."""
        config = PathConfig(base_dir="/custom/base")
        
        assert config.base_dir == "/custom/base"
        assert config.data_dir == "/custom/base/data"
        assert config.models_dir == "/custom/base/models"
    
    def test_custom_subdirectories(self):
        """Test setting custom subdirectories."""
        config = PathConfig(
            data_dir="custom_data",
            models_dir="custom_models",
            logs_dir="custom_logs"
        )
        
        assert config.data_dir == "custom_data"
        assert config.models_dir == "custom_models"
        assert config.logs_dir == "custom_logs"
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'CHURN_BASE_DIR': '/tmp/churn',
            'CHURN_DATA_DIR': '/tmp/churn/my_data',
            'CHURN_MODELS_DIR': '/tmp/churn/my_models'
        }):
            config = PathConfig.from_environment()
            
            assert config.base_dir == '/tmp/churn'
            assert config.data_dir == '/tmp/churn/my_data'
            assert config.models_dir == '/tmp/churn/my_models'
    
    def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        config = PathConfig(base_dir="/base")
        
        model_path = config.get_model_path("test_model.joblib")
        assert model_path == "/base/models/test_model.joblib"
        
        data_path = config.get_data_path("raw", "test.csv")
        assert data_path == "/base/data/raw/test.csv"
    
    def test_absolute_path_passthrough(self):
        """Test that absolute paths are not modified."""
        config = PathConfig()
        
        abs_path = "/absolute/path/to/file.csv"
        result = config.get_data_path(abs_path)
        assert result == abs_path


class TestPathHelperFunctions:
    """Test cases for path helper functions."""
    
    def test_get_model_path_default(self):
        """Test getting model path with default configuration."""
        with patch('src.path_config._global_config') as mock_config:
            mock_config.get_model_path.return_value = "models/test.joblib"
            
            result = get_model_path("test.joblib")
            assert result == "models/test.joblib"
            mock_config.get_model_path.assert_called_once_with("test.joblib")
    
    def test_get_data_path_with_subdirs(self):
        """Test getting data path with subdirectories."""
        with patch('src.path_config._global_config') as mock_config:
            mock_config.get_data_path.return_value = "data/processed/features.csv"
            
            result = get_data_path("processed", "features.csv")
            assert result == "data/processed/features.csv"
            mock_config.get_data_path.assert_called_once_with("processed", "features.csv")
    
    def test_standard_paths(self):
        """Test all standard path helper functions."""
        with patch('src.path_config._global_config') as mock_config:
            # Setup mock returns
            mock_config.get_model_path.side_effect = lambda x: f"models/{x}"
            mock_config.get_data_path.side_effect = lambda *args: f"data/{'/'.join(args)}"
            mock_config.get_log_path.side_effect = lambda x: f"logs/{x}"
            
            # Test standard paths
            assert get_feature_columns_path() == "models/feature_columns.json"
            assert get_preprocessor_path() == "models/preprocessor.joblib"
            assert get_processed_features_path() == "data/processed/processed_features.csv"
            assert get_processed_target_path() == "data/processed/processed_target.csv"
            assert get_mlflow_run_id_path() == "models/mlflow_run_id.txt"


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    def test_configure_from_environment_variables(self):
        """Test configuring paths from environment variables."""
        env_vars = {
            'CHURN_BASE_DIR': '/app',
            'CHURN_DATA_DIR': '/app/data',
            'CHURN_MODELS_DIR': '/app/models',
            'CHURN_LOGS_DIR': '/app/logs',
            'CHURN_MODEL_PATH': '/app/models/custom_model.joblib',
            'CHURN_FEATURES_PATH': '/app/data/custom_features.csv'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            configure_paths_from_env()
            
            # Test that paths are configured correctly
            assert get_model_path() == '/app/models/custom_model.joblib'
            assert get_processed_features_path() == '/app/data/custom_features.csv'
    
    def test_partial_environment_override(self):
        """Test that partial environment variables work with defaults."""
        with patch.dict(os.environ, {'CHURN_MODELS_DIR': '/custom/models'}, clear=True):
            configure_paths_from_env()
            
            # Custom models dir should be used
            model_path = get_model_path("test.joblib")
            assert "/custom/models" in model_path
    
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        with patch.dict(os.environ, {'CHURN_BASE_DIR': ''}, clear=True):
            # Should fall back to defaults without error
            configure_paths_from_env()
            
            # Should still work with defaults
            result = get_model_path("test.joblib")
            assert result is not None


class TestPathConfigIntegration:
    """Integration tests for path configuration."""
    
    def test_docker_environment_simulation(self):
        """Test configuration for Docker deployment."""
        docker_env = {
            'CHURN_BASE_DIR': '/app',
            'CHURN_DATA_DIR': '/app/data',
            'CHURN_MODELS_DIR': '/app/models',
            'CHURN_LOGS_DIR': '/app/logs'
        }
        
        with patch.dict(os.environ, docker_env):
            config = PathConfig.from_environment()
            
            assert config.base_dir == '/app'
            assert config.get_model_path('model.joblib') == '/app/models/model.joblib'
            assert config.get_data_path('processed', 'data.csv') == '/app/data/processed/data.csv'
    
    def test_development_environment_simulation(self):
        """Test configuration for development environment."""
        dev_env = {
            'CHURN_BASE_DIR': './dev_workspace',
            'CHURN_MODELS_DIR': './dev_workspace/models'
        }
        
        with patch.dict(os.environ, dev_env):
            config = PathConfig.from_environment()
            
            assert config.base_dir == './dev_workspace'
            assert 'dev_workspace/models' in config.get_model_path('model.joblib')
    
    def test_production_environment_simulation(self):
        """Test configuration for production environment."""
        prod_env = {
            'CHURN_BASE_DIR': '/opt/churn-predictor',
            'CHURN_DATA_DIR': '/data/churn',
            'CHURN_MODELS_DIR': '/models/churn',
            'CHURN_LOGS_DIR': '/var/log/churn'
        }
        
        with patch.dict(os.environ, prod_env):
            config = PathConfig.from_environment()
            
            assert config.base_dir == '/opt/churn-predictor'
            assert config.data_dir == '/data/churn'
            assert config.models_dir == '/models/churn'
            assert config.logs_dir == '/var/log/churn'


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code."""
    
    def test_constants_still_work(self):
        """Test that existing constants still function."""
        # Import should work without errors
        from src.constants import MODEL_PATH, PROCESSED_FEATURES_PATH
        
        # Values should be strings (paths)
        assert isinstance(MODEL_PATH, str)
        assert isinstance(PROCESSED_FEATURES_PATH, str)
    
    def test_gradual_migration_support(self):
        """Test that old and new path systems can coexist."""
        # Configure new system
        configure_paths_from_env()
        
        # Both old and new approaches should work
        old_path = "models/churn_model.joblib"  # Direct string
        new_path = get_model_path("churn_model.joblib")  # Configurable
        
        # Should be compatible
        assert isinstance(old_path, str)
        assert isinstance(new_path, str)


if __name__ == "__main__":
    pytest.main([__file__])