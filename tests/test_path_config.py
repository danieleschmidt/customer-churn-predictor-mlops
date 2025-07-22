"""
Tests for configurable path system.

This module tests the ability to configure all file paths via environment variables
and configuration files, supporting different deployment environments.
"""

import os
import tempfile
import threading
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        assert config.data_dir == "./data"
        assert config.models_dir == "./models"
        assert config.logs_dir == "./logs"
        assert config.processed_dir == "./data/processed"
    
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
        # Create a specific config for testing
        test_config = PathConfig(base_dir="/test")
        
        result = get_model_path("test.joblib", config=test_config)
        assert result == "/test/models/test.joblib"
    
    def test_get_data_path_with_subdirs(self):
        """Test getting data path with subdirectories."""
        test_config = PathConfig(base_dir="/test")
        
        result = get_data_path("processed", "features.csv", config=test_config)
        assert result == "/test/data/processed/features.csv"
    
    def test_standard_paths(self):
        """Test all standard path helper functions."""
        test_config = PathConfig(base_dir="/test")
        
        # Test standard paths with explicit config
        assert get_feature_columns_path(config=test_config) == "/test/models/feature_columns.json"
        assert get_preprocessor_path(config=test_config) == "/test/models/preprocessor.joblib"
        assert get_processed_features_path(config=test_config) == "/test/data/processed/processed_features.csv"
        assert get_processed_target_path(config=test_config) == "/test/data/processed/processed_target.csv"
        assert get_mlflow_run_id_path(config=test_config) == "/test/models/mlflow_run_id.txt"
    
    def test_paths_without_explicit_config(self):
        """Test that path functions work without explicit config (backward compatibility)."""
        # Clear environment to ensure clean defaults
        with patch.dict(os.environ, {}, clear=True):
            # These should work without explicit config, using defaults
            model_path = get_model_path("test.joblib")
            data_path = get_data_path("processed", "test.csv")
            
            # Should contain expected path components
            assert "models/test.joblib" in model_path
            assert "data/processed/test.csv" in data_path


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    def test_configure_from_environment_variables(self):
        """Test configuring paths from environment variables."""
        env_vars = {
            'CHURN_BASE_DIR': '/app',
            'CHURN_DATA_DIR': '/app/data',
            'CHURN_MODELS_DIR': '/app/models',
            'CHURN_LOGS_DIR': '/app/logs',
            'CHURN_FEATURE_COLUMNS_PATH': '/app/models/custom_feature_columns.json',
            'CHURN_PROCESSED_FEATURES_PATH': '/app/data/custom_features.csv'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Create config from environment
            config = PathConfig.from_environment()
            
            # Test that paths are configured correctly using the environment config
            assert get_model_path(config=config) == '/app/models/churn_model.joblib'
            assert get_processed_features_path(config=config) == '/app/data/custom_features.csv'
            assert get_feature_columns_path(config=config) == '/app/models/custom_feature_columns.json'
    
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


class TestThreadSafetyAndInstanceManagement:
    """Test thread safety and instance management without global state."""
    
    def test_pathconfig_instance_isolation(self):
        """Test that PathConfig instances are isolated from each other."""
        config1 = PathConfig(base_dir="/path1")
        config2 = PathConfig(base_dir="/path2")
        
        # Instances should be independent
        assert config1.base_dir != config2.base_dir
        assert config1.get_model_path("test.joblib") != config2.get_model_path("test.joblib")
    
    def test_thread_safety_with_separate_instances(self):
        """Test that separate PathConfig instances work safely in concurrent threads."""
        results = []
        errors = []
        
        def create_and_use_config(thread_id: int):
            try:
                # Each thread gets its own config instance
                config = PathConfig(base_dir=f"/thread{thread_id}")
                model_path = config.get_model_path(f"model{thread_id}.joblib")
                data_path = config.get_data_path("processed", f"data{thread_id}.csv")
                
                # Simulate some processing time
                time.sleep(0.01)
                
                results.append({
                    'thread_id': thread_id,
                    'model_path': model_path,
                    'data_path': data_path,
                    'base_dir': config.base_dir
                })
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Run multiple threads concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_and_use_config, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        # Each thread should have unique paths
        model_paths = [r['model_path'] for r in results]
        assert len(set(model_paths)) == 10  # All unique
        
        # Verify correct paths were generated
        for result in results:
            tid = result['thread_id']
            assert f"/thread{tid}" in result['model_path']
            assert f"model{tid}.joblib" in result['model_path']
            assert f"data{tid}.csv" in result['data_path']
    
    def test_concurrent_pathconfig_creation_from_environment(self):
        """Test concurrent creation of PathConfig instances from environment."""
        test_env = {
            'CHURN_BASE_DIR': '/test/concurrent',
            'CHURN_DATA_DIR': '/test/concurrent/data',
            'CHURN_MODELS_DIR': '/test/concurrent/models'
        }
        
        results = []
        errors = []
        
        def create_config_from_env():
            try:
                with patch.dict(os.environ, test_env):
                    config = PathConfig.from_environment()
                    results.append({
                        'base_dir': config.base_dir,
                        'data_dir': config.data_dir,
                        'models_dir': config.models_dir
                    })
            except Exception as e:
                errors.append(str(e))
        
        # Use ThreadPoolExecutor for better control
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_config_from_env) for _ in range(10)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        # All results should be identical (same environment)
        first_result = results[0]
        for result in results:
            assert result == first_result
    
    def test_pathconfig_method_thread_safety(self):
        """Test that PathConfig methods are thread-safe when called concurrently."""
        config = PathConfig(base_dir="/shared")
        results = []
        errors = []
        
        def call_config_methods(worker_id: int):
            try:
                paths = {
                    'model': config.get_model_path(f"model_{worker_id}.joblib"),
                    'data': config.get_data_path("processed", f"data_{worker_id}.csv"),
                    'log': config.get_log_path(f"log_{worker_id}.log"),
                    'processed': config.get_processed_path(f"processed_{worker_id}.csv")
                }
                results.append((worker_id, paths))
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Run concurrent method calls on the same instance
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(call_config_methods, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()
        
        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        
        # Verify all paths contain the shared base directory
        for worker_id, paths in results:
            for path_type, path in paths.items():
                assert "/shared" in path
                assert f"_{worker_id}." in path  # Unique identifier should be present
    
    def test_dependency_injection_pattern(self):
        """Test that PathConfig can be used with dependency injection pattern."""
        
        # Simulate a service that accepts PathConfig via dependency injection
        class ModelService:
            def __init__(self, path_config: PathConfig):
                self.path_config = path_config
            
            def get_model_location(self, model_name: str) -> str:
                return self.path_config.get_model_path(f"{model_name}.joblib")
            
            def get_training_data_location(self) -> str:
                return self.path_config.get_processed_path("training_data.csv")
        
        # Test with different configurations
        dev_config = PathConfig(base_dir="/dev")
        prod_config = PathConfig(base_dir="/prod")
        
        dev_service = ModelService(dev_config)
        prod_service = ModelService(prod_config)
        
        # Services should use their respective configurations
        dev_path = dev_service.get_model_location("churn_model")
        prod_path = prod_service.get_model_location("churn_model")
        
        assert "/dev" in dev_path
        assert "/prod" in prod_path
        assert dev_path != prod_path
        
        # Training data paths should also be different
        dev_training = dev_service.get_training_data_location()
        prod_training = prod_service.get_training_data_location()
        
        assert "/dev" in dev_training
        assert "/prod" in prod_training
        assert dev_training != prod_training


if __name__ == "__main__":
    pytest.main([__file__])