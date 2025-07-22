"""
Tests for constants module.

This module tests both legacy constants and new factory functions
for path generation with proper dependency injection support.
"""

import os
import pytest
from unittest.mock import patch, Mock

from src.constants import (
    # Static constants
    RUN_ID_ENV_VAR,
    MODEL_ARTIFACT_PATH,
    DEFAULT_THRESHOLD,
    THRESHOLD_ENV_VAR,
    # Factory functions
    get_model_path_constant,
    get_feature_columns_path_constant,
    get_run_id_path_constant,
    get_preprocessor_path_constant,
    get_processed_features_path_constant,
    get_processed_target_path_constant,
    # Legacy constants
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    RUN_ID_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_FEATURES_PATH,
    PROCESSED_TARGET_PATH,
)
from src.path_config import PathConfig


class TestStaticConstants:
    """Test static constants that don't depend on path configuration."""
    
    def test_run_id_env_var(self):
        """Test that RUN_ID_ENV_VAR is correct."""
        assert RUN_ID_ENV_VAR == "MLFLOW_RUN_ID"
        assert isinstance(RUN_ID_ENV_VAR, str)
    
    def test_model_artifact_path(self):
        """Test that MODEL_ARTIFACT_PATH is correct."""
        assert MODEL_ARTIFACT_PATH == "churn_model"
        assert isinstance(MODEL_ARTIFACT_PATH, str)
    
    def test_default_threshold(self):
        """Test that DEFAULT_THRESHOLD is valid."""
        assert DEFAULT_THRESHOLD == 0.8
        assert isinstance(DEFAULT_THRESHOLD, (int, float))
        assert 0.0 <= DEFAULT_THRESHOLD <= 1.0
    
    def test_threshold_env_var(self):
        """Test that THRESHOLD_ENV_VAR is correct."""
        assert THRESHOLD_ENV_VAR == "CHURN_THRESHOLD"
        assert isinstance(THRESHOLD_ENV_VAR, str)


class TestLegacyConstants:
    """Test legacy path constants for backwards compatibility."""
    
    def test_legacy_constants_are_strings(self):
        """Test that all legacy constants are valid path strings."""
        legacy_constants = [
            MODEL_PATH,
            FEATURE_COLUMNS_PATH,
            RUN_ID_PATH,
            PREPROCESSOR_PATH,
            PROCESSED_FEATURES_PATH,
            PROCESSED_TARGET_PATH,
        ]
        
        for constant in legacy_constants:
            assert isinstance(constant, str)
            assert len(constant) > 0
            # Should contain some path-like structure
            assert "/" in constant or "\\" in constant
    
    def test_model_path_contains_expected_filename(self):
        """Test that MODEL_PATH contains expected model filename."""
        assert "churn_model.joblib" in MODEL_PATH
    
    def test_feature_columns_path_contains_expected_filename(self):
        """Test that FEATURE_COLUMNS_PATH contains expected filename."""
        assert "feature_columns.json" in FEATURE_COLUMNS_PATH
    
    def test_preprocessor_path_contains_expected_filename(self):
        """Test that PREPROCESSOR_PATH contains expected filename."""
        assert "preprocessor.joblib" in PREPROCESSOR_PATH
    
    def test_processed_paths_contain_expected_filenames(self):
        """Test processed data paths contain expected filenames."""
        assert "processed_features.csv" in PROCESSED_FEATURES_PATH
        assert "processed_target.csv" in PROCESSED_TARGET_PATH
    
    def test_mlflow_run_id_path_contains_expected_filename(self):
        """Test that RUN_ID_PATH contains expected filename."""
        assert "mlflow_run_id.txt" in RUN_ID_PATH


class TestFactoryFunctions:
    """Test the new factory functions for path generation."""
    
    def test_get_model_path_constant_default(self):
        """Test model path factory with default config."""
        path = get_model_path_constant()
        
        assert isinstance(path, str)
        assert "churn_model.joblib" in path
        assert len(path) > 0
    
    def test_get_model_path_constant_with_custom_config(self):
        """Test model path factory with custom config."""
        config = PathConfig(base_dir="/custom")
        path = get_model_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/custom/models/churn_model.joblib" == path
    
    def test_get_feature_columns_path_constant_default(self):
        """Test feature columns path factory with default config."""
        path = get_feature_columns_path_constant()
        
        assert isinstance(path, str)
        assert "feature_columns.json" in path
    
    def test_get_feature_columns_path_constant_with_custom_config(self):
        """Test feature columns path factory with custom config."""
        config = PathConfig(base_dir="/test")
        path = get_feature_columns_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/test/models/feature_columns.json" == path
    
    def test_get_run_id_path_constant_default(self):
        """Test run ID path factory with default config."""
        path = get_run_id_path_constant()
        
        assert isinstance(path, str)
        assert "mlflow_run_id.txt" in path
    
    def test_get_run_id_path_constant_with_custom_config(self):
        """Test run ID path factory with custom config."""
        config = PathConfig(base_dir="/test")
        path = get_run_id_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/test/models/mlflow_run_id.txt" == path
    
    def test_get_preprocessor_path_constant_default(self):
        """Test preprocessor path factory with default config."""
        path = get_preprocessor_path_constant()
        
        assert isinstance(path, str)
        assert "preprocessor.joblib" in path
    
    def test_get_preprocessor_path_constant_with_custom_config(self):
        """Test preprocessor path factory with custom config."""
        config = PathConfig(base_dir="/test")
        path = get_preprocessor_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/test/models/preprocessor.joblib" == path
    
    def test_get_processed_features_path_constant_default(self):
        """Test processed features path factory with default config."""
        path = get_processed_features_path_constant()
        
        assert isinstance(path, str)
        assert "processed_features.csv" in path
    
    def test_get_processed_features_path_constant_with_custom_config(self):
        """Test processed features path factory with custom config."""
        config = PathConfig(base_dir="/test")
        path = get_processed_features_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/test/data/processed/processed_features.csv" == path
    
    def test_get_processed_target_path_constant_default(self):
        """Test processed target path factory with default config."""
        path = get_processed_target_path_constant()
        
        assert isinstance(path, str)
        assert "processed_target.csv" in path
    
    def test_get_processed_target_path_constant_with_custom_config(self):
        """Test processed target path factory with custom config."""
        config = PathConfig(base_dir="/test")
        path = get_processed_target_path_constant(config=config)
        
        assert isinstance(path, str)
        assert "/test/data/processed/processed_target.csv" == path


class TestFactoryFunctionConsistency:
    """Test consistency between factory functions and legacy constants."""
    
    def test_factory_functions_match_legacy_without_config(self):
        """Test that factory functions without config match legacy constants."""
        # Clear environment to ensure clean comparison
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_path_constant() == MODEL_PATH
            assert get_feature_columns_path_constant() == FEATURE_COLUMNS_PATH
            assert get_run_id_path_constant() == RUN_ID_PATH
            assert get_preprocessor_path_constant() == PREPROCESSOR_PATH
            assert get_processed_features_path_constant() == PROCESSED_FEATURES_PATH
            assert get_processed_target_path_constant() == PROCESSED_TARGET_PATH
    
    def test_factory_functions_differ_with_custom_config(self):
        """Test that factory functions with custom config differ from legacy."""
        config = PathConfig(base_dir="/different")
        
        assert get_model_path_constant(config=config) != MODEL_PATH
        assert get_feature_columns_path_constant(config=config) != FEATURE_COLUMNS_PATH
        assert get_run_id_path_constant(config=config) != RUN_ID_PATH
        assert get_preprocessor_path_constant(config=config) != PREPROCESSOR_PATH
        assert get_processed_features_path_constant(config=config) != PROCESSED_FEATURES_PATH
        assert get_processed_target_path_constant(config=config) != PROCESSED_TARGET_PATH


class TestEnvironmentVariableHandling:
    """Test how factory functions handle environment variables."""
    
    def test_factory_functions_respect_environment_variables(self):
        """Test that factory functions respect environment variables."""
        test_env = {
            'CHURN_BASE_DIR': '/env/test',
            'CHURN_MODELS_DIR': '/env/test/models',
            'CHURN_DATA_DIR': '/env/test/data'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # Should use environment configuration when no config provided
            model_path = get_model_path_constant()
            features_path = get_processed_features_path_constant()
            
            assert "/env/test/models" in model_path
            assert "/env/test/data" in features_path
    
    def test_explicit_config_overrides_environment(self):
        """Test that explicit config overrides environment variables."""
        test_env = {
            'CHURN_BASE_DIR': '/env/test',
            'CHURN_MODELS_DIR': '/env/test/models'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            explicit_config = PathConfig(base_dir="/explicit")
            model_path = get_model_path_constant(config=explicit_config)
            
            # Should use explicit config, not environment
            assert "/explicit/models" in model_path
            assert "/env/test" not in model_path


class TestPathGeneration:
    """Test path generation logic."""
    
    def test_all_factory_functions_return_absolute_paths_with_custom_config(self):
        """Test that all factory functions return absolute paths with custom config."""
        config = PathConfig(base_dir="/absolute/test")
        
        factory_functions = [
            get_model_path_constant,
            get_feature_columns_path_constant,
            get_run_id_path_constant,
            get_preprocessor_path_constant,
            get_processed_features_path_constant,
            get_processed_target_path_constant,
        ]
        
        for factory_function in factory_functions:
            path = factory_function(config=config)
            assert os.path.isabs(path), f"Path not absolute: {path}"
            assert path.startswith("/absolute/test"), f"Path doesn't start with base: {path}"
    
    def test_factory_functions_generate_different_paths(self):
        """Test that factory functions generate different paths."""
        config = PathConfig(base_dir="/test")
        
        paths = [
            get_model_path_constant(config=config),
            get_feature_columns_path_constant(config=config),
            get_run_id_path_constant(config=config),
            get_preprocessor_path_constant(config=config),
            get_processed_features_path_constant(config=config),
            get_processed_target_path_constant(config=config),
        ]
        
        # All paths should be unique
        assert len(paths) == len(set(paths)), "Factory functions generated duplicate paths"
        
        # All paths should exist and be non-empty
        for path in paths:
            assert isinstance(path, str)
            assert len(path) > 0
    
    def test_factory_functions_handle_none_config_gracefully(self):
        """Test that factory functions handle None config gracefully."""
        factory_functions = [
            get_model_path_constant,
            get_feature_columns_path_constant,
            get_run_id_path_constant,
            get_preprocessor_path_constant,
            get_processed_features_path_constant,
            get_processed_target_path_constant,
        ]
        
        for factory_function in factory_functions:
            # Should not raise exceptions
            path = factory_function(config=None)
            assert isinstance(path, str)
            assert len(path) > 0


class TestBackwardsCompatibilityImport:
    """Test that backwards compatibility imports work."""
    
    def test_can_import_all_legacy_constants(self):
        """Test that all legacy constants can be imported."""
        # This test ensures import statements work
        from src.constants import (
            MODEL_PATH,
            FEATURE_COLUMNS_PATH, 
            RUN_ID_PATH,
            PREPROCESSOR_PATH,
            PROCESSED_FEATURES_PATH,
            PROCESSED_TARGET_PATH
        )
        
        # All should be strings
        constants = [
            MODEL_PATH, FEATURE_COLUMNS_PATH, RUN_ID_PATH,
            PREPROCESSOR_PATH, PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH
        ]
        
        for constant in constants:
            assert isinstance(constant, str)
            assert len(constant) > 0
    
    def test_can_import_all_factory_functions(self):
        """Test that all factory functions can be imported."""
        from src.constants import (
            get_model_path_constant,
            get_feature_columns_path_constant,
            get_run_id_path_constant,
            get_preprocessor_path_constant,
            get_processed_features_path_constant,
            get_processed_target_path_constant
        )
        
        # All should be callable
        functions = [
            get_model_path_constant, get_feature_columns_path_constant,
            get_run_id_path_constant, get_preprocessor_path_constant,
            get_processed_features_path_constant, get_processed_target_path_constant
        ]
        
        for func in functions:
            assert callable(func)


if __name__ == "__main__":
    pytest.main([__file__])