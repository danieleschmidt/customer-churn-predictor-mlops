"""
Constants for the churn prediction system.

This module provides static constants and factory functions for creating paths.
For consistent path behavior, applications should use dependency injection
with PathConfig instances rather than these module-level constants.
"""

from typing import Optional
from .path_config import (
    PathConfig,
    get_model_path,
    get_feature_columns_path,
    get_preprocessor_path,
    get_processed_features_path,
    get_processed_target_path,
    get_mlflow_run_id_path
)

# Static constants (non-path related)
RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
MODEL_ARTIFACT_PATH = "churn_model"

# Accuracy threshold used for monitoring. This value can be overridden
# by the `CHURN_THRESHOLD` environment variable or a CLI argument.
DEFAULT_THRESHOLD = 0.8
THRESHOLD_ENV_VAR = "CHURN_THRESHOLD"

# Path factory functions (use these instead of static constants)
def get_model_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get model file path using provided config or default."""
    return get_model_path("churn_model.joblib", config=config)

def get_feature_columns_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get feature columns file path using provided config or default."""
    return get_feature_columns_path(config=config)

def get_run_id_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get MLflow run ID file path using provided config or default."""
    return get_mlflow_run_id_path(config=config)

def get_preprocessor_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get preprocessor file path using provided config or default."""
    return get_preprocessor_path(config=config)

def get_processed_features_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get processed features file path using provided config or default."""
    return get_processed_features_path(config=config)

def get_processed_target_path_constant(config: Optional[PathConfig] = None) -> str:
    """Get processed target file path using provided config or default."""
    return get_processed_target_path(config=config)

# Backwards compatibility - these are deprecated but maintained for legacy code
# Use the factory functions above for new code
MODEL_PATH = get_model_path("churn_model.joblib")
FEATURE_COLUMNS_PATH = get_feature_columns_path()
RUN_ID_PATH = get_mlflow_run_id_path()
PREPROCESSOR_PATH = get_preprocessor_path()
PROCESSED_FEATURES_PATH = get_processed_features_path()
PROCESSED_TARGET_PATH = get_processed_target_path()
