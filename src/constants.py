"""
Constants for the churn prediction system.

This module now uses configurable paths that can be overridden via environment variables.
The constants maintain backwards compatibility while supporting flexible deployment.
"""

from .path_config import (
    get_model_path,
    get_feature_columns_path,
    get_preprocessor_path,
    get_processed_features_path,
    get_processed_target_path,
    get_mlflow_run_id_path
)

# Configurable paths - these will use environment variables if set
MODEL_PATH = get_model_path("churn_model.joblib")
FEATURE_COLUMNS_PATH = get_feature_columns_path()
RUN_ID_PATH = get_mlflow_run_id_path()
PREPROCESSOR_PATH = get_preprocessor_path()
PROCESSED_FEATURES_PATH = get_processed_features_path()
PROCESSED_TARGET_PATH = get_processed_target_path()

# Static constants (non-path related)
RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
MODEL_ARTIFACT_PATH = "churn_model"

# Accuracy threshold used for monitoring. This value can be overridden
# by the `CHURN_THRESHOLD` environment variable or a CLI argument.
DEFAULT_THRESHOLD = 0.8
THRESHOLD_ENV_VAR = "CHURN_THRESHOLD"
