"""
Configurable path management system.

This module provides a centralized way to manage file paths that can be configured
via environment variables, supporting different deployment environments like
development, staging, production, and containerized deployments.
"""

import os
from pathlib import Path
from typing import Optional, Union
from .logging_config import get_logger

logger = get_logger(__name__)


class PathConfig:
    """
    Manages configurable file paths for the churn prediction system.
    
    Supports environment-based configuration for different deployment scenarios:
    - Development: Local directories
    - Docker: Container-based paths
    - Production: System-wide installation paths
    """
    
    def __init__(
        self,
        base_dir: str = ".",
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        processed_dir: Optional[str] = None
    ):
        """
        Initialize path configuration.
        
        Args:
            base_dir: Base directory for all paths
            data_dir: Directory for data files (default: base_dir/data)
            models_dir: Directory for model artifacts (default: base_dir/models)
            logs_dir: Directory for log files (default: base_dir/logs)
            processed_dir: Directory for processed data (default: data_dir/processed)
        """
        self.base_dir = base_dir
        self.data_dir = data_dir or os.path.join(base_dir, "data")
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        self.logs_dir = logs_dir or os.path.join(base_dir, "logs")
        self.processed_dir = processed_dir or os.path.join(self.data_dir, "processed")
        
        logger.debug(f"PathConfig initialized with base_dir={base_dir}")
    
    @classmethod
    def from_environment(cls) -> 'PathConfig':
        """
        Create PathConfig from environment variables.
        
        Environment variables:
            CHURN_BASE_DIR: Base directory for all paths
            CHURN_DATA_DIR: Data directory
            CHURN_MODELS_DIR: Models directory
            CHURN_LOGS_DIR: Logs directory
            CHURN_PROCESSED_DIR: Processed data directory
        
        Returns:
            PathConfig instance configured from environment
        """
        base_dir = os.getenv('CHURN_BASE_DIR', '.')
        
        # Allow absolute paths or relative to base_dir
        data_dir = os.getenv('CHURN_DATA_DIR')
        if data_dir and not os.path.isabs(data_dir):
            data_dir = os.path.join(base_dir, data_dir)
        
        models_dir = os.getenv('CHURN_MODELS_DIR')
        if models_dir and not os.path.isabs(models_dir):
            models_dir = os.path.join(base_dir, models_dir)
        
        logs_dir = os.getenv('CHURN_LOGS_DIR')
        if logs_dir and not os.path.isabs(logs_dir):
            logs_dir = os.path.join(base_dir, logs_dir)
        
        processed_dir = os.getenv('CHURN_PROCESSED_DIR')
        if processed_dir and not os.path.isabs(processed_dir):
            if data_dir:
                processed_dir = os.path.join(data_dir, processed_dir)
            else:
                processed_dir = os.path.join(base_dir, 'data', processed_dir)
        
        logger.info(f"Configured paths from environment: base_dir={base_dir}")
        
        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            processed_dir=processed_dir
        )
    
    def get_model_path(self, filename: str = "churn_model.joblib") -> str:
        """
        Get path for model file.
        
        Args:
            filename: Model filename
            
        Returns:
            Full path to model file
        """
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.models_dir, filename)
    
    def get_data_path(self, *path_parts: str) -> str:
        """
        Get path for data file with optional subdirectories.
        
        Args:
            *path_parts: Path components (subdirs and filename)
            
        Returns:
            Full path to data file
        """
        if len(path_parts) == 1 and os.path.isabs(path_parts[0]):
            return path_parts[0]
        return os.path.join(self.data_dir, *path_parts)
    
    def get_log_path(self, filename: str) -> str:
        """
        Get path for log file.
        
        Args:
            filename: Log filename
            
        Returns:
            Full path to log file
        """
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.logs_dir, filename)
    
    def get_processed_path(self, filename: str) -> str:
        """
        Get path for processed data file.
        
        Args:
            filename: Processed data filename
            
        Returns:
            Full path to processed data file
        """
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.processed_dir, filename)
    
    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        directories = [self.data_dir, self.models_dir, self.logs_dir, self.processed_dir]
        
        for directory in directories:
            if directory:  # Skip None values
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")


# Default configuration instance factory
def get_default_path_config() -> PathConfig:
    """
    Get a default path configuration instance.
    
    This creates a new PathConfig instance each time it's called.
    For consistent behavior, applications should create their own
    PathConfig instance and pass it around via dependency injection.
    
    Returns:
        New PathConfig instance configured from environment if available,
        otherwise with defaults
    """
    # Check if environment variables are set, use them if available
    if any(key.startswith('CHURN_') for key in os.environ):
        return PathConfig.from_environment()
    return PathConfig()


# Backwards compatibility function (deprecated)
def configure_paths_from_env() -> PathConfig:
    """
    Configure paths from environment variables.
    
    Returns:
        PathConfig instance configured from environment
        
    Note: This function is deprecated. Use PathConfig.from_environment() 
    and dependency injection instead.
    """
    logger.warning(
        "configure_paths_from_env() is deprecated. "
        "Use PathConfig.from_environment() and dependency injection instead."
    )
    return PathConfig.from_environment()


# Convenience functions for common paths with optional config parameter
def get_model_path(
    filename: str = "churn_model.joblib", 
    config: Optional[PathConfig] = None
) -> str:
    """
    Get model file path.
    
    Args:
        filename: Model filename
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to model file
    """
    if config is None:
        config = get_default_path_config()
    return config.get_model_path(filename)


def get_data_path(*path_parts: str, config: Optional[PathConfig] = None) -> str:
    """
    Get data file path.
    
    Args:
        *path_parts: Path components (subdirs and filename)
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to data file
    """
    if config is None:
        config = get_default_path_config()
    return config.get_data_path(*path_parts)


def get_log_path(filename: str, config: Optional[PathConfig] = None) -> str:
    """
    Get log file path.
    
    Args:
        filename: Log filename
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to log file
    """
    if config is None:
        config = get_default_path_config()
    return config.get_log_path(filename)


def get_processed_path(filename: str, config: Optional[PathConfig] = None) -> str:
    """
    Get processed data file path.
    
    Args:
        filename: Processed data filename
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to processed data file
    """
    if config is None:
        config = get_default_path_config()
    return config.get_processed_path(filename)


# Standard path getters with environment variable support
def get_feature_columns_path(config: Optional[PathConfig] = None) -> str:
    """
    Get feature columns JSON file path.
    
    Args:
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to feature columns file
    """
    # Check for specific environment variable first
    env_path = os.getenv('CHURN_FEATURE_COLUMNS_PATH')
    if env_path:
        return env_path
    return get_model_path("feature_columns.json", config=config)


def get_preprocessor_path(config: Optional[PathConfig] = None) -> str:
    """
    Get preprocessor file path.
    
    Args:
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to preprocessor file
    """
    env_path = os.getenv('CHURN_PREPROCESSOR_PATH')
    if env_path:
        return env_path
    return get_model_path("preprocessor.joblib", config=config)


def get_processed_features_path(config: Optional[PathConfig] = None) -> str:
    """
    Get processed features CSV file path.
    
    Args:
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to processed features file
    """
    env_path = os.getenv('CHURN_PROCESSED_FEATURES_PATH')
    if env_path:
        return env_path
    return get_processed_path("processed_features.csv", config=config)


def get_processed_target_path(config: Optional[PathConfig] = None) -> str:
    """
    Get processed target CSV file path.
    
    Args:
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to processed target file
    """
    env_path = os.getenv('CHURN_PROCESSED_TARGET_PATH')
    if env_path:
        return env_path
    return get_processed_path("processed_target.csv", config=config)


def get_mlflow_run_id_path(config: Optional[PathConfig] = None) -> str:
    """
    Get MLflow run ID file path.
    
    Args:
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to MLflow run ID file
    """
    env_path = os.getenv('CHURN_MLFLOW_RUN_ID_PATH')
    if env_path:
        return env_path
    return get_model_path("mlflow_run_id.txt", config=config)


def get_raw_data_path(
    filename: str = "customer_data.csv", 
    config: Optional[PathConfig] = None
) -> str:
    """
    Get raw data file path.
    
    Args:
        filename: Raw data filename
        config: PathConfig instance (creates default if None)
        
    Returns:
        Full path to raw data file
    """
    env_path = os.getenv('CHURN_RAW_DATA_PATH')
    if env_path:
        return env_path
    return get_data_path("raw", filename, config=config)


# Note: Environment configuration is now handled per-instance basis
# The get_default_path_config() function automatically detects and uses
# environment variables when available


def get_environment_example() -> str:
    """
    Get example environment variable configuration.
    
    Returns:
        String with example environment variable settings
    """
    return """
# Example environment variable configuration for churn prediction paths

# Base directory (all others are relative to this unless absolute)
export CHURN_BASE_DIR="/opt/churn-predictor"

# Main directories
export CHURN_DATA_DIR="/data/churn"
export CHURN_MODELS_DIR="/models/churn"
export CHURN_LOGS_DIR="/var/log/churn"

# Specific file paths (optional, will use defaults if not set)
export CHURN_PROCESSED_FEATURES_PATH="/data/churn/processed/features.csv"
export CHURN_PROCESSED_TARGET_PATH="/data/churn/processed/target.csv"
export CHURN_FEATURE_COLUMNS_PATH="/models/churn/feature_columns.json"
export CHURN_PREPROCESSOR_PATH="/models/churn/preprocessor.joblib"
export CHURN_MLFLOW_RUN_ID_PATH="/models/churn/mlflow_run_id.txt"
export CHURN_RAW_DATA_PATH="/data/churn/raw/customer_data.csv"

# Docker example:
# export CHURN_BASE_DIR="/app"
# export CHURN_DATA_DIR="/app/data"
# export CHURN_MODELS_DIR="/app/models"
# export CHURN_LOGS_DIR="/app/logs"

# Development example:
# export CHURN_BASE_DIR="./workspace"
# export CHURN_DATA_DIR="./workspace/data"
# export CHURN_MODELS_DIR="./workspace/models"
"""