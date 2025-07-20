"""
Environment Variable Configuration and Validation

This module provides centralized configuration and validation for all environment
variables used throughout the application.
"""

import os
import logging
from typing import Optional, Union, Any
from dataclasses import dataclass, field

from src.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentValidationError(Exception):
    """Raised when environment variable validation fails."""
    pass


@dataclass
class EnvConfig:
    """
    Centralized environment variable configuration with validation.
    
    All environment variables are validated when the class is instantiated.
    Invalid values will raise EnvironmentValidationError with specific guidance.
    """
    
    # MLflow Configuration
    mlflow_run_id: Optional[str] = field(default=None)
    
    # Model Performance
    churn_threshold: float = field(default=0.8)
    
    # Logging Configuration  
    log_level: str = field(default="INFO")
    log_file: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate all environment variables after initialization."""
        self._load_from_environment()
        self._validate_all()
    
    def _load_from_environment(self) -> None:
        """Load values from environment variables."""
        # MLflow Run ID
        env_run_id = os.environ.get("MLFLOW_RUN_ID")
        if env_run_id:
            self.mlflow_run_id = env_run_id
        
        # Churn Threshold
        env_threshold = os.environ.get("CHURN_THRESHOLD")
        if env_threshold:
            try:
                self.churn_threshold = float(env_threshold)
            except ValueError:
                raise EnvironmentValidationError(
                    f"CHURN_THRESHOLD must be a valid float, got: {env_threshold}"
                )
        
        # Log Level
        env_log_level = os.environ.get("LOG_LEVEL")
        if env_log_level:
            self.log_level = env_log_level
        
        # Log File
        env_log_file = os.environ.get("LOG_FILE")
        if env_log_file:
            self.log_file = env_log_file
    
    def _validate_all(self) -> None:
        """Validate all configuration values."""
        self._validate_mlflow_run_id()
        self._validate_churn_threshold()
        self._validate_log_level()
        self._validate_log_file()
    
    def _validate_mlflow_run_id(self) -> None:
        """Validate MLflow run ID format."""
        if self.mlflow_run_id is None:
            return
        
        # Basic format validation for MLflow run ID (32-character hex string)
        if not isinstance(self.mlflow_run_id, str):
            raise EnvironmentValidationError(
                f"MLFLOW_RUN_ID must be a string, got: {type(self.mlflow_run_id)}"
            )
        
        if len(self.mlflow_run_id) != 32:
            raise EnvironmentValidationError(
                f"MLFLOW_RUN_ID must be 32 characters long, got: {len(self.mlflow_run_id)}"
            )
        
        if not all(c in '0123456789abcdefABCDEF' for c in self.mlflow_run_id):
            raise EnvironmentValidationError(
                f"MLFLOW_RUN_ID must contain only hexadecimal characters, got: {self.mlflow_run_id}"
            )
    
    def _validate_churn_threshold(self) -> None:
        """Validate churn threshold is between 0.0 and 1.0."""
        if not isinstance(self.churn_threshold, (int, float)):
            raise EnvironmentValidationError(
                f"CHURN_THRESHOLD must be a number, got: {type(self.churn_threshold)}"
            )
        
        if not 0.0 <= self.churn_threshold <= 1.0:
            raise EnvironmentValidationError(
                f"CHURN_THRESHOLD must be between 0.0 and 1.0, got: {self.churn_threshold}"
            )
    
    def _validate_log_level(self) -> None:
        """Validate log level is a valid logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if not isinstance(self.log_level, str):
            raise EnvironmentValidationError(
                f"LOG_LEVEL must be a string, got: {type(self.log_level)}"
            )
        
        if self.log_level.upper() not in valid_levels:
            raise EnvironmentValidationError(
                f"LOG_LEVEL must be one of {valid_levels}, got: {self.log_level}"
            )
        
        # Normalize to uppercase
        self.log_level = self.log_level.upper()
    
    def _validate_log_file(self) -> None:
        """Validate log file path and permissions."""
        if self.log_file is None:
            return
        
        if not isinstance(self.log_file, str):
            raise EnvironmentValidationError(
                f"LOG_FILE must be a string, got: {type(self.log_file)}"
            )
        
        # Check if the parent directory exists
        parent_dir = os.path.dirname(self.log_file)
        if parent_dir and not os.path.exists(parent_dir):
            raise EnvironmentValidationError(
                f"LOG_FILE parent directory does not exist: {parent_dir}"
            )
        
        # Check if we can write to the location
        if os.path.exists(self.log_file):
            if not os.access(self.log_file, os.W_OK):
                raise EnvironmentValidationError(
                    f"LOG_FILE is not writable: {self.log_file}"
                )
        else:
            # Check if we can write to the parent directory
            test_dir = parent_dir if parent_dir else os.getcwd()
            if not os.access(test_dir, os.W_OK):
                raise EnvironmentValidationError(
                    f"Cannot write to LOG_FILE directory: {test_dir}"
                )
    
    def get_log_level_numeric(self) -> int:
        """Get the numeric log level for the logging module."""
        return getattr(logging, self.log_level)
    
    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            'mlflow_run_id': self.mlflow_run_id,
            'churn_threshold': self.churn_threshold,
            'log_level': self.log_level,
            'log_file': self.log_file,
        }


def validate_environment() -> EnvConfig:
    """
    Validate all environment variables and return configuration.
    
    This function should be called at application startup to ensure
    all environment variables are valid before proceeding.
    
    Returns:
        EnvConfig: Validated environment configuration
        
    Raises:
        EnvironmentValidationError: If any environment variable is invalid
    """
    try:
        config = EnvConfig()
        logger.info("Environment variable validation successful")
        logger.debug(f"Configuration: {config.to_dict()}")
        return config
    except EnvironmentValidationError:
        logger.error("Environment variable validation failed")
        raise


# Global configuration instance (validated on import)
# Import this to get validated configuration throughout the application
try:
    env_config = validate_environment()
except EnvironmentValidationError as e:
    logger.error(f"Failed to validate environment variables: {e}")
    # Re-raise to prevent application startup with invalid config
    raise


__all__ = ['env_config', 'validate_environment', 'EnvironmentValidationError', 'EnvConfig']