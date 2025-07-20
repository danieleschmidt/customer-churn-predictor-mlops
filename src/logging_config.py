"""
Centralized logging configuration for the churn prediction system.

This module provides a unified logging interface that replaces print statements
throughout the codebase with structured, configurable logging.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_rotation: bool = True,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file for file output
        format_string: Custom format string for log messages
        enable_rotation: Whether to enable log file rotation
        max_bytes: Maximum size for each log file (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    # Default format string if not provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Read configuration from environment variables
    env_level = os.getenv('LOG_LEVEL', '').upper()
    env_log_file = os.getenv('LOG_FILE')
    
    # Override with environment variables if available
    if env_level:
        try:
            level = getattr(logging, env_level)
        except AttributeError:
            # Invalid level in environment variable, use default
            pass
    
    if env_log_file:
        log_file = env_log_file
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        if enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for our modules
    churn_logger = logging.getLogger('src')
    churn_logger.setLevel(level)
    
    scripts_logger = logging.getLogger('scripts')
    scripts_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_mlflow_logging(level: int = logging.WARNING) -> None:
    """
    Configure MLflow logging to reduce noise.
    
    Args:
        level: Logging level for MLflow (default: WARNING)
    """
    mlflow_loggers = [
        'mlflow',
        'mlflow.tracking',
        'mlflow.store',
        'mlflow.utils'
    ]
    
    for logger_name in mlflow_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            return result
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


# Auto-configure logging when module is imported
def _auto_configure() -> None:
    """Auto-configure logging when the module is first imported."""
    # Only setup if no handlers are configured yet
    if not logging.getLogger().handlers:
        setup_logging()
        
        # Configure MLflow to be less verbose
        configure_mlflow_logging()


# Perform auto-configuration
_auto_configure()