"""
Input validation utilities for secure file operations and data processing.

This module provides comprehensive validation for file paths, data types, and ranges
to prevent security vulnerabilities and ensure data integrity.
"""

import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class PathValidator:
    """Validates file paths for security and existence."""
    
    def __init__(self, allowed_directories: Optional[List[str]] = None, 
                 allowed_extensions: Optional[List[str]] = None):
        """
        Initialize path validator.
        
        Args:
            allowed_directories: List of allowed directory paths (default: project subdirs)
            allowed_extensions: List of allowed file extensions (default: common data formats)
        """
        self.allowed_directories = allowed_directories or [
            "data/", 
            "models/", 
            "logs/", 
            "tests/",
            "/tmp/",
            "scripts/"
        ]
        self.allowed_extensions = allowed_extensions or [
            ".csv", ".json", ".joblib", ".pkl", ".txt", ".yaml", ".yml", ".log"
        ]
    
    def validate_path(self, path: Union[str, Path], 
                     must_exist: bool = False,
                     allow_create: bool = True,
                     check_parent: bool = True) -> Path:
        """
        Validate a file path for security and accessibility.
        
        Args:
            path: File path to validate
            must_exist: Whether the file must exist
            allow_create: Whether creating new files is allowed
            check_parent: Whether to validate parent directory exists
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path validation fails
        """
        if not path:
            raise ValidationError("Path cannot be empty")
            
        # Convert to Path object and resolve
        try:
            path_obj = Path(path)
            abs_path = path_obj.resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid path format: {e}")
        
        # Check for path traversal attempts
        if ".." in path_obj.parts:
            raise ValidationError("Path traversal detected (..)")
        
        # Validate against allowed directories
        path_str = str(abs_path)
        allowed = any(
            path_str.startswith(os.path.abspath(allowed_dir)) 
            for allowed_dir in self.allowed_directories
        )
        
        if not allowed:
            raise ValidationError(
                f"Path not in allowed directories: {path}. "
                f"Allowed: {self.allowed_directories}"
            )
        
        # Check file extension
        if abs_path.suffix and abs_path.suffix not in self.allowed_extensions:
            raise ValidationError(
                f"File extension not allowed: {abs_path.suffix}. "
                f"Allowed: {self.allowed_extensions}"
            )
        
        # Check existence requirements
        if must_exist and not abs_path.exists():
            raise ValidationError(f"Required file does not exist: {abs_path}")
        
        # Check parent directory exists if creating new files
        if check_parent and not abs_path.parent.exists():
            if allow_create:
                try:
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {abs_path.parent}")
                except OSError as e:
                    raise ValidationError(f"Cannot create parent directory: {e}")
            else:
                raise ValidationError(f"Parent directory does not exist: {abs_path.parent}")
        
        return abs_path


class DataValidator:
    """Validates data types, ranges, and formats."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1,
                          max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Validate pandas DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
        if max_rows is not None and len(df) > max_rows:
            raise ValidationError(f"DataFrame exceeds maximum {max_rows} rows, got {len(df)}")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Check for completely empty DataFrame
        if df.empty:
            raise ValidationError("DataFrame cannot be completely empty")
        
        return df
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], 
                              min_value: Optional[float] = None,
                              max_value: Optional[float] = None,
                              name: str = "value") -> Union[int, float]:
        """
        Validate numeric value is within specified range.
        
        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            name: Name of the value for error messages
            
        Returns:
            Validated numeric value
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {value}")
        
        return value
    
    @staticmethod
    def validate_string(value: str, 
                       max_length: Optional[int] = None,
                       allowed_chars: Optional[str] = None,
                       name: str = "string") -> str:
        """
        Validate string content and format.
        
        Args:
            value: String to validate
            max_length: Maximum allowed length
            allowed_chars: Pattern of allowed characters (regex)
            name: Name of the value for error messages
            
        Returns:
            Validated string
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{name} must be a string, got {type(value)}")
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(f"{name} exceeds maximum length {max_length}")
        
        if allowed_chars is not None:
            import re
            if not re.match(allowed_chars, value):
                raise ValidationError(f"{name} contains invalid characters")
        
        return value


class MLValidator:
    """Validates machine learning specific inputs."""
    
    @staticmethod
    def validate_model_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate machine learning model hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        validated = {}
        
        # Common validation rules for scikit-learn parameters
        validators = {
            'C': lambda x: DataValidator.validate_numeric_range(x, 0.0001, 1000, 'C'),
            'max_iter': lambda x: DataValidator.validate_numeric_range(int(x), 1, 10000, 'max_iter'),
            'random_state': lambda x: DataValidator.validate_numeric_range(int(x), 0, 2**31-1, 'random_state'),
            'test_size': lambda x: DataValidator.validate_numeric_range(x, 0.01, 0.99, 'test_size'),
            'threshold': lambda x: DataValidator.validate_numeric_range(x, 0.0, 1.0, 'threshold'),
            'penalty': lambda x: DataValidator.validate_string(x, allowed_chars=r'^(l1|l2|elasticnet|none)$', name='penalty'),
            'solver': lambda x: DataValidator.validate_string(x, allowed_chars=r'^(newton-cg|lbfgs|liblinear|sag|saga)$', name='solver')
        }
        
        for key, value in params.items():
            if key in validators:
                try:
                    validated[key] = validators[key](value)
                except ValueError as e:
                    raise ValidationError(f"Invalid {key}: {e}")
            else:
                # For unknown parameters, at least check they're not None
                if value is None:
                    raise ValidationError(f"Parameter {key} cannot be None")
                validated[key] = value
        
        return validated


def safe_read_csv(file_path: Union[str, Path], 
                  validator: Optional[PathValidator] = None,
                  **pandas_kwargs) -> pd.DataFrame:
    """
    Safely read CSV file with path validation.
    
    Args:
        file_path: Path to CSV file
        validator: PathValidator instance (creates default if None)
        **pandas_kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame loaded from CSV
        
    Raises:
        ValidationError: If path validation fails
    """
    if validator is None:
        validator = PathValidator()
    
    validated_path = validator.validate_path(file_path, must_exist=True)
    
    try:
        df = pd.read_csv(validated_path, **pandas_kwargs)
        return DataValidator.validate_dataframe(df)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV {validated_path}: {e}")


def safe_write_csv(df: pd.DataFrame, 
                   file_path: Union[str, Path],
                   validator: Optional[PathValidator] = None,
                   **pandas_kwargs) -> Path:
    """
    Safely write DataFrame to CSV with path validation.
    
    Args:
        df: DataFrame to write
        file_path: Output path for CSV file
        validator: PathValidator instance (creates default if None)
        **pandas_kwargs: Additional arguments for df.to_csv
        
    Returns:
        Validated output path
        
    Raises:
        ValidationError: If validation fails
    """
    if validator is None:
        validator = PathValidator()
    
    # Validate DataFrame
    DataValidator.validate_dataframe(df)
    
    # Validate output path
    validated_path = validator.validate_path(file_path, allow_create=True)
    
    try:
        df.to_csv(validated_path, index=False, **pandas_kwargs)
        logger.info(f"Successfully wrote {len(df)} rows to {validated_path}")
        return validated_path
    except Exception as e:
        raise ValidationError(f"Failed to write CSV {validated_path}: {e}")


# Global default validator instances
DEFAULT_PATH_VALIDATOR = PathValidator()
DEFAULT_DATA_VALIDATOR = DataValidator()
DEFAULT_ML_VALIDATOR = MLValidator()