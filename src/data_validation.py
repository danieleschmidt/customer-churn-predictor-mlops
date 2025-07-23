"""
Comprehensive data validation framework for customer churn prediction.

This module provides extensive validation capabilities including:
- Schema validation for customer data structure
- Business rule validation for data integrity  
- Data quality checks and outlier detection
- ML-specific validation for prediction inputs
- Configurable validation rules and reporting

The framework follows the principle of fail-fast validation while providing
detailed error reporting and warnings for data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from src.logging_config import get_logger
from src.validation import safe_read_csv

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


@dataclass
class ValidationReport:
    """
    Comprehensive validation report containing errors, warnings, and statistics.
    
    Attributes:
        errors: List of validation errors that prevent processing
        warnings: List of warnings about data quality issues
        statistics: Dictionary of data quality statistics
    """
    errors: List[str]
    warnings: List[str]
    statistics: Optional[Dict[str, Any]] = None
    
    @property
    def is_valid(self) -> bool:
        """Return True if validation passed (no errors)."""
        return len(self.errors) == 0
    
    @property
    def error_count(self) -> int:
        """Return number of validation errors."""
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        """Return number of validation warnings."""
        return len(self.warnings)
    
    def get_summary(self) -> str:
        """Generate a human-readable validation summary."""
        if self.is_valid:
            summary = "✅ Validation passed"
            if self.warning_count > 0:
                summary += f" with {self.warning_count} warning(s)"
        else:
            summary = f"❌ Validation failed with {self.error_count} error(s)"
            if self.warning_count > 0:
                summary += f" and {self.warning_count} warning(s)"
        
        return summary
    
    def get_detailed_report(self) -> str:
        """Generate detailed validation report."""
        lines = [self.get_summary(), ""]
        
        if self.errors:
            lines.append("🚨 ERRORS:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
            lines.append("")
        
        if self.statistics:
            lines.append("📊 STATISTICS:")
            for key, value in self.statistics.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def create_data_schema() -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive schema definition for customer churn data.
    
    Returns:
        Dictionary defining validation rules for each column
    """
    return {
        'customerID': {
            'type': 'string',
            'required': True,
            'unique': True,
            'nullable': False,
            'pattern': r'^[\w-]+$',  # Alphanumeric and dashes
            'description': 'Unique customer identifier'
        },
        'gender': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Male', 'Female'],
            'description': 'Customer gender'
        },
        'SeniorCitizen': {
            'type': 'integer',
            'required': True,
            'nullable': False,
            'allowed_values': [0, 1],
            'description': 'Senior citizen flag (0=No, 1=Yes)'
        },
        'Partner': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No'],
            'description': 'Has partner'
        },
        'Dependents': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No'],
            'description': 'Has dependents'
        },
        'tenure': {
            'type': 'integer',
            'required': True,
            'nullable': False,
            'min_value': 0,
            'max_value': 100,  # Reasonable upper bound
            'description': 'Number of months customer has stayed with company'
        },
        'PhoneService': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No'],
            'description': 'Has phone service'
        },
        'MultipleLines': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No phone service'],
            'description': 'Has multiple phone lines'
        },
        'InternetService': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['DSL', 'Fiber optic', 'No'],
            'description': 'Type of internet service'
        },
        'OnlineSecurity': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has online security service'
        },
        'OnlineBackup': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has online backup service'
        },
        'DeviceProtection': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has device protection service'
        },
        'TechSupport': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has tech support service'
        },
        'StreamingTV': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has streaming TV service'
        },
        'StreamingMovies': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No', 'No internet service'],
            'description': 'Has streaming movies service'
        },
        'Contract': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Month-to-month', 'One year', 'Two year'],
            'description': 'Contract type'
        },
        'PaperlessBilling': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': ['Yes', 'No'],
            'description': 'Uses paperless billing'
        },
        'PaymentMethod': {
            'type': 'categorical',
            'required': True,
            'nullable': False,
            'allowed_values': [
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ],
            'description': 'Payment method'
        },
        'MonthlyCharges': {
            'type': 'float',
            'required': True,
            'nullable': False,
            'min_value': 0.0,
            'max_value': 200.0,  # Reasonable upper bound
            'description': 'Monthly charges amount'
        },
        'TotalCharges': {
            'type': 'float',
            'required': True,
            'nullable': True,  # Can be missing for new customers
            'min_value': 0.0,
            'max_value': 10000.0,  # Reasonable upper bound
            'description': 'Total charges to date'
        },
        'Churn': {
            'type': 'categorical',
            'required': False,  # Not required for prediction inputs
            'nullable': False,
            'allowed_values': ['Yes', 'No'],
            'description': 'Customer churned (target variable)'
        }
    }


class ChurnDataValidator:
    """
    Comprehensive validator for customer churn prediction data.
    
    Provides extensive validation including schema validation, business rules,
    data quality checks, and ML-specific validation.
    """
    
    def __init__(self, schema: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize validator with schema.
        
        Args:
            schema: Custom schema definition. If None, uses default schema.
        """
        self.schema = schema or create_data_schema()
        logger.debug(f"ChurnDataValidator initialized with {len(self.schema)} schema rules")
    
    def validate(self, data: pd.DataFrame, 
                check_distribution: bool = False,
                check_business_rules: bool = True) -> ValidationReport:
        """
        Perform comprehensive validation of customer data.
        
        Args:
            data: DataFrame to validate
            check_distribution: Whether to check feature distributions
            check_business_rules: Whether to apply business rule validation
            
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        statistics = {}
        
        logger.info(f"Validating DataFrame with {len(data)} rows and {len(data.columns)} columns")
        
        # Basic structure validation
        errors.extend(self._validate_structure(data))
        
        # Schema validation
        errors.extend(self._validate_schema(data))
        
        # Data type validation
        errors.extend(self._validate_data_types(data))
        
        # Business rule validation
        if check_business_rules:
            errors.extend(self._validate_business_rules(data))
        
        # Data quality checks (warnings)
        warnings.extend(self._check_data_quality(data))
        
        # Outlier detection (warnings)
        warnings.extend(self._detect_outliers(data))
        
        # Feature distribution checks
        if check_distribution:
            warnings.extend(self._check_feature_distributions(data))
        
        # Generate statistics
        statistics = self._generate_statistics(data)
        
        report = ValidationReport(errors, warnings, statistics)
        logger.info(f"Validation completed: {report.get_summary()}")
        
        return report
    
    def validate_for_prediction(self, data: pd.DataFrame) -> ValidationReport:
        """
        Validate data for prediction (doesn't require target variable).
        
        Args:
            data: DataFrame to validate for prediction
            
        Returns:
            ValidationReport with validation results
        """
        logger.info("Validating data for prediction")
        
        # Create modified schema without Churn requirement
        prediction_schema = self.schema.copy()
        if 'Churn' in prediction_schema:
            prediction_schema['Churn']['required'] = False
        
        # Temporarily use prediction schema
        original_schema = self.schema
        self.schema = prediction_schema
        
        try:
            report = self.validate(data, check_business_rules=True, check_distribution=True)
        finally:
            # Restore original schema
            self.schema = original_schema
        
        return report
    
    def _validate_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic DataFrame structure."""
        errors = []
        
        if data.empty:
            errors.append("DataFrame is empty")
            return errors
        
        if len(data.columns) == 0:
            errors.append("DataFrame has no columns")
        
        # Check for required columns
        required_columns = [col for col, rules in self.schema.items() 
                           if rules.get('required', False)]
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {sorted(missing_columns)}")
        
        return errors
    
    def _validate_schema(self, data: pd.DataFrame) -> List[str]:
        """Validate data against schema rules."""
        errors = []
        
        for column, rules in self.schema.items():
            if column not in data.columns:
                continue
            
            # Check uniqueness
            if rules.get('unique', False):
                duplicates = data[column].duplicated().sum()
                if duplicates > 0:
                    errors.append(f"Column '{column}' has {duplicates} duplicate values")
            
            # Check null values
            if not rules.get('nullable', True):
                null_count = data[column].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{column}' has {null_count} null values but is not nullable")
        
        return errors
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[str]:
        """Validate data types and value constraints."""
        errors = []
        
        for column, rules in self.schema.items():
            if column not in data.columns:
                continue
            
            series = data[column].dropna()  # Skip null values for type checking
            if len(series) == 0:
                continue
            
            # Validate categorical values
            if rules['type'] == 'categorical' and 'allowed_values' in rules:
                invalid_values = set(series.unique()) - set(rules['allowed_values'])
                if invalid_values:
                    errors.append(f"Column '{column}' has invalid values: {sorted(invalid_values)}")
            
            # Validate numeric ranges
            elif rules['type'] in ['integer', 'float']:
                try:
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    
                    # Check for conversion errors
                    conversion_errors = numeric_series.isnull().sum() - series.isnull().sum()
                    if conversion_errors > 0:
                        errors.append(f"Column '{column}' has {conversion_errors} values that cannot be converted to {rules['type']}")
                    
                    # Check value ranges
                    valid_numeric = numeric_series.dropna()
                    if len(valid_numeric) > 0:
                        if 'min_value' in rules:
                            below_min = (valid_numeric < rules['min_value']).sum()
                            if below_min > 0:
                                errors.append(f"Column '{column}' has {below_min} values below minimum {rules['min_value']}")
                        
                        if 'max_value' in rules:
                            above_max = (valid_numeric > rules['max_value']).sum()
                            if above_max > 0:
                                errors.append(f"Column '{column}' has {above_max} values above maximum {rules['max_value']}")
                        
                        # Check allowed values for integer types
                        if rules['type'] == 'integer' and 'allowed_values' in rules:
                            invalid_ints = set(valid_numeric.unique()) - set(rules['allowed_values'])
                            if invalid_ints:
                                errors.append(f"Column '{column}' has invalid integer values: {sorted(invalid_ints)}")
                
                except Exception as e:
                    errors.append(f"Error validating numeric column '{column}': {str(e)}")
        
        return errors
    
    def _validate_business_rules(self, data: pd.DataFrame) -> List[str]:
        """Validate business logic and consistency rules."""
        errors = []
        
        # Rule 1: TotalCharges should be >= MonthlyCharges for customers with tenure >= 1
        if all(col in data.columns for col in ['TotalCharges', 'MonthlyCharges', 'tenure']):
            try:
                mask = (data['tenure'] >= 1) & data['TotalCharges'].notna() & data['MonthlyCharges'].notna()
                invalid_total = data[mask & (data['TotalCharges'] < data['MonthlyCharges'])]
                if len(invalid_total) > 0:
                    errors.append(f"Business rule violation: {len(invalid_total)} customers have TotalCharges less than MonthlyCharges despite tenure >= 1")
            except Exception as e:
                logger.warning(f"Error checking TotalCharges business rule: {e}")
        
        # Rule 2: Internet service consistency
        if all(col in data.columns for col in ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']):
            try:
                no_internet_mask = data['InternetService'] == 'No'
                internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                   'TechSupport', 'StreamingTV', 'StreamingMovies']
                
                for service in internet_services:
                    if service in data.columns:
                        # If no internet service, these should be 'No internet service'
                        inconsistent = data[no_internet_mask & (data[service] != 'No internet service')]
                        if len(inconsistent) > 0:
                            errors.append(f"Business rule violation: {len(inconsistent)} customers have no internet service but {service} is not 'No internet service'")
            except Exception as e:
                logger.warning(f"Error checking internet service consistency: {e}")
        
        # Rule 3: Phone service consistency
        if all(col in data.columns for col in ['PhoneService', 'MultipleLines']):
            try:
                no_phone_mask = data['PhoneService'] == 'No'
                inconsistent = data[no_phone_mask & (data['MultipleLines'] != 'No phone service')]
                if len(inconsistent) > 0:
                    errors.append(f"Business rule violation: {len(inconsistent)} customers have no phone service but MultipleLines is not 'No phone service'")
            except Exception as e:
                logger.warning(f"Error checking phone service consistency: {e}")
        
        return errors
    
    def _check_data_quality(self, data: pd.DataFrame) -> List[str]:
        """Check data quality and generate warnings."""
        warnings = []
        
        # Check missing data rates
        for column in data.columns:
            missing_rate = data[column].isnull().mean()
            if missing_rate > 0.1:  # More than 10% missing
                warnings.append(f"Column '{column}' has {missing_rate:.1%} missing values")
        
        # Check data distribution issues
        for column, rules in self.schema.items():
            if column not in data.columns or rules['type'] != 'categorical':
                continue
            
            # Check for highly imbalanced categorical variables
            value_counts = data[column].value_counts()
            if len(value_counts) > 1:
                max_proportion = value_counts.iloc[0] / len(data)
                if max_proportion > 0.95:
                    warnings.append(f"Column '{column}' is highly imbalanced: {max_proportion:.1%} are '{value_counts.index[0]}'")
        
        return warnings
    
    def _detect_outliers(self, data: pd.DataFrame) -> List[str]:
        """Detect outliers in numeric columns."""
        warnings = []
        
        numeric_columns = [col for col, rules in self.schema.items() 
                          if col in data.columns and rules['type'] in ['integer', 'float']]
        
        for column in numeric_columns:
            try:
                series = pd.to_numeric(data[column], errors='coerce').dropna()
                if len(series) < 4:  # Need minimum data for outlier detection
                    continue
                
                # Use IQR method for outlier detection
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                if len(outliers) > 0:
                    outlier_rate = len(outliers) / len(series)
                    warnings.append(f"Column '{column}' has {len(outliers)} outliers ({outlier_rate:.1%} of data)")
            
            except Exception as e:
                logger.debug(f"Error detecting outliers in column '{column}': {e}")
        
        return warnings
    
    def _check_feature_distributions(self, data: pd.DataFrame) -> List[str]:
        """Check feature distributions for ML validation."""
        warnings = []
        
        # This is a placeholder for more sophisticated distribution checking
        # In practice, you would compare against reference distributions from training data
        numeric_columns = [col for col, rules in self.schema.items() 
                          if col in data.columns and rules['type'] in ['integer', 'float']]
        
        for column in numeric_columns:
            try:
                series = pd.to_numeric(data[column], errors='coerce').dropna()
                if len(series) == 0:
                    continue
                
                # Check for extreme skewness
                if len(series) > 3:
                    skewness = abs(series.skew())
                    if skewness > 2:
                        warnings.append(f"Column '{column}' has high skewness ({skewness:.2f}), may affect model performance")
            
            except Exception as e:
                logger.debug(f"Error checking distribution for column '{column}': {e}")
        
        return warnings
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality statistics."""
        stats = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_data_rate': data.isnull().mean().mean(),
            'duplicate_rows': data.duplicated().sum()
        }
        
        # Add column-specific statistics
        for column in data.columns:
            if column in self.schema:
                col_stats = {
                    'missing_rate': data[column].isnull().mean(),
                    'unique_values': data[column].nunique()
                }
                
                if self.schema[column]['type'] in ['integer', 'float']:
                    numeric_data = pd.to_numeric(data[column], errors='coerce')
                    col_stats.update({
                        'mean': numeric_data.mean(),
                        'std': numeric_data.std(),
                        'min': numeric_data.min(),
                        'max': numeric_data.max()
                    })
                
                stats[f'{column}_stats'] = col_stats
        
        return stats


def validate_customer_data(data: Union[pd.DataFrame, str, Path], 
                          for_prediction: bool = False,
                          **kwargs) -> ValidationReport:
    """
    Validate customer churn data with comprehensive checks.
    
    Args:
        data: DataFrame or path to CSV file containing customer data
        for_prediction: If True, validates for prediction (no target required)
        **kwargs: Additional arguments passed to validator
        
    Returns:
        ValidationReport with validation results
        
    Raises:
        ValidationError: If file cannot be read or critical validation fails
    """
    logger.info("Starting customer data validation")
    
    # Load data if path provided
    if isinstance(data, (str, Path)):
        try:
            data = safe_read_csv(str(data))
            logger.info(f"Loaded data from {data} with shape {data.shape}")
        except Exception as e:
            raise ValidationError(f"Failed to load data: {e}")
    
    # Validate data type
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame or file path, got {type(data)}")
    
    # Perform validation
    validator = ChurnDataValidator()
    
    if for_prediction:
        report = validator.validate_for_prediction(data)
    else:
        report = validator.validate(data, **kwargs)
    
    logger.info(f"Validation completed: {report.get_summary()}")
    return report


# Convenience functions for common validation tasks
def validate_training_data(data: Union[pd.DataFrame, str, Path]) -> ValidationReport:
    """Validate data for model training (requires target variable)."""
    return validate_customer_data(data, for_prediction=False, check_business_rules=True)


def validate_prediction_data(data: Union[pd.DataFrame, str, Path]) -> ValidationReport:
    """Validate data for prediction (target variable optional)."""
    return validate_customer_data(data, for_prediction=True, check_distribution=True)


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Validating {file_path}...")
        report = validate_customer_data(file_path)
        print(report.get_detailed_report())
    else:
        print("Usage: python data_validation.py <data_file.csv>")