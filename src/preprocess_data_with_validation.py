"""
Enhanced data preprocessing with comprehensive validation.

This module extends the original preprocessing functionality with:
- Comprehensive data validation before processing
- Detailed validation reporting
- Data quality checks and warnings
- Business rule validation
- Secure preprocessing pipeline

The validation ensures data integrity and prevents processing of invalid data
that could lead to poor model performance or security issues.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib
from typing import Union, Tuple, Optional
from pathlib import Path

from .constants import PREPROCESSOR_PATH
from .validation import safe_read_csv, ValidationError, DEFAULT_PATH_VALIDATOR
from .data_validation import (
    validate_training_data, 
    ValidationReport, 
    ValidationError as DataValidationError
)
from .logging_config import get_logger

logger = get_logger(__name__)


def preprocess_with_validation(
    df_path: str, 
    *,
    return_preprocessor: bool = False, 
    save_preprocessor: bool = False,
    skip_validation: bool = False,
    validation_report_path: Optional[str] = None
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, ColumnTransformer]]:
    """
    Enhanced preprocessing with comprehensive data validation.
    
    This function performs the same preprocessing as the original preprocess()
    function but adds comprehensive data validation before processing.
    
    Args:
        df_path: Path to the raw CSV data file
        return_preprocessor: Whether to return the fitted ColumnTransformer
        save_preprocessor: If True, save the fitted preprocessor to PREPROCESSOR_PATH
        skip_validation: If True, skip data validation (not recommended)
        validation_report_path: Optional path to save validation report
        
    Returns:
        Tuple of (processed_features, target) or (processed_features, target, preprocessor)
        
    Raises:
        ValidationError: If file access validation fails
        DataValidationError: If data validation fails
        ValueError: If preprocessing fails
    """
    logger.info(f"🔄 Starting preprocessing with validation for: {df_path}")
    
    # Step 1: Secure file loading
    try:
        df: pd.DataFrame = safe_read_csv(df_path)
        logger.info(f"📂 Loaded data with shape: {df.shape}")
    except ValidationError as e:
        logger.error(f"❌ File access validation failed: {e}")
        raise ValueError(f"Failed to read input data safely: {e}") from e
    
    # Step 2: Comprehensive data validation
    if not skip_validation:
        logger.info("🔍 Performing comprehensive data validation...")
        try:
            validation_report = validate_training_data(df)
            
            # Save validation report if requested
            if validation_report_path:
                report_content = validation_report.get_detailed_report()
                with open(validation_report_path, 'w') as f:
                    f.write(report_content)
                logger.info(f"📄 Validation report saved to: {validation_report_path}")
            
            # Check validation results
            if not validation_report.is_valid:
                error_summary = f"Data validation failed with {validation_report.error_count} errors"
                logger.error(f"❌ {error_summary}")
                
                # Log first few errors for debugging
                for i, error in enumerate(validation_report.errors[:3], 1):
                    logger.error(f"  Error {i}: {error}")
                
                if len(validation_report.errors) > 3:
                    logger.error(f"  ... and {len(validation_report.errors) - 3} more errors")
                
                raise DataValidationError(
                    f"{error_summary}. Use skip_validation=True to override (not recommended). "
                    f"Errors: {validation_report.errors[:3]}"
                )
            
            # Log warnings if present
            if validation_report.warnings:
                logger.warning(f"⚠️  Data validation passed with {len(validation_report.warnings)} warnings")
                for warning in validation_report.warnings[:3]:
                    logger.warning(f"  {warning}")
            else:
                logger.info("✅ Data validation passed with no issues")
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            logger.error(f"❌ Error during data validation: {e}")
            raise DataValidationError(f"Data validation failed: {e}") from e
    
    else:
        logger.warning("⚠️  Data validation skipped - processing may fail with invalid data")
    
    # Step 3: Original preprocessing logic with enhanced error handling
    logger.info("🔧 Starting data preprocessing...")
    
    try:
        # Convert 'TotalCharges' to numeric, coercing errors, and fill NaNs
        logger.debug("Converting TotalCharges to numeric")
        total_charges_before = df["TotalCharges"].dtype
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        # Count and log conversion issues
        na_count = df["TotalCharges"].isna().sum()
        if na_count > 0:
            logger.warning(f"⚠️  Converted {na_count} invalid TotalCharges values to NaN")
        
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        logger.debug(f"TotalCharges converted from {total_charges_before} to {df['TotalCharges'].dtype}")
        
        # Encode target variable 'Churn'
        logger.debug("Encoding target variable 'Churn'")
        if "Churn" not in df.columns:
            raise ValueError("Target column 'Churn' not found in data")
        
        label_encoder_churn = LabelEncoder()
        churn_before = df["Churn"].unique()
        df["Churn"] = label_encoder_churn.fit_transform(df["Churn"])
        y: pd.Series = df["Churn"]
        logger.debug(f"Churn encoded: {churn_before} -> {df['Churn'].unique()}")
        
        # Separate features (X)
        logger.debug("Separating features from target")
        X: pd.DataFrame = df.drop("Churn", axis=1)
        
        if "customerID" in X.columns:
            X = X.drop("customerID", axis=1)  # Drop customerID as it's an identifier
            logger.debug("Dropped customerID column")
        
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=["object"]).columns
        numerical_features = X.select_dtypes(include=np.number).columns
        
        logger.info(f"📊 Feature analysis:")
        logger.info(f"  Categorical features ({len(categorical_features)}): {list(categorical_features)}")
        logger.info(f"  Numerical features ({len(numerical_features)}): {list(numerical_features)}")
        
        # Create preprocessing pipelines
        logger.debug("Creating preprocessing pipeline")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                )
            ],
            remainder="passthrough",  # Keep numerical columns
        )
        
        # Fit and transform
        logger.debug("Fitting and transforming features")
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        try:
            ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
                categorical_features
            )
            all_feature_names = list(ohe_feature_names) + list(numerical_features)
            logger.debug(f"Generated {len(all_feature_names)} feature columns after preprocessing")
        except Exception as e:
            logger.warning(f"Could not generate feature names: {e}")
            all_feature_names = None
        
        # Convert to DataFrame with proper column names
        if all_feature_names:
            X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)
        else:
            X_processed = pd.DataFrame(X_processed, index=X.index)
        
        logger.info(f"✅ Preprocessing completed. Output shape: {X_processed.shape}")
        
        # Save preprocessor if requested
        if save_preprocessor:
            try:
                # Ensure models directory exists 
                Path(PREPROCESSOR_PATH).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(preprocessor, PREPROCESSOR_PATH)
                logger.info(f"💾 Preprocessor saved to: {PREPROCESSOR_PATH}")
            except Exception as e:
                logger.error(f"❌ Failed to save preprocessor: {e}")
                raise ValueError(f"Failed to save preprocessor: {e}") from e
        
        # Return results
        if return_preprocessor:
            return X_processed, y, preprocessor
        else:
            return X_processed, y
            
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise ValueError(f"Preprocessing failed: {e}") from e


def preprocess(
    df_path: str, *, return_preprocessor: bool = False, save_preprocessor: bool = False
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, ColumnTransformer]]:
    """
    Original preprocessing function (backwards compatibility).
    
    This function maintains backwards compatibility with the original preprocessing
    function while adding a warning about the enhanced version.
    
    For new code, consider using preprocess_with_validation() instead.
    """
    logger.warning(
        "Using legacy preprocessing function. Consider using preprocess_with_validation() "
        "for enhanced data validation and error handling."
    )
    
    # Call the original implementation logic
    try:
        df: pd.DataFrame = safe_read_csv(df_path)
    except ValidationError as e:
        raise ValueError(f"Failed to read input data safely: {e}") from e

    # Convert 'TotalCharges' to numeric, coercing errors, and fill NaNs
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target variable 'Churn'
    label_encoder_churn: LabelEncoder = LabelEncoder()
    df["Churn"] = label_encoder_churn.fit_transform(df["Churn"])
    y: pd.Series = df["Churn"]

    # Separate features (X)
    X: pd.DataFrame = df.drop("Churn", axis=1)
    X = X.drop("customerID", axis=1)  # Drop customerID as it's an identifier

    # Identify categorical and numerical features
    categorical_features: pd.Index = X.select_dtypes(include=["object"]).columns
    numerical_features: pd.Index = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for categorical and numerical features
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",  # Keep numerical columns not explicitly transformed
    )

    X_processed: np.ndarray = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    all_feature_names = list(ohe_feature_names) + list(numerical_features)

    # Convert to DataFrame with proper column names
    X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

    # Save preprocessor if needed
    if save_preprocessor:
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

    if return_preprocessor:
        return X_processed, y, preprocessor
    else:
        return X_processed, y


# Export the enhanced function as the primary interface
__all__ = [
    'preprocess_with_validation',
    'preprocess',  # Backwards compatibility
    'ValidationError',
    'DataValidationError'
]


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Preprocessing {input_file} with validation...")
        
        try:
            X, y = preprocess_with_validation(
                input_file, 
                validation_report_path="validation_report.txt"
            )
            print(f"✅ Success! Processed features shape: {X.shape}, target shape: {y.shape}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python preprocess_data_with_validation.py <data_file.csv>")