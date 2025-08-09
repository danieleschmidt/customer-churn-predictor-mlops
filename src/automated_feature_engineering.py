"""
Automated Feature Engineering Pipeline with Advanced Transformations.

This module provides comprehensive automated feature engineering capabilities including:
- Intelligent feature detection and transformation
- Advanced feature creation and selection
- Domain-specific feature engineering patterns
- Temporal and categorical feature handling
- Feature interaction discovery
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, chi2, mutual_info_classif
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import joblib

from .logging_config import get_logger
from .data_validation import validate_customer_data

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for automated feature engineering."""
    # Basic transformations
    enable_scaling: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust, quantile
    
    # Missing value handling
    handle_missing: bool = True
    missing_strategy: str = "smart"  # mean, median, mode, smart, advanced
    
    # Categorical encoding
    categorical_encoding: str = "auto"  # onehot, ordinal, target, auto
    max_categories: int = 10
    
    # Numerical transformations
    enable_log_transform: bool = True
    enable_power_transform: bool = True
    enable_binning: bool = True
    
    # Feature creation
    create_interactions: bool = True
    max_interaction_degree: int = 2
    create_polynomials: bool = False
    polynomial_degree: int = 2
    
    # Temporal features
    extract_temporal_features: bool = True
    temporal_granularities: List[str] = None
    
    # Feature selection
    enable_feature_selection: bool = True
    selection_method: str = "auto"  # univariate, rfe, importance, auto
    max_features: Optional[int] = None
    
    # Advanced features
    create_aggregations: bool = True
    create_domain_features: bool = True
    detect_outliers: bool = True
    
    # Performance settings
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class FeatureMetadata:
    """Metadata about engineered features."""
    name: str
    original_name: Optional[str]
    feature_type: str  # numerical, categorical, temporal, engineered
    transformation: str
    importance_score: float
    correlation_with_target: float
    missing_ratio: float
    unique_values: int
    data_type: str
    creation_method: str


class AdvancedMissingValueImputer(BaseEstimator, TransformerMixin):
    """Advanced missing value imputation with multiple strategies."""
    
    def __init__(self, strategy="smart", random_state=42):
        self.strategy = strategy
        self.random_state = random_state
        self.imputers = {}
        
    def fit(self, X, y=None):
        """Fit imputers for each column."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if self.strategy == "smart":
                    # Choose best strategy per column
                    if X[col].dtype in ['int64', 'float64']:
                        # For numerical: use median for skewed, mean for normal
                        skewness = abs(X[col].skew())
                        self.imputers[col] = ('median' if skewness > 1 else 'mean', 
                                            X[col].median() if skewness > 1 else X[col].mean())
                    else:
                        # For categorical: use mode
                        mode_value = X[col].mode()
                        self.imputers[col] = ('mode', mode_value[0] if len(mode_value) > 0 else 'unknown')
                        
                elif self.strategy == "advanced":
                    # Use predictive imputation
                    if X[col].dtype in ['int64', 'float64']:
                        # Simple regression-based imputation
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=5)
                        # This is simplified - full implementation would be more sophisticated
                        self.imputers[col] = ('knn', imputer)
                    else:
                        # For categorical: use mode
                        mode_value = X[col].mode()
                        self.imputers[col] = ('mode', mode_value[0] if len(mode_value) > 0 else 'unknown')
                else:
                    # Simple strategies
                    if self.strategy == "mean" and X[col].dtype in ['int64', 'float64']:
                        self.imputers[col] = ('mean', X[col].mean())
                    elif self.strategy == "median" and X[col].dtype in ['int64', 'float64']:
                        self.imputers[col] = ('median', X[col].median())
                    else:
                        mode_value = X[col].mode()
                        self.imputers[col] = ('mode', mode_value[0] if len(mode_value) > 0 else 'unknown')
        
        return self
    
    def transform(self, X):
        """Transform data by imputing missing values."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (method, value) in self.imputers.items():
            if col in X.columns:
                if method in ['mean', 'median', 'mode']:
                    X[col].fillna(value, inplace=True)
                elif method == 'knn':
                    # Apply KNN imputation (simplified)
                    mask = X[col].isnull()
                    if mask.any():
                        X[col].fillna(X[col].median(), inplace=True)  # Fallback
        
        return X


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Smart categorical encoding with multiple strategies."""
    
    def __init__(self, encoding_method="auto", max_categories=10, random_state=42):
        self.encoding_method = encoding_method
        self.max_categories = max_categories
        self.random_state = random_state
        self.encoders = {}
        self.encoding_strategies = {}
        
    def fit(self, X, y=None):
        """Fit encoders for categorical columns."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if self.encoding_method == "auto":
                # Choose encoding strategy based on characteristics
                if unique_count <= 2:
                    strategy = "label"
                elif unique_count <= self.max_categories:
                    strategy = "onehot"
                elif y is not None and unique_count <= 20:
                    strategy = "target"
                else:
                    strategy = "ordinal"
            else:
                strategy = self.encoding_method
            
            self.encoding_strategies[col] = strategy
            
            # Fit appropriate encoder
            if strategy == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
            elif strategy == "ordinal" or strategy == "label":
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
                self.encoders[col] = encoder
            elif strategy == "target" and y is not None:
                # Target encoding (simplified)
                target_means = X.groupby(col)[col].count()  # Placeholder
                self.encoders[col] = target_means
        
        return self
    
    def transform(self, X):
        """Transform categorical columns."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X.columns:
                strategy = self.encoding_strategies[col]
                
                if strategy == "onehot":
                    # One-hot encoding
                    encoded = encoder.transform(X[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X = pd.concat([X.drop(col, axis=1), encoded_df], axis=1)
                elif strategy in ["ordinal", "label"]:
                    # Label/Ordinal encoding
                    X[col] = encoder.transform(X[col].astype(str))
                elif strategy == "target":
                    # Target encoding (simplified)
                    X[col] = X[col].map(encoder).fillna(encoder.mean())
        
        return X


class AdvancedNumericalTransformer(BaseEstimator, TransformerMixin):
    """Advanced numerical feature transformations."""
    
    def __init__(self, 
                 enable_log=True, 
                 enable_power=True, 
                 enable_binning=True,
                 random_state=42):
        self.enable_log = enable_log
        self.enable_power = enable_power
        self.enable_binning = enable_binning
        self.random_state = random_state
        self.transformations = {}
        
    def fit(self, X, y=None):
        """Fit transformations for numerical columns."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue
                
            transformations = []
            
            # Analyze distribution
            skewness = abs(col_data.skew())
            kurtosis = abs(col_data.kurtosis())
            
            # Log transformation for highly skewed data
            if self.enable_log and skewness > 2 and col_data.min() > 0:
                transformations.append("log")
            
            # Power transformation for non-normal data
            if self.enable_power and (skewness > 1.5 or kurtosis > 3):
                # Determine best power transformation
                if col_data.min() > 0:
                    transformations.append("boxcox")
                else:
                    transformations.append("yeojohnson")
            
            # Binning for highly skewed or sparse data
            if self.enable_binning and (skewness > 3 or col_data.nunique() / len(col_data) < 0.1):
                n_bins = min(10, max(3, col_data.nunique() // 10))
                bin_edges = pd.qcut(col_data, q=n_bins, duplicates='drop', retbins=True)[1]
                transformations.append(("binning", bin_edges))
            
            if transformations:
                self.transformations[col] = transformations
                
        return self
    
    def transform(self, X):
        """Apply numerical transformations."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, transformations in self.transformations.items():
            if col in X.columns:
                original_col = X[col].copy()
                
                for transform in transformations:
                    if transform == "log":
                        # Log transformation
                        X[f"{col}_log"] = np.log1p(np.maximum(X[col], 0))
                    elif transform == "boxcox":
                        # Box-Cox transformation
                        try:
                            positive_data = np.maximum(X[col], 1e-6)
                            X[f"{col}_boxcox"], _ = boxcox(positive_data)
                        except:
                            pass  # Skip if transformation fails
                    elif transform == "yeojohnson":
                        # Yeo-Johnson transformation
                        try:
                            X[f"{col}_yeojohnson"], _ = yeojohnson(X[col])
                        except:
                            pass  # Skip if transformation fails
                    elif isinstance(transform, tuple) and transform[0] == "binning":
                        # Binning transformation
                        _, bin_edges = transform
                        X[f"{col}_binned"] = pd.cut(X[col], bins=bin_edges, 
                                                   labels=False, duplicates='drop')
        
        return X


class FeatureInteractionGenerator(BaseEstimator, TransformerMixin):
    """Generate feature interactions automatically."""
    
    def __init__(self, max_degree=2, max_features=50, random_state=42):
        self.max_degree = max_degree
        self.max_features = max_features
        self.random_state = random_state
        self.selected_interactions = []
        
    def fit(self, X, y=None):
        """Identify important feature interactions."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return self
            
        # Generate all possible interactions up to max_degree
        from itertools import combinations
        
        interactions = []
        for degree in range(2, self.max_degree + 1):
            for combo in combinations(numerical_cols, degree):
                interactions.append(combo)
        
        # If we have target variable, score interactions
        if y is not None and len(interactions) > 0:
            interaction_scores = []
            
            for interaction in interactions[:100]:  # Limit for performance
                try:
                    # Create interaction feature
                    if len(interaction) == 2:
                        col1, col2 = interaction
                        interaction_feature = X[col1] * X[col2]
                    else:
                        # Higher-order interactions
                        interaction_feature = X[interaction[0]]
                        for col in interaction[1:]:
                            interaction_feature = interaction_feature * X[col]
                    
                    # Calculate correlation with target
                    if y.dtype in ['int64', 'float64']:
                        correlation = abs(np.corrcoef(interaction_feature.fillna(0), y)[0, 1])
                    else:
                        # For categorical targets, use mutual information
                        from sklearn.feature_selection import mutual_info_classif
                        correlation = mutual_info_classif(
                            interaction_feature.fillna(0).values.reshape(-1, 1), 
                            y, random_state=self.random_state
                        )[0]
                    
                    if not np.isnan(correlation):
                        interaction_scores.append((interaction, correlation))
                        
                except Exception as e:
                    continue
            
            # Select top interactions
            interaction_scores.sort(key=lambda x: x[1], reverse=True)
            self.selected_interactions = [x[0] for x in interaction_scores[:self.max_features]]
        else:
            # Without target, select a few random interactions
            np.random.seed(self.random_state)
            self.selected_interactions = list(np.random.choice(
                interactions, size=min(10, len(interactions)), replace=False
            ))
        
        return self
    
    def transform(self, X):
        """Create interaction features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for interaction in self.selected_interactions:
            try:
                if len(interaction) == 2:
                    col1, col2 = interaction
                    feature_name = f"{col1}_x_{col2}"
                    X[feature_name] = X[col1] * X[col2]
                else:
                    # Higher-order interactions
                    feature_name = "_x_".join(interaction)
                    interaction_feature = X[interaction[0]]
                    for col in interaction[1:]:
                        interaction_feature = interaction_feature * X[col]
                    X[feature_name] = interaction_feature
            except Exception as e:
                logger.warning(f"Failed to create interaction {interaction}: {e}")
                continue
        
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""
    
    def __init__(self, granularities=None):
        if granularities is None:
            granularities = ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter']
        self.granularities = granularities
        self.temporal_columns = []
        
    def fit(self, X, y=None):
        """Identify temporal columns."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.temporal_columns = []
        for col in X.columns:
            if X[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                try:
                    # Try to parse as datetime
                    pd.to_datetime(X[col].head(100))
                    self.temporal_columns.append(col)
                except:
                    pass
        
        return self
    
    def transform(self, X):
        """Extract temporal features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col in self.temporal_columns:
            if col in X.columns:
                try:
                    # Convert to datetime if not already
                    if X[col].dtype != 'datetime64[ns]':
                        X[col] = pd.to_datetime(X[col])
                    
                    # Extract features
                    for granularity in self.granularities:
                        if granularity == 'year':
                            X[f"{col}_year"] = X[col].dt.year
                        elif granularity == 'month':
                            X[f"{col}_month"] = X[col].dt.month
                        elif granularity == 'day':
                            X[f"{col}_day"] = X[col].dt.day
                        elif granularity == 'dayofweek':
                            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                        elif granularity == 'hour':
                            X[f"{col}_hour"] = X[col].dt.hour
                        elif granularity == 'quarter':
                            X[f"{col}_quarter"] = X[col].dt.quarter
                        elif granularity == 'dayofyear':
                            X[f"{col}_dayofyear"] = X[col].dt.dayofyear
                        elif granularity == 'week':
                            X[f"{col}_week"] = X[col].dt.isocalendar().week
                    
                    # Create cyclical features for seasonal patterns
                    X[f"{col}_month_sin"] = np.sin(2 * np.pi * X[col].dt.month / 12)
                    X[f"{col}_month_cos"] = np.cos(2 * np.pi * X[col].dt.month / 12)
                    X[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * X[col].dt.dayofweek / 7)
                    X[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * X[col].dt.dayofweek / 7)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract temporal features from {col}: {e}")
                    continue
        
        return X


class DomainSpecificFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate domain-specific features for customer churn prediction."""
    
    def __init__(self):
        self.feature_generators = []
        
    def fit(self, X, y=None):
        """Identify opportunities for domain-specific features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Analyze column names for customer churn domain
        columns_lower = [col.lower() for col in X.columns]
        
        # Customer value features
        if any(term in col for col in columns_lower for term in ['charge', 'price', 'cost', 'revenue']):
            self.feature_generators.append("customer_value")
        
        # Usage pattern features
        if any(term in col for col in columns_lower for term in ['usage', 'minutes', 'data', 'calls']):
            self.feature_generators.append("usage_patterns")
        
        # Service features
        if any(term in col for col in columns_lower for term in ['service', 'plan', 'contract']):
            self.feature_generators.append("service_features")
        
        # Demographic features
        if any(term in col for col in columns_lower for term in ['age', 'senior', 'gender']):
            self.feature_generators.append("demographic_features")
        
        return self
    
    def transform(self, X):
        """Generate domain-specific features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if "customer_value" in self.feature_generators:
            # Customer value features
            charge_cols = [col for col in X.columns if 'charge' in col.lower()]
            if len(charge_cols) >= 2:
                # Average charges
                X['avg_charges'] = X[charge_cols].mean(axis=1)
                # Charge ratio (monthly vs total)
                monthly_cols = [col for col in charge_cols if 'monthly' in col.lower()]
                total_cols = [col for col in charge_cols if 'total' in col.lower()]
                if monthly_cols and total_cols:
                    X['monthly_to_total_ratio'] = X[monthly_cols[0]] / (X[total_cols[0]] + 1e-6)
        
        if "usage_patterns" in self.feature_generators:
            # Usage pattern features
            usage_cols = [col for col in X.columns if any(term in col.lower() 
                         for term in ['usage', 'minutes', 'data', 'calls'])]
            if usage_cols:
                # Total usage
                X['total_usage'] = X[usage_cols].sum(axis=1)
                # Usage diversity (number of different services used)
                X['usage_diversity'] = (X[usage_cols] > 0).sum(axis=1)
        
        if "service_features" in self.feature_generators:
            # Service-related features
            service_cols = [col for col in X.columns if any(term in col.lower() 
                           for term in ['service', 'plan', 'contract'])]
            # Count of services
            if service_cols:
                X['num_services'] = (X[service_cols] != 'No').sum(axis=1)
        
        if "demographic_features" in self.feature_generators:
            # Demographic interaction features
            if 'SeniorCitizen' in X.columns and 'tenure' in X.columns:
                X['senior_tenure_interaction'] = X['SeniorCitizen'] * X['tenure']
        
        return X


class AutomatedFeatureEngineer:
    """Main automated feature engineering pipeline."""
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.pipeline = None
        self.feature_metadata = {}
        self.original_columns = []
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the feature engineering pipeline and transform data."""
        logger.info("Starting automated feature engineering...")
        
        self.original_columns = X.columns.tolist()
        
        # Build pipeline
        transformers = []
        
        # 1. Missing value imputation
        if self.config.handle_missing:
            imputer = AdvancedMissingValueImputer(
                strategy=self.config.missing_strategy,
                random_state=self.config.random_state
            )
            transformers.append(("imputer", imputer))
        
        # 2. Temporal feature extraction
        if self.config.extract_temporal_features:
            temporal_extractor = TemporalFeatureExtractor(
                granularities=self.config.temporal_granularities
            )
            transformers.append(("temporal", temporal_extractor))
        
        # 3. Categorical encoding
        categorical_encoder = SmartCategoricalEncoder(
            encoding_method=self.config.categorical_encoding,
            max_categories=self.config.max_categories,
            random_state=self.config.random_state
        )
        transformers.append(("categorical", categorical_encoder))
        
        # 4. Numerical transformations
        numerical_transformer = AdvancedNumericalTransformer(
            enable_log=self.config.enable_log_transform,
            enable_power=self.config.enable_power_transform,
            enable_binning=self.config.enable_binning,
            random_state=self.config.random_state
        )
        transformers.append(("numerical", numerical_transformer))
        
        # 5. Domain-specific features
        if self.config.create_domain_features:
            domain_generator = DomainSpecificFeatureGenerator()
            transformers.append(("domain", domain_generator))
        
        # 6. Feature interactions
        if self.config.create_interactions:
            interaction_generator = FeatureInteractionGenerator(
                max_degree=self.config.max_interaction_degree,
                max_features=50,
                random_state=self.config.random_state
            )
            transformers.append(("interactions", interaction_generator))
        
        # 7. Polynomial features
        if self.config.create_polynomials:
            poly_transformer = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                interaction_only=True,
                include_bias=False
            )
            transformers.append(("polynomials", poly_transformer))
        
        # 8. Scaling
        if self.config.enable_scaling:
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                scaler = RobustScaler()
            elif self.config.scaling_method == "quantile":
                scaler = QuantileTransformer(random_state=self.config.random_state)
            else:
                scaler = StandardScaler()
            
            transformers.append(("scaler", scaler))
        
        # Create pipeline
        self.pipeline = Pipeline(transformers)
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # Convert back to DataFrame if needed
        if not isinstance(X_transformed, pd.DataFrame):
            # Try to get feature names
            try:
                feature_names = self.pipeline.named_steps['scaler'].get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
            X_transformed = pd.DataFrame(
                X_transformed, 
                columns=feature_names,
                index=X.index
            )
        
        # Generate feature metadata
        self._generate_feature_metadata(X, X_transformed, y)
        
        logger.info(f"Feature engineering completed: {len(self.original_columns)} -> {X_transformed.shape[1]} features")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        X_transformed = self.pipeline.transform(X)
        
        # Convert back to DataFrame if needed
        if not isinstance(X_transformed, pd.DataFrame):
            try:
                feature_names = self.pipeline.named_steps['scaler'].get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=feature_names,
                index=X.index
            )
        
        return X_transformed
    
    def _generate_feature_metadata(self, 
                                 X_original: pd.DataFrame, 
                                 X_transformed: pd.DataFrame,
                                 y: Optional[pd.Series] = None) -> None:
        """Generate metadata for engineered features."""
        
        self.feature_metadata = {}
        
        for col in X_transformed.columns:
            metadata = FeatureMetadata(
                name=col,
                original_name=col if col in X_original.columns else None,
                feature_type=self._determine_feature_type(col, X_transformed[col]),
                transformation=self._determine_transformation(col),
                importance_score=0.0,  # Will be calculated if target available
                correlation_with_target=0.0,
                missing_ratio=X_transformed[col].isnull().sum() / len(X_transformed),
                unique_values=X_transformed[col].nunique(),
                data_type=str(X_transformed[col].dtype),
                creation_method="automated"
            )
            
            # Calculate correlation with target if available
            if y is not None:
                try:
                    if X_transformed[col].dtype in ['int64', 'float64'] and y.dtype in ['int64', 'float64']:
                        correlation = abs(np.corrcoef(X_transformed[col].fillna(0), y)[0, 1])
                        metadata.correlation_with_target = correlation if not np.isnan(correlation) else 0.0
                except:
                    pass
            
            self.feature_metadata[col] = metadata
    
    def _determine_feature_type(self, col_name: str, col_data: pd.Series) -> str:
        """Determine the type of a feature."""
        if any(keyword in col_name.lower() for keyword in ['_x_', 'interaction', 'poly']):
            return "engineered"
        elif any(keyword in col_name.lower() for keyword in ['year', 'month', 'day', 'hour', 'time']):
            return "temporal"
        elif col_data.dtype in ['int64', 'float64']:
            return "numerical"
        else:
            return "categorical"
    
    def _determine_transformation(self, col_name: str) -> str:
        """Determine the transformation applied to a feature."""
        transformations = []
        
        if '_log' in col_name:
            transformations.append('log')
        if '_boxcox' in col_name:
            transformations.append('boxcox')
        if '_binned' in col_name:
            transformations.append('binned')
        if '_x_' in col_name:
            transformations.append('interaction')
        if any(suffix in col_name for suffix in ['_sin', '_cos']):
            transformations.append('cyclical')
        
        return ','.join(transformations) if transformations else 'none'
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance/correlation."""
        rankings = []
        for name, metadata in self.feature_metadata.items():
            score = max(metadata.importance_score, metadata.correlation_with_target)
            rankings.append((name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the feature engineering pipeline."""
        pipeline_data = {
            'pipeline': self.pipeline,
            'config': asdict(self.config),
            'feature_metadata': {name: asdict(meta) for name, meta in self.feature_metadata.items()},
            'original_columns': self.original_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Feature engineering pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load a feature engineering pipeline."""
        pipeline_data = joblib.load(filepath)
        
        self.pipeline = pipeline_data['pipeline']
        self.config = FeatureEngineeringConfig(**pipeline_data['config'])
        self.feature_metadata = {
            name: FeatureMetadata(**meta) 
            for name, meta in pipeline_data['feature_metadata'].items()
        }
        self.original_columns = pipeline_data['original_columns']
        
        logger.info(f"Feature engineering pipeline loaded from {filepath}")


def engineer_features(X: pd.DataFrame, 
                     y: Optional[pd.Series] = None,
                     config: Optional[FeatureEngineeringConfig] = None,
                     save_pipeline: Optional[str] = None) -> Tuple[pd.DataFrame, AutomatedFeatureEngineer]:
    """
    Perform automated feature engineering on a dataset.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        config: Feature engineering configuration
        save_pipeline: Path to save the pipeline (optional)
        
    Returns:
        Tuple of (transformed_features, feature_engineer)
    """
    
    # Initialize feature engineer
    engineer = AutomatedFeatureEngineer(config)
    
    # Perform feature engineering
    X_transformed = engineer.fit_transform(X, y)
    
    # Save pipeline if requested
    if save_pipeline:
        engineer.save_pipeline(save_pipeline)
    
    return X_transformed, engineer


if __name__ == "__main__":
    print("Automated Feature Engineering Pipeline")
    print("This module provides comprehensive automated feature engineering capabilities.")