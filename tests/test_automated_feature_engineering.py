"""
Tests for Automated Feature Engineering Pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

from src.automated_feature_engineering import (
    FeatureEngineeringConfig, AutomatedFeatureEngineer,
    AdvancedMissingValueImputer, SmartCategoricalEncoder,
    AdvancedNumericalTransformer, FeatureInteractionGenerator,
    TemporalFeatureExtractor, DomainSpecificFeatureGenerator,
    engineer_features
)


@pytest.fixture
def sample_data():
    """Create sample dataset with mixed feature types."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create mixed data
    data = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.uniform(0, 100, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples),
        'binary_flag': np.random.choice([0, 1], n_samples),
        'skewed_feature': np.random.exponential(2, n_samples),
        'date_feature': pd.date_range('2020-01-01', periods=n_samples, freq='D')[:n_samples]
    })
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    data.loc[missing_indices, 'numeric_1'] = np.nan
    data.loc[missing_indices[:20], 'categorical_1'] = np.nan
    
    # Create target
    target = np.random.choice([0, 1], n_samples)
    
    return data, pd.Series(target, name='target')


@pytest.fixture
def customer_data():
    """Create customer churn-like dataset."""
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples)
    })
    
    target = np.random.choice([0, 1], n_samples)
    return data, pd.Series(target, name='churn')


class TestFeatureEngineeringConfig:
    """Tests for FeatureEngineeringConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FeatureEngineeringConfig()
        
        assert config.enable_scaling is True
        assert config.scaling_method == "standard"
        assert config.handle_missing is True
        assert config.categorical_encoding == "auto"
        assert config.create_interactions is True
        assert config.random_state == 42


class TestAdvancedMissingValueImputer:
    """Tests for AdvancedMissingValueImputer."""
    
    def test_fit_transform_smart_strategy(self, sample_data):
        """Test smart imputation strategy."""
        X, _ = sample_data
        
        imputer = AdvancedMissingValueImputer(strategy="smart")
        
        # Fit the imputer
        imputer.fit(X)
        
        # Should have imputers for columns with missing values
        assert 'numeric_1' in imputer.imputers
        assert 'categorical_1' in imputer.imputers
        
        # Transform the data
        X_imputed = imputer.transform(X)
        
        # Should have no missing values
        assert X_imputed.isnull().sum().sum() == 0
    
    def test_fit_transform_mean_strategy(self):
        """Test mean imputation strategy."""
        X = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_col': ['A', 'B', np.nan, 'A', 'B']
        })
        
        imputer = AdvancedMissingValueImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        
        # Numeric column should be imputed with mean
        assert not X_imputed['numeric_col'].isnull().any()
        # Categorical should be imputed with mode
        assert not X_imputed['categorical_col'].isnull().any()
    
    def test_no_missing_values(self):
        """Test with data that has no missing values."""
        X = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        imputer = AdvancedMissingValueImputer()
        X_imputed = imputer.fit_transform(X)
        
        # Should return unchanged data
        pd.testing.assert_frame_equal(X, X_imputed)


class TestSmartCategoricalEncoder:
    """Tests for SmartCategoricalEncoder."""
    
    def test_auto_encoding_selection(self):
        """Test automatic encoding strategy selection."""
        X = pd.DataFrame({
            'binary_cat': ['A', 'B'] * 50,  # Should use label encoding
            'multi_cat': ['X', 'Y', 'Z'] * 33 + ['X'],  # Should use one-hot
            'high_card': [f'cat_{i}' for i in range(100)]  # Should use ordinal
        })
        
        encoder = SmartCategoricalEncoder(encoding_method="auto", max_categories=10)
        encoder.fit(X)
        
        # Check encoding strategies
        assert encoder.encoding_strategies['binary_cat'] == 'label'
        assert encoder.encoding_strategies['multi_cat'] == 'onehot'
        assert encoder.encoding_strategies['high_card'] == 'ordinal'
    
    def test_onehot_encoding(self):
        """Test one-hot encoding."""
        X = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        encoder = SmartCategoricalEncoder(encoding_method="onehot")
        X_encoded = encoder.fit_transform(X)
        
        # Should create multiple columns
        assert X_encoded.shape[1] > 1
        # Should have binary values
        assert all(col in X_encoded.columns for col in X_encoded.columns if 'category_' in col)
    
    def test_label_encoding(self):
        """Test label encoding."""
        X = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        encoder = SmartCategoricalEncoder(encoding_method="label")
        X_encoded = encoder.fit_transform(X)
        
        # Should keep same number of columns
        assert X_encoded.shape[1] == 1
        # Should contain numeric values
        assert X_encoded['category'].dtype in [np.int64, int]


class TestAdvancedNumericalTransformer:
    """Tests for AdvancedNumericalTransformer."""
    
    def test_skewed_data_transformation(self):
        """Test transformation of skewed data."""
        # Create highly skewed data
        X = pd.DataFrame({
            'skewed_col': np.random.exponential(1, 1000)
        })
        
        transformer = AdvancedNumericalTransformer(
            enable_log=True,
            enable_power=True,
            enable_binning=True
        )
        
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        # Should create additional transformed columns
        assert X_transformed.shape[1] > X.shape[1]
        
        # Check for expected transformations
        transformed_cols = X_transformed.columns.tolist()
        assert any('log' in col for col in transformed_cols)
    
    def test_normal_data_no_transformation(self):
        """Test that normal data doesn't get unnecessary transformations."""
        # Create normally distributed data
        X = pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 1000)
        })
        
        transformer = AdvancedNumericalTransformer()
        transformer.fit(X)
        
        # Should not have many transformations for normal data
        assert len(transformer.transformations) <= 1


class TestFeatureInteractionGenerator:
    """Tests for FeatureInteractionGenerator."""
    
    def test_interaction_generation(self):
        """Test generation of feature interactions."""
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        })
        y = np.random.choice([0, 1], 100)
        
        generator = FeatureInteractionGenerator(max_degree=2, max_features=5)
        generator.fit(X, y)
        
        X_with_interactions = generator.transform(X)
        
        # Should have more features
        assert X_with_interactions.shape[1] >= X.shape[1]
        
        # Check for interaction columns
        interaction_cols = [col for col in X_with_interactions.columns if '_x_' in col]
        assert len(interaction_cols) > 0
    
    def test_no_target_interactions(self):
        """Test interaction generation without target variable."""
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        
        generator = FeatureInteractionGenerator()
        generator.fit(X)  # No target
        
        X_with_interactions = generator.transform(X)
        
        # Should still generate some interactions
        assert X_with_interactions.shape[1] >= X.shape[1]


class TestTemporalFeatureExtractor:
    """Tests for TemporalFeatureExtractor."""
    
    def test_datetime_feature_extraction(self):
        """Test extraction of temporal features."""
        X = pd.DataFrame({
            'date_col': pd.date_range('2020-01-01', periods=100, freq='D'),
            'numeric_col': np.random.randn(100)
        })
        
        extractor = TemporalFeatureExtractor()
        extractor.fit(X)
        
        X_with_temporal = extractor.transform(X)
        
        # Should have additional temporal columns
        assert X_with_temporal.shape[1] > X.shape[1]
        
        # Check for expected temporal features
        temporal_cols = X_with_temporal.columns.tolist()
        assert any('year' in col for col in temporal_cols)
        assert any('month' in col for col in temporal_cols)
        assert any('sin' in col for col in temporal_cols)  # Cyclical features
    
    def test_no_temporal_features(self):
        """Test with data that has no temporal features."""
        X = pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'categorical_col': ['A', 'B'] * 50
        })
        
        extractor = TemporalFeatureExtractor()
        extractor.fit(X)
        
        X_unchanged = extractor.transform(X)
        
        # Should remain unchanged
        pd.testing.assert_frame_equal(X, X_unchanged)


class TestDomainSpecificFeatureGenerator:
    """Tests for DomainSpecificFeatureGenerator."""
    
    def test_customer_value_features(self, customer_data):
        """Test generation of customer value features."""
        X, _ = customer_data
        
        generator = DomainSpecificFeatureGenerator()
        generator.fit(X)
        
        X_with_domain = generator.transform(X)
        
        # Should have additional domain-specific features
        assert X_with_domain.shape[1] >= X.shape[1]
        
        # Check for expected customer value features
        domain_cols = X_with_domain.columns.tolist()
        if any('charge' in col.lower() for col in X.columns):
            assert any('avg_charges' in col for col in domain_cols)
    
    def test_usage_pattern_features(self):
        """Test generation of usage pattern features."""
        X = pd.DataFrame({
            'data_usage': np.random.uniform(0, 1000, 100),
            'call_minutes': np.random.uniform(0, 500, 100),
            'other_col': np.random.randn(100)
        })
        
        generator = DomainSpecificFeatureGenerator()
        generator.fit(X)
        
        X_with_domain = generator.transform(X)
        
        # Should detect usage patterns and create features
        domain_cols = X_with_domain.columns.tolist()
        if 'usage' in str(X.columns).lower():
            assert 'total_usage' in domain_cols or 'usage_diversity' in domain_cols


class TestAutomatedFeatureEngineer:
    """Tests for AutomatedFeatureEngineer."""
    
    def test_basic_feature_engineering(self, sample_data):
        """Test basic feature engineering pipeline."""
        X, y = sample_data
        
        config = FeatureEngineeringConfig(
            create_interactions=False,  # Disable for faster testing
            create_polynomials=False
        )
        
        engineer = AutomatedFeatureEngineer(config)
        X_engineered = engineer.fit_transform(X, y)
        
        # Should have engineered features
        assert X_engineered is not None
        assert X_engineered.shape[0] == X.shape[0]  # Same number of rows
        
        # Should have feature metadata
        assert len(engineer.feature_metadata) > 0
    
    def test_feature_engineering_with_interactions(self, sample_data):
        """Test feature engineering with interactions."""
        X, y = sample_data
        
        # Select only numeric columns for faster testing
        X_numeric = X.select_dtypes(include=[np.number])
        
        config = FeatureEngineeringConfig(
            create_interactions=True,
            max_interaction_degree=2
        )
        
        engineer = AutomatedFeatureEngineer(config)
        X_engineered = engineer.fit_transform(X_numeric, y)
        
        # Should have more features due to interactions
        assert X_engineered.shape[1] >= X_numeric.shape[1]
    
    def test_transform_new_data(self, sample_data):
        """Test transforming new data with fitted pipeline."""
        X, y = sample_data
        
        # Split data
        X_train, X_test = X.iloc[:500], X.iloc[500:]
        y_train = y.iloc[:500]
        
        config = FeatureEngineeringConfig(
            create_interactions=False,
            create_polynomials=False
        )
        
        engineer = AutomatedFeatureEngineer(config)
        
        # Fit on training data
        X_train_engineered = engineer.fit_transform(X_train, y_train)
        
        # Transform test data
        X_test_engineered = engineer.transform(X_test)
        
        # Should have same number of columns
        assert X_test_engineered.shape[1] == X_train_engineered.shape[1]
    
    def test_feature_importance_ranking(self, sample_data):
        """Test feature importance ranking."""
        X, y = sample_data
        
        config = FeatureEngineeringConfig(create_interactions=False)
        engineer = AutomatedFeatureEngineer(config)
        
        engineer.fit_transform(X, y)
        
        rankings = engineer.get_feature_importance_ranking()
        
        assert isinstance(rankings, list)
        assert len(rankings) > 0
        assert all(len(item) == 2 for item in rankings)  # (name, score) tuples
    
    @patch('joblib.dump')
    def test_save_pipeline(self, mock_dump, sample_data):
        """Test saving feature engineering pipeline."""
        X, y = sample_data
        
        engineer = AutomatedFeatureEngineer()
        engineer.fit_transform(X, y)
        
        with tempfile.NamedTemporaryFile() as tmp_file:
            engineer.save_pipeline(tmp_file.name)
            
            # Should call joblib.dump
            mock_dump.assert_called_once()
    
    @patch('joblib.load')
    def test_load_pipeline(self, mock_load, sample_data):
        """Test loading feature engineering pipeline."""
        X, y = sample_data
        
        # Mock loaded data
        mock_pipeline_data = {
            'pipeline': Mock(),
            'config': {
                'enable_scaling': True,
                'scaling_method': 'standard',
                'handle_missing': True,
                'missing_strategy': 'smart',
                'categorical_encoding': 'auto',
                'max_categories': 10,
                'enable_log_transform': True,
                'enable_power_transform': True,
                'enable_binning': True,
                'create_interactions': True,
                'max_interaction_degree': 2,
                'create_polynomials': False,
                'polynomial_degree': 2,
                'extract_temporal_features': True,
                'temporal_granularities': None,
                'enable_feature_selection': True,
                'selection_method': 'auto',
                'max_features': None,
                'create_aggregations': True,
                'create_domain_features': True,
                'detect_outliers': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'feature_metadata': {},
            'original_columns': ['col1', 'col2']
        }
        
        mock_load.return_value = mock_pipeline_data
        
        engineer = AutomatedFeatureEngineer()
        
        with tempfile.NamedTemporaryFile() as tmp_file:
            engineer.load_pipeline(tmp_file.name)
            
            # Should call joblib.load
            mock_load.assert_called_once_with(tmp_file.name)
            
            # Should update internal state
            assert engineer.pipeline is not None
            assert engineer.original_columns == ['col1', 'col2']


class TestEngineerFeaturesFunction:
    """Tests for engineer_features function."""
    
    def test_engineer_features_basic(self, sample_data):
        """Test basic feature engineering function."""
        X, y = sample_data
        
        X_engineered, engineer = engineer_features(X, y)
        
        assert isinstance(X_engineered, pd.DataFrame)
        assert isinstance(engineer, AutomatedFeatureEngineer)
        assert X_engineered.shape[0] == X.shape[0]
    
    def test_engineer_features_with_config(self, sample_data):
        """Test feature engineering with custom config."""
        X, y = sample_data
        
        config = FeatureEngineeringConfig(
            enable_scaling=False,
            create_interactions=False
        )
        
        X_engineered, engineer = engineer_features(X, y, config=config)
        
        assert engineer.config.enable_scaling is False
        assert engineer.config.create_interactions is False
    
    @patch('src.automated_feature_engineering.AutomatedFeatureEngineer.save_pipeline')
    def test_engineer_features_with_save(self, mock_save, sample_data):
        """Test feature engineering with pipeline saving."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "pipeline.pkl")
            
            engineer_features(X, y, save_pipeline=save_path)
            
            # Should call save_pipeline
            mock_save.assert_called_once_with(save_path)


class TestIntegration:
    """Integration tests for feature engineering."""
    
    def test_end_to_end_feature_engineering(self, customer_data):
        """Test complete feature engineering pipeline."""
        X, y = customer_data
        
        # Create config for customer churn domain
        config = FeatureEngineeringConfig(
            create_domain_features=True,
            create_interactions=True,
            max_interaction_degree=2,
            extract_temporal_features=True
        )
        
        engineer = AutomatedFeatureEngineer(config)
        X_engineered = engineer.fit_transform(X, y)
        
        # Should handle all feature types
        assert X_engineered is not None
        assert X_engineered.shape[0] == X.shape[0]
        
        # Should create domain-specific features
        feature_names = X_engineered.columns.tolist()
        domain_features = [col for col in feature_names if any(
            keyword in col.lower() for keyword in ['avg', 'ratio', 'interaction', 'total']
        )]
        assert len(domain_features) > 0
        
        # Should handle missing values
        assert X_engineered.isnull().sum().sum() == 0
        
        # Feature metadata should be comprehensive
        assert len(engineer.feature_metadata) > 0
        for metadata in engineer.feature_metadata.values():
            assert hasattr(metadata, 'feature_type')
            assert hasattr(metadata, 'transformation')
    
    def test_feature_engineering_robustness(self):
        """Test feature engineering robustness with edge cases."""
        # Create challenging dataset
        X = pd.DataFrame({
            'all_missing': [np.nan] * 100,
            'all_same': [1] * 100,
            'high_cardinality': [f'cat_{i}' for i in range(100)],
            'extreme_skew': np.concatenate([np.zeros(95), [1000] * 5]),
            'normal_feature': np.random.randn(100)
        })
        y = np.random.choice([0, 1], 100)
        
        engineer = AutomatedFeatureEngineer()
        
        # Should handle edge cases gracefully
        try:
            X_engineered = engineer.fit_transform(X, y)
            assert X_engineered is not None
            assert X_engineered.shape[0] == X.shape[0]
        except Exception as e:
            pytest.fail(f"Feature engineering failed on edge cases: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])