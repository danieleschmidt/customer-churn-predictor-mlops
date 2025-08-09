"""
Tests for Intelligent Model Selection Framework.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os

from src.intelligent_model_selection import (
    MetaLearningEngine, ModelConfigurationRegistry, IntelligentModelSelector,
    DataCharacteristics, ModelSelectionResult, auto_select_model
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


@pytest.fixture
def meta_engine():
    """Create MetaLearningEngine instance."""
    return MetaLearningEngine()


@pytest.fixture
def model_registry():
    """Create ModelConfigurationRegistry instance."""
    return ModelConfigurationRegistry()


@pytest.fixture
def model_selector():
    """Create IntelligentModelSelector instance."""
    return IntelligentModelSelector(cv_folds=3, n_iter=5)


class TestMetaLearningEngine:
    """Tests for MetaLearningEngine."""
    
    def test_analyze_data_characteristics(self, meta_engine, sample_data):
        """Test data characteristics analysis."""
        X, y = sample_data
        
        characteristics = meta_engine.analyze_data_characteristics(X, y)
        
        assert isinstance(characteristics, DataCharacteristics)
        assert characteristics.n_samples == 1000
        assert characteristics.n_features == 20
        assert characteristics.n_classes == 2
        assert 0 <= characteristics.dimensionality_ratio <= 1
        assert characteristics.dataset_complexity in ["simple", "medium", "complex"]
    
    def test_recommend_models_high_dimensional(self, meta_engine):
        """Test model recommendations for high-dimensional data."""
        # Create high-dimensional characteristics
        characteristics = DataCharacteristics(
            n_samples=500,
            n_features=100,  # High dimensional
            n_classes=2,
            class_balance={"0": 0.5, "1": 0.5},
            feature_types={},
            missing_values_pct=0.0,
            correlation_strength=0.3,
            dimensionality_ratio=0.2,  # High ratio
            noise_level=0.1,
            dataset_complexity="complex",
            domain_type="tabular"
        )
        
        recommendations = meta_engine.recommend_models(characteristics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should avoid KNN for high-dimensional data
        assert "knn" not in recommendations
    
    def test_recommend_models_small_sample(self, meta_engine):
        """Test model recommendations for small sample data."""
        characteristics = DataCharacteristics(
            n_samples=200,  # Small sample
            n_features=10,
            n_classes=2,
            class_balance={"0": 0.5, "1": 0.5},
            feature_types={},
            missing_values_pct=0.0,
            correlation_strength=0.3,
            dimensionality_ratio=0.05,
            noise_level=0.1,
            dataset_complexity="simple",
            domain_type="tabular"
        )
        
        recommendations = meta_engine.recommend_models(characteristics)
        
        assert isinstance(recommendations, list)
        # Should avoid complex models for small samples
        assert "neural_network" not in recommendations
    
    def test_calculate_meta_features(self, meta_engine, sample_data):
        """Test meta-features calculation."""
        X, y = sample_data
        
        meta_features = meta_engine.calculate_meta_features(X, y)
        
        assert isinstance(meta_features, dict)
        assert "n_samples" in meta_features
        assert "n_features" in meta_features
        assert "n_classes" in meta_features
        assert "class_entropy" in meta_features
        assert meta_features["n_samples"] == 1000
        assert meta_features["n_features"] == 20


class TestModelConfigurationRegistry:
    """Tests for ModelConfigurationRegistry."""
    
    def test_initialization(self, model_registry):
        """Test registry initialization."""
        configurations = model_registry.configurations
        
        assert len(configurations) > 0
        assert "logistic_regression" in configurations
        assert "random_forest" in configurations
        assert "gradient_boosting" in configurations
    
    def test_get_configuration(self, model_registry):
        """Test getting model configuration."""
        config = model_registry.get_configuration("logistic_regression")
        
        assert config.name == "Logistic Regression"
        assert config.estimator_class is not None
        assert isinstance(config.default_params, dict)
        assert isinstance(config.param_distributions, dict)
        assert isinstance(config.complexity_score, int)
    
    def test_get_nonexistent_configuration(self, model_registry):
        """Test getting non-existent configuration."""
        with pytest.raises(ValueError, match="not found in registry"):
            model_registry.get_configuration("nonexistent_model")
    
    def test_get_recommended_models(self, model_registry):
        """Test getting recommended models."""
        characteristics = DataCharacteristics(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            class_balance={"0": 0.5, "1": 0.5},
            feature_types={},
            missing_values_pct=0.0,
            correlation_strength=0.3,
            dimensionality_ratio=0.01,
            noise_level=0.1,
            dataset_complexity="simple",
            domain_type="tabular"
        )
        
        recommendations = model_registry.get_recommended_models(characteristics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestIntelligentModelSelector:
    """Tests for IntelligentModelSelector."""
    
    def test_initialization(self, model_selector):
        """Test selector initialization."""
        assert model_selector.cv_folds == 3
        assert model_selector.n_iter == 5
        assert model_selector.meta_engine is not None
        assert model_selector.model_registry is not None
    
    @pytest.mark.slow
    def test_select_best_model(self, model_selector, sample_data):
        """Test model selection process."""
        X, y = sample_data
        
        # Use fast strategy for testing
        result = model_selector.select_best_model(
            X, y,
            candidate_models=["logistic_regression", "random_forest"],
            selection_strategy="fast"
        )
        
        assert isinstance(result, ModelSelectionResult)
        assert result.best_model is not None
        assert result.best_model_name in ["logistic_regression", "random_forest"]
        assert isinstance(result.best_score, float)
        assert 0 <= result.best_score <= 1
        assert result.best_params is not None
        assert len(result.model_rankings) > 0
    
    def test_create_preprocessing_pipeline(self, model_selector, sample_data):
        """Test preprocessing pipeline creation."""
        X, y = sample_data
        
        # Add some missing values for testing
        X_with_na = X.copy()
        X_with_na.iloc[0, 0] = np.nan
        
        data_chars = model_selector.meta_engine.analyze_data_characteristics(X_with_na, y)
        pipeline = model_selector._create_preprocessing_pipeline(X_with_na, data_chars)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    @patch('src.intelligent_model_selection.RandomizedSearchCV')
    def test_evaluate_model(self, mock_search, model_selector, sample_data):
        """Test single model evaluation."""
        X, y = sample_data
        
        # Mock the search results
        mock_search_instance = Mock()
        mock_search_instance.best_estimator_ = Mock()
        mock_search_instance.best_score_ = 0.85
        mock_search_instance.best_params_ = {"C": 1.0}
        mock_search_instance.cv_results_ = {
            'std_test_score': np.array([0.05, 0.03, 0.04]),
            'split0_test_score': np.array([0.8, 0.85, 0.82])
        }
        mock_search_instance.best_index_ = 1
        mock_search.return_value = mock_search_instance
        
        preprocessor = Mock()
        
        result = model_selector._evaluate_model(
            "logistic_regression", X, y, preprocessor, 5, 3
        )
        
        assert result["name"] == "logistic_regression"
        assert result["score"] == 0.85
        assert result["params"] == {"C": 1.0}
    
    def test_extract_feature_importance(self, model_selector):
        """Test feature importance extraction."""
        # Mock model with feature_importances_
        mock_model = Mock()
        mock_classifier = Mock()
        mock_classifier.feature_importances_ = np.array([0.3, 0.2, 0.5])
        mock_model.named_steps = {"classifier": mock_classifier}
        
        feature_names = ["feature_0", "feature_1", "feature_2"]
        
        importance = model_selector._extract_feature_importance(mock_model, feature_names)
        
        assert isinstance(importance, dict)
        assert len(importance) == 3
    
    def test_save_and_load_model_selection_result(self, model_selector, sample_data):
        """Test saving and loading model selection results."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock result
            mock_model = Mock()
            mock_preprocessor = Mock()
            
            result = ModelSelectionResult(
                best_model=mock_model,
                best_model_name="test_model",
                best_score=0.85,
                best_params={"param": "value"},
                model_rankings=[],
                selection_strategy="test",
                data_characteristics=model_selector.meta_engine.analyze_data_characteristics(X, y),
                preprocessing_pipeline=mock_preprocessor,
                feature_importance={"feature_0": 0.5},
                cross_validation_scores={"test_model": [0.8, 0.85, 0.9]},
                training_time=10.5,
                selection_confidence=0.9,
                ensemble_models=None,
                meta_features={"test": 1.0}
            )
            
            filepath = os.path.join(temp_dir, "test_result.json")
            
            with patch('joblib.dump') as mock_dump:
                model_selector.save_model_selection_result(result, filepath)
                assert mock_dump.called
                
                # Check that result file was created
                assert os.path.exists(filepath)


class TestAutoSelectModel:
    """Tests for auto_select_model function."""
    
    @pytest.mark.slow
    def test_auto_select_model_fast(self, sample_data):
        """Test auto model selection with fast strategy."""
        X, y = sample_data
        
        result = auto_select_model(
            X, y,
            strategy="fast",
            enable_ensemble=False
        )
        
        assert isinstance(result, ModelSelectionResult)
        assert result.best_model is not None
        assert result.selection_strategy == "fast"
    
    @patch('src.intelligent_model_selection.IntelligentModelSelector')
    def test_auto_select_model_with_output_dir(self, mock_selector_class, sample_data):
        """Test auto model selection with output directory."""
        X, y = sample_data
        
        mock_selector = Mock()
        mock_result = Mock(spec=ModelSelectionResult)
        mock_selector.select_best_model.return_value = mock_result
        mock_selector_class.return_value = mock_selector
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = auto_select_model(X, y, output_dir=temp_dir)
            
            # Should save the pipeline
            mock_selector.save_model_selection_result.assert_called_once()


class TestIntegration:
    """Integration tests for model selection."""
    
    @pytest.mark.slow
    def test_end_to_end_model_selection(self, sample_data):
        """Test complete model selection pipeline."""
        X, y = sample_data
        
        # Run complete model selection
        selector = IntelligentModelSelector(cv_folds=3, n_iter=3)
        
        result = selector.select_best_model(
            X, y,
            candidate_models=["logistic_regression"],
            selection_strategy="fast"
        )
        
        # Test that we can use the selected model
        assert result.best_model is not None
        
        # Make predictions with the model
        predictions = result.best_model.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_selection_with_categorical_data(self):
        """Test model selection with categorical features."""
        # Create data with categorical features
        data = pd.DataFrame({
            'numeric_feature': np.random.randn(100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        X = data[['numeric_feature', 'categorical_feature']]
        y = data['target']
        
        selector = IntelligentModelSelector(cv_folds=3, n_iter=3)
        
        # This should handle categorical data appropriately
        result = selector.select_best_model(
            X, y,
            candidate_models=["logistic_regression"],
            selection_strategy="fast"
        )
        
        assert result.best_model is not None
    
    def test_model_selection_with_imbalanced_data(self):
        """Test model selection with imbalanced dataset."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        meta_engine = MetaLearningEngine()
        characteristics = meta_engine.analyze_data_characteristics(X_df, y_series)
        
        # Should detect imbalanced data
        assert min(characteristics.class_balance.values()) < 0.3
        
        # Should recommend appropriate models
        recommendations = meta_engine.recommend_models(characteristics)
        assert "random_forest" in recommendations or "gradient_boosting" in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])