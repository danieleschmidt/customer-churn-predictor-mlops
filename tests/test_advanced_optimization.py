"""
Tests for advanced model optimization system.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.advanced_optimization import (
    BayesianOptimizer, AdvancedFeatureSelector, AutoMLPipeline,
    optimize_model_hyperparameters, save_optimization_results
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, 
        n_informative=8, n_redundant=2, random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


class TestBayesianOptimizer:
    """Test Bayesian optimization functionality."""
    
    def test_optimizer_initialization(self):
        """Test Bayesian optimizer initialization."""
        optimizer = BayesianOptimizer()
        assert optimizer.acquisition_function == "expected_improvement"
        assert optimizer.n_random_starts == 10
        assert optimizer.history == []
        assert optimizer.best_score == -np.inf
    
    def test_optimizer_simple_function(self):
        """Test optimization of simple function."""
        def simple_objective(params):
            x = params.get('x', 0)
            return -(x - 2)**2 + 4  # Maximum at x=2
        
        optimizer = BayesianOptimizer(n_random_starts=5)
        param_bounds = {'x': (0, 5)}
        
        result = optimizer.optimize(simple_objective, param_bounds, n_calls=10)
        
        assert result.best_params is not None
        assert result.best_score > 0
        assert len(result.optimization_history) > 0


class TestAdvancedFeatureSelector:
    """Test advanced feature selection."""
    
    def test_feature_selector_initialization(self):
        """Test feature selector initialization."""
        selector = AdvancedFeatureSelector(n_features=5)
        assert selector.n_features == 5
        assert 'mutual_info' in selector.methods
        assert 'f_classif' in selector.methods
    
    def test_feature_selection(self, sample_data):
        """Test feature selection process."""
        X, y = sample_data
        
        selector = AdvancedFeatureSelector(n_features=5)
        X_selected = selector.fit_select(X, y)
        
        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X.shape[0]
        assert len(selector.selected_features) == 5
        
        # Selected features should be subset of original features
        assert set(X_selected.columns).issubset(set(X.columns))


class TestAutoMLPipeline:
    """Test AutoML pipeline functionality."""
    
    def test_automl_initialization(self):
        """Test AutoML pipeline initialization."""
        automl = AutoMLPipeline(time_budget=60)
        assert automl.time_budget == 60
        assert automl.cv_folds == 5
        assert len(automl.model_candidates) > 0
    
    def test_automl_fit(self, sample_data):
        """Test AutoML fitting process."""
        X, y = sample_data
        
        # Use short time budget for testing
        automl = AutoMLPipeline(time_budget=30, cv_folds=3)
        result = automl.fit(X, y)
        
        assert result.best_model is not None
        assert result.best_score > 0
        assert result.best_params is not None
        assert result.optimization_time > 0
        assert len(result.feature_importance) > 0


class TestModelOptimization:
    """Test main optimization functions."""
    
    def test_optimize_model_hyperparameters(self, sample_data):
        """Test hyperparameter optimization."""
        X, y = sample_data
        
        result = optimize_model_hyperparameters(
            X, y, model_type="auto", optimization_budget=30
        )
        
        assert result.best_model is not None
        assert result.best_score > 0.5  # Should be better than random
        assert result.best_params is not None
        assert len(result.cross_validation_scores) > 0
        assert result.optimization_time > 0
    
    def test_save_optimization_results(self, sample_data, tmp_path):
        """Test saving optimization results."""
        X, y = sample_data
        
        result = optimize_model_hyperparameters(
            X, y, model_type="auto", optimization_budget=30
        )
        
        output_dir = str(tmp_path)
        result_file = save_optimization_results(result, output_dir)
        
        assert result_file is not None
        assert tmp_path.exists()
        
        # Check that files were created
        json_files = list(tmp_path.glob("*.json"))
        joblib_files = list(tmp_path.glob("*.joblib"))
        
        assert len(json_files) > 0
        assert len(joblib_files) > 0


@pytest.mark.integration
class TestIntegrationOptimization:
    """Integration tests for optimization system."""
    
    def test_end_to_end_optimization(self, sample_data):
        """Test complete optimization workflow."""
        X, y = sample_data
        
        # Run full optimization
        result = optimize_model_hyperparameters(
            X, y, model_type="auto", optimization_budget=60
        )
        
        # Verify result quality
        assert result.best_score > 0.7  # Should achieve good performance
        assert result.best_model is not None
        
        # Test model can make predictions
        predictions = result.best_model.predict(X[:10])
        probabilities = result.best_model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_optimization_with_different_budgets(self, sample_data):
        """Test optimization with different time budgets."""
        X, y = sample_data
        
        # Test with small budget
        result_small = optimize_model_hyperparameters(
            X, y, model_type="auto", optimization_budget=15
        )
        
        # Test with larger budget
        result_large = optimize_model_hyperparameters(
            X, y, model_type="auto", optimization_budget=60
        )
        
        # Both should work
        assert result_small.best_model is not None
        assert result_large.best_model is not None
        
        # Larger budget might (but not guaranteed to) achieve better results
        assert result_small.best_score >= 0.5
        assert result_large.best_score >= 0.5