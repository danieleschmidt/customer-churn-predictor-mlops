"""
Tests for research framework and benchmarking system.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification

from src.research_framework import (
    ResearchAlgorithmRegistry, StatisticalAnalysisEngine, BenchmarkDatasetGenerator,
    ResearchBenchmarkingFramework, run_research_benchmark, ExperimentalDesign
)


@pytest.fixture
def sample_experimental_design():
    """Create sample experimental design."""
    return ExperimentalDesign(
        algorithms=["logistic_regression", "random_forest"],
        datasets=["test_dataset"],
        evaluation_metrics=["accuracy", "f1_score"],
        cv_strategy="stratified_kfold",
        cv_folds=3,  # Small for testing
        cv_repeats=1,
        random_seeds=[42],
        significance_level=0.05,
        effect_size_threshold=0.2,
        power_analysis=False,
        preprocessing_steps=[],
        hyperparameter_optimization=False
    )


class TestResearchAlgorithmRegistry:
    """Test algorithm registry functionality."""
    
    def test_registry_initialization(self):
        """Test algorithm registry initialization."""
        registry = ResearchAlgorithmRegistry()
        
        assert len(registry.algorithms) > 0
        assert 'logistic_regression' in registry.algorithms
        assert 'random_forest' in registry.algorithms
    
    def test_get_algorithm(self):
        """Test getting algorithm configuration."""
        registry = ResearchAlgorithmRegistry()
        
        alg_config = registry.get_algorithm('logistic_regression')
        
        assert 'estimator' in alg_config
        assert 'default_params' in alg_config
        assert 'param_grid' in alg_config
        assert 'theoretical_properties' in alg_config
    
    def test_get_nonexistent_algorithm(self):
        """Test getting non-existent algorithm."""
        registry = ResearchAlgorithmRegistry()
        
        with pytest.raises(ValueError):
            registry.get_algorithm('nonexistent_algorithm')
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        registry = ResearchAlgorithmRegistry()
        algorithms = registry.list_algorithms()
        
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert 'logistic_regression' in algorithms


class TestStatisticalAnalysisEngine:
    """Test statistical analysis functionality."""
    
    def test_engine_initialization(self):
        """Test statistical engine initialization."""
        engine = StatisticalAnalysisEngine()
        assert engine.significance_level == 0.05
    
    def test_friedman_test_insufficient_data(self):
        """Test Friedman test with insufficient data."""
        engine = StatisticalAnalysisEngine()
        
        # Insufficient algorithms
        results = {'alg1': [0.8, 0.9], 'alg2': [0.7, 0.8]}
        test_result = engine.friedman_test(results)
        
        assert not test_result['significant']
        assert 'Insufficient' in test_result.get('message', '')
    
    def test_friedman_test_sufficient_data(self):
        """Test Friedman test with sufficient data."""
        engine = StatisticalAnalysisEngine()
        
        # Create mock results
        np.random.seed(42)
        results = {
            'alg1': np.random.normal(0.8, 0.1, 10).tolist(),
            'alg2': np.random.normal(0.7, 0.1, 10).tolist(),
            'alg3': np.random.normal(0.75, 0.1, 10).tolist()
        }
        
        test_result = engine.friedman_test(results)
        
        assert 'statistic' in test_result
        assert 'p_value' in test_result
        assert isinstance(test_result['significant'], bool)
    
    def test_pairwise_tests(self):
        """Test pairwise statistical tests."""
        engine = StatisticalAnalysisEngine()
        
        # Create mock results with clear difference
        results = {
            'alg1': [0.9, 0.85, 0.88, 0.92, 0.87],
            'alg2': [0.6, 0.65, 0.62, 0.68, 0.63]
        }
        
        pairwise_results = engine.pairwise_tests(results)
        
        assert len(pairwise_results) == 1  # One pair
        pair_key = ('alg1', 'alg2')
        assert pair_key in pairwise_results
        
        pair_result = pairwise_results[pair_key]
        assert 'p_value' in pair_result
        assert 'effect_size' in pair_result
        assert 'winner' in pair_result
    
    def test_rank_algorithms(self):
        """Test algorithm ranking."""
        engine = StatisticalAnalysisEngine()
        
        results = {
            'alg1': [0.8, 0.82, 0.79],
            'alg2': [0.9, 0.88, 0.91],
            'alg3': [0.7, 0.72, 0.68]
        }
        
        ranking = engine.rank_algorithms(results, 'test_metric')
        
        assert 'ranking' in ranking
        assert 'best_algorithm' in ranking
        assert len(ranking['ranking']) == 3
        
        # Should be sorted by performance
        rankings = ranking['ranking']
        assert rankings[0]['mean_score'] >= rankings[1]['mean_score']
        assert rankings[1]['mean_score'] >= rankings[2]['mean_score']


class TestBenchmarkDatasetGenerator:
    """Test benchmark dataset generation."""
    
    def test_create_synthetic_datasets(self):
        """Test synthetic dataset creation."""
        datasets = BenchmarkDatasetGenerator.create_synthetic_datasets(n_datasets=3)
        
        assert len(datasets) == 3
        
        for dataset in datasets:
            assert hasattr(dataset, 'name')
            assert hasattr(dataset, 'X')
            assert hasattr(dataset, 'y')
            assert hasattr(dataset, 'n_samples')
            assert hasattr(dataset, 'n_features')
            
            # Verify data shapes
            assert len(dataset.X) == dataset.n_samples
            assert len(dataset.y) == dataset.n_samples
            assert dataset.X.shape[1] == dataset.n_features
    
    def test_dataset_properties(self):
        """Test dataset properties are calculated correctly."""
        datasets = BenchmarkDatasetGenerator.create_synthetic_datasets(n_datasets=1)
        dataset = datasets[0]
        
        assert dataset.task_type == "classification"
        assert dataset.domain == "synthetic"
        assert 0 <= dataset.difficulty_score <= 1
        assert isinstance(dataset.class_balance, dict)
        assert sum(dataset.class_balance.values()) == pytest.approx(1.0, rel=1e-2)


class TestResearchBenchmarkingFramework:
    """Test research benchmarking framework."""
    
    def test_framework_initialization(self, sample_experimental_design):
        """Test framework initialization."""
        framework = ResearchBenchmarkingFramework(sample_experimental_design)
        
        assert framework.experimental_design == sample_experimental_design
        assert framework.algorithm_registry is not None
        assert framework.statistical_engine is not None
    
    def test_default_experimental_design(self):
        """Test default experimental design creation."""
        framework = ResearchBenchmarkingFramework()
        
        design = framework.experimental_design
        assert len(design.algorithms) > 0
        assert len(design.evaluation_metrics) > 0
        assert design.cv_folds > 0
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment')
    def test_single_experiment(self, mock_set_exp, mock_create_exp, sample_experimental_design):
        """Test running single experiment."""
        framework = ResearchBenchmarkingFramework(sample_experimental_design)
        
        # Create test dataset
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        from src.research_framework import BenchmarkDataset
        dataset = BenchmarkDataset(
            name="test_dataset",
            X=X_df,
            y=y_series,
            description="Test dataset",
            domain="test",
            task_type="classification",
            n_samples=100,
            n_features=5,
            n_classes=2,
            class_balance={"0": 0.5, "1": 0.5},
            difficulty_score=0.5
        )
        
        with patch('mlflow.start_run'), patch('mlflow.log_param'), patch('mlflow.log_metric'):
            result = framework._run_single_experiment('logistic_regression', dataset)
        
        assert result.algorithm_name == 'logistic_regression'
        assert result.dataset_name == 'test_dataset'
        assert 'accuracy' in result.cv_scores
        assert 'accuracy' in result.mean_scores
        assert result.mean_scores['accuracy'] > 0


@pytest.mark.integration
class TestResearchIntegration:
    """Integration tests for research framework."""
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment')
    def test_mini_benchmark(self, mock_set_exp, mock_create_exp):
        """Test minimal benchmark run."""
        # Create simple experimental design
        design = ExperimentalDesign(
            algorithms=["logistic_regression"],
            datasets=[],
            evaluation_metrics=["accuracy"],
            cv_strategy="stratified_kfold",
            cv_folds=2,
            cv_repeats=1,
            random_seeds=[42],
            significance_level=0.05,
            effect_size_threshold=0.2,
            power_analysis=False,
            preprocessing_steps=[],
            hyperparameter_optimization=False
        )
        
        framework = ResearchBenchmarkingFramework(design)
        
        # Create single test dataset
        datasets = BenchmarkDatasetGenerator.create_synthetic_datasets(n_datasets=1)
        
        with patch('mlflow.start_run'), patch('mlflow.log_param'), patch('mlflow.log_metric'):
            result = framework.run_comprehensive_benchmark(datasets)
        
        assert len(result.algorithm_results) > 0
        assert result.publication_summary is not None
        assert len(result.recommendations) > 0
    
    def test_run_research_benchmark_function(self):
        """Test main research benchmark function."""
        with patch('mlflow.create_experiment'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run'), \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'):
            
            result = run_research_benchmark(
                algorithms=["logistic_regression"],
                n_synthetic_datasets=1,
                output_dir=tempfile.mkdtemp()
            )
        
        assert result is not None
        assert len(result.algorithm_results) > 0
    
    def test_generate_research_report(self, sample_experimental_design):
        """Test research report generation."""
        framework = ResearchBenchmarkingFramework(sample_experimental_design)
        
        # Create mock result
        from src.research_framework import ComparativeAnalysisResult, AlgorithmResult
        
        mock_algorithm_result = AlgorithmResult(
            algorithm_name="test_alg",
            dataset_name="test_dataset",
            cv_scores={"accuracy": [0.8, 0.85, 0.82]},
            mean_scores={"accuracy": 0.82},
            std_scores={"accuracy": 0.02},
            training_times=[1.0],
            prediction_times=[0.01],
            model_complexity=10,
            hyperparameters={},
            statistical_properties={}
        )
        
        mock_result = ComparativeAnalysisResult(
            experimental_design=sample_experimental_design,
            algorithm_results=[mock_algorithm_result],
            pairwise_comparisons={},
            statistical_tests={},
            ranking_analysis={"accuracy": {"ranking": [{"algorithm": "test_alg", "mean_score": 0.82, "std_score": 0.02}]}},
            effect_sizes={},
            performance_profiles={},
            recommendations=["Test recommendation"],
            publication_summary={"experiment_summary": {"n_algorithms": 1}}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = framework.generate_research_report(mock_result, temp_dir)
            
            assert report_file is not None
            assert temp_dir in report_file


class TestComplexityAnalysis:
    """Test complexity-performance analysis."""
    
    def test_complexity_analysis_logic(self, sample_experimental_design):
        """Test complexity analysis functionality."""
        framework = ResearchBenchmarkingFramework(sample_experimental_design)
        
        # Create mock results with different complexities
        from src.research_framework import AlgorithmResult
        
        results = [
            AlgorithmResult(
                algorithm_name="simple_alg",
                dataset_name="test",
                cv_scores={"accuracy": [0.8, 0.82]},
                mean_scores={"accuracy": 0.81},
                std_scores={"accuracy": 0.01},
                training_times=[1.0],
                prediction_times=[0.001],
                model_complexity=5,
                hyperparameters={},
                statistical_properties={}
            ),
            AlgorithmResult(
                algorithm_name="complex_alg",
                dataset_name="test",
                cv_scores={"accuracy": [0.85, 0.87]},
                mean_scores={"accuracy": 0.86},
                std_scores={"accuracy": 0.01},
                training_times=[10.0],
                prediction_times=[0.1],
                model_complexity=100,
                hyperparameters={},
                statistical_properties={}
            )
        ]
        
        analysis = framework._analyze_complexity_performance_tradeoff(results)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0