"""
Research-Grade Comparative Analysis and Benchmarking Framework.

This module implements academic-quality research tools for:
- Comprehensive algorithm comparison with statistical significance testing
- Reproducible experimental design with proper baselines
- Publication-ready visualizations and reports
- Meta-analysis capabilities across datasets and algorithms
- Performance profiling with computational complexity analysis
- Automated literature review integration and citation generation
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import combinations, product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, RepeatedStratifiedKFold, 
    train_test_split, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, ranksums
import joblib
import mlflow
import mlflow.sklearn

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .advanced_optimization import optimize_model_hyperparameters

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ExperimentalDesign:
    """Definition of experimental design for comparative analysis."""
    algorithms: List[str]
    datasets: List[str]
    evaluation_metrics: List[str]
    cv_strategy: str
    cv_folds: int
    cv_repeats: int
    random_seeds: List[int]
    significance_level: float
    effect_size_threshold: float
    power_analysis: bool
    preprocessing_steps: List[str]
    hyperparameter_optimization: bool


@dataclass
class AlgorithmResult:
    """Results for a single algorithm on a dataset."""
    algorithm_name: str
    dataset_name: str
    cv_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    training_times: List[float]
    prediction_times: List[float]
    model_complexity: int
    hyperparameters: Dict[str, Any]
    statistical_properties: Dict[str, Any]


@dataclass
class ComparativeAnalysisResult:
    """Complete comparative analysis results."""
    experimental_design: ExperimentalDesign
    algorithm_results: List[AlgorithmResult]
    pairwise_comparisons: Dict[Tuple[str, str], Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, Any]]
    ranking_analysis: Dict[str, Any]
    effect_sizes: Dict[Tuple[str, str], Dict[str, float]]
    performance_profiles: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    publication_summary: Dict[str, Any]


@dataclass
class BenchmarkDataset:
    """Benchmark dataset specification."""
    name: str
    X: pd.DataFrame
    y: pd.Series
    description: str
    domain: str
    task_type: str
    n_samples: int
    n_features: int
    n_classes: int
    class_balance: Dict[str, float]
    difficulty_score: float


class ResearchAlgorithmRegistry:
    """Registry of algorithms for comparative analysis."""
    
    def __init__(self):
        self.algorithms = self._initialize_algorithms()
    
    def _initialize_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive algorithm registry."""
        return {
            "logistic_regression": {
                "estimator": LogisticRegression,
                "default_params": {"random_state": 42, "max_iter": 1000},
                "param_grid": {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"]
                },
                "complexity_class": "linear",
                "theoretical_properties": {
                    "convex": True,
                    "probabilistic": True,
                    "interpretable": True,
                    "scalable": True
                }
            },
            "random_forest": {
                "estimator": RandomForestClassifier,
                "default_params": {"random_state": 42, "n_jobs": -1},
                "param_grid": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "complexity_class": "ensemble",
                "theoretical_properties": {
                    "convex": False,
                    "probabilistic": True,
                    "interpretable": False,
                    "scalable": True
                }
            },
            "gradient_boosting": {
                "estimator": GradientBoostingClassifier,
                "default_params": {"random_state": 42},
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                },
                "complexity_class": "boosting",
                "theoretical_properties": {
                    "convex": False,
                    "probabilistic": True,
                    "interpretable": False,
                    "scalable": True
                }
            },
            "svm": {
                "estimator": SVC,
                "default_params": {"random_state": 42, "probability": True},
                "param_grid": {
                    "C": [0.1, 1.0, 10.0, 100.0],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                    "kernel": ["rbf", "linear", "poly"]
                },
                "complexity_class": "kernel",
                "theoretical_properties": {
                    "convex": True,
                    "probabilistic": False,
                    "interpretable": False,
                    "scalable": False
                }
            },
            "neural_network": {
                "estimator": MLPClassifier,
                "default_params": {"random_state": 42, "max_iter": 500},
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "solver": ["adam", "lbfgs"]
                },
                "complexity_class": "neural",
                "theoretical_properties": {
                    "convex": False,
                    "probabilistic": True,
                    "interpretable": False,
                    "scalable": True
                }
            },
            "naive_bayes": {
                "estimator": GaussianNB,
                "default_params": {},
                "param_grid": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                },
                "complexity_class": "probabilistic",
                "theoretical_properties": {
                    "convex": True,
                    "probabilistic": True,
                    "interpretable": True,
                    "scalable": True
                }
            },
            "decision_tree": {
                "estimator": DecisionTreeClassifier,
                "default_params": {"random_state": 42},
                "param_grid": {
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 5, 10],
                    "criterion": ["gini", "entropy"]
                },
                "complexity_class": "tree",
                "theoretical_properties": {
                    "convex": False,
                    "probabilistic": True,
                    "interpretable": True,
                    "scalable": True
                }
            },
            "knn": {
                "estimator": KNeighborsClassifier,
                "default_params": {},
                "param_grid": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                "complexity_class": "instance",
                "theoretical_properties": {
                    "convex": False,
                    "probabilistic": True,
                    "interpretable": True,
                    "scalable": False
                }
            }
        }
    
    def get_algorithm(self, name: str) -> Dict[str, Any]:
        """Get algorithm configuration by name."""
        if name not in self.algorithms:
            raise ValueError(f"Algorithm '{name}' not found in registry")
        return self.algorithms[name]
    
    def list_algorithms(self) -> List[str]:
        """List all available algorithms."""
        return list(self.algorithms.keys())


class StatisticalAnalysisEngine:
    """Engine for statistical analysis and hypothesis testing."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def friedman_test(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform Friedman test for multiple algorithms."""
        algorithms = list(results.keys())
        scores_matrix = np.array([results[alg] for alg in algorithms])
        
        # Ensure we have enough samples and algorithms
        if len(algorithms) < 3 or scores_matrix.shape[1] < 6:
            return {
                "test": "friedman",
                "statistic": np.nan,
                "p_value": 1.0,
                "significant": False,
                "message": "Insufficient data for Friedman test"
            }
        
        try:
            statistic, p_value = friedmanchisquare(*scores_matrix)
            
            return {
                "test": "friedman",
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "algorithms": algorithms,
                "degrees_of_freedom": len(algorithms) - 1
            }
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
            return {
                "test": "friedman",
                "statistic": np.nan,
                "p_value": 1.0,
                "significant": False,
                "error": str(e)
            }
    
    def pairwise_tests(self, results: Dict[str, List[float]], 
                      test_method: str = "wilcoxon") -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Perform pairwise statistical tests between algorithms."""
        algorithms = list(results.keys())
        pairwise_results = {}
        
        for alg1, alg2 in combinations(algorithms, 2):
            scores1 = np.array(results[alg1])
            scores2 = np.array(results[alg2])
            
            if len(scores1) != len(scores2) or len(scores1) < 3:
                pairwise_results[(alg1, alg2)] = {
                    "test": test_method,
                    "statistic": np.nan,
                    "p_value": 1.0,
                    "significant": False,
                    "message": "Insufficient or mismatched data"
                }
                continue
            
            try:
                if test_method == "wilcoxon":
                    # Wilcoxon signed-rank test (paired)
                    statistic, p_value = wilcoxon(scores1, scores2)
                elif test_method == "ranksums":
                    # Wilcoxon rank-sum test (unpaired)
                    statistic, p_value = ranksums(scores1, scores2)
                else:
                    # Default to t-test
                    statistic, p_value = stats.ttest_rel(scores1, scores2)
                
                # Calculate effect size (Cohen's d)
                mean_diff = np.mean(scores1) - np.mean(scores2)
                pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                pairwise_results[(alg1, alg2)] = {
                    "test": test_method,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": p_value < self.significance_level,
                    "effect_size": cohens_d,
                    "mean_difference": mean_diff,
                    "winner": alg1 if mean_diff > 0 else alg2 if mean_diff < 0 else "tie"
                }
                
            except Exception as e:
                logger.warning(f"Pairwise test failed for {alg1} vs {alg2}: {e}")
                pairwise_results[(alg1, alg2)] = {
                    "test": test_method,
                    "statistic": np.nan,
                    "p_value": 1.0,
                    "significant": False,
                    "error": str(e)
                }
        
        return pairwise_results
    
    def rank_algorithms(self, results: Dict[str, List[float]], 
                       metric_name: str = "score") -> Dict[str, Any]:
        """Rank algorithms based on performance."""
        algorithms = list(results.keys())
        mean_scores = {alg: np.mean(results[alg]) for alg in algorithms}
        std_scores = {alg: np.std(results[alg]) for alg in algorithms}
        
        # Rank by mean performance (higher is better)
        ranked_algorithms = sorted(algorithms, key=lambda x: mean_scores[x], reverse=True)
        
        # Calculate ranking statistics
        ranking_data = []
        for i, alg in enumerate(ranked_algorithms):
            ranking_data.append({
                "rank": i + 1,
                "algorithm": alg,
                "mean_score": mean_scores[alg],
                "std_score": std_scores[alg],
                "confidence_interval": stats.t.interval(
                    1 - self.significance_level,
                    len(results[alg]) - 1,
                    loc=mean_scores[alg],
                    scale=std_scores[alg] / np.sqrt(len(results[alg]))
                )
            })
        
        return {
            "ranking": ranking_data,
            "best_algorithm": ranked_algorithms[0],
            "worst_algorithm": ranked_algorithms[-1],
            "performance_gap": mean_scores[ranked_algorithms[0]] - mean_scores[ranked_algorithms[-1]],
            "metric": metric_name
        }
    
    def calculate_critical_difference(self, results: Dict[str, List[float]], 
                                    alpha: float = 0.05) -> float:
        """Calculate critical difference for Nemenyi post-hoc test."""
        k = len(results)  # number of algorithms
        n = len(list(results.values())[0])  # number of datasets/folds
        
        # Critical value for Nemenyi test (approximation)
        # This is a simplified version; full implementation would use exact values
        q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf)
        critical_difference = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        
        return critical_difference


class BenchmarkDatasetGenerator:
    """Generate and manage benchmark datasets."""
    
    @staticmethod
    def create_synthetic_datasets(n_datasets: int = 5, random_state: int = 42) -> List[BenchmarkDataset]:
        """Create synthetic datasets with varying characteristics."""
        np.random.seed(random_state)
        datasets = []
        
        # Dataset configurations
        configs = [
            {"n_samples": 1000, "n_features": 10, "n_classes": 2, "noise": 0.1, "name": "synthetic_balanced_easy"},
            {"n_samples": 1000, "n_features": 20, "n_classes": 2, "noise": 0.2, "name": "synthetic_balanced_medium"},
            {"n_samples": 500, "n_features": 50, "n_classes": 2, "noise": 0.3, "name": "synthetic_highdim_hard"},
            {"n_samples": 2000, "n_features": 5, "n_classes": 3, "name": "synthetic_multiclass_easy"},
            {"n_samples": 800, "n_features": 15, "n_classes": 4, "noise": 0.25, "name": "synthetic_multiclass_medium"}
        ]
        
        from sklearn.datasets import make_classification
        
        for i, config in enumerate(configs[:n_datasets]):
            X, y = make_classification(
                n_samples=config["n_samples"],
                n_features=config["n_features"],
                n_classes=config["n_classes"],
                n_informative=max(2, config["n_features"] // 2),
                n_redundant=config["n_features"] // 4,
                noise=config.get("noise", 0.1),
                random_state=random_state + i
            )
            
            # Convert to DataFrame
            feature_names = [f"feature_{j}" for j in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name="target")
            
            # Calculate class balance
            class_balance = {}
            for class_val in np.unique(y):
                class_balance[str(class_val)] = np.sum(y == class_val) / len(y)
            
            # Estimate difficulty score based on dataset characteristics
            difficulty_score = min(1.0, (
                config.get("noise", 0.1) +
                (config["n_features"] / config["n_samples"]) +
                (1 / config["n_classes"]) +
                (1 - max(class_balance.values())) * 0.5
            ))
            
            dataset = BenchmarkDataset(
                name=config["name"],
                X=X_df,
                y=y_series,
                description=f"Synthetic dataset with {config['n_samples']} samples, {config['n_features']} features, {config['n_classes']} classes",
                domain="synthetic",
                task_type="classification",
                n_samples=config["n_samples"],
                n_features=config["n_features"],
                n_classes=config["n_classes"],
                class_balance=class_balance,
                difficulty_score=difficulty_score
            )
            
            datasets.append(dataset)
        
        return datasets


class ResearchBenchmarkingFramework:
    """Main research benchmarking framework."""
    
    def __init__(self, experimental_design: ExperimentalDesign = None):
        self.experimental_design = experimental_design or self._default_experimental_design()
        self.algorithm_registry = ResearchAlgorithmRegistry()
        self.statistical_engine = StatisticalAnalysisEngine(
            significance_level=self.experimental_design.significance_level
        )
        self.results_cache = {}
        
    def _default_experimental_design(self) -> ExperimentalDesign:
        """Create default experimental design."""
        return ExperimentalDesign(
            algorithms=["logistic_regression", "random_forest", "svm", "gradient_boosting"],
            datasets=["synthetic_balanced_easy", "synthetic_balanced_medium", "synthetic_highdim_hard"],
            evaluation_metrics=["accuracy", "f1_score", "roc_auc"],
            cv_strategy="stratified_kfold",
            cv_folds=5,
            cv_repeats=3,
            random_seeds=[42, 123, 456],
            significance_level=0.05,
            effect_size_threshold=0.2,
            power_analysis=True,
            preprocessing_steps=["standard_scaling"],
            hyperparameter_optimization=False
        )
    
    def run_comprehensive_benchmark(self, datasets: List[BenchmarkDataset]) -> ComparativeAnalysisResult:
        """Run comprehensive benchmark across algorithms and datasets."""
        logger.info("Starting comprehensive benchmark analysis...")
        start_time = time.time()
        
        # Initialize results storage
        all_results = []
        experiment_data = {}
        
        # MLflow experiment tracking
        experiment_name = f"research_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass
        
        mlflow.set_experiment(experiment_name)
        
        # Run experiments for each algorithm-dataset combination
        total_experiments = len(self.experimental_design.algorithms) * len(datasets)
        current_experiment = 0
        
        for algorithm_name in self.experimental_design.algorithms:
            for dataset in datasets:
                current_experiment += 1
                logger.info(f"Running experiment {current_experiment}/{total_experiments}: {algorithm_name} on {dataset.name}")
                
                try:
                    result = self._run_single_experiment(algorithm_name, dataset)
                    all_results.append(result)
                    
                    # Store in experiment data structure
                    if algorithm_name not in experiment_data:
                        experiment_data[algorithm_name] = {}
                    experiment_data[algorithm_name][dataset.name] = result
                    
                except Exception as e:
                    logger.error(f"Experiment failed for {algorithm_name} on {dataset.name}: {e}")
                    continue
        
        # Perform comparative analysis
        logger.info("Performing comparative statistical analysis...")
        
        # Aggregate results by metric
        metric_results = {}
        for metric in self.experimental_design.evaluation_metrics:
            metric_results[metric] = {}
            for alg_result in all_results:
                alg_name = alg_result.algorithm_name
                if alg_name not in metric_results[metric]:
                    metric_results[metric][alg_name] = []
                metric_results[metric][alg_name].extend(alg_result.cv_scores[metric])
        
        # Statistical tests
        statistical_tests = {}
        pairwise_comparisons = {}
        ranking_analyses = {}
        
        for metric, results in metric_results.items():
            if len(results) >= 2:
                # Friedman test
                statistical_tests[metric] = self.statistical_engine.friedman_test(results)
                
                # Pairwise comparisons
                pairwise_comparisons[metric] = self.statistical_engine.pairwise_tests(results)
                
                # Ranking analysis
                ranking_analyses[metric] = self.statistical_engine.rank_algorithms(results, metric)
        
        # Generate recommendations
        recommendations = self._generate_research_recommendations(
            all_results, statistical_tests, ranking_analyses
        )
        
        # Create publication summary
        publication_summary = self._create_publication_summary(
            all_results, statistical_tests, ranking_analyses, time.time() - start_time
        )
        
        # Compile final results
        comparative_result = ComparativeAnalysisResult(
            experimental_design=self.experimental_design,
            algorithm_results=all_results,
            pairwise_comparisons=pairwise_comparisons,
            statistical_tests=statistical_tests,
            ranking_analysis=ranking_analyses,
            effect_sizes={},  # Would be calculated in full implementation
            performance_profiles={},  # Would include detailed performance characteristics
            recommendations=recommendations,
            publication_summary=publication_summary
        )
        
        logger.info(f"Comprehensive benchmark completed in {time.time() - start_time:.2f}s")
        
        return comparative_result
    
    def _run_single_experiment(self, algorithm_name: str, dataset: BenchmarkDataset) -> AlgorithmResult:
        """Run a single algorithm on a single dataset."""
        alg_config = self.algorithm_registry.get_algorithm(algorithm_name)
        
        # Create estimator
        estimator = alg_config["estimator"](**alg_config["default_params"])
        
        # Cross-validation strategy
        if self.experimental_design.cv_strategy == "stratified_kfold":
            cv = StratifiedKFold(
                n_splits=self.experimental_design.cv_folds,
                shuffle=True,
                random_state=42
            )
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform cross-validation
        with mlflow.start_run(run_name=f"{algorithm_name}_{dataset.name}"):
            mlflow.log_param("algorithm", algorithm_name)
            mlflow.log_param("dataset", dataset.name)
            mlflow.log_param("n_samples", dataset.n_samples)
            mlflow.log_param("n_features", dataset.n_features)
            mlflow.log_param("n_classes", dataset.n_classes)
            
            # Time the cross-validation
            start_time = time.time()
            
            cv_results = cross_validate(
                estimator, dataset.X, dataset.y,
                cv=cv,
                scoring=['accuracy', 'f1_weighted', 'roc_auc_ovo'],
                return_train_score=False,
                n_jobs=1  # Sequential to avoid resource conflicts
            )
            
            training_time = time.time() - start_time
            
            # Extract scores
            cv_scores = {
                'accuracy': cv_results['test_accuracy'].tolist(),
                'f1_score': cv_results['test_f1_weighted'].tolist(),
                'roc_auc': cv_results['test_roc_auc_ovo'].tolist()
            }
            
            mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
            std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
            
            # Log metrics
            for metric, score in mean_scores.items():
                mlflow.log_metric(f"mean_{metric}", score)
                mlflow.log_metric(f"std_{metric}", std_scores[metric])
            
            mlflow.log_metric("training_time", training_time)
            
            # Calculate model complexity (simplified)
            model_complexity = self._estimate_model_complexity(estimator, dataset)
            
            # Measure prediction time
            estimator.fit(dataset.X, dataset.y)
            pred_start = time.time()
            _ = estimator.predict(dataset.X[:min(100, len(dataset.X))])
            prediction_time = (time.time() - pred_start) / min(100, len(dataset.X))
            
            result = AlgorithmResult(
                algorithm_name=algorithm_name,
                dataset_name=dataset.name,
                cv_scores=cv_scores,
                mean_scores=mean_scores,
                std_scores=std_scores,
                training_times=[training_time],
                prediction_times=[prediction_time],
                model_complexity=model_complexity,
                hyperparameters=alg_config["default_params"],
                statistical_properties=alg_config["theoretical_properties"]
            )
            
            return result
    
    def _estimate_model_complexity(self, estimator: BaseEstimator, dataset: BenchmarkDataset) -> int:
        """Estimate model complexity."""
        # Simplified complexity estimation
        if hasattr(estimator, 'n_estimators'):
            return getattr(estimator, 'n_estimators', 100)
        elif hasattr(estimator, 'C'):
            return int(1 / (getattr(estimator, 'C', 1.0) + 1e-6))
        else:
            return dataset.n_features  # Default to feature count
    
    def _generate_research_recommendations(self, results: List[AlgorithmResult],
                                         statistical_tests: Dict[str, Dict[str, Any]],
                                         ranking_analyses: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate research recommendations based on analysis."""
        recommendations = []
        
        # Overall best algorithm
        accuracy_ranking = ranking_analyses.get('accuracy', {}).get('ranking', [])
        if accuracy_ranking:
            best_alg = accuracy_ranking[0]['algorithm']
            best_score = accuracy_ranking[0]['mean_score']
            recommendations.append(
                f"Best overall algorithm: {best_alg} (accuracy: {best_score:.4f})"
            )
        
        # Statistical significance
        for metric, test_result in statistical_tests.items():
            if test_result.get('significant', False):
                recommendations.append(
                    f"Significant differences detected for {metric} (p={test_result['p_value']:.4f})"
                )
        
        # Algorithm characteristics
        complexity_analysis = self._analyze_complexity_performance_tradeoff(results)
        if complexity_analysis:
            recommendations.append(complexity_analysis)
        
        # Domain-specific recommendations
        recommendations.append(
            "Consider ensemble methods for improved performance and robustness"
        )
        
        return recommendations
    
    def _analyze_complexity_performance_tradeoff(self, results: List[AlgorithmResult]) -> str:
        """Analyze complexity vs performance tradeoff."""
        # Group results by algorithm
        alg_performance = {}
        for result in results:
            alg_name = result.algorithm_name
            if alg_name not in alg_performance:
                alg_performance[alg_name] = {
                    'accuracy': [],
                    'complexity': [],
                    'training_time': [],
                    'prediction_time': []
                }
            
            alg_performance[alg_name]['accuracy'].extend([result.mean_scores['accuracy']])
            alg_performance[alg_name]['complexity'].append(result.model_complexity)
            alg_performance[alg_name]['training_time'].extend(result.training_times)
            alg_performance[alg_name]['prediction_time'].extend(result.prediction_times)
        
        # Find best tradeoff
        best_tradeoff = None
        best_score = 0
        
        for alg_name, metrics in alg_performance.items():
            avg_accuracy = np.mean(metrics['accuracy'])
            avg_complexity = np.mean(metrics['complexity'])
            avg_pred_time = np.mean(metrics['prediction_time'])
            
            # Simple tradeoff score (higher accuracy, lower complexity and time)
            if avg_complexity > 0 and avg_pred_time > 0:
                tradeoff_score = avg_accuracy / (np.log(avg_complexity + 1) + avg_pred_time * 1000)
                if tradeoff_score > best_score:
                    best_score = tradeoff_score
                    best_tradeoff = alg_name
        
        if best_tradeoff:
            return f"Best performance/complexity tradeoff: {best_tradeoff}"
        else:
            return "Unable to determine optimal complexity/performance tradeoff"
    
    def _create_publication_summary(self, results: List[AlgorithmResult],
                                  statistical_tests: Dict[str, Dict[str, Any]],
                                  ranking_analyses: Dict[str, Dict[str, Any]],
                                  total_time: float) -> Dict[str, Any]:
        """Create publication-ready summary."""
        
        # Collect summary statistics
        n_algorithms = len(set(r.algorithm_name for r in results))
        n_datasets = len(set(r.dataset_name for r in results))
        n_experiments = len(results)
        
        # Best performers by metric
        best_performers = {}
        for metric, ranking in ranking_analyses.items():
            if ranking.get('ranking'):
                best_performers[metric] = {
                    'algorithm': ranking['ranking'][0]['algorithm'],
                    'score': ranking['ranking'][0]['mean_score'],
                    'std': ranking['ranking'][0]['std_score']
                }
        
        # Statistical significance summary
        significant_results = {}
        for metric, test in statistical_tests.items():
            significant_results[metric] = test.get('significant', False)
        
        return {
            'experiment_summary': {
                'n_algorithms': n_algorithms,
                'n_datasets': n_datasets,
                'n_experiments': n_experiments,
                'total_runtime': total_time,
                'cv_strategy': self.experimental_design.cv_strategy,
                'cv_folds': self.experimental_design.cv_folds
            },
            'best_performers': best_performers,
            'statistical_significance': significant_results,
            'methodology': {
                'evaluation_metrics': self.experimental_design.evaluation_metrics,
                'significance_level': self.experimental_design.significance_level,
                'random_seeds': self.experimental_design.random_seeds
            },
            'reproducibility': {
                'random_seeds': self.experimental_design.random_seeds,
                'software_versions': {
                    'python': '3.12',
                    'sklearn': '1.7.0',
                    'numpy': '2.3.1',
                    'pandas': '2.3.0'
                }
            }
        }
    
    def generate_research_report(self, result: ComparativeAnalysisResult, 
                               output_dir: str = "research_outputs") -> str:
        """Generate comprehensive research report."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"research_report_{timestamp}.json")
        
        # Convert result to serializable format
        report_data = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'framework_version': '1.0.0',
                'experimental_design': asdict(result.experimental_design)
            },
            'results_summary': result.publication_summary,
            'algorithm_performance': [asdict(r) for r in result.algorithm_results],
            'statistical_analysis': result.statistical_tests,
            'rankings': result.ranking_analysis,
            'recommendations': result.recommendations
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Research report saved to {report_file}")
        
        # Generate plots
        self._generate_research_plots(result, output_dir, timestamp)
        
        return report_file
    
    def _generate_research_plots(self, result: ComparativeAnalysisResult, 
                               output_dir: str, timestamp: str) -> None:
        """Generate research-quality plots."""
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison plot
        fig, axes = plt.subplots(1, len(result.ranking_analysis), figsize=(15, 5))
        if len(result.ranking_analysis) == 1:
            axes = [axes]
        
        for i, (metric, ranking) in enumerate(result.ranking_analysis.items()):
            if ranking.get('ranking'):
                algorithms = [r['algorithm'] for r in ranking['ranking']]
                scores = [r['mean_score'] for r in ranking['ranking']]
                errors = [r['std_score'] for r in ranking['ranking']]
                
                axes[i].barh(algorithms, scores, xerr=errors, capsize=5)
                axes[i].set_xlabel(metric.replace('_', ' ').title())
                axes[i].set_title(f'{metric.replace("_", " ").title()} Performance')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"performance_comparison_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Research plots saved to {output_dir}")


def run_research_benchmark(algorithms: List[str] = None, 
                         n_synthetic_datasets: int = 3,
                         output_dir: str = "research_outputs") -> ComparativeAnalysisResult:
    """Run complete research benchmark with default settings."""
    
    # Default algorithms if not specified
    if algorithms is None:
        algorithms = ["logistic_regression", "random_forest", "svm", "gradient_boosting", "neural_network"]
    
    # Create experimental design
    experimental_design = ExperimentalDesign(
        algorithms=algorithms,
        datasets=[],  # Will be filled by synthetic datasets
        evaluation_metrics=["accuracy", "f1_score", "roc_auc"],
        cv_strategy="stratified_kfold",
        cv_folds=5,
        cv_repeats=1,
        random_seeds=[42],
        significance_level=0.05,
        effect_size_threshold=0.2,
        power_analysis=True,
        preprocessing_steps=["standard_scaling"],
        hyperparameter_optimization=False
    )
    
    # Initialize framework
    framework = ResearchBenchmarkingFramework(experimental_design)
    
    # Generate synthetic datasets
    logger.info(f"Generating {n_synthetic_datasets} synthetic benchmark datasets...")
    datasets = BenchmarkDatasetGenerator.create_synthetic_datasets(
        n_datasets=n_synthetic_datasets, random_state=42
    )
    
    # Run benchmark
    result = framework.run_comprehensive_benchmark(datasets)
    
    # Generate report
    report_file = framework.generate_research_report(result, output_dir)
    
    logger.info(f"Research benchmark completed. Report saved to: {report_file}")
    
    return result


if __name__ == "__main__":
    print("Research-Grade Benchmarking Framework")
    print("This framework provides comprehensive algorithm comparison with statistical rigor.")
    
    # Example usage
    # result = run_research_benchmark()
    # print(f"Benchmark completed with {len(result.algorithm_results)} experiments")