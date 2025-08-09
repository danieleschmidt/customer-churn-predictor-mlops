"""
Intelligent Model Selection Framework with AutoML Capabilities.

This module provides automated model selection and hyperparameter optimization
using advanced meta-learning, progressive search strategies, and ensemble methods.
Designed for autonomous model selection across different data characteristics.
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, RandomizedSearchCV,
    train_test_split, validation_curve
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import mlflow
import mlflow.sklearn
from scipy.stats import uniform, randint

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .data_validation import validate_prediction_data

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DataCharacteristics:
    """Characteristics of a dataset for model selection."""
    n_samples: int
    n_features: int
    n_classes: int
    class_balance: Dict[str, float]
    feature_types: Dict[str, str]  # numerical, categorical, etc.
    missing_values_pct: float
    correlation_strength: float
    dimensionality_ratio: float  # n_features / n_samples
    noise_level: float
    dataset_complexity: str  # simple, medium, complex
    domain_type: str  # tabular, text, image, etc.


@dataclass
class ModelConfiguration:
    """Configuration for a machine learning model."""
    name: str
    estimator_class: type
    default_params: Dict[str, Any]
    param_distributions: Dict[str, Any]
    complexity_score: int  # 1-10 scale
    training_speed: int  # 1-10 scale (10 = fastest)
    interpretability: int  # 1-10 scale (10 = most interpretable)
    scalability: int  # 1-10 scale (10 = most scalable)
    recommended_data_size: Tuple[int, int]  # (min_samples, max_samples)
    recommended_features: Tuple[int, int]  # (min_features, max_features)
    handles_missing: bool
    handles_categorical: bool
    supports_multiclass: bool
    theoretical_properties: Dict[str, bool]


@dataclass
class ModelSelectionResult:
    """Result from intelligent model selection."""
    best_model: BaseEstimator
    best_model_name: str
    best_score: float
    best_params: Dict[str, Any]
    model_rankings: List[Dict[str, Any]]
    selection_strategy: str
    data_characteristics: DataCharacteristics
    preprocessing_pipeline: Pipeline
    feature_importance: Optional[Dict[str, float]]
    cross_validation_scores: Dict[str, List[float]]
    training_time: float
    selection_confidence: float
    ensemble_models: Optional[List[BaseEstimator]]
    meta_features: Dict[str, float]


class MetaLearningEngine:
    """Meta-learning engine for intelligent model selection."""
    
    def __init__(self):
        self.meta_knowledge = self._initialize_meta_knowledge()
        self.performance_history = {}
        
    def _initialize_meta_knowledge(self) -> Dict[str, Any]:
        """Initialize meta-learning knowledge base."""
        return {
            "data_patterns": {
                "high_dimensional": {
                    "recommended_models": ["random_forest", "gradient_boosting", "linear_models"],
                    "avoid_models": ["knn", "naive_bayes"],
                    "preprocessing": ["feature_selection", "dimensionality_reduction"]
                },
                "small_sample": {
                    "recommended_models": ["naive_bayes", "logistic_regression", "svm"],
                    "avoid_models": ["neural_network", "ensemble_large"],
                    "preprocessing": ["regularization", "cross_validation"]
                },
                "imbalanced": {
                    "recommended_models": ["random_forest", "gradient_boosting"],
                    "avoid_models": ["knn", "linear_discriminant"],
                    "preprocessing": ["resampling", "class_weight_balancing"]
                },
                "noisy": {
                    "recommended_models": ["ensemble_methods", "robust_models"],
                    "avoid_models": ["knn", "decision_tree"],
                    "preprocessing": ["outlier_removal", "robust_scaling"]
                },
                "linear_separable": {
                    "recommended_models": ["logistic_regression", "svm_linear", "linear_discriminant"],
                    "avoid_models": ["ensemble_complex", "neural_network"],
                    "preprocessing": ["standard_scaling"]
                }
            },
            "complexity_mapping": {
                "simple": {"max_complexity": 3, "prefer_interpretable": True},
                "medium": {"max_complexity": 6, "prefer_interpretable": False},
                "complex": {"max_complexity": 10, "prefer_interpretable": False}
            }
        }
    
    def analyze_data_characteristics(self, X: pd.DataFrame, y: pd.Series) -> DataCharacteristics:
        """Analyze dataset characteristics for model selection."""
        n_samples, n_features = X.shape
        n_classes = len(y.unique())
        
        # Class balance analysis
        class_balance = {}
        value_counts = y.value_counts()
        for class_val, count in value_counts.items():
            class_balance[str(class_val)] = count / len(y)
        
        # Feature type analysis
        feature_types = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'
        
        # Missing values analysis
        missing_pct = (X.isnull().sum().sum() / (n_samples * n_features)) * 100
        
        # Correlation analysis (for numerical features only)
        numeric_cols = [col for col, dtype in feature_types.items() if dtype == 'numerical']
        if len(numeric_cols) > 1:
            correlation_matrix = X[numeric_cols].corr().abs()
            correlation_strength = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        else:
            correlation_strength = 0.0
        
        # Dimensionality ratio
        dimensionality_ratio = n_features / n_samples
        
        # Estimate noise level (simplified)
        noise_level = min(1.0, missing_pct / 100 + (1 - max(class_balance.values())))
        
        # Dataset complexity assessment
        complexity_factors = [
            dimensionality_ratio > 0.1,  # High dimensional
            n_classes > 2,  # Multiclass
            missing_pct > 5,  # Significant missing data
            min(class_balance.values()) < 0.3,  # Class imbalance
            correlation_strength > 0.8  # High correlation
        ]
        
        complexity_count = sum(complexity_factors)
        if complexity_count <= 1:
            dataset_complexity = "simple"
        elif complexity_count <= 3:
            dataset_complexity = "medium"
        else:
            dataset_complexity = "complex"
        
        return DataCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_balance=class_balance,
            feature_types=feature_types,
            missing_values_pct=missing_pct,
            correlation_strength=correlation_strength,
            dimensionality_ratio=dimensionality_ratio,
            noise_level=noise_level,
            dataset_complexity=dataset_complexity,
            domain_type="tabular"
        )
    
    def recommend_models(self, data_chars: DataCharacteristics) -> List[str]:
        """Recommend models based on data characteristics."""
        recommended = set()
        avoided = set()
        
        # Apply meta-learning rules
        if data_chars.dimensionality_ratio > 0.1:
            pattern = self.meta_knowledge["data_patterns"]["high_dimensional"]
            recommended.update(pattern["recommended_models"])
            avoided.update(pattern["avoid_models"])
        
        if data_chars.n_samples < 1000:
            pattern = self.meta_knowledge["data_patterns"]["small_sample"]
            recommended.update(pattern["recommended_models"])
            avoided.update(pattern["avoid_models"])
        
        if min(data_chars.class_balance.values()) < 0.3:
            pattern = self.meta_knowledge["data_patterns"]["imbalanced"]
            recommended.update(pattern["recommended_models"])
            avoided.update(pattern["avoid_models"])
        
        if data_chars.noise_level > 0.3:
            pattern = self.meta_knowledge["data_patterns"]["noisy"]
            recommended.update(pattern["recommended_models"])
            avoided.update(pattern["avoid_models"])
        
        # Remove avoided models
        final_recommendations = list(recommended - avoided)
        
        # If no specific recommendations, use default set
        if not final_recommendations:
            final_recommendations = ["logistic_regression", "random_forest", "gradient_boosting"]
        
        return final_recommendations
    
    def calculate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate meta-features for the dataset."""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        meta_features = {}
        
        # Basic statistics
        meta_features['n_samples'] = len(X)
        meta_features['n_features'] = len(X.columns)
        meta_features['n_classes'] = len(y.unique())
        meta_features['samples_per_feature'] = len(X) / len(X.columns)
        
        # Class distribution features
        class_counts = y.value_counts()
        meta_features['class_entropy'] = -(class_counts / len(y) * np.log2(class_counts / len(y))).sum()
        meta_features['minority_class_ratio'] = class_counts.min() / len(y)
        meta_features['majority_class_ratio'] = class_counts.max() / len(y)
        
        # Feature distribution features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            meta_features['mean_skewness'] = abs(X[numeric_cols].skew()).mean()
            meta_features['mean_kurtosis'] = abs(X[numeric_cols].kurtosis()).mean()
            meta_features['mean_correlation'] = abs(X[numeric_cols].corr()).values[np.triu_indices_from(X[numeric_cols].corr().values, k=1)].mean()
        else:
            meta_features['mean_skewness'] = 0.0
            meta_features['mean_kurtosis'] = 0.0
            meta_features['mean_correlation'] = 0.0
        
        # Information theoretic features
        try:
            # Encode categorical variables for mutual information
            X_encoded = X.copy()
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            mutual_info = mutual_info_classif(X_encoded, y, random_state=42)
            meta_features['mean_mutual_info'] = np.mean(mutual_info)
            meta_features['max_mutual_info'] = np.max(mutual_info)
            meta_features['std_mutual_info'] = np.std(mutual_info)
        except Exception as e:
            logger.warning(f"Failed to calculate mutual information: {e}")
            meta_features['mean_mutual_info'] = 0.0
            meta_features['max_mutual_info'] = 0.0
            meta_features['std_mutual_info'] = 0.0
        
        return meta_features


class ModelConfigurationRegistry:
    """Registry of model configurations with meta-information."""
    
    def __init__(self):
        self.configurations = self._initialize_configurations()
    
    def _initialize_configurations(self) -> Dict[str, ModelConfiguration]:
        """Initialize comprehensive model configurations."""
        configs = {}
        
        # Logistic Regression
        configs["logistic_regression"] = ModelConfiguration(
            name="Logistic Regression",
            estimator_class=LogisticRegression,
            default_params={"random_state": 42, "max_iter": 1000},
            param_distributions={
                "C": uniform(0.01, 100),
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga", "lbfgs"],
                "class_weight": [None, "balanced"]
            },
            complexity_score=2,
            training_speed=9,
            interpretability=9,
            scalability=8,
            recommended_data_size=(100, 100000),
            recommended_features=(1, 1000),
            handles_missing=False,
            handles_categorical=False,
            supports_multiclass=True,
            theoretical_properties={"linear": True, "probabilistic": True}
        )
        
        # Random Forest
        configs["random_forest"] = ModelConfiguration(
            name="Random Forest",
            estimator_class=RandomForestClassifier,
            default_params={"random_state": 42, "n_jobs": -1},
            param_distributions={
                "n_estimators": randint(50, 500),
                "max_depth": [None] + list(range(3, 20)),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced", "balanced_subsample"]
            },
            complexity_score=6,
            training_speed=6,
            interpretability=5,
            scalability=7,
            recommended_data_size=(100, 1000000),
            recommended_features=(1, 10000),
            handles_missing=True,
            handles_categorical=True,
            supports_multiclass=True,
            theoretical_properties={"ensemble": True, "non_linear": True}
        )
        
        # Gradient Boosting
        configs["gradient_boosting"] = ModelConfiguration(
            name="Gradient Boosting",
            estimator_class=GradientBoostingClassifier,
            default_params={"random_state": 42},
            param_distributions={
                "n_estimators": randint(50, 300),
                "learning_rate": uniform(0.01, 0.3),
                "max_depth": randint(3, 10),
                "subsample": uniform(0.6, 0.4),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10)
            },
            complexity_score=8,
            training_speed=4,
            interpretability=3,
            scalability=6,
            recommended_data_size=(500, 500000),
            recommended_features=(1, 5000),
            handles_missing=True,
            handles_categorical=False,
            supports_multiclass=True,
            theoretical_properties={"boosting": True, "non_linear": True}
        )
        
        # Support Vector Machine
        configs["svm"] = ModelConfiguration(
            name="Support Vector Machine",
            estimator_class=SVC,
            default_params={"random_state": 42, "probability": True},
            param_distributions={
                "C": uniform(0.1, 100),
                "gamma": ["scale", "auto"] + list(uniform(0.001, 1).rvs(10)),
                "kernel": ["rbf", "linear", "poly"],
                "class_weight": [None, "balanced"]
            },
            complexity_score=7,
            training_speed=3,
            interpretability=2,
            scalability=3,
            recommended_data_size=(100, 10000),
            recommended_features=(1, 1000),
            handles_missing=False,
            handles_categorical=False,
            supports_multiclass=True,
            theoretical_properties={"kernel_method": True, "non_linear": True}
        )
        
        # Neural Network
        configs["neural_network"] = ModelConfiguration(
            name="Multi-layer Perceptron",
            estimator_class=MLPClassifier,
            default_params={"random_state": 42, "max_iter": 500},
            param_distributions={
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                "alpha": uniform(0.0001, 0.01),
                "learning_rate": ["constant", "adaptive"],
                "solver": ["adam", "lbfgs"],
                "activation": ["relu", "tanh", "logistic"]
            },
            complexity_score=9,
            training_speed=5,
            interpretability=1,
            scalability=7,
            recommended_data_size=(500, 100000),
            recommended_features=(1, 5000),
            handles_missing=False,
            handles_categorical=False,
            supports_multiclass=True,
            theoretical_properties={"neural_network": True, "non_linear": True}
        )
        
        # Naive Bayes
        configs["naive_bayes"] = ModelConfiguration(
            name="Gaussian Naive Bayes",
            estimator_class=GaussianNB,
            default_params={},
            param_distributions={
                "var_smoothing": uniform(1e-10, 1e-5)
            },
            complexity_score=1,
            training_speed=10,
            interpretability=8,
            scalability=9,
            recommended_data_size=(50, 50000),
            recommended_features=(1, 100),
            handles_missing=False,
            handles_categorical=False,
            supports_multiclass=True,
            theoretical_properties={"probabilistic": True, "linear": True}
        )
        
        # Extra Trees
        configs["extra_trees"] = ModelConfiguration(
            name="Extra Trees",
            estimator_class=ExtraTreesClassifier,
            default_params={"random_state": 42, "n_jobs": -1},
            param_distributions={
                "n_estimators": randint(50, 500),
                "max_depth": [None] + list(range(3, 20)),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"]
            },
            complexity_score=6,
            training_speed=7,
            interpretability=4,
            scalability=8,
            recommended_data_size=(100, 1000000),
            recommended_features=(1, 10000),
            handles_missing=True,
            handles_categorical=True,
            supports_multiclass=True,
            theoretical_properties={"ensemble": True, "non_linear": True}
        )
        
        return configs
    
    def get_configuration(self, model_name: str) -> ModelConfiguration:
        """Get model configuration by name."""
        if model_name not in self.configurations:
            raise ValueError(f"Model '{model_name}' not found in registry")
        return self.configurations[model_name]
    
    def get_recommended_models(self, data_chars: DataCharacteristics) -> List[str]:
        """Get models recommended for given data characteristics."""
        recommended = []
        
        for name, config in self.configurations.items():
            # Check data size compatibility
            min_samples, max_samples = config.recommended_data_size
            min_features, max_features = config.recommended_features
            
            if (min_samples <= data_chars.n_samples <= max_samples and
                min_features <= data_chars.n_features <= max_features):
                recommended.append(name)
        
        return recommended if recommended else ["logistic_regression", "random_forest"]


class IntelligentModelSelector:
    """Main intelligent model selection system."""
    
    def __init__(self, 
                 cv_folds: int = 5,
                 n_iter: int = 50,
                 scoring: str = "f1_weighted",
                 random_state: int = 42,
                 n_jobs: int = -1,
                 enable_ensemble: bool = True):
        """
        Initialize intelligent model selector.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for hyperparameter search
            scoring: Scoring metric for model evaluation
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            enable_ensemble: Whether to create ensemble models
        """
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.enable_ensemble = enable_ensemble
        
        self.meta_engine = MetaLearningEngine()
        self.model_registry = ModelConfigurationRegistry()
        self.selection_history = []
        
    def select_best_model(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         candidate_models: Optional[List[str]] = None,
                         preprocessing_options: Optional[List[str]] = None,
                         selection_strategy: str = "comprehensive") -> ModelSelectionResult:
        """
        Select the best model for the given dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            candidate_models: List of models to consider
            preprocessing_options: Preprocessing options to try
            selection_strategy: Selection strategy ("fast", "balanced", "comprehensive")
            
        Returns:
            ModelSelectionResult with best model and analysis
        """
        logger.info("Starting intelligent model selection...")
        start_time = time.time()
        
        # Analyze data characteristics
        logger.info("Analyzing data characteristics...")
        data_chars = self.meta_engine.analyze_data_characteristics(X, y)
        meta_features = self.meta_engine.calculate_meta_features(X, y)
        
        logger.info(f"Dataset complexity: {data_chars.dataset_complexity}")
        logger.info(f"Samples: {data_chars.n_samples}, Features: {data_chars.n_features}, Classes: {data_chars.n_classes}")
        
        # Get candidate models
        if candidate_models is None:
            candidate_models = self.meta_engine.recommend_models(data_chars)
            logger.info(f"Recommended models: {candidate_models}")
        
        # Add registry-based recommendations
        registry_models = self.model_registry.get_recommended_models(data_chars)
        candidate_models = list(set(candidate_models + registry_models))
        
        # Adjust search intensity based on strategy
        if selection_strategy == "fast":
            search_iterations = max(10, self.n_iter // 3)
            cv_folds = 3
        elif selection_strategy == "comprehensive":
            search_iterations = self.n_iter * 2
            cv_folds = self.cv_folds
        else:  # balanced
            search_iterations = self.n_iter
            cv_folds = self.cv_folds
        
        # Create preprocessing pipeline
        preprocessor = self._create_preprocessing_pipeline(X, data_chars, preprocessing_options)
        
        # Evaluate candidate models
        logger.info(f"Evaluating {len(candidate_models)} candidate models...")
        model_results = []
        
        for model_name in candidate_models:
            try:
                logger.info(f"Evaluating {model_name}...")
                result = self._evaluate_model(
                    model_name, X, y, preprocessor, 
                    search_iterations, cv_folds
                )
                model_results.append(result)
                logger.info(f"{model_name}: {result['score']:.4f} ± {result['score_std']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if not model_results:
            raise ValueError("No models could be evaluated successfully")
        
        # Sort by performance
        model_results.sort(key=lambda x: x['score'], reverse=True)
        best_result = model_results[0]
        
        logger.info(f"Best model: {best_result['name']} with score {best_result['score']:.4f}")
        
        # Create ensemble if enabled
        ensemble_models = None
        if self.enable_ensemble and len(model_results) >= 3:
            ensemble_models = self._create_ensemble_models(model_results[:3], X, y, preprocessor)
        
        # Calculate selection confidence
        if len(model_results) >= 2:
            selection_confidence = (best_result['score'] - model_results[1]['score']) / best_result['score']
        else:
            selection_confidence = 1.0
        
        # Extract feature importance
        feature_importance = self._extract_feature_importance(best_result['model'], X.columns)
        
        selection_result = ModelSelectionResult(
            best_model=best_result['model'],
            best_model_name=best_result['name'],
            best_score=best_result['score'],
            best_params=best_result['params'],
            model_rankings=model_results,
            selection_strategy=selection_strategy,
            data_characteristics=data_chars,
            preprocessing_pipeline=preprocessor,
            feature_importance=feature_importance,
            cross_validation_scores={best_result['name']: best_result['cv_scores']},
            training_time=time.time() - start_time,
            selection_confidence=selection_confidence,
            ensemble_models=ensemble_models,
            meta_features=meta_features
        )
        
        # Store in history
        self.selection_history.append(selection_result)
        
        logger.info(f"Model selection completed in {selection_result.training_time:.2f}s")
        return selection_result
    
    def _create_preprocessing_pipeline(self, 
                                     X: pd.DataFrame, 
                                     data_chars: DataCharacteristics,
                                     preprocessing_options: Optional[List[str]] = None) -> Pipeline:
        """Create appropriate preprocessing pipeline."""
        
        # Determine preprocessing steps based on data characteristics
        preprocessing_steps = []
        
        # Handle missing values if present
        if data_chars.missing_values_pct > 0:
            from sklearn.impute import SimpleImputer
            preprocessing_steps.append(
                ("imputer", SimpleImputer(strategy="median"))
            )
        
        # Feature scaling based on data characteristics
        if data_chars.noise_level > 0.3:
            scaler = RobustScaler()
        elif data_chars.n_features > 100:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        preprocessing_steps.append(("scaler", scaler))
        
        # Feature selection for high-dimensional data
        if data_chars.dimensionality_ratio > 0.1 and data_chars.n_features > 50:
            k_features = min(int(data_chars.n_samples * 0.1), 100)
            preprocessing_steps.append(
                ("feature_selection", SelectKBest(f_classif, k=k_features))
            )
        
        return Pipeline(preprocessing_steps)
    
    def _evaluate_model(self, 
                       model_name: str, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       preprocessor: Pipeline,
                       n_iter: int,
                       cv_folds: int) -> Dict[str, Any]:
        """Evaluate a single model with hyperparameter optimization."""
        
        config = self.model_registry.get_configuration(model_name)
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", config.estimator_class(**config.default_params))
        ])
        
        # Prepare parameter distributions for search
        param_distributions = {}
        for param, distribution in config.param_distributions.items():
            param_distributions[f"classifier__{param}"] = distribution
        
        # Perform randomized search
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            return_train_score=False
        )
        
        search.fit(X, y)
        
        return {
            "name": model_name,
            "model": search.best_estimator_,
            "score": search.best_score_,
            "score_std": search.cv_results_['std_test_score'][search.best_index_],
            "params": search.best_params_,
            "cv_scores": search.cv_results_['split0_test_score'],  # Simplified
            "configuration": config
        }
    
    def _create_ensemble_models(self, 
                              top_models: List[Dict[str, Any]],
                              X: pd.DataFrame,
                              y: pd.Series,
                              preprocessor: Pipeline) -> List[BaseEstimator]:
        """Create ensemble models from top performers."""
        
        ensemble_models = []
        
        try:
            # Voting classifier (soft voting)
            estimators = [(result['name'], result['model'].named_steps['classifier']) 
                         for result in top_models]
            
            voting_pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("voting", VotingClassifier(estimators, voting='soft'))
            ])
            
            voting_pipeline.fit(X, y)
            ensemble_models.append(voting_pipeline)
            
            logger.info("Created voting ensemble")
            
        except Exception as e:
            logger.warning(f"Failed to create voting ensemble: {e}")
        
        return ensemble_models
    
    def _extract_feature_importance(self, 
                                  model: Pipeline, 
                                  feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from the model."""
        
        try:
            classifier = model.named_steps['classifier']
            
            # Get feature importance
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
            else:
                return None
            
            # Handle preprocessing transformations
            preprocessor = model.named_steps['preprocessor']
            
            # Get feature names after preprocessing
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    processed_feature_names = preprocessor.get_feature_names_out(feature_names)
                except:
                    processed_feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                processed_feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create feature importance dictionary
            feature_importance = {}
            for name, importance in zip(processed_feature_names, importances):
                feature_importance[name] = float(importance)
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
            return None
    
    def save_model_selection_result(self, 
                                   result: ModelSelectionResult, 
                                   filepath: str) -> None:
        """Save model selection result to file."""
        
        # Prepare serializable data
        result_data = {
            "best_model_name": result.best_model_name,
            "best_score": result.best_score,
            "best_params": result.best_params,
            "selection_strategy": result.selection_strategy,
            "data_characteristics": asdict(result.data_characteristics),
            "feature_importance": result.feature_importance,
            "training_time": result.training_time,
            "selection_confidence": result.selection_confidence,
            "meta_features": result.meta_features,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save model and preprocessor separately
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{result.best_model_name}_model.joblib"
        joblib.dump(result.best_model, model_path)
        
        preprocessor_path = model_dir / f"{result.best_model_name}_preprocessor.joblib"
        joblib.dump(result.preprocessing_pipeline, preprocessor_path)
        
        result_data["model_path"] = str(model_path)
        result_data["preprocessor_path"] = str(preprocessor_path)
        
        # Save ensemble models if available
        if result.ensemble_models:
            ensemble_paths = []
            for i, ensemble_model in enumerate(result.ensemble_models):
                ensemble_path = model_dir / f"{result.best_model_name}_ensemble_{i}.joblib"
                joblib.dump(ensemble_model, ensemble_path)
                ensemble_paths.append(str(ensemble_path))
            result_data["ensemble_paths"] = ensemble_paths
        
        # Save result data
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info(f"Model selection result saved to {filepath}")
    
    def load_model_selection_result(self, filepath: str) -> Dict[str, Any]:
        """Load model selection result from file."""
        
        with open(filepath, 'r') as f:
            result_data = json.load(f)
        
        # Load models
        if "model_path" in result_data and os.path.exists(result_data["model_path"]):
            result_data["best_model"] = joblib.load(result_data["model_path"])
        
        if "preprocessor_path" in result_data and os.path.exists(result_data["preprocessor_path"]):
            result_data["preprocessing_pipeline"] = joblib.load(result_data["preprocessor_path"])
        
        # Load ensemble models if available
        if "ensemble_paths" in result_data:
            ensemble_models = []
            for ensemble_path in result_data["ensemble_paths"]:
                if os.path.exists(ensemble_path):
                    ensemble_models.append(joblib.load(ensemble_path))
            result_data["ensemble_models"] = ensemble_models
        
        logger.info(f"Model selection result loaded from {filepath}")
        return result_data


def auto_select_model(X: pd.DataFrame, 
                     y: pd.Series,
                     strategy: str = "balanced",
                     enable_ensemble: bool = True,
                     output_dir: Optional[str] = None) -> ModelSelectionResult:
    """
    Automatically select the best model for a dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        strategy: Selection strategy ("fast", "balanced", "comprehensive")
        enable_ensemble: Whether to create ensemble models
        output_dir: Directory to save results
        
    Returns:
        ModelSelectionResult with best model and analysis
    """
    
    # Initialize selector
    selector = IntelligentModelSelector(
        cv_folds=5 if strategy != "fast" else 3,
        n_iter=50 if strategy == "balanced" else (20 if strategy == "fast" else 100),
        enable_ensemble=enable_ensemble
    )
    
    # Perform model selection
    result = selector.select_best_model(X, y, selection_strategy=strategy)
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(output_dir, f"model_selection_result_{timestamp}.json")
        selector.save_model_selection_result(result, result_path)
    
    return result


if __name__ == "__main__":
    print("Intelligent Model Selection Framework")
    print("This framework provides automated model selection with meta-learning.")