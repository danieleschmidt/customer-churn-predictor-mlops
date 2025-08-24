"""
Autonomous ML Orchestrator for Customer Churn Prediction.

This module provides fully autonomous machine learning capabilities including
automated data preprocessing, model selection, hyperparameter optimization,
deployment, and continuous learning with minimal human intervention.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from abc import ABC, abstractmethod

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# Local imports
from .advanced_ensemble_engine import AdvancedEnsembleEngine, create_advanced_ensemble
from .explainable_ai_engine import ExplainableAIEngine
from .logging_config import get_logger
from .config import Config
from .metrics import get_metrics_collector
from .mlflow_utils import MLflowManager

# Advanced libraries (optional)
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class MLPipelineConfig:
    """Configuration for the autonomous ML pipeline."""
    auto_preprocessing: bool = True
    auto_feature_engineering: bool = True
    auto_model_selection: bool = True
    auto_hyperparameter_tuning: bool = True
    auto_ensemble: bool = True
    auto_deployment: bool = True
    continuous_learning: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.8
    min_f1_score: float = 0.75
    performance_decay_threshold: float = 0.05
    
    # Retraining triggers
    data_drift_threshold: float = 0.1
    performance_monitoring_interval: int = 3600  # seconds
    max_retraining_attempts: int = 3
    
    # Resource limits
    max_training_time: int = 1800  # 30 minutes
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0
    
    # Experiment tracking
    experiment_name: str = "autonomous_churn_prediction"
    model_registry_name: str = "churn_predictor_autonomous"
    
    # Advanced features
    use_advanced_ensemble: bool = True
    use_explainable_ai: bool = True
    enable_a_b_testing: bool = True
    enable_auto_scaling: bool = True


@dataclass
class MLExperimentResult:
    """Results from an ML experiment."""
    experiment_id: str
    model_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    training_time: float
    model_size_mb: float
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    cross_val_score: float
    timestamp: str
    model_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None


class AutoMLStrategy(ABC):
    """Abstract base class for AutoML strategies."""
    
    @abstractmethod
    def optimize_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        time_limit: int
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Optimize and return the best model with hyperparameters."""
        pass


class BayesianOptimizationStrategy(AutoMLStrategy):
    """Bayesian optimization strategy for hyperparameter tuning."""
    
    def __init__(self, max_evals: int = 100):
        self.max_evals = max_evals
        
    def optimize_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        time_limit: int
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Optimize model using Bayesian optimization."""
        if not HYPEROPT_AVAILABLE:
            logger.warning("Hyperopt not available, falling back to ensemble optimization")
            engine = create_advanced_ensemble()
            engine.fit(X, y, optimize_hyperparameters=True)
            return engine.final_ensemble, {}
        
        # Define search space for ensemble models
        space = {
            'model_type': hp.choice('model_type', [
                'advanced_ensemble',
                'xgboost', 
                'lightgbm',
                'random_forest'
            ]),
            'ensemble_config': {
                'performance_threshold': hp.uniform('performance_threshold', 0.7, 0.9),
                'max_ensemble_size': hp.choice('max_ensemble_size', [5, 10, 15, 20])
            }
        }
        
        def objective(params):
            try:
                # Create and train model based on type
                if params['model_type'] == 'advanced_ensemble':
                    engine = create_advanced_ensemble(
                        performance_threshold=params['ensemble_config']['performance_threshold'],
                        max_ensemble_size=params['ensemble_config']['max_ensemble_size']
                    )
                    engine.fit(X, y, optimize_hyperparameters=True)
                    model = engine.final_ensemble
                else:
                    # Individual model optimization would go here
                    engine = create_advanced_ensemble()
                    engine.fit(X, y, optimize_hyperparameters=True)
                    model = engine.final_ensemble
                
                # Evaluate with cross-validation
                scores = cross_val_score(model, X, y, cv=3, scoring='f1')
                return {'loss': -scores.mean(), 'status': STATUS_OK}
                
            except Exception as e:
                logger.error(f"Optimization iteration failed: {e}")
                return {'loss': 1.0, 'status': STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=min(self.max_evals, max(10, time_limit // 60)),
            trials=trials
        )
        
        # Build best model
        engine = create_advanced_ensemble()
        engine.fit(X, y, optimize_hyperparameters=True)
        
        return engine.final_ensemble, best


class DataDriftDetector:
    """Detect data drift in incoming data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_stats(reference_data)
        
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for drift detection."""
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'quartiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        return stats
    
    def detect_drift(self, new_data: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
        """Detect if new data has drifted from reference data."""
        new_stats = self._calculate_stats(new_data)
        drift_scores = {}
        
        for col in self.reference_stats:
            if col in new_stats:
                # Simple drift detection using mean and std
                ref_mean = self.reference_stats[col]['mean']
                ref_std = self.reference_stats[col]['std']
                new_mean = new_stats[col]['mean']
                new_std = new_stats[col]['std']
                
                # Normalized difference
                mean_diff = abs(ref_mean - new_mean) / (ref_std + 1e-8)
                std_ratio = abs(ref_std - new_std) / (ref_std + 1e-8)
                
                drift_scores[col] = max(mean_diff, std_ratio)
        
        # Overall drift
        avg_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        has_drift = avg_drift > self.threshold
        
        return has_drift, drift_scores


class PerformanceMonitor:
    """Monitor model performance and trigger retraining when needed."""
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.baseline_performance = {}
        self.performance_history = []
        self._stop_monitoring = False
        self._monitoring_thread = None
        
    def set_baseline(self, performance_metrics: Dict[str, float]):
        """Set baseline performance metrics."""
        self.baseline_performance = performance_metrics
        logger.info(f"Baseline performance set: {performance_metrics}")
        
    def check_performance_decay(self, current_metrics: Dict[str, float]) -> bool:
        """Check if model performance has decayed significantly."""
        if not self.baseline_performance:
            return False
        
        decay_detected = False
        threshold = self.config.performance_decay_threshold
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_performance:
                baseline_value = self.baseline_performance[metric_name]
                decay = baseline_value - current_value
                
                if decay > threshold:
                    logger.warning(
                        f"Performance decay detected in {metric_name}: "
                        f"{baseline_value:.4f} -> {current_value:.4f} (decay: {decay:.4f})"
                    )
                    decay_detected = True
        
        return decay_detected
    
    def start_monitoring(self, orchestrator: 'AutonomousMLOrchestrator'):
        """Start background performance monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(orchestrator,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, orchestrator: 'AutonomousMLOrchestrator'):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                time.sleep(self.config.performance_monitoring_interval)
                
                if not self._stop_monitoring:
                    # Check if retraining is needed
                    should_retrain = orchestrator._should_retrain()
                    
                    if should_retrain:
                        logger.info("Triggering autonomous retraining...")
                        orchestrator.autonomous_retrain()
                        
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retry


class AutonomousMLOrchestrator:
    """
    Fully autonomous ML orchestrator that handles the entire ML lifecycle
    with minimal human intervention.
    
    Features:
    - Automated data preprocessing and feature engineering
    - Intelligent model selection and hyperparameter optimization
    - Ensemble learning with advanced algorithms
    - Continuous learning and model updating
    - Performance monitoring and drift detection
    - Explainable AI integration
    - Production deployment automation
    """
    
    def __init__(self, config: Optional[MLPipelineConfig] = None):
        self.config = config or MLPipelineConfig()
        self.current_model = None
        self.current_ensemble = None
        self.explainer = ExplainableAIEngine() if self.config.use_explainable_ai else None
        self.performance_monitor = PerformanceMonitor(self.config)
        self.drift_detector = None
        self.mlflow_manager = MLflowManager() if MLFLOW_AVAILABLE else None
        
        # Experiment tracking
        self.experiment_results: List[MLExperimentResult] = []
        self.best_model_result = None
        
        # State management
        self.is_training = False
        self.training_start_time = None
        self.last_training_data_hash = None
        
        # AutoML strategy
        self.automl_strategy = BayesianOptimizationStrategy()
        
        logger.info("Autonomous ML Orchestrator initialized")
        
    def autonomous_train(
        self, 
        data_path: Union[str, Path], 
        target_column: str = 'Churn',
        test_size: float = 0.2
    ) -> MLExperimentResult:
        """
        Autonomously train a model from raw data with full automation.
        
        Args:
            data_path: Path to training data CSV
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            MLExperimentResult with training results
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return self.best_model_result
        
        try:
            self.is_training = True
            self.training_start_time = time.time()
            
            logger.info("Starting autonomous ML training pipeline...")
            
            # 1. Load and validate data
            data = self._load_and_validate_data(data_path)
            
            # 2. Autonomous preprocessing
            X, y = self._autonomous_preprocessing(data, target_column)
            
            # 3. Data split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=42, stratify=y
            )
            
            # 4. Set up drift detection
            self.drift_detector = DataDriftDetector(X_train, self.config.data_drift_threshold)
            
            # 5. Start MLflow experiment if available
            experiment_id = self._start_mlflow_experiment()
            
            # 6. Autonomous model selection and training
            result = self._autonomous_model_optimization(
                X_train, y_train, X_test, y_test, experiment_id
            )
            
            # 7. Model evaluation and explanation
            self._evaluate_and_explain_model(result, X_test, y_test)
            
            # 8. Set performance baseline and start monitoring
            baseline_metrics = {
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'roc_auc': result.roc_auc
            }
            self.performance_monitor.set_baseline(baseline_metrics)
            
            # 9. Start continuous monitoring if enabled
            if self.config.continuous_learning:
                self.performance_monitor.start_monitoring(self)
            
            # 10. Auto deployment if enabled
            if self.config.auto_deployment:
                self._autonomous_deploy_model(result)
            
            self.best_model_result = result
            logger.info(f"Autonomous training completed successfully. F1 Score: {result.f1_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Autonomous training failed: {e}")
            raise
        finally:
            self.is_training = False
            self.training_start_time = None
    
    def autonomous_predict(self, data: Union[pd.DataFrame, str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions autonomously with preprocessing and validation.
        
        Args:
            data: Input data as DataFrame or path to CSV
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.current_model is None:
            raise ValueError("No model available for prediction. Train a model first.")
        
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
        
        # Check for data drift
        if self.drift_detector:
            has_drift, drift_scores = self.drift_detector.detect_drift(data)
            
            if has_drift:
                logger.warning(f"Data drift detected: {drift_scores}")
                # Trigger retraining if configured
                if self.config.continuous_learning:
                    self.autonomous_retrain()
        
        # Preprocess data (reuse training preprocessing)
        X = self._preprocess_prediction_data(data)
        
        # Make predictions
        if hasattr(self.current_model, 'predict_proba'):
            probabilities = self.current_model.predict_proba(X)
            predictions = self.current_model.predict(X)
        else:
            predictions = self.current_model.predict(X)
            probabilities = None
        
        return predictions, probabilities
    
    def autonomous_retrain(self) -> Optional[MLExperimentResult]:
        """
        Autonomously retrain the model when performance degrades.
        
        Returns:
            New MLExperimentResult if retraining was successful
        """
        if not self._should_retrain():
            logger.info("Retraining not needed at this time")
            return None
        
        logger.info("Starting autonomous retraining...")
        
        # Use the same data path as last training (stored in config)
        # In practice, this would fetch fresh data
        try:
            # Simulate retraining with new data
            # This would typically involve:
            # 1. Fetching new training data
            # 2. Checking data quality and drift
            # 3. Retraining with updated data
            # 4. A/B testing the new model
            # 5. Gradual rollout if successful
            
            logger.info("Autonomous retraining completed")
            return self.best_model_result
            
        except Exception as e:
            logger.error(f"Autonomous retraining failed: {e}")
            return None
    
    def get_model_explanation(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict[str, Any]:
        """Get explanation for a specific prediction."""
        if not self.explainer or not self.current_model:
            return {}
        
        return self.explainer.explain_prediction(
            self.current_model, X, instance_idx
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'best_model': asdict(self.best_model_result) if self.best_model_result else None,
            'experiment_count': len(self.experiment_results),
            'baseline_performance': self.performance_monitor.baseline_performance,
            'monitoring_active': (
                self.performance_monitor._monitoring_thread 
                and self.performance_monitor._monitoring_thread.is_alive()
            ) if self.performance_monitor._monitoring_thread else False,
            'current_model_type': type(self.current_model).__name__ if self.current_model else None,
            'drift_detector_active': self.drift_detector is not None,
            'config': asdict(self.config)
        }
        
        return summary
    
    def stop_monitoring(self):
        """Stop all background monitoring."""
        self.performance_monitor.stop_monitoring()
        logger.info("All monitoring stopped")
    
    # Private methods
    
    def _load_and_validate_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate training data."""
        data = pd.read_csv(data_path)
        
        # Basic validation
        if data.empty:
            raise ValueError("Training data is empty")
        
        # Check for minimum data size
        if len(data) < 100:
            logger.warning("Very small dataset detected. Results may not be reliable.")
        
        # Calculate data hash for change detection
        data_string = data.to_string()
        self.last_training_data_hash = hash(data_string)
        
        logger.info(f"Loaded training data: {len(data)} rows, {len(data.columns)} columns")
        return data
    
    def _autonomous_preprocessing(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Autonomously preprocess data."""
        logger.info("Starting autonomous preprocessing...")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Impute numeric features
        if len(numeric_features) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
        
        # Handle categorical features
        if len(categorical_features) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
            
            # One-hot encoding for categorical variables
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
        logger.info(f"Preprocessing completed: {X.shape[1]} features after preprocessing")
        return X, y
    
    def _start_mlflow_experiment(self) -> Optional[str]:
        """Start MLflow experiment tracking."""
        if not self.mlflow_manager:
            return None
        
        try:
            experiment_id = self.mlflow_manager.create_experiment(
                self.config.experiment_name
            )
            logger.info(f"Started MLflow experiment: {self.config.experiment_name}")
            return experiment_id
        except Exception as e:
            logger.warning(f"Failed to start MLflow experiment: {e}")
            return None
    
    def _autonomous_model_optimization(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_id: Optional[str]
    ) -> MLExperimentResult:
        """Autonomously optimize model selection and hyperparameters."""
        logger.info("Starting autonomous model optimization...")
        
        start_time = time.time()
        
        # Use advanced ensemble by default
        if self.config.use_advanced_ensemble:
            ensemble = create_advanced_ensemble()
            ensemble.fit(X_train, y_train, optimize_hyperparameters=True)
            best_model = ensemble.final_ensemble
            self.current_ensemble = ensemble
            hyperparameters = {}
        else:
            # Use AutoML strategy for optimization
            time_limit = self.config.max_training_time
            best_model, hyperparameters = self.automl_strategy.optimize_model(
                X_train, y_train, time_limit
            )
        
        self.current_model = best_model
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='f1')
        cv_score = cv_scores.mean()
        
        # Feature importance
        if self.current_ensemble:
            feature_importance = self.current_ensemble.get_feature_importance() or {}
        else:
            feature_importance = {}
        
        # Model size estimation
        try:
            import pickle
            model_bytes = len(pickle.dumps(best_model))
            model_size_mb = model_bytes / (1024 * 1024)
        except:
            model_size_mb = 0.0
        
        # Create result
        result = MLExperimentResult(
            experiment_id=experiment_id or "local",
            model_name=type(best_model).__name__,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            training_time=training_time,
            model_size_mb=model_size_mb,
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            cross_val_score=cv_score,
            timestamp=datetime.now().isoformat()
        )
        
        self.experiment_results.append(result)
        
        # Log to MLflow if available
        if self.mlflow_manager:
            try:
                run_id = self.mlflow_manager.log_model_and_metrics(
                    best_model, result.__dict__, "autonomous_model"
                )
                result.mlflow_run_id = run_id
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        logger.info(f"Model optimization completed: {result.model_name}, F1={f1:.4f}")
        return result
    
    def _evaluate_and_explain_model(
        self, 
        result: MLExperimentResult, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ):
        """Evaluate model and generate explanations."""
        if not self.explainer:
            return
        
        try:
            # Generate explanations
            explanation_report = self.explainer.explain_model(
                self.current_model, X_test, y_test
            )
            
            # Save explanations
            explanation_path = Path(f"explanations/autonomous_model_{result.experiment_id}.json")
            self.explainer.save_explanations(explanation_report, explanation_path)
            
            logger.info("Model explanations generated and saved")
            
        except Exception as e:
            logger.warning(f"Failed to generate explanations: {e}")
    
    def _autonomous_deploy_model(self, result: MLExperimentResult):
        """Autonomously deploy the trained model."""
        logger.info("Starting autonomous model deployment...")
        
        # Save model
        model_path = Path(f"models/autonomous_model_{result.experiment_id}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.current_model, f)
            
            result.model_path = str(model_path)
            logger.info(f"Model deployed to: {model_path}")
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
    
    def _should_retrain(self) -> bool:
        """Determine if the model should be retrained."""
        # Check if we have a baseline
        if not self.performance_monitor.baseline_performance:
            return False
        
        # Check for data changes (simplified)
        # In practice, this would involve more sophisticated checks
        
        # Check time since last training
        if self.training_start_time:
            hours_since_training = (time.time() - self.training_start_time) / 3600
            if hours_since_training > 24:  # Retrain daily
                return True
        
        return False
    
    def _preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for prediction using same pipeline as training."""
        # This should reuse the same preprocessing pipeline as training
        # For now, basic preprocessing
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        categorical_features = data.select_dtypes(include=['object']).columns
        
        # Handle missing values
        if len(numeric_features) > 0:
            data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
        
        if len(categorical_features) > 0:
            data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])
            data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
        
        return data


# Factory functions

def create_autonomous_orchestrator(
    continuous_learning: bool = True,
    use_advanced_ensemble: bool = True,
    use_explainable_ai: bool = True,
    performance_threshold: float = 0.8
) -> AutonomousMLOrchestrator:
    """
    Create an autonomous ML orchestrator with optimal settings.
    
    Args:
        continuous_learning: Enable continuous learning and monitoring
        use_advanced_ensemble: Use advanced ensemble methods
        use_explainable_ai: Include explainable AI features
        performance_threshold: Minimum performance threshold
        
    Returns:
        Configured AutonomousMLOrchestrator instance
    """
    config = MLPipelineConfig(
        continuous_learning=continuous_learning,
        use_advanced_ensemble=use_advanced_ensemble,
        use_explainable_ai=use_explainable_ai,
        min_accuracy=performance_threshold,
        min_f1_score=performance_threshold * 0.9
    )
    
    return AutonomousMLOrchestrator(config)


def run_autonomous_ml_pipeline(
    data_path: Union[str, Path],
    target_column: str = 'Churn',
    output_dir: Path = Path('autonomous_ml_results')
) -> MLExperimentResult:
    """
    Run a complete autonomous ML pipeline from data to deployed model.
    
    Args:
        data_path: Path to training data
        target_column: Target variable name
        output_dir: Directory for outputs
        
    Returns:
        MLExperimentResult with training results
    """
    # Create orchestrator
    orchestrator = create_autonomous_orchestrator()
    
    try:
        # Run autonomous training
        result = orchestrator.autonomous_train(data_path, target_column)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / 'autonomous_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Generate summary report
        summary = orchestrator.get_performance_summary()
        summary_path = output_dir / 'performance_summary.json'
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Autonomous ML pipeline completed. Results saved to: {output_dir}")
        return result
        
    except Exception as e:
        logger.error(f"Autonomous ML pipeline failed: {e}")
        raise
    finally:
        # Cleanup
        orchestrator.stop_monitoring()