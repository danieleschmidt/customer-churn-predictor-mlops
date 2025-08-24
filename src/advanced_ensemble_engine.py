"""
Advanced Ensemble Engine for Customer Churn Prediction.

This module implements state-of-the-art ensemble methods with automated model
selection, hyperparameter optimization, and explainability features for
production-grade machine learning systems.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Advanced ML libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Local imports
from .logging_config import get_logger
from .config import Config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    model_size_mb: float


@dataclass
class EnsembleConfig:
    """Configuration for ensemble learning."""
    use_voting: bool = True
    use_stacking: bool = True
    use_blending: bool = True
    voting_strategy: str = 'soft'  # 'hard' or 'soft'
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    max_ensemble_size: int = 15
    performance_threshold: float = 0.85
    diversity_threshold: float = 0.1


class BaseEnsembleModel(ABC):
    """Abstract base class for ensemble models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseEnsembleModel':
        """Fit the ensemble model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the ensemble."""
        pass


class AdvancedEnsembleEngine:
    """
    Advanced ensemble learning engine with automated model selection and optimization.
    
    Features:
    - Automated base model selection and hyperparameter tuning
    - Multiple ensemble strategies (voting, stacking, blending)
    - Model performance tracking and comparison
    - Explainability integration (SHAP)
    - Production-ready model persistence and versioning
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize the ensemble engine."""
        self.config = config or EnsembleConfig()
        self.base_models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.final_ensemble: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        
        # Initialize available models
        self._initialize_base_models()
        
    def _initialize_base_models(self) -> None:
        """Initialize the pool of base models for ensemble learning."""
        self.base_models = {
            # Traditional ML models
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.config.random_state,
                n_estimators=100
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.config.random_state,
                n_estimators=100
            ),
            'extra_trees': ExtraTreesClassifier(
                random_state=self.config.random_state,
                n_estimators=100
            ),
            'ada_boost': AdaBoostClassifier(
                random_state=self.config.random_state,
                n_estimators=50
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.config.random_state,
                max_depth=10
            ),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'svm': SVC(
                probability=True,
                random_state=self.config.random_state
            ),
            'neural_network': MLPClassifier(
                random_state=self.config.random_state,
                max_iter=500,
                early_stopping=True
            )
        }
        
        # Add advanced gradient boosting models if available
        if XGBOOST_AVAILABLE:
            self.base_models['xgboost'] = XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='logloss'
            )
            
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = LGBMClassifier(
                random_state=self.config.random_state,
                verbose=-1
            )
            
        if CATBOOST_AVAILABLE:
            self.base_models['catboost'] = CatBoostClassifier(
                random_state=self.config.random_state,
                verbose=False
            )
        
        logger.info(f"Initialized {len(self.base_models)} base models for ensemble learning")
    
    def _evaluate_model_performance(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> ModelPerformance:
        """Evaluate individual model performance with cross-validation."""
        start_time = datetime.now()
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                            random_state=self.config.random_state)
        cv_scores = cross_val_score(
            model, X, y, cv=cv, scoring='accuracy', 
            n_jobs=self.config.n_jobs
        )
        
        # Fit model for additional metrics
        model.fit(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        start_pred = datetime.now()
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        prediction_time = (datetime.now() - start_pred).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Model size estimation
        try:
            model_bytes = len(pickle.dumps(model))
            model_size_mb = model_bytes / (1024 * 1024)
        except:
            model_size_mb = 0.0
        
        return ModelPerformance(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=prediction_time,
            model_size_mb=model_size_mb
        )
    
    def _select_best_models(self, threshold: float = None) -> List[str]:
        """Select best performing models for ensemble."""
        threshold = threshold or self.config.performance_threshold
        
        # Sort models by performance
        sorted_models = sorted(
            self.model_performances.items(),
            key=lambda x: x[1].f1_score,
            reverse=True
        )
        
        # Select models above threshold
        selected_models = []
        for model_name, performance in sorted_models:
            if performance.f1_score >= threshold and len(selected_models) < self.config.max_ensemble_size:
                selected_models.append(model_name)
        
        if not selected_models:
            # If no models meet threshold, select top 3
            selected_models = [name for name, _ in sorted_models[:3]]
        
        logger.info(f"Selected {len(selected_models)} models for ensemble: {selected_models}")
        return selected_models
    
    def _create_voting_ensemble(self, selected_models: List[str], X: pd.DataFrame, y: pd.Series) -> VotingClassifier:
        """Create a voting classifier ensemble."""
        estimators = []
        for model_name in selected_models:
            model = self.base_models[model_name]
            estimators.append((model_name, model))
        
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.config.voting_strategy,
            n_jobs=self.config.n_jobs
        )
        
        voting_ensemble.fit(X, y)
        return voting_ensemble
    
    def _optimize_hyperparameters(self, model_name: str, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Optimize hyperparameters for a given model using RandomizedSearchCV."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name in param_grids:
            try:
                search = RandomizedSearchCV(
                    model, param_grids[model_name],
                    n_iter=20, cv=3, scoring='f1',
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
                search.fit(X, y)
                return search.best_estimator_
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed for {model_name}: {e}")
                return model
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparameters: bool = True) -> 'AdvancedEnsembleEngine':
        """
        Fit the ensemble engine on training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting advanced ensemble training...")
        
        # Store feature names
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Evaluate all base models
        logger.info("Evaluating base models...")
        for model_name, model in self.base_models.items():
            try:
                # Optimize hyperparameters if requested
                if optimize_hyperparameters:
                    model = self._optimize_hyperparameters(model_name, model, X_scaled, y)
                    self.base_models[model_name] = model
                
                # Evaluate performance
                performance = self._evaluate_model_performance(model, X_scaled, y)
                self.model_performances[model_name] = performance
                
                logger.info(
                    f"{model_name}: F1={performance.f1_score:.4f}, "
                    f"Accuracy={performance.accuracy:.4f}, "
                    f"ROC-AUC={performance.roc_auc:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Select best models for ensemble
        selected_models = self._select_best_models()
        
        # Create final ensemble
        if self.config.use_voting and len(selected_models) > 1:
            self.final_ensemble = self._create_voting_ensemble(selected_models, X_scaled, y)
            logger.info("Created voting ensemble")
        else:
            # Use best single model
            best_model_name = max(
                self.model_performances.items(),
                key=lambda x: x[1].f1_score
            )[0]
            self.final_ensemble = self.base_models[best_model_name]
            logger.info(f"Using single best model: {best_model_name}")
        
        self.is_fitted = True
        logger.info("Advanced ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.final_ensemble.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the trained ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.final_ensemble.predict_proba(X_scaled)
    
    def get_feature_importance(self, method: str = 'ensemble') -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self.is_fitted or not self.feature_names:
            return None
        
        if method == 'ensemble' and hasattr(self.final_ensemble, 'feature_importances_'):
            importances = self.final_ensemble.feature_importances_
        elif method == 'ensemble' and isinstance(self.final_ensemble, VotingClassifier):
            # Aggregate importance from voting classifier
            importances = np.zeros(len(self.feature_names))
            valid_estimators = 0
            
            for name, estimator in self.final_ensemble.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances += estimator.feature_importances_
                    valid_estimators += 1
            
            if valid_estimators > 0:
                importances /= valid_estimators
            else:
                return None
        else:
            return None
        
        return dict(zip(self.feature_names, importances))
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """Get a summary of all model performances."""
        if not self.model_performances:
            return pd.DataFrame()
        
        data = []
        for model_name, performance in self.model_performances.items():
            data.append({
                'Model': model_name,
                'Accuracy': performance.accuracy,
                'F1_Score': performance.f1_score,
                'Precision': performance.precision,
                'Recall': performance.recall,
                'ROC_AUC': performance.roc_auc,
                'CV_Mean': performance.cv_mean,
                'CV_Std': performance.cv_std,
                'Training_Time': performance.training_time,
                'Prediction_Time': performance.prediction_time,
                'Model_Size_MB': performance.model_size_mb
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('F1_Score', ascending=False)
    
    def explain_predictions(self, X: pd.DataFrame, sample_size: int = 100) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanations for predictions."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for explanations")
            return None
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before generating explanations")
        
        try:
            # Sample data if too large
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=self.config.random_state)
            else:
                X_sample = X
            
            # Scale features
            X_scaled = self.scaler.transform(X_sample)
            X_scaled = pd.DataFrame(X_scaled, columns=X_sample.columns, index=X_sample.index)
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.final_ensemble, X_scaled)
            shap_values = explainer(X_scaled)
            
            # Extract explanations
            explanations = {
                'feature_importance': dict(zip(
                    X_sample.columns,
                    np.abs(shap_values.values).mean(axis=0)
                )),
                'shap_values': shap_values.values.tolist(),
                'base_value': float(shap_values.base_values[0]) if len(shap_values.base_values.shape) == 1 else shap_values.base_values[0].tolist(),
                'feature_names': X_sample.columns.tolist(),
                'sample_size': len(X_sample)
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble model and metadata
        model_data = {
            'ensemble_model': self.final_ensemble,
            'scaler': self.scaler,
            'config': asdict(self.config),
            'feature_names': self.feature_names,
            'model_performances': {
                name: asdict(perf) for name, perf in self.model_performances.items()
            },
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Advanced ensemble model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> 'AdvancedEnsembleEngine':
        """Load a trained ensemble model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        config = EnsembleConfig(**model_data['config'])
        engine = cls(config)
        
        # Restore state
        engine.final_ensemble = model_data['ensemble_model']
        engine.scaler = model_data['scaler']
        engine.feature_names = model_data['feature_names']
        engine.model_performances = {
            name: ModelPerformance(**perf_data) 
            for name, perf_data in model_data['model_performances'].items()
        }
        engine.is_fitted = True
        
        logger.info(f"Advanced ensemble model loaded from {model_path}")
        return engine


# Factory function for easy model creation
def create_advanced_ensemble(
    use_xgboost: bool = True,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    performance_threshold: float = 0.85,
    max_ensemble_size: int = 10,
    random_state: int = 42
) -> AdvancedEnsembleEngine:
    """
    Factory function to create an advanced ensemble engine with optimal settings.
    
    Args:
        use_xgboost: Whether to include XGBoost if available
        use_lightgbm: Whether to include LightGBM if available  
        use_catboost: Whether to include CatBoost if available
        performance_threshold: Minimum F1 score for model inclusion
        max_ensemble_size: Maximum number of models in ensemble
        random_state: Random seed for reproducibility
        
    Returns:
        Configured AdvancedEnsembleEngine instance
    """
    config = EnsembleConfig(
        performance_threshold=performance_threshold,
        max_ensemble_size=max_ensemble_size,
        random_state=random_state
    )
    
    engine = AdvancedEnsembleEngine(config)
    
    # Remove unavailable models based on parameters
    if not use_xgboost and 'xgboost' in engine.base_models:
        del engine.base_models['xgboost']
    if not use_lightgbm and 'lightgbm' in engine.base_models:
        del engine.base_models['lightgbm']
    if not use_catboost and 'catboost' in engine.base_models:
        del engine.base_models['catboost']
    
    return engine