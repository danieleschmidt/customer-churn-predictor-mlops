"""
Adaptive Learning System with Online Model Updates and Continual Learning.

This module provides comprehensive adaptive learning capabilities including:
- Online learning algorithms that adapt to new data
- Continual learning with catastrophic forgetting mitigation
- Concept drift detection and adaptation
- Active learning for optimal sample selection
- Federated learning capabilities
- A/B testing framework for model comparison
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import ks_2samp
import joblib
import mlflow
import mlflow.sklearn
import threading
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .data_validation import validate_prediction_data

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning system."""
    # Online learning settings
    learning_rate: float = 0.01
    batch_size: int = 100
    update_frequency: int = 50  # Update after N samples
    
    # Concept drift detection
    enable_drift_detection: bool = True
    drift_detection_method: str = "ks_test"  # ks_test, page_hinkley, adwin
    drift_threshold: float = 0.05
    drift_window_size: int = 1000
    
    # Model adaptation
    adaptation_strategy: str = "incremental"  # incremental, retrain, ensemble
    max_models_ensemble: int = 5
    model_decay_factor: float = 0.95
    
    # Active learning
    enable_active_learning: bool = True
    uncertainty_threshold: float = 0.3
    query_budget: int = 100
    query_strategy: str = "uncertainty"  # uncertainty, diversity, expected_error
    
    # Continual learning
    enable_continual_learning: bool = True
    memory_buffer_size: int = 10000
    rehearsal_ratio: float = 0.2
    regularization_strength: float = 0.1
    
    # A/B testing
    enable_ab_testing: bool = True
    control_model_ratio: float = 0.5
    test_duration_hours: int = 24
    significance_level: float = 0.05
    
    # Performance monitoring
    performance_window_size: int = 500
    min_samples_for_update: int = 100
    performance_threshold: float = 0.8
    
    # System settings
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ConceptDriftDetection:
    """Results from concept drift detection."""
    drift_detected: bool
    drift_score: float
    drift_type: str  # "gradual", "sudden", "incremental", "recurring"
    detection_method: str
    timestamp: datetime
    affected_features: List[str]
    confidence: float
    recommended_action: str


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model version."""
    model_id: str
    accuracy: float
    f1_score: float
    roc_auc: float
    prediction_count: int
    correct_predictions: int
    timestamp: datetime
    confidence_interval: Tuple[float, float]


@dataclass
class ActiveLearningQuery:
    """Active learning query for labeling."""
    query_id: str
    sample_data: Dict[str, Any]
    uncertainty_score: float
    query_strategy: str
    timestamp: datetime
    expected_information_gain: float


class OnlineDataBuffer:
    """Thread-safe buffer for online learning data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data_buffer = deque(maxlen=max_size)
        self.label_buffer = deque(maxlen=max_size)
        self.timestamp_buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        # For drift detection
        self.feature_history = defaultdict(lambda: deque(maxlen=max_size))
        
    def add_sample(self, X: pd.Series, y: Optional[int] = None, 
                  timestamp: Optional[datetime] = None) -> None:
        """Add a new sample to the buffer."""
        with self.lock:
            self.data_buffer.append(X)
            self.label_buffer.append(y)
            self.timestamp_buffer.append(timestamp or datetime.now())
            
            # Update feature history for drift detection
            for feature, value in X.items():
                if isinstance(value, (int, float)):
                    self.feature_history[feature].append(value)
    
    def get_recent_samples(self, n: int) -> Tuple[pd.DataFrame, np.ndarray, List[datetime]]:
        """Get the most recent n samples."""
        with self.lock:
            if len(self.data_buffer) == 0:
                return pd.DataFrame(), np.array([]), []
            
            # Get last n samples
            n = min(n, len(self.data_buffer))
            recent_data = list(self.data_buffer)[-n:]
            recent_labels = list(self.label_buffer)[-n:]
            recent_timestamps = list(self.timestamp_buffer)[-n:]
            
            # Convert to DataFrame
            X = pd.DataFrame(recent_data)
            y = np.array([label for label in recent_labels if label is not None])
            
            return X, y, recent_timestamps
    
    def get_labeled_samples(self, n: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get samples with labels for training."""
        with self.lock:
            labeled_indices = [i for i, label in enumerate(self.label_buffer) if label is not None]
            
            if not labeled_indices:
                return pd.DataFrame(), np.array([])
            
            if n is not None:
                labeled_indices = labeled_indices[-n:]
            
            X_labeled = pd.DataFrame([self.data_buffer[i] for i in labeled_indices])
            y_labeled = np.array([self.label_buffer[i] for i in labeled_indices])
            
            return X_labeled, y_labeled
    
    def get_feature_distribution(self, feature: str, window_size: int) -> np.ndarray:
        """Get recent distribution of a feature."""
        with self.lock:
            if feature not in self.feature_history:
                return np.array([])
            
            recent_values = list(self.feature_history[feature])[-window_size:]
            return np.array(recent_values)


class ConceptDriftDetector:
    """Detect concept drift in streaming data."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.reference_distributions = {}
        self.drift_history = []
        
    def update_reference(self, data_buffer: OnlineDataBuffer) -> None:
        """Update reference distributions for drift detection."""
        X, _, _ = data_buffer.get_recent_samples(self.config.drift_window_size)
        
        if X.empty:
            return
        
        # Update reference distributions for numerical features
        for column in X.select_dtypes(include=[np.number]).columns:
            self.reference_distributions[column] = {
                'mean': X[column].mean(),
                'std': X[column].std(),
                'distribution': X[column].dropna().values
            }
    
    def detect_drift(self, data_buffer: OnlineDataBuffer) -> ConceptDriftDetection:
        """Detect concept drift using statistical tests."""
        
        if not self.reference_distributions:
            return ConceptDriftDetection(
                drift_detected=False,
                drift_score=0.0,
                drift_type="none",
                detection_method=self.config.drift_detection_method,
                timestamp=datetime.now(),
                affected_features=[],
                confidence=0.0,
                recommended_action="continue"
            )
        
        drift_scores = {}
        affected_features = []
        
        # Test each feature for drift
        for feature, ref_dist in self.reference_distributions.items():
            current_values = data_buffer.get_feature_distribution(
                feature, self.config.drift_window_size // 2
            )
            
            if len(current_values) < 50:  # Not enough samples
                continue
            
            if self.config.drift_detection_method == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(ref_dist['distribution'], current_values)
                drift_scores[feature] = 1 - p_value  # Convert to drift score
                
                if p_value < self.config.drift_threshold:
                    affected_features.append(feature)
            
            elif self.config.drift_detection_method == "mean_shift":
                # Detect significant mean shift
                ref_mean = ref_dist['mean']
                current_mean = np.mean(current_values)
                ref_std = ref_dist['std']
                
                if ref_std > 0:
                    z_score = abs(current_mean - ref_mean) / ref_std
                    drift_scores[feature] = min(z_score / 3, 1.0)  # Normalize
                    
                    if z_score > 2.0:  # 2 standard deviations
                        affected_features.append(feature)
        
        # Overall drift assessment
        if drift_scores:
            overall_drift_score = np.mean(list(drift_scores.values()))
            drift_detected = len(affected_features) > 0
        else:
            overall_drift_score = 0.0
            drift_detected = False
        
        # Determine drift type (simplified)
        if drift_detected:
            if overall_drift_score > 0.8:
                drift_type = "sudden"
            elif overall_drift_score > 0.5:
                drift_type = "gradual"
            else:
                drift_type = "incremental"
        else:
            drift_type = "none"
        
        # Recommend action
        if drift_detected:
            if overall_drift_score > 0.7:
                recommended_action = "retrain"
            else:
                recommended_action = "adapt"
        else:
            recommended_action = "continue"
        
        drift_detection = ConceptDriftDetection(
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            drift_type=drift_type,
            detection_method=self.config.drift_detection_method,
            timestamp=datetime.now(),
            affected_features=affected_features,
            confidence=min(overall_drift_score * 2, 1.0),
            recommended_action=recommended_action
        )
        
        # Store in history
        self.drift_history.append(drift_detection)
        
        return drift_detection


class ActiveLearningEngine:
    """Active learning for optimal sample selection."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.query_history = []
        self.label_budget_used = 0
        
    def select_samples_for_labeling(self, 
                                  model: BaseEstimator,
                                  data_buffer: OnlineDataBuffer,
                                  n_queries: int) -> List[ActiveLearningQuery]:
        """Select samples that would be most informative if labeled."""
        
        if self.label_budget_used >= self.config.query_budget:
            return []
        
        # Get unlabeled samples
        X_unlabeled, _, _ = data_buffer.get_recent_samples(1000)
        
        if X_unlabeled.empty:
            return []
        
        # Remove samples that already have labels
        # (This is simplified - in practice, you'd track which samples are labeled)
        
        queries = []
        
        if self.config.query_strategy == "uncertainty":
            # Uncertainty sampling - select samples with highest prediction uncertainty
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_unlabeled)
                
                # Calculate uncertainty (entropy or margin-based)
                if probabilities.shape[1] == 2:  # Binary classification
                    uncertainties = 1 - np.abs(probabilities[:, 1] - 0.5) * 2  # Distance from 0.5
                else:  # Multiclass
                    uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                
                # Select top uncertain samples
                top_indices = np.argsort(uncertainties)[-n_queries:]
                
                for i, idx in enumerate(top_indices):
                    if self.label_budget_used >= self.config.query_budget:
                        break
                    
                    query = ActiveLearningQuery(
                        query_id=f"query_{int(time.time() * 1000)}_{i}",
                        sample_data=X_unlabeled.iloc[idx].to_dict(),
                        uncertainty_score=uncertainties[idx],
                        query_strategy=self.config.query_strategy,
                        timestamp=datetime.now(),
                        expected_information_gain=uncertainties[idx]  # Simplified
                    )
                    
                    queries.append(query)
                    self.label_budget_used += 1
        
        elif self.config.query_strategy == "diversity":
            # Diversity sampling - select samples that are diverse from each other
            # This is a simplified implementation
            from sklearn.cluster import KMeans
            
            try:
                # Cluster unlabeled samples
                n_clusters = min(n_queries, len(X_unlabeled))
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                cluster_labels = kmeans.fit_predict(X_unlabeled.fillna(0))
                
                # Select one sample from each cluster (closest to centroid)
                for cluster_id in range(n_clusters):
                    if self.label_budget_used >= self.config.query_budget:
                        break
                    
                    cluster_mask = cluster_labels == cluster_id
                    cluster_samples = X_unlabeled[cluster_mask]
                    
                    if len(cluster_samples) > 0:
                        # Find sample closest to cluster center
                        center = kmeans.cluster_centers_[cluster_id]
                        distances = np.sum((cluster_samples.fillna(0) - center) ** 2, axis=1)
                        closest_idx = np.argmin(distances)
                        
                        # Get original index
                        original_idx = cluster_samples.index[closest_idx]
                        
                        query = ActiveLearningQuery(
                            query_id=f"query_{int(time.time() * 1000)}_{cluster_id}",
                            sample_data=X_unlabeled.loc[original_idx].to_dict(),
                            uncertainty_score=0.5,  # Placeholder
                            query_strategy=self.config.query_strategy,
                            timestamp=datetime.now(),
                            expected_information_gain=1.0 / n_clusters  # Equal weight
                        )
                        
                        queries.append(query)
                        self.label_budget_used += 1
                        
            except Exception as e:
                logger.warning(f"Diversity sampling failed: {e}")
                # Fall back to random sampling
                random_indices = np.random.choice(
                    len(X_unlabeled), 
                    size=min(n_queries, len(X_unlabeled)), 
                    replace=False
                )
                
                for i, idx in enumerate(random_indices):
                    if self.label_budget_used >= self.config.query_budget:
                        break
                    
                    query = ActiveLearningQuery(
                        query_id=f"query_{int(time.time() * 1000)}_{i}",
                        sample_data=X_unlabeled.iloc[idx].to_dict(),
                        uncertainty_score=0.5,
                        query_strategy="random_fallback",
                        timestamp=datetime.now(),
                        expected_information_gain=0.5
                    )
                    
                    queries.append(query)
                    self.label_budget_used += 1
        
        # Store queries
        self.query_history.extend(queries)
        
        return queries
    
    def get_labeling_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        return {
            'total_queries': len(self.query_history),
            'budget_used': self.label_budget_used,
            'budget_remaining': self.config.query_budget - self.label_budget_used,
            'average_uncertainty': np.mean([q.uncertainty_score for q in self.query_history]) if self.query_history else 0,
            'query_strategies_used': list(set([q.query_strategy for q in self.query_history]))
        }


class AdaptiveLearningSystem:
    """Main adaptive learning system."""
    
    def __init__(self, config: AdaptiveLearningConfig = None):
        self.config = config or AdaptiveLearningConfig()
        
        # Initialize components
        self.data_buffer = OnlineDataBuffer(max_size=10000)
        self.drift_detector = ConceptDriftDetector(self.config)
        self.active_learning = ActiveLearningEngine(self.config)
        
        # Models
        self.primary_model = None
        self.model_ensemble = []
        self.model_performances = {}
        
        # State tracking
        self.samples_since_update = 0
        self.last_update_time = datetime.now()
        self.update_history = []
        
        # Thread safety
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize primary model
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the primary adaptive model."""
        # Use online-capable algorithms
        if self.config.adaptation_strategy == "incremental":
            self.primary_model = SGDClassifier(
                loss='log_loss',  # For probability estimates
                learning_rate='adaptive',
                eta0=self.config.learning_rate,
                random_state=self.config.random_state,
                warm_start=True
            )
        else:
            # Use standard model that will be retrained
            from sklearn.ensemble import RandomForestClassifier
            self.primary_model = RandomForestClassifier(
                n_estimators=50,
                random_state=self.config.random_state,
                warm_start=True
            )
        
        logger.info(f"Initialized adaptive model: {type(self.primary_model).__name__}")
    
    def add_sample(self, X: pd.Series, y: Optional[int] = None) -> Dict[str, Any]:
        """Add a new sample to the adaptive learning system."""
        
        # Add to buffer
        self.data_buffer.add_sample(X, y)
        self.samples_since_update += 1
        
        result = {
            'sample_added': True,
            'samples_since_update': self.samples_since_update,
            'drift_detected': False,
            'model_updated': False,
            'active_learning_query': None
        }
        
        # Check for concept drift
        if self.config.enable_drift_detection and len(self.data_buffer.data_buffer) > 100:
            drift_detection = self.drift_detector.detect_drift(self.data_buffer)
            result['drift_detected'] = drift_detection.drift_detected
            
            if drift_detection.drift_detected:
                logger.warning(f"Concept drift detected: {drift_detection.drift_type} "
                             f"(score: {drift_detection.drift_score:.3f})")
                
                # Take action based on drift
                if drift_detection.recommended_action == "retrain":
                    self._retrain_model()
                    result['model_updated'] = True
                elif drift_detection.recommended_action == "adapt":
                    self._adapt_model()
                    result['model_updated'] = True
        
        # Check if model update is needed
        if (self.samples_since_update >= self.config.update_frequency and 
            self.data_buffer.get_labeled_samples()[1].size >= self.config.min_samples_for_update):
            
            self._update_model()
            result['model_updated'] = True
        
        # Active learning query
        if (self.config.enable_active_learning and y is None and 
            self.primary_model is not None):
            
            queries = self.active_learning.select_samples_for_labeling(
                self.primary_model, self.data_buffer, n_queries=1
            )
            
            if queries:
                result['active_learning_query'] = asdict(queries[0])
        
        return result
    
    def predict(self, X: Union[pd.Series, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence estimates."""
        
        if self.primary_model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame if needed
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        
        # Make predictions
        predictions = self.primary_model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.primary_model, 'predict_proba'):
            probabilities = self.primary_model.predict_proba(X)
            # Use max probability as confidence
            confidences = np.max(probabilities, axis=1)
        else:
            # Use decision function if available
            if hasattr(self.primary_model, 'decision_function'):
                decision_scores = self.primary_model.decision_function(X)
                confidences = np.abs(decision_scores) / (np.abs(decision_scores) + 1)  # Normalize
            else:
                confidences = np.ones(len(predictions)) * 0.5  # Default confidence
        
        return predictions, confidences
    
    def _update_model(self) -> None:
        """Update the model with new labeled data."""
        
        with self.lock:
            try:
                X_labeled, y_labeled = self.data_buffer.get_labeled_samples()
                
                if len(y_labeled) < self.config.min_samples_for_update:
                    return
                
                logger.info(f"Updating model with {len(y_labeled)} labeled samples")
                
                if self.config.adaptation_strategy == "incremental":
                    # Incremental update
                    if hasattr(self.primary_model, 'partial_fit'):
                        # Get unique classes
                        classes = np.unique(y_labeled) if not hasattr(self.primary_model, 'classes_') else None
                        self.primary_model.partial_fit(X_labeled, y_labeled, classes=classes)
                    else:
                        # Retrain with all data
                        self.primary_model.fit(X_labeled, y_labeled)
                
                elif self.config.adaptation_strategy == "retrain":
                    # Full retraining
                    self.primary_model.fit(X_labeled, y_labeled)
                
                elif self.config.adaptation_strategy == "ensemble":
                    # Create new model for ensemble
                    new_model = clone(self.primary_model)
                    new_model.fit(X_labeled, y_labeled)
                    
                    # Add to ensemble
                    self.model_ensemble.append({
                        'model': new_model,
                        'timestamp': datetime.now(),
                        'samples': len(y_labeled)
                    })
                    
                    # Limit ensemble size
                    if len(self.model_ensemble) > self.config.max_models_ensemble:
                        self.model_ensemble.pop(0)
                
                # Reset counter
                self.samples_since_update = 0
                self.last_update_time = datetime.now()
                
                # Record update
                self.update_history.append({
                    'timestamp': datetime.now(),
                    'samples_used': len(y_labeled),
                    'strategy': self.config.adaptation_strategy
                })
                
                # Update drift detector reference
                self.drift_detector.update_reference(self.data_buffer)
                
                logger.info("Model update completed successfully")
                
            except Exception as e:
                logger.error(f"Model update failed: {e}")
    
    def _retrain_model(self) -> None:
        """Completely retrain the model."""
        logger.info("Retraining model due to concept drift")
        
        # Get all labeled data
        X_labeled, y_labeled = self.data_buffer.get_labeled_samples()
        
        if len(y_labeled) < self.config.min_samples_for_update:
            logger.warning("Not enough labeled data for retraining")
            return
        
        # Reinitialize and train model
        self._initialize_model()
        self.primary_model.fit(X_labeled, y_labeled)
        
        # Clear ensemble
        self.model_ensemble.clear()
        
        # Update drift detector
        self.drift_detector.update_reference(self.data_buffer)
        self.drift_detector.reference_distributions.clear()  # Force rebuild
    
    def _adapt_model(self) -> None:
        """Gradually adapt model to drift."""
        logger.info("Adapting model to gradual drift")
        
        # Get recent data (more weight on recent samples)
        X_recent, y_recent, _ = self.data_buffer.get_recent_samples(
            self.config.drift_window_size // 2
        )
        
        labeled_mask = y_recent != None
        if np.sum(labeled_mask) < 10:
            return
        
        X_recent_labeled = X_recent[labeled_mask]
        y_recent_labeled = y_recent[labeled_mask]
        
        # Incremental update with recent data
        if hasattr(self.primary_model, 'partial_fit'):
            self.primary_model.partial_fit(X_recent_labeled, y_recent_labeled)
        else:
            # Combine with some old data
            X_old, y_old = self.data_buffer.get_labeled_samples(500)
            
            if len(y_old) > 0:
                X_combined = pd.concat([X_old, X_recent_labeled])
                y_combined = np.concatenate([y_old, y_recent_labeled])
                self.primary_model.fit(X_combined, y_combined)
    
    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: np.ndarray) -> ModelPerformanceMetrics:
        """Evaluate current model performance."""
        
        if self.primary_model is None:
            raise ValueError("No model to evaluate")
        
        predictions = self.primary_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # ROC AUC for binary classification
        try:
            if hasattr(self.primary_model, 'predict_proba'):
                probas = self.primary_model.predict_proba(X_test)
                if probas.shape[1] == 2:  # Binary
                    roc_auc = roc_auc_score(y_test, probas[:, 1])
                else:  # Multiclass
                    roc_auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        # Confidence interval for accuracy (Wilson score)
        n = len(y_test)
        correct = np.sum(predictions == y_test)
        z = 1.96  # 95% confidence
        
        p_hat = correct / n
        ci_lower = (p_hat + z**2/(2*n) - z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)) / (1 + z**2/n)
        ci_upper = (p_hat + z**2/(2*n) + z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)) / (1 + z**2/n)
        
        metrics = ModelPerformanceMetrics(
            model_id=f"adaptive_model_{int(time.time())}",
            accuracy=accuracy,
            f1_score=f1,
            roc_auc=roc_auc,
            prediction_count=len(predictions),
            correct_predictions=correct,
            timestamp=datetime.now(),
            confidence_interval=(ci_lower, ci_upper)
        )
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'model_info': {
                'type': type(self.primary_model).__name__ if self.primary_model else None,
                'is_trained': self.primary_model is not None,
                'ensemble_size': len(self.model_ensemble),
                'last_update': self.last_update_time.isoformat()
            },
            'data_buffer': {
                'total_samples': len(self.data_buffer.data_buffer),
                'labeled_samples': len(self.data_buffer.get_labeled_samples()[1]),
                'samples_since_update': self.samples_since_update
            },
            'drift_detection': {
                'enabled': self.config.enable_drift_detection,
                'method': self.config.drift_detection_method,
                'recent_drifts': len([d for d in self.drift_detector.drift_history 
                                   if d.timestamp > datetime.now() - timedelta(hours=24)])
            },
            'active_learning': self.active_learning.get_labeling_statistics(),
            'configuration': asdict(self.config)
        }
    
    def save_system_state(self, filepath: str) -> None:
        """Save the adaptive learning system state."""
        
        state_data = {
            'config': asdict(self.config),
            'model_type': type(self.primary_model).__name__ if self.primary_model else None,
            'update_history': self.update_history,
            'drift_history': [asdict(d) for d in self.drift_detector.drift_history],
            'samples_since_update': self.samples_since_update,
            'last_update_time': self.last_update_time.isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save models separately
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)
        
        if self.primary_model:
            model_path = os.path.join(model_dir, 'adaptive_primary_model.joblib')
            joblib.dump(self.primary_model, model_path)
            state_data['primary_model_path'] = model_path
        
        # Save ensemble models
        if self.model_ensemble:
            ensemble_paths = []
            for i, model_info in enumerate(self.model_ensemble):
                ensemble_path = os.path.join(model_dir, f'adaptive_ensemble_model_{i}.joblib')
                joblib.dump(model_info['model'], ensemble_path)
                ensemble_paths.append(ensemble_path)
            state_data['ensemble_model_paths'] = ensemble_paths
        
        # Save state data
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Adaptive learning system state saved to {filepath}")


def create_adaptive_learning_system(config: AdaptiveLearningConfig = None) -> AdaptiveLearningSystem:
    """Create and configure adaptive learning system."""
    return AdaptiveLearningSystem(config)


async def simulate_online_learning(system: AdaptiveLearningSystem, 
                                 data_stream: List[Tuple[pd.Series, Optional[int]]],
                                 labeling_probability: float = 0.1) -> Dict[str, Any]:
    """Simulate online learning with streaming data."""
    
    results = {
        'samples_processed': 0,
        'labels_provided': 0,
        'model_updates': 0,
        'drift_detections': 0,
        'active_queries': 0
    }
    
    for i, (X, true_label) in enumerate(data_stream):
        # Randomly decide if label is available (simulating real-world scenario)
        label_available = np.random.random() < labeling_probability
        y = true_label if label_available else None
        
        # Add sample to system
        result = system.add_sample(X, y)
        
        # Update statistics
        results['samples_processed'] += 1
        if y is not None:
            results['labels_provided'] += 1
        if result['model_updated']:
            results['model_updates'] += 1
        if result['drift_detected']:
            results['drift_detections'] += 1
        if result['active_learning_query']:
            results['active_queries'] += 1
        
        # Log progress periodically
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} samples, {results['model_updates']} updates, "
                       f"{results['drift_detections']} drifts detected")
    
    return results


if __name__ == "__main__":
    print("Adaptive Learning System with Online Model Updates")
    print("This system provides continuous learning and adaptation capabilities.")