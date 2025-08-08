"""
Tests for autonomous retraining system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.autonomous_retraining import (
    StatisticalDriftDetector, PerformanceMonitor, AutonomousRetrainingSystem,
    run_autonomous_monitoring_cycle
)


@pytest.fixture
def sample_baseline_data():
    """Create baseline dataset."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2,
        n_informative=8, n_redundant=2, random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


@pytest.fixture
def sample_drift_data():
    """Create data with drift."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        n_informative=8, n_redundant=2, random_state=123  # Different seed
    )
    
    # Add systematic shift to simulate drift
    X[:, 0] += 2.0  # Shift first feature
    X[:, 1] *= 1.5  # Scale second feature
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


class TestStatisticalDriftDetector:
    """Test drift detection functionality."""
    
    def test_detector_initialization(self):
        """Test drift detector initialization."""
        detector = StatisticalDriftDetector()
        assert detector.significance_level == 0.05
        assert detector.min_samples == 100
        assert detector.reference_distributions == {}
    
    def test_fit_reference(self, sample_baseline_data):
        """Test fitting reference distributions."""
        X, y = sample_baseline_data
        
        detector = StatisticalDriftDetector()
        detector.fit_reference(X)
        
        assert len(detector.reference_distributions) == X.shape[1]
        
        # Check that distributions are properly stored
        for col in X.columns:
            assert col in detector.reference_distributions
            ref_dist = detector.reference_distributions[col]
            assert ref_dist['type'] == 'numerical'
            assert 'mean' in ref_dist
            assert 'std' in ref_dist
    
    def test_detect_no_drift(self, sample_baseline_data):
        """Test detection when no drift is present."""
        X, y = sample_baseline_data
        
        detector = StatisticalDriftDetector()
        detector.fit_reference(X)
        
        # Use same data (no drift)
        result = detector.detect_drift(X)
        
        assert not result.drift_detected
        assert result.drift_score < 0.3
        assert result.drift_type in ['no_drift', 'mild_drift']
    
    def test_detect_drift(self, sample_baseline_data, sample_drift_data):
        """Test detection when drift is present."""
        X_baseline, _ = sample_baseline_data
        X_drift, _ = sample_drift_data
        
        detector = StatisticalDriftDetector()
        detector.fit_reference(X_baseline)
        
        # Detect drift in new data
        result = detector.detect_drift(X_drift)
        
        # Should detect some level of drift
        assert result.drift_score > 0.1
        assert len(result.affected_features) > 0
        assert result.confidence_level > 0


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.82}
        monitor = PerformanceMonitor(baseline_metrics)
        
        assert monitor.baseline_metrics == baseline_metrics
        assert 'accuracy_drop' in monitor.alert_thresholds
    
    def test_evaluate_good_performance(self):
        """Test evaluation with good performance."""
        baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.82}
        monitor = PerformanceMonitor(baseline_metrics)
        
        # Create mock prediction data
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])  # Perfect predictions
        
        report = monitor.evaluate_performance(y_true, y_pred)
        
        assert report.current_accuracy == 1.0
        assert report.alert_level == 'green'
        assert report.performance_degradation <= 0
    
    def test_evaluate_degraded_performance(self):
        """Test evaluation with degraded performance."""
        baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.82}
        monitor = PerformanceMonitor(baseline_metrics)
        
        # Create mock prediction data with poor performance
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])  # All wrong
        
        report = monitor.evaluate_performance(y_true, y_pred)
        
        assert report.current_accuracy == 0.0
        assert report.alert_level == 'red'
        assert report.performance_degradation > 0.5


class TestAutonomousRetrainingSystem:
    """Test autonomous retraining system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = AutonomousRetrainingSystem()
        
        assert system.config is not None
        assert system.drift_detector is not None
        assert system.performance_monitor is None  # Not initialized yet
    
    def test_initialize_baseline(self, sample_baseline_data):
        """Test baseline initialization."""
        X, y = sample_baseline_data
        
        # Create simple model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X, y, model)
        
        assert system.performance_monitor is not None
        assert system.last_training_time is not None
        assert len(system.drift_detector.reference_distributions) > 0
    
    def test_assess_retraining_no_issues(self, sample_baseline_data):
        """Test retraining assessment with no issues."""
        X, y = sample_baseline_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X, y, model)
        
        # Use same data (no drift, good performance)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        decision = system.assess_retraining_need(X, y, y_pred, y_pred_proba)
        
        assert decision.priority == 'low'
        assert not decision.should_retrain or decision.recommended_strategy != 'emergency_retrain'
    
    def test_assess_retraining_with_drift(self, sample_baseline_data, sample_drift_data):
        """Test retraining assessment with drift."""
        X_baseline, y_baseline = sample_baseline_data
        X_drift, y_drift = sample_drift_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_baseline, y_baseline)
        
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X_baseline, y_baseline, model)
        
        # Use drift data
        decision = system.assess_retraining_need(X_drift)
        
        # Should detect need for retraining due to drift
        assert decision.priority in ['medium', 'high', 'critical']
        assert 'drift' in decision.reason.lower()
    
    @patch('src.autonomous_retraining.optimize_model_hyperparameters')
    def test_execute_retraining(self, mock_optimize, sample_baseline_data):
        """Test retraining execution."""
        X, y = sample_baseline_data
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.best_model = LogisticRegression()
        mock_result.best_score = 0.87
        mock_result.optimization_time = 30.0
        mock_result.model_complexity = 10
        mock_result.cross_validation_scores = [0.85, 0.87, 0.89]
        mock_optimize.return_value = mock_result
        
        system = AutonomousRetrainingSystem()
        
        result = system.execute_retraining(X, y, retraining_strategy="scheduled_retrain")
        
        assert result is not None
        assert mock_optimize.called
        assert system.last_training_time is not None
        assert len(system.retraining_history) > 0


class TestMonitoringCycle:
    """Test complete monitoring cycle."""
    
    def test_monitoring_cycle_no_retrain(self, sample_baseline_data):
        """Test monitoring cycle without retraining."""
        X, y = sample_baseline_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X, y, model)
        
        y_pred = model.predict(X)
        
        result = run_autonomous_monitoring_cycle(
            X_current=X,
            y_true=y,
            y_pred=y_pred,
            retraining_system=system,
            auto_retrain=False
        )
        
        assert 'retraining_decision' in result
        assert 'retraining_executed' in result
        assert not result['retraining_executed']
    
    def test_get_system_status(self, sample_baseline_data):
        """Test getting system status."""
        X, y = sample_baseline_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X, y, model)
        
        status = system.get_system_status()
        
        assert 'config' in status
        assert 'last_training_time' in status
        assert 'system_ready' in status
        assert status['system_ready'] is True


@pytest.mark.integration
class TestIntegrationRetraining:
    """Integration tests for retraining system."""
    
    def test_complete_drift_response(self, sample_baseline_data, sample_drift_data):
        """Test complete drift detection and response."""
        X_baseline, y_baseline = sample_baseline_data
        X_drift, y_drift = sample_drift_data
        
        # Train initial model
        model = LogisticRegression(random_state=42)
        model.fit(X_baseline, y_baseline)
        
        # Initialize system
        system = AutonomousRetrainingSystem()
        system.initialize_baseline(X_baseline, y_baseline, model)
        
        # Test with drift data
        y_pred_drift = model.predict(X_drift)
        
        decision = system.assess_retraining_need(X_drift, y_drift, y_pred_drift)
        
        # Should recommend retraining
        assert decision.priority in ['medium', 'high', 'critical']
        
        # Execute retraining if recommended
        if decision.should_retrain:
            # Mock retraining for integration test
            system.last_training_time = datetime.utcnow()
            system.retraining_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'reason': decision.reason,
                'strategy': decision.recommended_strategy
            })
        
        # Verify system state
        status = system.get_system_status()
        assert status['system_ready'] is True