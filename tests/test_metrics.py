"""
Tests for Prometheus metrics functionality.

This module tests the metrics collection and exposition system
for monitoring application performance and model behavior.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.metrics import (
    MetricsCollector,
    get_metrics_collector,
    get_prometheus_metrics,
    record_prediction_latency,
    record_prediction_count,
    record_model_accuracy,
    record_cache_hit_rate
)


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
    
    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        self.assertIsNotNone(self.collector._prediction_latency)
        self.assertIsNotNone(self.collector._prediction_count)
        self.assertIsNotNone(self.collector._model_accuracy)
        self.assertIsNotNone(self.collector._cache_hit_rate)
        self.assertIsNotNone(self.collector._health_check_duration)
    
    def test_record_prediction_latency(self):
        """Test recording prediction latency metrics."""
        # Record some latencies
        self.collector.record_prediction_latency(0.5, "batch")
        self.collector.record_prediction_latency(0.3, "single")
        self.collector.record_prediction_latency(0.8, "batch")
        
        # Check that histogram was updated
        metrics = self.collector.get_metrics()
        
        # Should have prediction latency metrics
        latency_metrics = [m for m in metrics if 'prediction_latency' in m]
        self.assertGreater(len(latency_metrics), 0)
    
    def test_record_prediction_count(self):
        """Test recording prediction count metrics."""
        # Record predictions
        self.collector.record_prediction_count(1, "success", "batch")
        self.collector.record_prediction_count(5, "success", "single")
        self.collector.record_prediction_count(1, "error", "batch")
        
        # Check counter metrics
        metrics = self.collector.get_metrics()
        count_metrics = [m for m in metrics if 'prediction_count' in m]
        self.assertGreater(len(count_metrics), 0)
    
    def test_record_model_accuracy(self):
        """Test recording model accuracy metrics."""
        # Record accuracy
        self.collector.record_model_accuracy(0.85, "test_set")
        self.collector.record_model_accuracy(0.92, "validation_set")
        
        # Check gauge metrics
        metrics = self.collector.get_metrics()
        accuracy_metrics = [m for m in metrics if 'model_accuracy' in m]
        self.assertGreater(len(accuracy_metrics), 0)
    
    def test_record_cache_metrics(self):
        """Test recording cache performance metrics."""
        # Record cache statistics
        cache_stats = {
            "hit_rate": 75.5,
            "entries": 5,
            "memory_used_mb": 150.2
        }
        self.collector.record_cache_metrics(cache_stats)
        
        # Check metrics
        metrics = self.collector.get_metrics()
        cache_metrics = [m for m in metrics if 'cache_' in m]
        self.assertGreater(len(cache_metrics), 0)
    
    def test_record_health_check_duration(self):
        """Test recording health check duration."""
        self.collector.record_health_check_duration(0.1, "basic", "healthy")
        self.collector.record_health_check_duration(0.5, "detailed", "degraded")
        
        metrics = self.collector.get_metrics()
        health_metrics = [m for m in metrics if 'health_check' in m]
        self.assertGreater(len(health_metrics), 0)
    
    def test_get_prometheus_format(self):
        """Test Prometheus format output."""
        # Record some metrics
        self.collector.record_prediction_latency(0.5, "batch")
        self.collector.record_prediction_count(1, "success", "batch")
        
        # Get Prometheus format
        prometheus_output = self.collector.get_prometheus_format()
        
        # Should be valid Prometheus format
        self.assertIsInstance(prometheus_output, str)
        self.assertIn('# HELP', prometheus_output)
        self.assertIn('# TYPE', prometheus_output)
        
        # Should contain our metrics
        self.assertIn('prediction_latency', prometheus_output)
        self.assertIn('prediction_count', prometheus_output)
    
    def test_metrics_with_labels(self):
        """Test metrics with different label combinations."""
        # Record metrics with various labels
        self.collector.record_prediction_latency(0.1, "single")
        self.collector.record_prediction_latency(0.5, "batch")
        self.collector.record_prediction_count(1, "success", "single")
        self.collector.record_prediction_count(10, "success", "batch")
        
        metrics = self.collector.get_metrics()
        
        # Should have metrics with different label values
        latency_metrics = [m for m in metrics if 'prediction_latency' in m]
        count_metrics = [m for m in metrics if 'prediction_count' in m]
        
        self.assertGreater(len(latency_metrics), 0)
        self.assertGreater(len(count_metrics), 0)


class TestMetricsFunctions(unittest.TestCase):
    """Test module-level metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear global collector
        from src.metrics import _global_collector
        if _global_collector:
            _global_collector = None
    
    def test_get_global_collector(self):
        """Test global collector retrieval."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return same instance
        self.assertIs(collector1, collector2)
        self.assertIsInstance(collector1, MetricsCollector)
    
    def test_record_prediction_latency_function(self):
        """Test module-level prediction latency recording."""
        start_time = time.time()
        time.sleep(0.01)  # Small delay
        
        record_prediction_latency(start_time, "test")
        
        # Should have recorded latency
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        
        latency_metrics = [m for m in metrics if 'prediction_latency' in m]
        self.assertGreater(len(latency_metrics), 0)
    
    def test_record_prediction_count_function(self):
        """Test module-level prediction count recording."""
        record_prediction_count(5, "success", "batch")
        record_prediction_count(1, "error", "single")
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        
        count_metrics = [m for m in metrics if 'prediction_count' in m]
        self.assertGreater(len(count_metrics), 0)
    
    def test_record_model_accuracy_function(self):
        """Test module-level model accuracy recording."""
        record_model_accuracy(0.89, "validation")
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        
        accuracy_metrics = [m for m in metrics if 'model_accuracy' in m]
        self.assertGreater(len(accuracy_metrics), 0)
    
    def test_record_cache_hit_rate_function(self):
        """Test module-level cache hit rate recording."""
        record_cache_hit_rate(85.5)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        
        cache_metrics = [m for m in metrics if 'cache_hit_rate' in m]
        self.assertGreater(len(cache_metrics), 0)
    
    def test_get_prometheus_metrics_function(self):
        """Test getting Prometheus metrics."""
        # Record some metrics
        record_prediction_latency(time.time() - 0.1, "test")
        record_prediction_count(1, "success", "test")
        
        # Get Prometheus format
        prometheus_output = get_prometheus_metrics()
        
        self.assertIsInstance(prometheus_output, str)
        self.assertIn('# HELP', prometheus_output)
        self.assertIn('prediction_', prometheus_output)


class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics system."""
    
    def test_metrics_collection_realistic(self):
        """Test metrics collection in realistic scenario."""
        collector = get_metrics_collector()
        
        # Simulate prediction workflow
        start_time = time.time()
        
        # Record various metrics
        record_prediction_count(10, "success", "batch")
        record_prediction_latency(start_time - 0.5, "batch")
        record_model_accuracy(0.87, "test_set")
        record_cache_hit_rate(78.5)
        
        # Get metrics
        prometheus_output = get_prometheus_metrics()
        
        # Should have comprehensive metrics
        self.assertIn('prediction_count_total', prometheus_output)
        self.assertIn('prediction_latency_seconds', prometheus_output)
        self.assertIn('model_accuracy', prometheus_output)
        self.assertIn('cache_hit_rate', prometheus_output)
    
    def test_concurrent_metrics_recording(self):
        """Test thread safety of metrics recording."""
        import threading
        
        def record_metrics(thread_id):
            for i in range(10):
                record_prediction_count(1, "success", f"thread_{thread_id}")
                record_prediction_latency(time.time() - 0.1, f"thread_{thread_id}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Metrics should be recorded without errors
        prometheus_output = get_prometheus_metrics()
        self.assertIn('prediction_count_total', prometheus_output)
        self.assertIn('prediction_latency_seconds', prometheus_output)


if __name__ == "__main__":
    unittest.main()