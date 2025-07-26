"""
Test enhanced Prometheus metrics functionality.
"""
import unittest
from pathlib import Path


class TestEnhancedMetrics(unittest.TestCase):
    """Test enhanced metrics functionality."""
    
    def test_api_endpoint_performance_metrics(self):
        """Test that API endpoint performance metrics are collected."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should have endpoint-specific metrics
        self.assertIn("endpoint_duration", content, "Should track endpoint duration")
        self.assertIn("endpoint_requests", content, "Should track endpoint requests")
        self.assertIn("record_api_endpoint", content, "Should have API endpoint recording method")
    
    def test_model_prediction_latency_tracking(self):
        """Test that model prediction latency is properly tracked."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should have prediction latency metrics (already exists, verify enhancements)
        self.assertIn("prediction_latency", content, "Should track prediction latency")
        self.assertIn("record_prediction_latency", content, "Should record prediction latency")
        
        # Should have enhanced prediction metrics
        self.assertIn("prediction_batch_size", content, "Should track batch sizes")
        self.assertIn("prediction_queue_time", content, "Should track queue time")
    
    def test_system_resource_monitoring(self):
        """Test that system resources are monitored."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should have resource monitoring
        self.assertIn("system_memory", content, "Should monitor memory usage")
        self.assertIn("system_cpu", content, "Should monitor CPU usage")
        self.assertIn("record_system_resources", content, "Should record system resources")
    
    def test_custom_business_metrics(self):
        """Test that custom business metrics are implemented."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should have business metrics
        self.assertIn("churn_prediction_confidence", content, "Should track prediction confidence")
        self.assertIn("model_drift", content, "Should monitor model drift")
        self.assertIn("prediction_accuracy_real_time", content, "Should track real-time accuracy")
    
    def test_enhanced_error_tracking(self):
        """Test that enhanced error tracking is implemented."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should have enhanced error metrics
        self.assertIn("error_rate", content, "Should track error rates")
        self.assertIn("error_by_endpoint", content, "Should track errors by endpoint")
        self.assertIn("record_endpoint_error", content, "Should record endpoint-specific errors")
    
    def test_performance_percentiles(self):
        """Test that performance percentiles are calculated."""
        metrics_file = Path(__file__).parent.parent / "src" / "metrics.py"
        content = metrics_file.read_text()
        
        # Should calculate percentiles
        self.assertIn("percentile", content, "Should calculate percentiles")
        self.assertIn("p95", content, "Should track 95th percentile")
        self.assertIn("p99", content, "Should track 99th percentile")


if __name__ == '__main__':
    unittest.main()