"""
Test enhanced health monitoring functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from pathlib import Path
import json
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TestEnhancedHealthMonitoring(unittest.TestCase):
    """Test enhanced health monitoring features."""
    
    def test_resource_usage_monitoring(self):
        """Test that system resource usage is monitored."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        health = checker.get_comprehensive_health()
        
        # Should include resource monitoring
        self.assertIn("resources", health["checks"], "Health check should include resource monitoring")
        
        resources = health["checks"]["resources"]
        self.assertIn("cpu_percent", resources, "Should monitor CPU usage")
        self.assertIn("memory_percent", resources, "Should monitor memory usage")
        self.assertIn("disk_usage", resources, "Should monitor disk usage")
    
    def test_mlflow_service_check(self):
        """Test MLflow service connectivity check."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        health = checker.get_comprehensive_health()
        
        # Should include MLflow service check
        self.assertIn("mlflow_service", health["checks"], "Health check should include MLflow service check")
        
        mlflow_check = health["checks"]["mlflow_service"]
        self.assertIn("service_available", mlflow_check, "Should check MLflow service availability")
        self.assertIn("tracking_uri", mlflow_check, "Should report tracking URI")
    
    def test_dependency_version_reporting(self):
        """Test that dependency versions are reported."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        health = checker.get_comprehensive_health()
        
        # Dependencies check should include versions
        deps = health["checks"]["dependencies"]
        self.assertIn("versions", deps, "Should report dependency versions")
        
        versions = deps["versions"]
        self.assertIsInstance(versions, dict, "Versions should be a dictionary")
        # Should have at least Python version
        self.assertIn("python", versions, "Should report Python version")
    
    def test_database_connectivity_placeholder(self):
        """Test database connectivity check (placeholder - no actual DB)."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        health = checker.get_comprehensive_health()
        
        # Should include database check (even if no DB configured)
        self.assertIn("database", health["checks"], "Health check should include database connectivity")
        
        db_check = health["checks"]["database"]
        self.assertIn("configured", db_check, "Should report if database is configured")
    
    def test_enhanced_error_reporting(self):
        """Test enhanced error reporting with more details."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        health = checker.get_comprehensive_health()
        
        # Each check should have detailed error information
        for check_name, check_result in health["checks"].items():
            self.assertIsInstance(check_result, dict, f"{check_name} check should return dict")
            
            # If there are errors, they should be detailed
            if "errors" in check_result and check_result["errors"]:
                self.assertIsInstance(check_result["errors"], list, f"{check_name} errors should be a list")
    
    def test_performance_metrics_in_health_check(self):
        """Test that performance metrics are included in health checks."""
        import sys
        sys.path.insert(0, 'src')
        from health_check import HealthChecker
        
        checker = HealthChecker()
        start_time = time.time()
        health = checker.get_comprehensive_health()
        end_time = time.time()
        
        # Should include timing information
        self.assertIn("performance", health, "Health check should include performance metrics")
        
        performance = health["performance"]
        self.assertIn("check_duration_ms", performance, "Should report check duration")
        self.assertIsInstance(performance["check_duration_ms"], (int, float), "Duration should be numeric")
        
        # Duration should be reasonable
        expected_duration = (end_time - start_time) * 1000
        actual_duration = performance["check_duration_ms"]
        self.assertLess(abs(actual_duration - expected_duration), 100, "Duration should be approximately correct")


if __name__ == '__main__':
    unittest.main()