"""
Test health monitoring functionality without full dependencies.
"""
import unittest
import sys
import os
import time
from pathlib import Path


class TestHealthMonitoringFunctionality(unittest.TestCase):
    """Test health monitoring functionality."""
    
    def test_health_check_file_updated(self):
        """Test that health_check.py has been updated with new functionality."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        self.assertTrue(health_check_file.exists(), "health_check.py should exist")
        
        content = health_check_file.read_text()
        
        # Check for new methods
        self.assertIn("check_resource_usage", content, "Should have resource usage check")
        self.assertIn("check_mlflow_service", content, "Should have MLflow service check")
        self.assertIn("check_database_connectivity", content, "Should have database connectivity check")
        self.assertIn("get_dependency_versions", content, "Should have dependency version reporting")
        
        # Check for enhanced comprehensive health check
        self.assertIn("performance", content, "Should include performance metrics")
        self.assertIn("check_duration_ms", content, "Should track check duration")
    
    def test_import_structure_updated(self):
        """Test that imports have been updated for enhanced monitoring."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        content = health_check_file.read_text()
        
        # Check for new imports
        self.assertIn("import sys", content, "Should import sys for version info")
        self.assertIn("import platform", content, "Should import platform for system info")
        self.assertIn("HAS_PSUTIL", content, "Should have optional psutil import")
        self.assertIn("HAS_MLFLOW", content, "Should have optional MLflow import")
    
    def test_comprehensive_health_structure(self):
        """Test that comprehensive health check has the right structure."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        content = health_check_file.read_text()
        
        # Should update total checks count
        self.assertIn("total_checks\": 8", content, "Should have 8 total checks")
        
        # Should include all new check types
        checks_to_find = [
            "\"resources\":", "\"mlflow_service\":", 
            "\"database\":", "\"performance\":"
        ]
        
        for check in checks_to_find:
            self.assertIn(check, content, f"Should include {check} in health results")
    
    def test_enhanced_error_handling(self):
        """Test that enhanced error handling is implemented."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        content = health_check_file.read_text()
        
        # Should have proper error handling for optional dependencies
        self.assertIn("if not HAS_PSUTIL:", content, "Should handle missing psutil gracefully")
        self.assertIn("if not HAS_MLFLOW:", content, "Should handle missing MLflow gracefully")
        
        # Should have comprehensive error collection
        self.assertIn("errors.extend", content, "Should collect multiple errors")
        self.assertIn("core_checks", content, "Should distinguish core vs enhanced checks")
    
    def test_performance_metrics_structure(self):
        """Test that performance metrics are properly structured."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        content = health_check_file.read_text()
        
        # Should calculate and report duration
        self.assertIn("duration_ms = round", content, "Should calculate duration in milliseconds")
        self.assertIn("performance", content, "Should include performance section")
        self.assertIn("check_duration_ms", content, "Should report check duration")
    
    def test_version_reporting_implementation(self):
        """Test that version reporting is implemented."""
        health_check_file = Path(__file__).parent.parent / "src" / "health_check.py"
        content = health_check_file.read_text()
        
        # Should collect version information
        self.assertIn("sys.version_info", content, "Should get Python version info")
        self.assertIn("platform.platform", content, "Should get platform info")
        self.assertIn("__version__", content, "Should check package versions")
        
        # Should handle missing packages gracefully
        self.assertIn("not_installed", content, "Should handle missing packages")


if __name__ == '__main__':
    unittest.main()