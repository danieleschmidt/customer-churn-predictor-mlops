"""
Tests for health check functionality.

This module tests the health check system used for monitoring and
container orchestration.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any

from src.health_check import (
    HealthChecker, 
    get_health_status, 
    get_comprehensive_health,
    get_readiness_status,
    is_healthy,
    is_ready
)
from src.path_config import PathConfig


class TestHealthChecker(unittest.TestCase):
    """Test HealthChecker class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.path_config = PathConfig(base_dir=Path(self.temp_dir))
        self.health_checker = HealthChecker(self.path_config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_basic_health(self):
        """Test basic health check returns expected structure."""
        health = self.health_checker.check_basic_health()
        
        # Check required fields
        self.assertIn("status", health)
        self.assertIn("timestamp", health)
        self.assertIn("uptime_seconds", health)
        self.assertIn("service", health)
        self.assertIn("version", health)
        
        # Check values
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["service"], "churn-predictor")
        self.assertEqual(health["version"], "1.0.0")
        self.assertIsInstance(health["uptime_seconds"], float)
        self.assertGreaterEqual(health["uptime_seconds"], 0)
    
    def test_check_model_availability_no_models_dir(self):
        """Test model availability check when models directory doesn't exist."""
        model_status = self.health_checker.check_model_availability()
        
        self.assertFalse(model_status["model_available"])
        self.assertIsNone(model_status["model_path"])
        self.assertIsNone(model_status["model_age_hours"])
        self.assertIn("Models directory does not exist", model_status["error"])
    
    def test_check_model_availability_no_models(self):
        """Test model availability check when no model files exist."""
        # Create models directory but no model files
        models_dir = self.path_config.get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_status = self.health_checker.check_model_availability()
        
        self.assertFalse(model_status["model_available"])
        self.assertIsNone(model_status["model_path"])
        self.assertIsNone(model_status["model_age_hours"])
        self.assertIn("No model files found", model_status["error"])
    
    def test_check_model_availability_with_model(self):
        """Test model availability check with existing model file."""
        # Create models directory and model file
        models_dir = self.path_config.get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = models_dir / "test_model.pkl"
        model_file.write_text("dummy model content")
        
        model_status = self.health_checker.check_model_availability()
        
        self.assertTrue(model_status["model_available"])
        self.assertEqual(model_status["model_path"], str(model_file))
        self.assertIsInstance(model_status["model_age_hours"], float)
        self.assertGreaterEqual(model_status["model_age_hours"], 0)
        self.assertIsNone(model_status["error"])
    
    def test_check_data_directories_missing(self):
        """Test data directories check when directories don't exist."""
        # Use empty temp directory where no data dirs exist
        empty_temp = tempfile.mkdtemp()
        try:
            empty_config = PathConfig(base_dir=Path(empty_temp))
            checker = HealthChecker(empty_config)
            
            status = checker.check_data_directories()
            
            self.assertFalse(status["data_dirs_accessible"])
            self.assertGreater(len(status["errors"]), 0)
            
            # Check that directories are marked as not existing
            for dir_name, dir_status in status["directories"].items():
                if "error" not in dir_status:
                    self.assertFalse(dir_status["exists"])
                    self.assertFalse(dir_status["readable"])
        finally:
            import shutil
            shutil.rmtree(empty_temp, ignore_errors=True)
    
    def test_check_data_directories_existing(self):
        """Test data directories check when directories exist."""
        # Create all required directories
        dirs_to_create = [
            self.path_config.get_data_dir(),
            self.path_config.get_raw_data_dir(),
            self.path_config.get_processed_data_dir(),
            self.path_config.get_models_dir()
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        status = self.health_checker.check_data_directories()
        
        self.assertTrue(status["data_dirs_accessible"])
        self.assertEqual(len(status["errors"]), 0)
        
        # Check that all directories are accessible
        for dir_name, dir_status in status["directories"].items():
            if "error" not in dir_status:
                self.assertTrue(dir_status["exists"])
                self.assertTrue(dir_status["readable"])
                self.assertTrue(dir_status["writable"])
    
    @patch("pathlib.Path.exists")
    @patch("os.access")
    def test_check_configuration_file_exists(self, mock_access, mock_exists):
        """Test configuration check when config file exists and is readable."""
        mock_exists.return_value = True
        mock_access.return_value = True
        
        config_status = self.health_checker.check_configuration()
        
        self.assertTrue(config_status["config_valid"])
        self.assertTrue(config_status["config_file_exists"])
        self.assertTrue(config_status["config_readable"])
        self.assertEqual(len(config_status["errors"]), 0)
    
    @patch("pathlib.Path.exists")
    def test_check_configuration_file_missing(self, mock_exists):
        """Test configuration check when config file doesn't exist."""
        mock_exists.return_value = False
        
        config_status = self.health_checker.check_configuration()
        
        self.assertTrue(config_status["config_valid"])  # Missing config is acceptable
        self.assertFalse(config_status["config_file_exists"])
        self.assertFalse(config_status["config_readable"])
        self.assertEqual(len(config_status["errors"]), 1)
        self.assertIn("Configuration file not found", config_status["errors"][0])
    
    @patch("pathlib.Path.exists")
    @patch("os.access")
    def test_check_configuration_file_not_readable(self, mock_access, mock_exists):
        """Test configuration check when config file exists but isn't readable."""
        mock_exists.return_value = True
        mock_access.return_value = False
        
        config_status = self.health_checker.check_configuration()
        
        self.assertFalse(config_status["config_valid"])
        self.assertTrue(config_status["config_file_exists"])
        self.assertFalse(config_status["config_readable"])
        self.assertGreater(len(config_status["errors"]), 0)
        self.assertIn("Configuration file not readable", config_status["errors"][0])
    
    def test_check_dependencies_all_available(self):
        """Test dependencies check when all packages are available."""
        deps_status = self.health_checker.check_dependencies()
        
        # Since we're running tests, critical packages should be available
        self.assertTrue(deps_status["dependencies_available"])
        self.assertEqual(len(deps_status["errors"]), 0)
        
        # Check that critical packages are marked as available
        critical_packages = ["pandas", "numpy", "scikit-learn", "joblib", "typer"]
        for package in critical_packages:
            self.assertEqual(deps_status["checked_dependencies"][package], "available")
    
    @patch("builtins.__import__")
    def test_check_dependencies_missing_package(self, mock_import):
        """Test dependencies check when a critical package is missing."""
        def side_effect(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return MagicMock()
        
        mock_import.side_effect = side_effect
        
        deps_status = self.health_checker.check_dependencies()
        
        self.assertFalse(deps_status["dependencies_available"])
        self.assertGreater(len(deps_status["errors"]), 0)
        self.assertIn("Missing package: pandas", deps_status["errors"][0])
        self.assertEqual(deps_status["checked_dependencies"]["pandas"], "missing")
    
    def test_get_comprehensive_health_healthy(self):
        """Test comprehensive health check with healthy system."""
        # Set up healthy conditions
        models_dir = self.path_config.get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "test_model.pkl").write_text("dummy model")
        
        # Create data directories
        for dir_path in [
            self.path_config.get_data_dir(),
            self.path_config.get_raw_data_dir(),
            self.path_config.get_processed_data_dir()
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        health = self.health_checker.get_comprehensive_health()
        
        self.assertEqual(health["overall_status"], "healthy")
        self.assertEqual(health["summary"]["healthy_checks"], 5)
        self.assertEqual(health["summary"]["total_checks"], 5)
        self.assertEqual(len(health["summary"]["errors"]), 0)
        
        # Check individual check results
        self.assertEqual(health["checks"]["basic"]["status"], "healthy")
        self.assertTrue(health["checks"]["model"]["model_available"])
        self.assertTrue(health["checks"]["data_directories"]["data_dirs_accessible"])
        self.assertTrue(health["checks"]["dependencies"]["dependencies_available"])
    
    def test_get_comprehensive_health_degraded(self):
        """Test comprehensive health check with degraded system."""
        # Set up degraded conditions (missing model, but other things work)
        for dir_path in [
            self.path_config.get_data_dir(),
            self.path_config.get_raw_data_dir(),
            self.path_config.get_processed_data_dir(),
            self.path_config.get_models_dir()
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        # Don't create model file - this will cause degraded status
        
        health = self.health_checker.get_comprehensive_health()
        
        self.assertEqual(health["overall_status"], "degraded")
        self.assertEqual(health["summary"]["healthy_checks"], 4)  # 4/5 checks pass
        self.assertEqual(health["summary"]["total_checks"], 5)
        self.assertGreater(len(health["summary"]["errors"]), 0)
    
    def test_get_readiness_check_ready(self):
        """Test readiness check when application is ready."""
        # Set up ready conditions
        models_dir = self.path_config.get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "test_model.pkl").write_text("dummy model")
        
        for dir_path in [
            self.path_config.get_data_dir(),
            self.path_config.get_raw_data_dir(),
            self.path_config.get_processed_data_dir()
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        readiness = self.health_checker.get_readiness_check()
        
        self.assertTrue(readiness["ready"])
        self.assertTrue(readiness["checks"]["model_available"])
        self.assertTrue(readiness["checks"]["data_accessible"])
        self.assertTrue(readiness["checks"]["dependencies_ready"])
    
    def test_get_readiness_check_not_ready(self):
        """Test readiness check when application is not ready."""
        # Don't set up directories/models - should fail readiness
        readiness = self.health_checker.get_readiness_check()
        
        self.assertFalse(readiness["ready"])
        self.assertFalse(readiness["checks"]["model_available"])
        self.assertFalse(readiness["checks"]["data_accessible"])


class TestHealthCheckFunctions(unittest.TestCase):
    """Test module-level health check functions."""
    
    def test_get_health_status(self):
        """Test basic health status function."""
        health = get_health_status()
        
        self.assertIn("status", health)
        self.assertIn("timestamp", health)
        self.assertIn("uptime_seconds", health)
        self.assertEqual(health["status"], "healthy")
    
    def test_get_comprehensive_health(self):
        """Test comprehensive health function."""
        health = get_comprehensive_health()
        
        self.assertIn("overall_status", health)
        self.assertIn("checks", health)
        self.assertIn("summary", health)
        
        # Should have all expected check categories
        expected_checks = ["basic", "model", "data_directories", "configuration", "dependencies"]
        for check in expected_checks:
            self.assertIn(check, health["checks"])
    
    def test_get_readiness_status(self):
        """Test readiness status function."""
        readiness = get_readiness_status()
        
        self.assertIn("ready", readiness)
        self.assertIn("timestamp", readiness)
        self.assertIn("checks", readiness)
        
        # Should have essential readiness checks
        expected_checks = ["model_available", "data_accessible", "dependencies_ready"]
        for check in expected_checks:
            self.assertIn(check, readiness["checks"])
    
    def test_is_healthy_true(self):
        """Test is_healthy function returns True for healthy system."""
        with patch("src.health_check.get_comprehensive_health") as mock_health:
            mock_health.return_value = {"overall_status": "healthy"}
            
            self.assertTrue(is_healthy())
    
    def test_is_healthy_degraded(self):
        """Test is_healthy function returns True for degraded system."""
        with patch("src.health_check.get_comprehensive_health") as mock_health:
            mock_health.return_value = {"overall_status": "degraded"}
            
            self.assertTrue(is_healthy())
    
    def test_is_healthy_false(self):
        """Test is_healthy function returns False for unhealthy system."""
        with patch("src.health_check.get_comprehensive_health") as mock_health:
            mock_health.return_value = {"overall_status": "unhealthy"}
            
            self.assertFalse(is_healthy())
    
    def test_is_healthy_exception(self):
        """Test is_healthy function returns False when exception occurs."""
        with patch("src.health_check.get_comprehensive_health") as mock_health:
            mock_health.side_effect = Exception("Test error")
            
            self.assertFalse(is_healthy())
    
    def test_is_ready_true(self):
        """Test is_ready function returns True when ready."""
        with patch("src.health_check.get_readiness_status") as mock_ready:
            mock_ready.return_value = {"ready": True}
            
            self.assertTrue(is_ready())
    
    def test_is_ready_false(self):
        """Test is_ready function returns False when not ready."""
        with patch("src.health_check.get_readiness_status") as mock_ready:
            mock_ready.return_value = {"ready": False}
            
            self.assertFalse(is_ready())
    
    def test_is_ready_exception(self):
        """Test is_ready function returns False when exception occurs."""
        with patch("src.health_check.get_readiness_status") as mock_ready:
            mock_ready.side_effect = Exception("Test error")
            
            self.assertFalse(is_ready())


class TestHealthCheckIntegration(unittest.TestCase):
    """Integration tests for health check system."""
    
    def test_health_check_with_real_dependencies(self):
        """Test health check with real dependencies."""
        # This test verifies that health checks work with actual dependencies
        health = get_comprehensive_health()
        
        # Should be able to run without errors
        self.assertIsInstance(health, dict)
        self.assertIn("overall_status", health)
        
        # Dependencies should be available (since we're running tests)
        deps_check = health["checks"]["dependencies"]
        self.assertTrue(deps_check["dependencies_available"])
    
    def test_readiness_check_realistic(self):
        """Test readiness check in realistic conditions."""
        readiness = get_readiness_status()
        
        # Should be able to run without errors
        self.assertIsInstance(readiness, dict)
        self.assertIn("ready", readiness)
        self.assertIn("checks", readiness)
        
        # Dependencies should be ready
        self.assertTrue(readiness["checks"]["dependencies_ready"])
    
    def test_health_checker_error_handling(self):
        """Test health checker handles errors gracefully."""
        # Test with invalid path config
        with patch("src.health_check.PathConfig") as mock_config:
            mock_config.side_effect = Exception("Configuration error")
            
            # Should not raise exception, but return error status
            health = get_comprehensive_health()
            self.assertIsInstance(health, dict)


if __name__ == "__main__":
    unittest.main()