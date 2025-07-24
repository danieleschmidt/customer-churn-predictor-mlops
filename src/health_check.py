"""
Health check functionality for the churn prediction application.

This module provides health check endpoints for monitoring application status,
system health, and dependency availability. Used for container orchestration
and monitoring systems.
"""

import json
import time
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .logging_config import get_logger
from .validation import safe_read_json
from .path_config import PathConfig

logger = get_logger(__name__)


class HealthChecker:
    """
    Health checker for the churn prediction application.
    
    Provides comprehensive health checks including:
    - Basic application readiness
    - Model availability
    - Data directory accessibility
    - Configuration validity
    - System resource checks
    """
    
    def __init__(self, path_config: Optional[PathConfig] = None):
        """
        Initialize health checker.
        
        Args:
            path_config: Optional PathConfig instance for path validation
        """
        self.path_config = path_config or PathConfig()
        self.start_time = time.time()
    
    def check_basic_health(self) -> Dict[str, Any]:
        """
        Perform basic health check.
        
        Returns:
            Dict containing basic health status
        """
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "service": "churn-predictor",
            "version": "1.0.0"
        }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """
        Check if trained model files are available.
        
        Returns:
            Dict containing model availability status
        """
        model_status = {
            "model_available": False,
            "model_path": None,
            "model_age_hours": None,
            "error": None
        }
        
        try:
            models_dir = self.path_config.get_models_dir()
            if not models_dir.exists():
                model_status["error"] = "Models directory does not exist"
                return model_status
            
            # Look for model files
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            
            if not model_files:
                model_status["error"] = "No model files found"
                return model_status
            
            # Get most recent model
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            model_age = (time.time() - latest_model.stat().st_mtime) / 3600  # hours
            
            model_status.update({
                "model_available": True,
                "model_path": str(latest_model),
                "model_age_hours": round(model_age, 2)
            })
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            model_status["error"] = str(e)
        
        return model_status
    
    def check_data_directories(self) -> Dict[str, Any]:
        """
        Check accessibility of data directories.
        
        Returns:
            Dict containing data directory status
        """
        directories_status = {
            "data_dirs_accessible": True,
            "directories": {},
            "errors": []
        }
        
        try:
            # Check key directories
            dirs_to_check = {
                "data_dir": self.path_config.get_data_dir(),
                "raw_data_dir": self.path_config.get_raw_data_dir(),
                "processed_data_dir": self.path_config.get_processed_data_dir(),
                "models_dir": self.path_config.get_models_dir()
            }
            
            for name, path in dirs_to_check.items():
                try:
                    status = {
                        "exists": path.exists(),
                        "readable": os.access(path, os.R_OK) if path.exists() else False,
                        "writable": os.access(path, os.W_OK) if path.exists() else False,
                        "path": str(path)
                    }
                    directories_status["directories"][name] = status
                    
                    if not status["exists"] or not status["readable"]:
                        directories_status["data_dirs_accessible"] = False
                        directories_status["errors"].append(f"{name}: Not accessible")
                        
                except Exception as e:
                    directories_status["data_dirs_accessible"] = False
                    directories_status["errors"].append(f"{name}: {str(e)}")
                    directories_status["directories"][name] = {"error": str(e)}
        
        except Exception as e:
            logger.error(f"Error checking data directories: {e}")
            directories_status["data_dirs_accessible"] = False
            directories_status["errors"].append(f"General error: {str(e)}")
        
        return directories_status
    
    def check_configuration(self) -> Dict[str, Any]:
        """
        Check application configuration validity.
        
        Returns:
            Dict containing configuration status
        """
        config_status = {
            "config_valid": True,
            "config_file_exists": False,
            "config_readable": False,
            "errors": []
        }
        
        try:
            config_file = Path("config.yml")
            config_status["config_file_exists"] = config_file.exists()
            
            if config_file.exists():
                config_status["config_readable"] = os.access(config_file, os.R_OK)
                
                if not config_status["config_readable"]:
                    config_status["config_valid"] = False
                    config_status["errors"].append("Configuration file not readable")
            else:
                # Config file is optional, but note its absence
                config_status["errors"].append("Configuration file not found (using defaults)")
        
        except Exception as e:
            logger.error(f"Error checking configuration: {e}")
            config_status["config_valid"] = False
            config_status["errors"].append(f"Configuration check error: {str(e)}")
        
        return config_status
    
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check critical dependencies availability.
        
        Returns:
            Dict containing dependency status
        """
        dependencies_status = {
            "dependencies_available": True,
            "checked_dependencies": {},
            "errors": []
        }
        
        # Check critical Python packages
        critical_packages = [
            "pandas", "numpy", "scikit-learn", 
            "joblib", "typer"
        ]
        
        for package in critical_packages:
            try:
                __import__(package)
                dependencies_status["checked_dependencies"][package] = "available"
            except ImportError:
                dependencies_status["dependencies_available"] = False
                dependencies_status["errors"].append(f"Missing package: {package}")
                dependencies_status["checked_dependencies"][package] = "missing"
        
        # Check optional MLflow
        try:
            __import__("mlflow")
            dependencies_status["checked_dependencies"]["mlflow"] = "available"
        except ImportError:
            dependencies_status["checked_dependencies"]["mlflow"] = "missing (optional)"
        
        return dependencies_status
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health check results.
        
        Returns:
            Dict containing all health check results
        """
        logger.info("Performing comprehensive health check")
        
        health_results = {
            "overall_status": "healthy",
            "checks": {
                "basic": self.check_basic_health(),
                "model": self.check_model_availability(),
                "data_directories": self.check_data_directories(),
                "configuration": self.check_configuration(),
                "dependencies": self.check_dependencies()
            },
            "summary": {
                "healthy_checks": 0,
                "total_checks": 5,
                "errors": []
            }
        }
        
        # Determine overall health
        checks = health_results["checks"]
        errors = []
        
        # Basic check is always healthy if we get here
        health_results["summary"]["healthy_checks"] += 1
        
        # Model check
        if checks["model"]["model_available"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.append(f"Model check failed: {checks['model'].get('error', 'Unknown error')}")
        
        # Data directories check
        if checks["data_directories"]["data_dirs_accessible"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["data_directories"]["errors"])
        
        # Configuration check
        if checks["configuration"]["config_valid"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["configuration"]["errors"])
        
        # Dependencies check
        if checks["dependencies"]["dependencies_available"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["dependencies"]["errors"])
        
        # Set overall status
        if health_results["summary"]["healthy_checks"] < 3:  # At least 3/5 checks must pass
            health_results["overall_status"] = "unhealthy"
        elif health_results["summary"]["healthy_checks"] < 5:
            health_results["overall_status"] = "degraded"
        
        health_results["summary"]["errors"] = errors
        
        logger.info(f"Health check completed: {health_results['overall_status']} "
                   f"({health_results['summary']['healthy_checks']}/5 checks passed)")
        
        return health_results
    
    def get_readiness_check(self) -> Dict[str, Any]:
        """
        Get readiness check for container orchestration.
        
        Returns:
            Dict containing readiness status (focused on essential services)
        """
        readiness = {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Essential checks for readiness
        model_check = self.check_model_availability()
        data_check = self.check_data_directories()
        deps_check = self.check_dependencies()
        
        readiness["checks"] = {
            "model_available": model_check["model_available"],
            "data_accessible": data_check["data_dirs_accessible"],
            "dependencies_ready": deps_check["dependencies_available"]
        }
        
        # Ready only if all essential checks pass
        readiness["ready"] = all(readiness["checks"].values())
        
        return readiness


def get_health_status() -> Dict[str, Any]:
    """
    Get basic health status.
    
    Returns:
        Dict containing health status
    """
    checker = HealthChecker()
    return checker.check_basic_health()


def get_comprehensive_health() -> Dict[str, Any]:
    """
    Get comprehensive health status.
    
    Returns:
        Dict containing comprehensive health status
    """
    checker = HealthChecker()
    return checker.get_comprehensive_health()


def get_readiness_status() -> Dict[str, Any]:
    """
    Get readiness status for container orchestration.
    
    Returns:
        Dict containing readiness status
    """
    checker = HealthChecker()
    return checker.get_readiness_check()


def is_healthy() -> bool:
    """
    Simple boolean health check.
    
    Returns:
        True if application is healthy, False otherwise
    """
    try:
        health = get_comprehensive_health()
        return health["overall_status"] in ["healthy", "degraded"]
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def is_ready() -> bool:
    """
    Simple boolean readiness check.
    
    Returns:
        True if application is ready, False otherwise
    """
    try:
        readiness = get_readiness_status()
        return readiness["ready"]
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False