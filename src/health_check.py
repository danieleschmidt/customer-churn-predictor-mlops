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
import sys
import platform
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .logging_config import get_logger
from .validation import safe_read_json
from .path_config import PathConfig
from .metrics import get_metrics_collector

# Optional dependencies for enhanced monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

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
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """
        Check system resource usage.
        
        Returns:
            Dict containing resource usage information
        """
        resource_status = {
            "resources_healthy": True,
            "cpu_percent": None,
            "memory_percent": None,
            "disk_usage": {},
            "load_average": None,
            "errors": []
        }
        
        if not HAS_PSUTIL:
            resource_status["resources_healthy"] = False
            resource_status["errors"].append("psutil not available for resource monitoring")
            return resource_status
        
        try:
            # CPU usage (averaged over 1 second)
            cpu_percent = psutil.cpu_percent(interval=1)
            resource_status["cpu_percent"] = round(cpu_percent, 2)
            
            # Memory usage
            memory = psutil.virtual_memory()
            resource_status["memory_percent"] = round(memory.percent, 2)
            
            # Disk usage for current directory
            disk = psutil.disk_usage('.')
            resource_status["disk_usage"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round((disk.used / disk.total) * 100, 2)
            }
            
            # Load average (Unix systems only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                resource_status["load_average"] = {
                    "1min": round(load_avg[0], 2),
                    "5min": round(load_avg[1], 2),
                    "15min": round(load_avg[2], 2)
                }
            
            # Check for resource thresholds
            if cpu_percent > 90:
                resource_status["resources_healthy"] = False
                resource_status["errors"].append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                resource_status["resources_healthy"] = False
                resource_status["errors"].append(f"High memory usage: {memory.percent}%")
            
            if resource_status["disk_usage"]["percent"] > 90:
                resource_status["resources_healthy"] = False
                resource_status["errors"].append(f"High disk usage: {resource_status['disk_usage']['percent']}%")
        
        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")
            resource_status["resources_healthy"] = False
            resource_status["errors"].append(f"Resource check error: {str(e)}")
        
        return resource_status
    
    def check_mlflow_service(self) -> Dict[str, Any]:
        """
        Check MLflow service connectivity.
        
        Returns:
            Dict containing MLflow service status
        """
        mlflow_status = {
            "service_available": False,
            "tracking_uri": None,
            "version": None,
            "experiments_accessible": False,
            "errors": []
        }
        
        if not HAS_MLFLOW:
            mlflow_status["errors"].append("MLflow not available")
            return mlflow_status
        
        try:
            # Get tracking URI
            tracking_uri = mlflow.get_tracking_uri()
            mlflow_status["tracking_uri"] = tracking_uri
            
            # Get MLflow version
            mlflow_status["version"] = mlflow.__version__
            
            # Try to access experiments (lightweight check)
            try:
                client = mlflow.tracking.MlflowClient()
                experiments = client.search_experiments(max_results=1)
                mlflow_status["experiments_accessible"] = True
                mlflow_status["service_available"] = True
            except Exception as e:
                mlflow_status["errors"].append(f"Cannot access experiments: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error checking MLflow service: {e}")
            mlflow_status["errors"].append(f"MLflow check error: {str(e)}")
        
        return mlflow_status
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """
        Check database connectivity (placeholder for future database integration).
        
        Returns:
            Dict containing database connectivity status
        """
        db_status = {
            "configured": False,
            "connected": False,
            "connection_pool_healthy": False,
            "database_name": None,
            "errors": []
        }
        
        # Check for common database environment variables
        db_vars = ['DATABASE_URL', 'DB_HOST', 'POSTGRES_URL', 'MYSQL_URL', 'MONGODB_URI']
        db_configured = any(os.getenv(var) for var in db_vars)
        
        if not db_configured:
            db_status["errors"].append("No database configuration found (optional)")
            return db_status
        
        db_status["configured"] = True
        
        # Placeholder for actual database connectivity checks
        # This would be implemented when database integration is added
        db_status["errors"].append("Database connectivity check not implemented yet")
        
        return db_status
    
    def get_dependency_versions(self) -> Dict[str, str]:
        """
        Get versions of key dependencies.
        
        Returns:
            Dict mapping package names to versions
        """
        versions = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        }
        
        # Check versions of key packages
        packages_to_check = [
            "pandas", "numpy", "scikit-learn", "joblib", 
            "typer", "fastapi", "uvicorn", "mlflow", "psutil"
        ]
        
        for package in packages_to_check:
            try:
                module = __import__(package)
                if hasattr(module, "__version__"):
                    versions[package] = module.__version__
                else:
                    versions[package] = "version_unknown"
            except ImportError:
                versions[package] = "not_installed"
        
        return versions
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health check results.
        
        Returns:
            Dict containing all health check results
        """
        logger.info("Performing comprehensive health check")
        start_time = time.time()
        
        # Perform all health checks
        basic_check = self.check_basic_health()
        model_check = self.check_model_availability()
        data_check = self.check_data_directories()
        config_check = self.check_configuration()
        deps_check = self.check_dependencies()
        resource_check = self.check_resource_usage()
        mlflow_check = self.check_mlflow_service()
        db_check = self.check_database_connectivity()
        
        # Add dependency versions to dependencies check
        deps_check["versions"] = self.get_dependency_versions()
        
        health_results = {
            "overall_status": "healthy",
            "checks": {
                "basic": basic_check,
                "model": model_check,
                "data_directories": data_check,
                "configuration": config_check,
                "dependencies": deps_check,
                "resources": resource_check,
                "mlflow_service": mlflow_check,
                "database": db_check
            },
            "summary": {
                "healthy_checks": 0,
                "total_checks": 8,
                "errors": []
            },
            "performance": {
                "check_duration_ms": 0  # Will be set at the end
            }
        }
        
        # Determine overall health
        checks = health_results["checks"]
        errors = []
        
        # Core checks (essential for operation)
        core_checks = ["basic", "model", "data_directories", "dependencies"]
        # Enhanced checks (important but not critical)
        enhanced_checks = ["configuration", "resources", "mlflow_service", "database"]
        
        core_healthy = 0
        enhanced_healthy = 0
        
        for check_name in core_checks:
            check_result = checks[check_name]
            if check_name == "basic":
                # Basic check is always healthy if we get here
                core_healthy += 1
            elif check_name == "model" and check_result.get("model_available", False):
                core_healthy += 1
            elif check_name == "data_directories" and check_result.get("data_dirs_accessible", False):
                core_healthy += 1
            elif check_name == "dependencies" and check_result.get("dependencies_available", False):
                core_healthy += 1
            else:
                if "errors" in check_result and check_result["errors"]:
                    errors.extend(check_result["errors"])
                elif "error" in check_result and check_result["error"]:
                    errors.append(f"{check_name}: {check_result['error']}")
        
        for check_name in enhanced_checks:
            check_result = checks[check_name]
            if check_name == "configuration" and check_result.get("config_valid", True):
                enhanced_healthy += 1
            elif check_name == "resources" and check_result.get("resources_healthy", True):
                enhanced_healthy += 1
            elif check_name == "mlflow_service" and check_result.get("service_available", False):
                enhanced_healthy += 1
            elif check_name == "database" and check_result.get("configured", False):
                enhanced_healthy += 1
            else:
                # Enhanced check failures are warnings, not errors
                if "errors" in check_result and check_result["errors"]:
                    # Only add non-optional errors
                    for error in check_result["errors"]:
                        if "optional" not in error.lower():
                            errors.append(f"{check_name}: {error}")
        
        total_healthy = core_healthy + enhanced_healthy
        health_results["summary"]["healthy_checks"] = total_healthy
        
        # Set overall status based on core checks primarily
        if core_healthy < 3:  # At least 3/4 core checks must pass
            health_results["overall_status"] = "unhealthy"
        elif core_healthy < 4 or total_healthy < 6:
            health_results["overall_status"] = "degraded"
        else:
            health_results["overall_status"] = "healthy"
        
        health_results["summary"]["errors"] = errors
        
        # Calculate performance metrics
        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000, 2)
        health_results["performance"]["check_duration_ms"] = duration_ms
        
        logger.info(f"Health check completed: {health_results['overall_status']} "
                   f"({total_healthy}/8 checks passed, {duration_ms}ms)")
        
        # Record health check metrics
        try:
            metrics_collector = get_metrics_collector()
            duration = end_time - start_time
            metrics_collector.record_health_check_duration(duration, "detailed", health_results['overall_status'])
        except Exception as e:
            logger.warning(f"Failed to record health check metrics: {e}")
        
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