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
import platform
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging_config import get_logger
from .validation import safe_read_json
from .path_config import PathConfig
from .metrics import get_metrics_collector

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
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """
        Check database connectivity (if applicable).
        
        Returns:
            Dict containing database connectivity status
        """
        db_status = {
            "database_connected": True,
            "database_type": "file_based",
            "connection_details": {},
            "errors": []
        }
        
        try:
            # For this MLOps system, check if MLflow tracking store is accessible
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            
            if mlflow_uri.startswith("file:"):
                # File-based tracking store
                tracking_path = Path(mlflow_uri.replace("file:", ""))
                db_status["connection_details"] = {
                    "tracking_uri": mlflow_uri,
                    "path_exists": tracking_path.exists(),
                    "path_writable": os.access(tracking_path.parent, os.W_OK) if tracking_path.parent.exists() else False
                }
                
                if not tracking_path.parent.exists():
                    db_status["database_connected"] = False
                    db_status["errors"].append("MLflow tracking directory parent not accessible")
                    
            elif mlflow_uri.startswith(("http://", "https://")):
                # Remote MLflow server
                db_status["database_type"] = "remote_mlflow"
                db_status["connection_details"] = {"tracking_uri": mlflow_uri}
                
                try:
                    # Try to connect to MLflow server
                    import mlflow
                    mlflow.set_tracking_uri(mlflow_uri)
                    # Simple check - list experiments (should not raise if connected)
                    experiments = mlflow.search_experiments(max_results=1)
                    db_status["connection_details"]["experiments_accessible"] = True
                except Exception as e:
                    db_status["database_connected"] = False
                    db_status["errors"].append(f"MLflow server connection failed: {str(e)}")
            else:
                # Other database types (PostgreSQL, MySQL, etc.)
                db_status["database_type"] = "external"
                db_status["connection_details"] = {"tracking_uri": mlflow_uri}
                # For now, assume connection is okay if URI is configured
                
        except Exception as e:
            logger.error(f"Error checking database connectivity: {e}")
            db_status["database_connected"] = False
            db_status["errors"].append(f"Database check error: {str(e)}")
        
        return db_status
    
    def check_mlflow_service(self) -> Dict[str, Any]:
        """
        Check MLflow service availability and status.
        
        Returns:
            Dict containing MLflow service status
        """
        mlflow_status = {
            "mlflow_available": False,
            "mlflow_version": None,
            "tracking_uri": None,
            "experiments_count": 0,
            "latest_run": None,
            "errors": []
        }
        
        try:
            # Check if MLflow is importable
            import mlflow
            mlflow_status["mlflow_version"] = mlflow.__version__
            mlflow_status["mlflow_available"] = True
            
            # Get tracking URI
            tracking_uri = mlflow.get_tracking_uri()
            mlflow_status["tracking_uri"] = tracking_uri
            
            # Try to access experiments
            try:
                experiments = mlflow.search_experiments()
                mlflow_status["experiments_count"] = len(experiments)
                
                # Get latest run info if available
                if experiments:
                    latest_experiment = experiments[0]
                    runs = mlflow.search_runs(
                        experiment_ids=[latest_experiment.experiment_id],
                        max_results=1,
                        order_by=["start_time DESC"]
                    )
                    
                    if not runs.empty:
                        latest_run = runs.iloc[0]
                        mlflow_status["latest_run"] = {
                            "run_id": latest_run["run_id"],
                            "status": latest_run["status"],
                            "start_time": str(latest_run["start_time"]),
                            "experiment_id": latest_run["experiment_id"]
                        }
                        
            except Exception as e:
                mlflow_status["errors"].append(f"MLflow tracking access error: {str(e)}")
                
        except ImportError:
            mlflow_status["errors"].append("MLflow not installed or not importable")
        except Exception as e:
            logger.error(f"Error checking MLflow service: {e}")
            mlflow_status["errors"].append(f"MLflow service check error: {str(e)}")
        
        return mlflow_status
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """
        Check system resource usage metrics.
        
        Returns:
            Dict containing resource usage information
        """
        resource_status = {
            "resources_healthy": True,
            "cpu": {},
            "memory": {},
            "disk": {},
            "system": {},
            "warnings": []
        }
        
        if not PSUTIL_AVAILABLE:
            resource_status["warnings"].append("psutil not available - resource monitoring limited")
            resource_status["system"] = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "psutil_available": False
            }
            # Still consider healthy if psutil is missing (graceful degradation)
            return resource_status
        
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            resource_status["cpu"] = {
                "usage_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            resource_status["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent
            }
            
            # Disk information (for current working directory)
            disk = psutil.disk_usage('.')
            resource_status["disk"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            }
            
            # System information
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            resource_status["system"] = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "boot_time": boot_time.isoformat(),
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2),
                "psutil_available": True
            }
            
            # Check for resource warnings
            if cpu_percent > 80:
                resource_status["warnings"].append(f"High CPU usage: {cpu_percent}%")
                resource_status["resources_healthy"] = False
                
            if memory.percent > 85:
                resource_status["warnings"].append(f"High memory usage: {memory.percent}%")
                resource_status["resources_healthy"] = False
                
            if resource_status["disk"]["usage_percent"] > 90:
                resource_status["warnings"].append(f"High disk usage: {resource_status['disk']['usage_percent']}%")
                resource_status["resources_healthy"] = False
                
        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")
            resource_status["resources_healthy"] = False
            resource_status["warnings"].append(f"Resource check error: {str(e)}")
        
        return resource_status
    
    def check_dependency_versions(self) -> Dict[str, Any]:
        """
        Check and report versions of critical dependencies.
        
        Returns:
            Dict containing dependency version information
        """
        version_status = {
            "versions_collected": True,
            "package_versions": {},
            "version_conflicts": [],
            "outdated_packages": [],
            "errors": []
        }
        
        # Critical packages to check
        critical_packages = [
            "pandas", "numpy", "scikit-learn", "joblib", 
            "typer", "fastapi", "uvicorn", "pydantic"
        ]
        
        # Optional packages
        optional_packages = ["mlflow", "pytest", "black", "flake8", "mypy"]
        
        try:
            for package in critical_packages + optional_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    version_status["package_versions"][package] = {
                        "version": version,
                        "status": "installed",
                        "required": package in critical_packages
                    }
                except ImportError:
                    version_status["package_versions"][package] = {
                        "version": None,
                        "status": "not_installed",
                        "required": package in critical_packages
                    }
                    if package in critical_packages:
                        version_status["errors"].append(f"Critical package missing: {package}")
                        
            # Check Python version compatibility
            python_version = platform.python_version()
            version_status["python_version"] = python_version
            
            # Basic version conflict detection (simplified)
            if version_status["package_versions"].get("pandas", {}).get("version", "").startswith("2."):
                numpy_version = version_status["package_versions"].get("numpy", {}).get("version", "")
                if numpy_version and numpy_version < "1.20":
                    version_status["version_conflicts"].append("Pandas 2.x requires NumPy >= 1.20")
                    
        except Exception as e:
            logger.error(f"Error checking dependency versions: {e}")
            version_status["versions_collected"] = False
            version_status["errors"].append(f"Version check error: {str(e)}")
        
        return version_status
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health check results.
        
        Returns:
            Dict containing all health check results
        """
        logger.info("Performing comprehensive health check")
        start_time = time.time()
        
        health_results = {
            "overall_status": "healthy",
            "checks": {
                "basic": self.check_basic_health(),
                "model": self.check_model_availability(),
                "data_directories": self.check_data_directories(),
                "configuration": self.check_configuration(),
                "dependencies": self.check_dependencies(),
                "database": self.check_database_connectivity(),
                "mlflow_service": self.check_mlflow_service(),
                "resource_usage": self.check_resource_usage(),
                "dependency_versions": self.check_dependency_versions()
            },
            "summary": {
                "healthy_checks": 0,
                "total_checks": 9,
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
        
        # Database connectivity check
        if checks["database"]["database_connected"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["database"]["errors"])
        
        # MLflow service check
        if checks["mlflow_service"]["mlflow_available"] and not checks["mlflow_service"]["errors"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["mlflow_service"]["errors"])
        
        # Resource usage check
        if checks["resource_usage"]["resources_healthy"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["resource_usage"]["warnings"])
        
        # Dependency versions check
        if checks["dependency_versions"]["versions_collected"] and not checks["dependency_versions"]["errors"]:
            health_results["summary"]["healthy_checks"] += 1
        else:
            errors.extend(checks["dependency_versions"]["errors"])
        
        # Set overall status (now with 9 total checks)
        if health_results["summary"]["healthy_checks"] < 5:  # At least 5/9 checks must pass
            health_results["overall_status"] = "unhealthy"
        elif health_results["summary"]["healthy_checks"] < 7:
            health_results["overall_status"] = "degraded"
        
        health_results["summary"]["errors"] = errors
        
        logger.info(f"Health check completed: {health_results['overall_status']} "
                   f"({health_results['summary']['healthy_checks']}/9 checks passed)")
        
        # Record health check metrics
        try:
            metrics_collector = get_metrics_collector()
            duration = time.time() - start_time
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