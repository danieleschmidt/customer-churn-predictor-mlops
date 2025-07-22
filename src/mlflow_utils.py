"""
MLflow utilities for artifact downloading and management.

This module provides centralized utilities for downloading MLflow artifacts,
eliminating code duplication across the codebase and providing consistent
error handling and logging.
"""

import os
import tempfile
from typing import Any, Optional, Union, Dict
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .logging_config import get_logger
from .constants import MODEL_ARTIFACT_PATH
from .validation import safe_read_json

logger = get_logger(__name__)


class MLflowError(Exception):
    """Raised when MLflow operations fail."""
    pass


class MLflowNotAvailableError(MLflowError):
    """Raised when MLflow is not available/installed."""
    pass


def _ensure_mlflow_available() -> None:
    """Ensure MLflow is available and raise error if not."""
    if not MLFLOW_AVAILABLE:
        raise MLflowNotAvailableError(
            "MLflow is not available. Please install it with: pip install mlflow"
        )


def download_model_from_mlflow(
    run_id: str, 
    model_artifact_path: str = MODEL_ARTIFACT_PATH
) -> Any:
    """
    Download a scikit-learn model from MLflow.
    
    Args:
        run_id: MLflow run ID to download from
        model_artifact_path: Path to model artifact in MLflow (default: MODEL_ARTIFACT_PATH)
        
    Returns:
        Loaded scikit-learn model
        
    Raises:
        MLflowError: If download fails
        MLflowNotAvailableError: If MLflow is not available
        ValueError: If run_id is invalid
    """
    _ensure_mlflow_available()
    
    if not run_id or not isinstance(run_id, str):
        raise ValueError(f"run_id must be a non-empty string, got: {run_id}")
    
    if not model_artifact_path:
        raise ValueError("model_artifact_path must be provided")
    
    try:
        logger.info(f"Downloading model from MLflow run {run_id}...")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_artifact_path}")
        logger.info("Model downloaded successfully from MLflow")
        return model
        
    except Exception as e:
        error_msg = f"Error downloading model from MLflow run {run_id}: {e}"
        logger.error(error_msg)
        raise MLflowError(error_msg) from e


def download_artifact_from_mlflow(
    run_id: str,
    artifact_path: str,
    destination_path: Optional[str] = None
) -> str:
    """
    Download an artifact from MLflow.
    
    Args:
        run_id: MLflow run ID to download from
        artifact_path: Path to artifact in MLflow
        destination_path: Local path to save artifact (optional)
        
    Returns:
        Path to downloaded artifact
        
    Raises:
        MLflowError: If download fails
        MLflowNotAvailableError: If MLflow is not available
        ValueError: If parameters are invalid
    """
    _ensure_mlflow_available()
    
    if not run_id or not isinstance(run_id, str):
        raise ValueError(f"run_id must be a non-empty string, got: {run_id}")
    
    if not artifact_path:
        raise ValueError("artifact_path must be provided")
    
    try:
        logger.info(f"Downloading artifact '{artifact_path}' from MLflow run {run_id}...")
        
        # Download to specified path or temporary location
        if destination_path:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            downloaded_path = mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{artifact_path}",
                dst_path=os.path.dirname(destination_path)
            )
            # Move to exact destination if needed
            if downloaded_path != destination_path:
                os.rename(downloaded_path, destination_path)
                downloaded_path = destination_path
        else:
            downloaded_path = mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{artifact_path}"
            )
        
        logger.info(f"Artifact downloaded successfully to: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        error_msg = f"Error downloading artifact '{artifact_path}' from MLflow run {run_id}: {e}"
        logger.error(error_msg)
        raise MLflowError(error_msg) from e


def download_preprocessor_from_mlflow(
    run_id: str,
    preprocessor_artifact_path: str = "preprocessor.joblib",
    destination_path: Optional[str] = None
) -> str:
    """
    Download preprocessor artifact from MLflow.
    
    Args:
        run_id: MLflow run ID to download from
        preprocessor_artifact_path: Path to preprocessor in MLflow
        destination_path: Local path to save preprocessor (optional)
        
    Returns:
        Path to downloaded preprocessor
        
    Raises:
        MLflowError: If download fails
        MLflowNotAvailableError: If MLflow is not available
    """
    return download_artifact_from_mlflow(
        run_id=run_id,
        artifact_path=preprocessor_artifact_path,
        destination_path=destination_path
    )


def download_feature_columns_from_mlflow(
    run_id: str,
    feature_columns_artifact_path: str = "feature_columns.json",
    destination_path: Optional[str] = None
) -> Dict:
    """
    Download feature columns JSON from MLflow and return parsed content.
    
    Args:
        run_id: MLflow run ID to download from
        feature_columns_artifact_path: Path to feature columns JSON in MLflow
        destination_path: Local path to save JSON (optional)
        
    Returns:
        Parsed JSON content as dictionary
        
    Raises:
        MLflowError: If download fails
        MLflowNotAvailableError: If MLflow is not available
    """
    downloaded_path = download_artifact_from_mlflow(
        run_id=run_id,
        artifact_path=feature_columns_artifact_path,
        destination_path=destination_path
    )
    
    try:
        # Use secure JSON reading
        feature_columns = safe_read_json(downloaded_path)
        logger.debug(f"Loaded {len(feature_columns)} feature columns from MLflow")
        return feature_columns
        
    except Exception as e:
        error_msg = f"Error parsing feature columns JSON from {downloaded_path}: {e}"
        logger.error(error_msg)
        raise MLflowError(error_msg) from e


def log_evaluation_metrics(
    metrics: Dict[str, Union[float, int]],
    run_name: str = "evaluation",
    artifacts: Optional[Dict[str, str]] = None
) -> str:
    """
    Log evaluation metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        run_name: Name for the MLflow run
        artifacts: Optional dictionary of artifact names and file paths to log
        
    Returns:
        MLflow run ID
        
    Raises:
        MLflowError: If logging fails
        MLflowNotAvailableError: If MLflow is not available
    """
    _ensure_mlflow_available()
    
    if not metrics or not isinstance(metrics, dict):
        raise ValueError("metrics must be a non-empty dictionary")
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    logger.warning(f"Skipping non-numeric metric {metric_name}: {metric_value}")
                    continue
                mlflow.log_metric(metric_name, metric_value)
                logger.debug(f"Logged metric {metric_name}: {metric_value}")
            
            # Log artifacts if provided
            if artifacts:
                for artifact_name, file_path in artifacts.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path)
                        logger.debug(f"Logged artifact {artifact_name}: {file_path}")
                    else:
                        logger.warning(f"Artifact file not found, skipping {artifact_name}: {file_path}")
            
            run_id = run.info.run_id
            logger.info(f"Logged evaluation metrics to MLflow run {run_id}")
            return run_id
            
    except Exception as e:
        error_msg = f"Error logging metrics to MLflow: {e}"
        logger.error(error_msg)
        raise MLflowError(error_msg) from e


class MLflowArtifactManager:
    """
    Context manager for MLflow artifact operations.
    
    Provides a convenient interface for downloading multiple artifacts
    from the same run with proper cleanup and error handling.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize artifact manager for specific run.
        
        Args:
            run_id: MLflow run ID to work with
        """
        self.run_id = run_id
        self.downloaded_artifacts = []
        self.temp_dir = None
        
    def __enter__(self):
        """Enter context manager."""
        _ensure_mlflow_available()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup."""
        # Clean up temporary files if any
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")
    
    def download_model(self, model_artifact_path: str = MODEL_ARTIFACT_PATH) -> Any:
        """Download model using this manager's run ID."""
        return download_model_from_mlflow(self.run_id, model_artifact_path)
    
    def download_preprocessor(
        self, 
        preprocessor_artifact_path: str = "preprocessor.joblib",
        destination_path: Optional[str] = None
    ) -> str:
        """Download preprocessor using this manager's run ID."""
        return download_preprocessor_from_mlflow(
            self.run_id, preprocessor_artifact_path, destination_path
        )
    
    def download_feature_columns(
        self,
        feature_columns_artifact_path: str = "feature_columns.json",
        destination_path: Optional[str] = None
    ) -> Dict:
        """Download feature columns using this manager's run ID."""
        return download_feature_columns_from_mlflow(
            self.run_id, feature_columns_artifact_path, destination_path
        )
    
    def download_artifact(
        self,
        artifact_path: str,
        destination_path: Optional[str] = None
    ) -> str:
        """Download any artifact using this manager's run ID."""
        return download_artifact_from_mlflow(
            self.run_id, artifact_path, destination_path
        )


def is_mlflow_available() -> bool:
    """
    Check if MLflow is available.
    
    Returns:
        True if MLflow is available, False otherwise
    """
    return MLFLOW_AVAILABLE


__all__ = [
    'MLflowError',
    'MLflowNotAvailableError',
    'MLflowArtifactManager',
    'download_model_from_mlflow',
    'download_artifact_from_mlflow', 
    'download_preprocessor_from_mlflow',
    'download_feature_columns_from_mlflow',
    'log_evaluation_metrics',
    'is_mlflow_available'
]