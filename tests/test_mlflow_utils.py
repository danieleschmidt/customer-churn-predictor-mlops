"""
Tests for MLflow utilities module.

This module tests the centralized MLflow artifact downloading and management
utilities that eliminate code duplication across the codebase.
"""

import os
import json
import tempfile
from unittest.mock import patch, Mock, MagicMock
import pytest

from src.mlflow_utils import (
    MLflowError,
    MLflowNotAvailableError,
    MLflowArtifactManager,
    download_model_from_mlflow,
    download_artifact_from_mlflow,
    download_preprocessor_from_mlflow,
    download_feature_columns_from_mlflow,
    log_evaluation_metrics,
    is_mlflow_available,
    _ensure_mlflow_available
)


class TestMLflowAvailability:
    """Test MLflow availability checking."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_is_mlflow_available_true(self):
        """Test MLflow availability check when available."""
        assert is_mlflow_available() is True
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_is_mlflow_available_false(self):
        """Test MLflow availability check when not available."""
        assert is_mlflow_available() is False
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_ensure_mlflow_available_when_available(self):
        """Test _ensure_mlflow_available when MLflow is available."""
        # Should not raise
        _ensure_mlflow_available()
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_ensure_mlflow_available_when_not_available(self):
        """Test _ensure_mlflow_available when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError) as excinfo:
            _ensure_mlflow_available()
        assert "MLflow is not available" in str(excinfo.value)


class TestDownloadModelFromMLflow:
    """Test model downloading from MLflow."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.sklearn.load_model')
    def test_download_model_success(self, mock_load_model):
        """Test successful model download."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        result = download_model_from_mlflow("test_run_id")
        
        assert result == mock_model
        mock_load_model.assert_called_once_with("runs:/test_run_id/churn_model")
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_download_model_mlflow_not_available(self):
        """Test model download when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            download_model_from_mlflow("test_run_id")
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_download_model_invalid_run_id(self):
        """Test model download with invalid run ID."""
        with pytest.raises(ValueError) as excinfo:
            download_model_from_mlflow("")
        assert "run_id must be a non-empty string" in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            download_model_from_mlflow(None)
        assert "run_id must be a non-empty string" in str(excinfo.value)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_download_model_invalid_artifact_path(self):
        """Test model download with invalid artifact path."""
        with pytest.raises(ValueError) as excinfo:
            download_model_from_mlflow("run_id", model_artifact_path="")
        assert "model_artifact_path must be provided" in str(excinfo.value)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.sklearn.load_model')
    def test_download_model_mlflow_error(self, mock_load_model):
        """Test model download MLflow error handling."""
        mock_load_model.side_effect = Exception("MLflow download failed")
        
        with pytest.raises(MLflowError) as excinfo:
            download_model_from_mlflow("test_run_id")
        
        assert "Error downloading model from MLflow run test_run_id" in str(excinfo.value)
        assert "MLflow download failed" in str(excinfo.value)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.sklearn.load_model')
    def test_download_model_custom_artifact_path(self, mock_load_model):
        """Test model download with custom artifact path."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        download_model_from_mlflow("test_run_id", "custom_model")
        
        mock_load_model.assert_called_once_with("runs:/test_run_id/custom_model")


class TestDownloadArtifactFromMLflow:
    """Test artifact downloading from MLflow."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.artifacts.download_artifacts')
    def test_download_artifact_success(self, mock_download):
        """Test successful artifact download."""
        mock_download.return_value = "/tmp/artifact.json"
        
        result = download_artifact_from_mlflow("test_run_id", "artifact.json")
        
        assert result == "/tmp/artifact.json"
        mock_download.assert_called_once_with("runs:/test_run_id/artifact.json")
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_download_artifact_mlflow_not_available(self):
        """Test artifact download when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            download_artifact_from_mlflow("test_run_id", "artifact.json")
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_download_artifact_invalid_parameters(self):
        """Test artifact download with invalid parameters."""
        with pytest.raises(ValueError) as excinfo:
            download_artifact_from_mlflow("", "artifact.json")
        assert "run_id must be a non-empty string" in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            download_artifact_from_mlflow("run_id", "")
        assert "artifact_path must be provided" in str(excinfo.value)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.artifacts.download_artifacts')
    @patch('os.makedirs')
    @patch('os.rename')
    def test_download_artifact_with_destination(self, mock_rename, mock_makedirs, mock_download):
        """Test artifact download with specific destination."""
        mock_download.return_value = "/tmp/downloaded_artifact.json"
        destination = "/custom/path/artifact.json"
        
        result = download_artifact_from_mlflow("test_run_id", "artifact.json", destination)
        
        assert result == destination
        mock_makedirs.assert_called_once_with("/custom/path", exist_ok=True)
        mock_download.assert_called_once_with("runs:/test_run_id/artifact.json", dst_path="/custom/path")
        mock_rename.assert_called_once_with("/tmp/downloaded_artifact.json", destination)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.artifacts.download_artifacts')
    def test_download_artifact_mlflow_error(self, mock_download):
        """Test artifact download MLflow error handling."""
        mock_download.side_effect = Exception("Artifact not found")
        
        with pytest.raises(MLflowError) as excinfo:
            download_artifact_from_mlflow("test_run_id", "missing.json")
        
        assert "Error downloading artifact 'missing.json'" in str(excinfo.value)
        assert "Artifact not found" in str(excinfo.value)


class TestDownloadPreprocessorFromMLflow:
    """Test preprocessor downloading from MLflow."""
    
    @patch('src.mlflow_utils.download_artifact_from_mlflow')
    def test_download_preprocessor_default_path(self, mock_download_artifact):
        """Test preprocessor download with default path."""
        mock_download_artifact.return_value = "/tmp/preprocessor.joblib"
        
        result = download_preprocessor_from_mlflow("test_run_id")
        
        assert result == "/tmp/preprocessor.joblib"
        mock_download_artifact.assert_called_once_with(
            run_id="test_run_id",
            artifact_path="preprocessor.joblib",
            destination_path=None
        )
    
    @patch('src.mlflow_utils.download_artifact_from_mlflow')
    def test_download_preprocessor_custom_path(self, mock_download_artifact):
        """Test preprocessor download with custom paths."""
        mock_download_artifact.return_value = "/custom/preprocessor.joblib"
        
        result = download_preprocessor_from_mlflow(
            "test_run_id",
            preprocessor_artifact_path="custom_preprocessor.joblib",
            destination_path="/custom/preprocessor.joblib"
        )
        
        assert result == "/custom/preprocessor.joblib"
        mock_download_artifact.assert_called_once_with(
            run_id="test_run_id",
            artifact_path="custom_preprocessor.joblib",
            destination_path="/custom/preprocessor.joblib"
        )


class TestDownloadFeatureColumnsFromMLflow:
    """Test feature columns downloading from MLflow."""
    
    @patch('src.mlflow_utils.download_artifact_from_mlflow')
    @patch('src.mlflow_utils.safe_read_json')
    def test_download_feature_columns_success(self, mock_safe_read, mock_download_artifact):
        """Test successful feature columns download."""
        mock_download_artifact.return_value = "/tmp/feature_columns.json"
        mock_safe_read.return_value = ["feature1", "feature2", "feature3"]
        
        result = download_feature_columns_from_mlflow("test_run_id")
        
        assert result == ["feature1", "feature2", "feature3"]
        mock_download_artifact.assert_called_once_with(
            run_id="test_run_id",
            artifact_path="feature_columns.json",
            destination_path=None
        )
        mock_safe_read.assert_called_once_with("/tmp/feature_columns.json")
    
    @patch('src.mlflow_utils.download_artifact_from_mlflow')
    @patch('src.mlflow_utils.safe_read_json')
    def test_download_feature_columns_json_error(self, mock_safe_read, mock_download_artifact):
        """Test feature columns download with JSON parsing error."""
        mock_download_artifact.return_value = "/tmp/feature_columns.json"
        mock_safe_read.side_effect = Exception("Invalid JSON")
        
        with pytest.raises(MLflowError) as excinfo:
            download_feature_columns_from_mlflow("test_run_id")
        
        assert "Error parsing feature columns JSON" in str(excinfo.value)
        assert "Invalid JSON" in str(excinfo.value)


class TestLogEvaluationMetrics:
    """Test evaluation metrics logging to MLflow."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.start_run')
    @patch('src.mlflow_utils.mlflow.log_metric')
    def test_log_metrics_success(self, mock_log_metric, mock_start_run):
        """Test successful metrics logging."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        metrics = {"accuracy": 0.95, "f1_score": 0.87}
        result = log_evaluation_metrics(metrics)
        
        assert result == "test_run_123"
        mock_start_run.assert_called_once_with(run_name="evaluation")
        
        # Check that metrics were logged
        assert mock_log_metric.call_count == 2
        mock_log_metric.assert_any_call("accuracy", 0.95)
        mock_log_metric.assert_any_call("f1_score", 0.87)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_log_metrics_mlflow_not_available(self):
        """Test metrics logging when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            log_evaluation_metrics({"accuracy": 0.95})
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_log_metrics_invalid_input(self):
        """Test metrics logging with invalid input."""
        with pytest.raises(ValueError) as excinfo:
            log_evaluation_metrics({})
        assert "metrics must be a non-empty dictionary" in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            log_evaluation_metrics(None)
        assert "metrics must be a non-empty dictionary" in str(excinfo.value)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.start_run')
    @patch('src.mlflow_utils.mlflow.log_metric')
    @patch('src.mlflow_utils.mlflow.log_artifact')
    def test_log_metrics_with_artifacts(self, mock_log_artifact, mock_log_metric, mock_start_run):
        """Test metrics logging with artifacts."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test content")
            tmp_file_path = tmp_file.name
        
        try:
            metrics = {"accuracy": 0.95}
            artifacts = {"report": tmp_file_path}
            
            result = log_evaluation_metrics(metrics, artifacts=artifacts)
            
            assert result == "test_run_123"
            mock_log_metric.assert_called_once_with("accuracy", 0.95)
            mock_log_artifact.assert_called_once_with(tmp_file_path)
            
        finally:
            os.unlink(tmp_file_path)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.start_run')
    @patch('src.mlflow_utils.mlflow.log_metric')
    def test_log_metrics_with_non_numeric_values(self, mock_log_metric, mock_start_run):
        """Test metrics logging with non-numeric values."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        metrics = {"accuracy": 0.95, "invalid": "string_value", "f1_score": 0.87}
        
        log_evaluation_metrics(metrics)
        
        # Should only log numeric metrics
        assert mock_log_metric.call_count == 2
        mock_log_metric.assert_any_call("accuracy", 0.95)
        mock_log_metric.assert_any_call("f1_score", 0.87)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.start_run')
    def test_log_metrics_mlflow_error(self, mock_start_run):
        """Test metrics logging MLflow error handling."""
        mock_start_run.side_effect = Exception("MLflow logging failed")
        
        with pytest.raises(MLflowError) as excinfo:
            log_evaluation_metrics({"accuracy": 0.95})
        
        assert "Error logging metrics to MLflow" in str(excinfo.value)
        assert "MLflow logging failed" in str(excinfo.value)


class TestMLflowArtifactManager:
    """Test MLflow artifact manager context manager."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    def test_artifact_manager_context_manager(self):
        """Test artifact manager as context manager."""
        with MLflowArtifactManager("test_run_id") as manager:
            assert manager.run_id == "test_run_id"
            assert manager.downloaded_artifacts == []
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_artifact_manager_mlflow_not_available(self):
        """Test artifact manager when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            with MLflowArtifactManager("test_run_id"):
                pass
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.download_model_from_mlflow')
    def test_artifact_manager_download_model(self, mock_download_model):
        """Test downloading model through artifact manager."""
        mock_model = MagicMock()
        mock_download_model.return_value = mock_model
        
        with MLflowArtifactManager("test_run_id") as manager:
            result = manager.download_model()
            
            assert result == mock_model
            mock_download_model.assert_called_once_with("test_run_id", "churn_model")
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.download_preprocessor_from_mlflow')
    def test_artifact_manager_download_preprocessor(self, mock_download_preprocessor):
        """Test downloading preprocessor through artifact manager."""
        mock_download_preprocessor.return_value = "/tmp/preprocessor.joblib"
        
        with MLflowArtifactManager("test_run_id") as manager:
            result = manager.download_preprocessor()
            
            assert result == "/tmp/preprocessor.joblib"
            mock_download_preprocessor.assert_called_once_with("test_run_id", "preprocessor.joblib", None)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.download_feature_columns_from_mlflow')
    def test_artifact_manager_download_feature_columns(self, mock_download_feature_columns):
        """Test downloading feature columns through artifact manager."""
        mock_download_feature_columns.return_value = ["feature1", "feature2"]
        
        with MLflowArtifactManager("test_run_id") as manager:
            result = manager.download_feature_columns()
            
            assert result == ["feature1", "feature2"]
            mock_download_feature_columns.assert_called_once_with("test_run_id", "feature_columns.json", None)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.download_artifact_from_mlflow')
    def test_artifact_manager_download_artifact(self, mock_download_artifact):
        """Test downloading generic artifact through artifact manager."""
        mock_download_artifact.return_value = "/tmp/artifact.json"
        
        with MLflowArtifactManager("test_run_id") as manager:
            result = manager.download_artifact("custom.json")
            
            assert result == "/tmp/artifact.json"
            mock_download_artifact.assert_called_once_with("test_run_id", "custom.json", None)
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_artifact_manager_cleanup(self, mock_rmtree, mock_exists):
        """Test artifact manager cleanup on exit."""
        mock_exists.return_value = True
        
        manager = MLflowArtifactManager("test_run_id")
        manager.temp_dir = "/tmp/test_dir"
        
        with manager:
            pass  # Context manager will cleanup on exit
        
        mock_rmtree.assert_called_once_with("/tmp/test_dir")


class TestMLflowExceptionClasses:
    """Test custom MLflow exception classes."""
    
    def test_mlflow_error_inheritance(self):
        """Test that MLflowError inherits from Exception."""
        error = MLflowError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"
    
    def test_mlflow_not_available_error_inheritance(self):
        """Test that MLflowNotAvailableError inherits from MLflowError."""
        error = MLflowNotAvailableError("mlflow not available")
        assert isinstance(error, MLflowError)
        assert isinstance(error, Exception)
        assert str(error) == "mlflow not available"


class TestIntegration:
    """Integration tests for MLflow utilities."""
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', True)
    @patch('src.mlflow_utils.mlflow.sklearn.load_model')
    @patch('src.mlflow_utils.mlflow.artifacts.download_artifacts')
    @patch('src.mlflow_utils.safe_read_json')
    def test_complete_artifact_workflow(self, mock_safe_read, mock_download_artifacts, mock_load_model):
        """Test complete workflow of downloading multiple artifacts."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_download_artifacts.return_value = "/tmp/artifact"
        mock_safe_read.return_value = ["feature1", "feature2"]
        
        run_id = "test_run_123"
        
        # Download all artifacts
        model = download_model_from_mlflow(run_id)
        preprocessor_path = download_preprocessor_from_mlflow(run_id)
        feature_columns = download_feature_columns_from_mlflow(run_id)
        
        # Verify results
        assert model == mock_model
        assert preprocessor_path == "/tmp/artifact"
        assert feature_columns == ["feature1", "feature2"]
        
        # Verify calls
        mock_load_model.assert_called_once_with(f"runs:/{run_id}/churn_model")
        mock_download_artifacts.assert_called()
        mock_safe_read.assert_called_once()
    
    @patch('src.mlflow_utils.MLFLOW_AVAILABLE', False)
    def test_workflow_without_mlflow(self):
        """Test that workflow fails gracefully without MLflow."""
        # All operations should raise MLflowNotAvailableError
        with pytest.raises(MLflowNotAvailableError):
            download_model_from_mlflow("run_id")
        
        with pytest.raises(MLflowNotAvailableError):
            download_artifact_from_mlflow("run_id", "artifact")
        
        with pytest.raises(MLflowNotAvailableError):
            log_evaluation_metrics({"accuracy": 0.95})


if __name__ == "__main__":
    pytest.main([__file__])