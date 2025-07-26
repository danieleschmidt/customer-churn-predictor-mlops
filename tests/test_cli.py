"""
Tests for CLI module.

This module tests the command-line interface functions that wrap
the various script operations.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from typer.testing import CliRunner

from src.cli import app
from src.validation import ValidationError


class TestCLICommands:
    """Test cases for CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.cli.run_preprocessing')
    def test_preprocess_command(self, mock_run_preprocessing):
        """Test the preprocess command."""
        mock_run_preprocessing.return_value = ("features.csv", "target.csv")
        
        result = self.runner.invoke(app, ["preprocess"])
        
        assert result.exit_code == 0
        mock_run_preprocessing.assert_called_once()
    
    @patch('src.cli.run_training')
    def test_train_command_default_params(self, mock_run_training):
        """Test the train command with default parameters."""
        result = self.runner.invoke(app, ["train"])
        
        assert result.exit_code == 0
        mock_run_training.assert_called_once_with(
            None, None,  # x_path, y_path
            solver="liblinear",
            C=1.0,
            penalty="l2",
            random_state=42,
            max_iter=100,
            test_size=0.2
        )
    
    @patch('src.cli.run_training')
    def test_train_command_custom_params(self, mock_run_training):
        """Test the train command with custom parameters."""
        result = self.runner.invoke(app, [
            "train",
            "--x-path", "custom_features.csv",
            "--y-path", "custom_target.csv", 
            "--solver", "lbfgs",
            "--c", "2.0",
            "--penalty", "l1",
            "--random-state", "123",
            "--max-iter", "200",
            "--test-size", "0.3"
        ])
        
        assert result.exit_code == 0
        mock_run_training.assert_called_once_with(
            "custom_features.csv", "custom_target.csv",
            solver="lbfgs",
            C=2.0,
            penalty="l1",
            random_state=123,
            max_iter=200,
            test_size=0.3
        )
    
    @patch('src.cli.run_evaluation')
    def test_evaluate_command_default_params(self, mock_run_evaluation):
        """Test the evaluate command with default parameters."""
        result = self.runner.invoke(app, ["evaluate"])
        
        assert result.exit_code == 0
        mock_run_evaluation.assert_called_once_with(
            model_path=None,
            X_path=None,
            y_path=None,
            output=None,
            run_id=None,
            detailed=False
        )
    
    @patch('src.cli.run_evaluation')
    def test_evaluate_command_custom_params(self, mock_run_evaluation):
        """Test the evaluate command with custom parameters."""
        result = self.runner.invoke(app, [
            "evaluate",
            "--model-path", "custom_model.joblib",
            "--x-path", "test_features.csv",
            "--y-path", "test_target.csv",
            "--run-id", "abc123",
            "--output", "results.json",
            "--detailed"
        ])
        
        assert result.exit_code == 0
        mock_run_evaluation.assert_called_once_with(
            model_path="custom_model.joblib",
            X_path="test_features.csv", 
            y_path="test_target.csv",
            output="results.json",
            run_id="abc123",
            detailed=True
        )
    
    @patch('src.cli.run_pipeline')
    def test_pipeline_command_default_params(self, mock_run_pipeline):
        """Test the pipeline command with default parameters."""
        result = self.runner.invoke(app, ["pipeline"])
        
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once_with(
            None,  # raw_path
            solver="liblinear",
            C=1.0,
            penalty="l2",
            random_state=42,
            max_iter=100,
            test_size=0.2
        )
    
    @patch('src.cli.run_pipeline')
    def test_pipeline_command_custom_params(self, mock_run_pipeline):
        """Test the pipeline command with custom parameters."""
        result = self.runner.invoke(app, [
            "pipeline",
            "--raw-path", "custom_data.csv",
            "--solver", "sag",
            "--c", "0.5",
            "--penalty", "l1"
        ])
        
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once_with(
            "custom_data.csv",
            solver="sag",
            C=0.5,
            penalty="l1",
            random_state=42,  # Default values
            max_iter=100,
            test_size=0.2
        )
    
    @patch('src.cli.monitor_and_retrain')
    def test_monitor_command_default_params(self, mock_monitor_and_retrain):
        """Test the monitor command with default parameters."""
        result = self.runner.invoke(app, ["monitor"])
        
        assert result.exit_code == 0
        # Should only pass non-None values
        expected_kwargs = {
            "solver": "liblinear",
            "C": 1.0,
            "penalty": "l2",
            "random_state": 42,
            "max_iter": 100,
            "test_size": 0.2
        }
        mock_monitor_and_retrain.assert_called_once_with(**expected_kwargs)
    
    @patch('src.cli.monitor_and_retrain')
    def test_monitor_command_with_threshold(self, mock_monitor_and_retrain):
        """Test the monitor command with custom threshold."""
        result = self.runner.invoke(app, [
            "monitor",
            "--threshold", "0.75",
            "--x-path", "test_features.csv",
            "--y-path", "test_target.csv"
        ])
        
        assert result.exit_code == 0
        expected_kwargs = {
            "threshold": 0.75,
            "X_path": "test_features.csv",
            "y_path": "test_target.csv", 
            "solver": "liblinear",
            "C": 1.0,
            "penalty": "l2",
            "random_state": 42,
            "max_iter": 100,
            "test_size": 0.2
        }
        mock_monitor_and_retrain.assert_called_once_with(**expected_kwargs)


class TestPredictCommand:
    """Test cases for the predict command with validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.cli.run_predictions')
    @patch('src.cli.DEFAULT_PATH_VALIDATOR')
    def test_predict_command_success(self, mock_validator, mock_run_predictions):
        """Test successful predict command."""
        # Mock validation to pass
        mock_validator.validate_path.return_value = None
        
        result = self.runner.invoke(app, [
            "predict",
            "input.csv",
            "--output-csv", "output.csv"
        ])
        
        assert result.exit_code == 0
        # Should validate both input and output paths
        assert mock_validator.validate_path.call_count == 2
        mock_validator.validate_path.assert_any_call("input.csv", must_exist=True)
        mock_validator.validate_path.assert_any_call("output.csv", allow_create=True)
        mock_run_predictions.assert_called_once_with("input.csv", "output.csv", run_id=None)
    
    @patch('src.cli.run_predictions')
    @patch('src.cli.DEFAULT_PATH_VALIDATOR')
    def test_predict_command_with_run_id(self, mock_validator, mock_run_predictions):
        """Test predict command with run ID."""
        mock_validator.validate_path.return_value = None
        
        result = self.runner.invoke(app, [
            "predict",
            "input.csv",
            "--run-id", "test-run-123"
        ])
        
        assert result.exit_code == 0
        mock_run_predictions.assert_called_once_with(
            "input.csv", "predictions.csv", run_id="test-run-123"
        )
    
    @patch('src.cli.run_predictions')
    @patch('src.cli.DEFAULT_PATH_VALIDATOR')
    def test_predict_command_validation_error(self, mock_validator, mock_run_predictions):
        """Test predict command with validation error."""
        # Mock validation to raise an error
        mock_validator.validate_path.side_effect = ValidationError("Invalid path")
        
        result = self.runner.invoke(app, [
            "predict",
            "invalid_input.csv"
        ])
        
        assert result.exit_code == 1
        assert "Validation error: Invalid path" in result.output
        # Should not call run_predictions if validation fails
        mock_run_predictions.assert_not_called()
    
    @patch('src.cli.run_predictions')
    @patch('src.cli.DEFAULT_PATH_VALIDATOR')
    def test_predict_command_validation_error_output_path(self, mock_validator, mock_run_predictions):
        """Test predict command with output path validation error."""
        def validate_side_effect(path, must_exist=False, allow_create=False):
            if "output" in path:
                raise ValidationError("Cannot create output file")
            return None
        
        mock_validator.validate_path.side_effect = validate_side_effect
        
        result = self.runner.invoke(app, [
            "predict",
            "input.csv",
            "--output-csv", "invalid_output.csv"
        ])
        
        assert result.exit_code == 1
        assert "Validation error: Cannot create output file" in result.output
        mock_run_predictions.assert_not_called()


class TestMainFunction:
    """Test cases for main function."""
    
    @patch('src.cli.app')
    def test_main_function_calls_app(self, mock_app):
        """Test that main function calls the Typer app."""
        from src.cli import main
        
        main()
        
        mock_app.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_help_command(self):
        """Test that help command works."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Customer churn prediction command-line interface" in result.stdout
        
        # Should list all available commands
        assert "preprocess" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
        assert "pipeline" in result.stdout
        assert "monitor" in result.stdout
        assert "predict" in result.stdout
    
    def test_individual_command_help(self):
        """Test help for individual commands."""
        commands = ["preprocess", "train", "evaluate", "pipeline", "monitor", "predict"]
        
        for command in commands:
            result = self.runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, f"Help failed for command: {command}"
            # Should contain command description or usage information
            assert len(result.stdout) > 0, f"Empty help for command: {command}"


class TestParameterTypes:
    """Test parameter type handling."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.runner = CliRunner()
    
    @patch('src.cli.run_training')
    def test_float_parameter_conversion(self, mock_run_training):
        """Test that float parameters are correctly converted."""
        result = self.runner.invoke(app, [
            "train",
            "--c", "0.5",
            "--test-size", "0.25"
        ])
        
        assert result.exit_code == 0
        args, kwargs = mock_run_training.call_args
        assert isinstance(kwargs['C'], float)
        assert kwargs['C'] == 0.5
        assert isinstance(kwargs['test_size'], float)
        assert kwargs['test_size'] == 0.25
    
    @patch('src.cli.run_training')
    def test_int_parameter_conversion(self, mock_run_training):
        """Test that int parameters are correctly converted."""
        result = self.runner.invoke(app, [
            "train",
            "--random-state", "999",
            "--max-iter", "500"
        ])
        
        assert result.exit_code == 0
        args, kwargs = mock_run_training.call_args
        assert isinstance(kwargs['random_state'], int)
        assert kwargs['random_state'] == 999
        assert isinstance(kwargs['max_iter'], int)
        assert kwargs['max_iter'] == 500
    
    def test_invalid_parameter_type(self):
        """Test handling of invalid parameter types."""
        result = self.runner.invoke(app, [
            "train",
            "--c", "invalid_float"
        ])
        
        # Should fail with type conversion error
        assert result.exit_code != 0
        assert "invalid" in result.stdout.lower() or "error" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__])