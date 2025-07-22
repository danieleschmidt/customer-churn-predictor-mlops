"""Tests for centralized logging system."""

import logging
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.logging_config import setup_logging, get_logger, configure_mlflow_logging, log_function_call


class TestLoggingSystem:
    """Test suite for centralized logging system."""
    
    def test_setup_logging_default_config(self):
        """Test default logging configuration."""
        with patch('logging.basicConfig') as mock_config:
            setup_logging()
            mock_config.assert_called_once()
    
    def test_setup_logging_with_custom_level(self):
        """Test logging setup with custom level."""
        with patch('logging.basicConfig') as mock_config:
            setup_logging(level=logging.DEBUG)
            # Verify DEBUG level was set
            args, kwargs = mock_config.call_args
            assert kwargs.get('level') == logging.DEBUG
    
    def test_setup_logging_with_file_output(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            with patch('logging.basicConfig') as mock_config:
                setup_logging(log_file=log_file)
                # Verify file handler configuration
                mock_config.assert_called_once()
                
        finally:
            os.unlink(log_file)
    
    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a proper logger instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__
    
    def test_logger_levels_work_correctly(self):
        """Test that different log levels work correctly."""
        logger = get_logger('test_logger')
        
        with patch.object(logger, 'debug') as mock_debug, \
             patch.object(logger, 'info') as mock_info, \
             patch.object(logger, 'warning') as mock_warning, \
             patch.object(logger, 'error') as mock_error:
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            mock_debug.assert_called_once_with("Debug message")
            mock_info.assert_called_once_with("Info message")
            mock_warning.assert_called_once_with("Warning message")
            mock_error.assert_called_once_with("Error message")
    
    def test_logging_format_includes_required_fields(self):
        """Test that log format includes timestamp, level, and message."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            setup_logging(log_file=log_file, level=logging.INFO)
            logger = get_logger('test_format')
            
            test_message = "Test log message"
            logger.info(test_message)
            
            # Read log file and verify format
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Should contain timestamp, level, logger name, and message
            assert 'INFO' in log_content
            assert 'test_format' in log_content
            assert test_message in log_content
            
        finally:
            os.unlink(log_file)
    
    def test_environment_variable_configuration(self):
        """Test logging configuration from environment variables."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG', 'LOG_FILE': 'test.log'}):
            with patch('logging.basicConfig') as mock_config:
                setup_logging()
                # Should use environment variable settings
                args, kwargs = mock_config.call_args
                assert kwargs.get('level') == logging.DEBUG
    
    def test_invalid_log_level_handling(self):
        """Test handling of invalid log level values."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID_LEVEL'}):
            with patch('logging.basicConfig') as mock_config:
                # Should not raise an exception and use default level
                setup_logging()
                mock_config.assert_called_once()
    
    def test_logger_hierarchy_works(self):
        """Test that logger hierarchy works correctly."""
        parent_logger = get_logger('parent')
        child_logger = get_logger('parent.child')
        
        assert child_logger.parent == parent_logger
    
    def test_concurrent_logger_access(self):
        """Test that multiple loggers can be created safely."""
        loggers = []
        for i in range(10):
            logger = get_logger(f'concurrent_test_{i}')
            loggers.append(logger)
        
        # All loggers should be unique instances
        logger_names = [logger.name for logger in loggers]
        assert len(set(logger_names)) == 10
    
    def test_log_rotation_configuration(self):
        """Test that log rotation is properly configured."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            setup_logging(log_file=log_file, enable_rotation=True)
            
            # Should create rotating file handler
            root_logger = logging.getLogger()
            handlers = root_logger.handlers
            
            # Should have at least one rotating file handler
            rotating_handlers = [h for h in handlers if hasattr(h, 'maxBytes')]
            assert len(rotating_handlers) > 0
            
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestMLflowLoggingConfiguration:
    """Test MLflow logging configuration."""
    
    def test_configure_mlflow_logging_default(self):
        """Test MLflow logging configuration with default level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_mlflow_logging()
            
            # Should configure specific MLflow loggers
            expected_loggers = ['mlflow', 'mlflow.tracking', 'mlflow.store', 'mlflow.utils']
            assert mock_get_logger.call_count == len(expected_loggers)
            
            for logger_name in expected_loggers:
                mock_get_logger.assert_any_call(logger_name)
            
            # Each logger should have WARNING level set
            assert mock_logger.setLevel.call_count == len(expected_loggers)
            mock_logger.setLevel.assert_called_with(logging.WARNING)
    
    def test_configure_mlflow_logging_custom_level(self):
        """Test MLflow logging configuration with custom level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            configure_mlflow_logging(level=logging.ERROR)
            
            # Should set ERROR level
            mock_logger.setLevel.assert_called_with(logging.ERROR)


class TestLogFunctionCallDecorator:
    """Test the log_function_call decorator."""
    
    def test_decorator_logs_function_call(self):
        """Test that decorator logs function calls."""
        @log_function_call
        def test_function(arg1, arg2=None):
            return "result"
        
        with patch('src.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function("value1", arg2="value2")
            
            assert result == "result"
            mock_get_logger.assert_called_once()
            
            # Should log function call and return
            assert mock_logger.debug.call_count == 2
            call_args = [call[0][0] for call in mock_logger.debug.call_args_list]
            
            # First call should log arguments
            assert "Calling test_function" in call_args[0]
            assert "arg1=('value1',)" in call_args[0] or "args=('value1',)" in call_args[0]
            assert "arg2='value2'" in call_args[0] or "kwargs={'arg2': 'value2'}" in call_args[0]
            
            # Second call should log return type
            assert "test_function returned: str" in call_args[1]
    
    def test_decorator_logs_exceptions(self):
        """Test that decorator logs exceptions."""
        @log_function_call
        def failing_function():
            raise ValueError("Test error")
        
        with patch('src.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                failing_function()
            
            # Should log function call and error
            assert mock_logger.debug.call_count >= 1  # Function call
            assert mock_logger.error.call_count == 1  # Exception
            
            error_call = mock_logger.error.call_args[0][0]
            assert "failing_function raised ValueError: Test error" in error_call
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @log_function_call
        def documented_function(arg):
            """This is a test function."""
            return arg * 2
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function."
    
    def test_decorator_works_with_no_args(self):
        """Test decorator with function that has no arguments."""
        @log_function_call
        def no_args_function():
            return 42
        
        with patch('src.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = no_args_function()
            
            assert result == 42
            assert mock_logger.debug.call_count == 2


class TestAutoConfiguration:
    """Test auto-configuration functionality."""
    
    @patch('src.logging_config.configure_mlflow_logging')
    @patch('src.logging_config.setup_logging')
    @patch('logging.getLogger')
    def test_auto_configure_when_no_handlers(self, mock_get_logger, mock_setup, mock_configure_mlflow):
        """Test auto-configuration when no handlers exist."""
        # Mock root logger with no handlers
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger
        
        from src.logging_config import _auto_configure
        _auto_configure()
        
        mock_setup.assert_called_once()
        mock_configure_mlflow.assert_called_once()
    
    @patch('src.logging_config.configure_mlflow_logging')
    @patch('src.logging_config.setup_logging')
    @patch('logging.getLogger')
    def test_auto_configure_skip_when_handlers_exist(self, mock_get_logger, mock_setup, mock_configure_mlflow):
        """Test auto-configuration is skipped when handlers exist."""
        # Mock root logger with existing handlers
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = [MagicMock()]  # Has handlers
        mock_get_logger.return_value = mock_root_logger
        
        from src.logging_config import _auto_configure
        _auto_configure()
        
        mock_setup.assert_not_called()
        mock_configure_mlflow.assert_not_called()


class TestLoggingConfigIntegration:
    """Integration tests for logging configuration."""
    
    def test_custom_format_string(self):
        """Test setup with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        
        with patch('logging.basicConfig') as mock_config, \
             patch('logging.getLogger') as mock_get_logger:
            
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            setup_logging(format_string=custom_format)
            
            # Should use custom format string
            mock_config.assert_called_once()
    
    def test_directory_creation_for_log_file(self):
        """Test that log directories are created when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_log_path = os.path.join(temp_dir, 'nested', 'logs', 'test.log')
            
            with patch('logging.basicConfig'), \
                 patch('logging.getLogger') as mock_get_logger:
                
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(log_file=nested_log_path)
                
                # Directory should be created
                assert os.path.exists(os.path.dirname(nested_log_path))
    
    def test_non_rotation_file_handler(self):
        """Test file handler without rotation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            with patch('logging.FileHandler') as mock_file_handler, \
                 patch('logging.getLogger') as mock_get_logger:
                
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(log_file=log_file, enable_rotation=False)
                
                # Should use FileHandler, not RotatingFileHandler
                mock_file_handler.assert_called_once_with(log_file)
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_specific_logger_configuration(self):
        """Test that specific loggers are configured."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_loggers = {}
            
            def get_logger_side_effect(name):
                if name not in mock_loggers:
                    mock_loggers[name] = MagicMock()
                return mock_loggers[name]
            
            mock_get_logger.side_effect = get_logger_side_effect
            
            setup_logging(level=logging.DEBUG)
            
            # Should configure src and scripts loggers
            assert 'src' in mock_loggers
            assert 'scripts' in mock_loggers
            
            mock_loggers['src'].setLevel.assert_called_with(logging.DEBUG)
            mock_loggers['scripts'].setLevel.assert_called_with(logging.DEBUG)


if __name__ == "__main__":
    pytest.main([__file__])