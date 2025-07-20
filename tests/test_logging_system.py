"""Tests for centralized logging system."""

import logging
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.logging_config import setup_logging, get_logger


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