"""
Tests for Error Handling and Recovery System.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.error_handling_recovery import (
    ErrorSeverity, ErrorCategory, RecoveryAction, ErrorEvent,
    CircuitBreakerConfig, CircuitBreakerState, CircuitBreaker,
    RetryConfig, RetryMechanism, ErrorClassifier, RecoveryStrategist,
    HealthcheckManager, ErrorHandler, CircuitBreakerOpenError,
    with_error_handling, error_context
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker("test_service")
        
        assert cb.name == "test_service"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.config.failure_threshold == 5
    
    def test_circuit_breaker_success(self):
        """Test successful function execution."""
        cb = CircuitBreaker("test_service")
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_failure_accumulation(self):
        """Test failure accumulation in circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_service", config)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Fail multiple times but not enough to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.failure_count == 2
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test_service", config)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Fail enough times to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(failing_func)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=2
        )
        cb = CircuitBreaker("test_service", config)
        
        def failing_func():
            raise ValueError("Test error")
        
        def success_func():
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should transition to half-open and allow success
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Another success should close the circuit
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test_service", config)
        
        @cb
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Test successful calls
        assert test_function(5) == 10
        assert test_function(0) == 0
        
        # Test failures
        with pytest.raises(ValueError):
            test_function(-1)
        with pytest.raises(ValueError):
            test_function(-2)
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpenError):
            test_function(5)
    
    def test_circuit_breaker_thread_safety(self):
        """Test circuit breaker thread safety."""
        cb = CircuitBreaker("test_service")
        results = []
        
        def test_func():
            return "success"
        
        def call_function():
            try:
                result = cb.call(test_func)
                results.append(result)
            except Exception as e:
                results.append(str(e))
        
        # Run multiple threads
        threads = [threading.Thread(target=call_function) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 10
        assert all(result == "success" for result in results)


class TestRetryMechanism:
    """Tests for RetryMechanism."""
    
    def test_retry_success_first_attempt(self):
        """Test successful function on first attempt."""
        retry = RetryMechanism()
        
        def success_func():
            return "success"
        
        result = retry.execute_with_retry(success_func)
        assert result == "success"
    
    def test_retry_success_after_failures(self):
        """Test success after some failures."""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry = RetryMechanism(config)
        
        attempt_count = 0
        
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = retry.execute_with_retry(flaky_func)
        assert result == "success"
        assert attempt_count == 3
    
    def test_retry_exhausted_attempts(self):
        """Test retry exhaustion."""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        retry = RetryMechanism(config)
        
        def always_fail():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            retry.execute_with_retry(always_fail)
    
    def test_retry_non_retriable_error(self):
        """Test that some errors are not retried."""
        retry = RetryMechanism()
        
        def invalid_input():
            raise ValueError("Invalid input")
        
        # ValueError should not be retried
        with pytest.raises(ValueError):
            retry.execute_with_retry(invalid_input)
    
    def test_retry_decorator(self):
        """Test retry as decorator."""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry = RetryMechanism(config)
        
        attempt_count = 0
        
        @retry
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Network error")
            return f"success on attempt {attempt_count}"
        
        result = flaky_function()
        assert "success" in result
        assert attempt_count == 2
    
    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        retry = RetryMechanism(config)
        
        # Test delay calculation
        delay1 = retry._calculate_delay(0)  # First retry
        delay2 = retry._calculate_delay(1)  # Second retry
        delay3 = retry._calculate_delay(2)  # Third retry
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
    
    def test_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        retry = RetryMechanism(config)
        
        delays = [retry._calculate_delay(0) for _ in range(10)]
        
        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        # All delays should be close to base delay
        assert all(0.5 <= d <= 1.5 for d in delays)


class TestErrorClassifier:
    """Tests for ErrorClassifier."""
    
    def test_data_error_classification(self):
        """Test classification of data-related errors."""
        classifier = ErrorClassifier()
        
        data_error = ValueError("Invalid dataframe column")
        category, severity = classifier.classify_error(data_error)
        
        assert category == ErrorCategory.DATA
        assert severity == ErrorSeverity.MEDIUM
    
    def test_model_error_classification(self):
        """Test classification of model-related errors."""
        classifier = ErrorClassifier()
        
        model_error = AttributeError("Model not fitted")
        category, severity = classifier.classify_error(model_error)
        
        assert category == ErrorCategory.MODEL
        assert severity == ErrorSeverity.HIGH
    
    def test_network_error_classification(self):
        """Test classification of network-related errors."""
        classifier = ErrorClassifier()
        
        network_error = ConnectionError("Network timeout")
        category, severity = classifier.classify_error(network_error)
        
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.MEDIUM
    
    def test_unknown_error_classification(self):
        """Test classification of unknown errors."""
        classifier = ErrorClassifier()
        
        unknown_error = RuntimeError("Unknown runtime error")
        category, severity = classifier.classify_error(unknown_error)
        
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM
    
    def test_context_based_classification(self):
        """Test classification with context information."""
        classifier = ErrorClassifier()
        
        error = ValueError("Some error")
        context = {
            'component': 'model_predictor',
            'is_critical_path': True
        }
        
        category, severity = classifier.classify_error(error, context)
        
        # Should be classified as critical due to context
        assert severity == ErrorSeverity.CRITICAL


class TestRecoveryStrategist:
    """Tests for RecoveryStrategist."""
    
    def test_data_error_recovery_actions(self):
        """Test recovery actions for data errors."""
        strategist = RecoveryStrategist()
        
        actions = strategist.get_recovery_actions(
            ErrorCategory.DATA, 
            ErrorSeverity.MEDIUM
        )
        
        assert RecoveryAction.FALLBACK in actions
        assert RecoveryAction.DEGRADE in actions
    
    def test_critical_error_recovery_actions(self):
        """Test recovery actions for critical errors."""
        strategist = RecoveryStrategist()
        
        actions = strategist.get_recovery_actions(
            ErrorCategory.SYSTEM,
            ErrorSeverity.CRITICAL
        )
        
        assert RecoveryAction.RESTART in actions
        assert RecoveryAction.ESCALATE in actions
    
    def test_network_error_recovery_actions(self):
        """Test recovery actions for network errors."""
        strategist = RecoveryStrategist()
        
        actions = strategist.get_recovery_actions(
            ErrorCategory.NETWORK,
            ErrorSeverity.LOW
        )
        
        assert RecoveryAction.RETRY in actions
    
    def test_unknown_category_fallback(self):
        """Test fallback for unknown error categories."""
        strategist = RecoveryStrategist()
        
        # Test with category not in strategies
        actions = strategist.get_recovery_actions(
            ErrorCategory.UNKNOWN,
            ErrorSeverity.HIGH
        )
        
        # Should return default actions
        assert RecoveryAction.RETRY in actions
        assert RecoveryAction.ALERT in actions


class TestHealthcheckManager:
    """Tests for HealthcheckManager."""
    
    def test_register_health_check(self):
        """Test registering health checks."""
        manager = HealthcheckManager(check_interval=1)
        
        def dummy_check():
            return True
        
        manager.register_health_check("test_check", dummy_check)
        
        assert "test_check" in manager.health_checks
    
    def test_health_check_execution(self):
        """Test health check execution."""
        manager = HealthcheckManager(check_interval=1)
        
        check_called = False
        
        def test_check():
            nonlocal check_called
            check_called = True
            return True
        
        manager.register_health_check("test_check", test_check)
        manager._perform_health_checks()
        
        assert check_called
        assert manager.health_status["test_check"]["healthy"] is True
    
    def test_health_check_failure(self):
        """Test health check failure handling."""
        manager = HealthcheckManager(check_interval=1)
        
        def failing_check():
            return False
        
        manager.register_health_check("failing_check", failing_check)
        manager._perform_health_checks()
        
        assert manager.health_status["failing_check"]["healthy"] is False
    
    def test_health_check_with_healing(self):
        """Test health check with healing function."""
        manager = HealthcheckManager(check_interval=1)
        
        system_healthy = False
        
        def health_check():
            return system_healthy
        
        def healing_function():
            nonlocal system_healthy
            system_healthy = True
            return True
        
        manager.register_health_check("test_check", health_check, healing_function)
        
        # Initial check should fail
        manager._perform_health_checks()
        assert manager.health_status["test_check"]["healthy"] is False
        assert manager.health_status["test_check"]["healed"] is True
        
        # System should be healed now
        assert system_healthy is True
    
    def test_health_monitoring_lifecycle(self):
        """Test health monitoring start/stop."""
        manager = HealthcheckManager(check_interval=0.1)
        
        def dummy_check():
            return True
        
        manager.register_health_check("test_check", dummy_check)
        
        # Start monitoring
        manager.start_monitoring()
        assert manager.running is True
        
        # Wait briefly for checks to run
        time.sleep(0.2)
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager.running is False


class TestErrorHandler:
    """Tests for ErrorHandler."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        
        assert handler.classifier is not None
        assert handler.recovery_strategist is not None
        assert handler.healthcheck_manager is not None
        assert len(handler.error_history) == 0
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        
        test_error = ValueError("Test error")
        context = {"component": "test_component"}
        
        error_event = handler.handle_error(test_error, context)
        
        assert isinstance(error_event, ErrorEvent)
        assert error_event.error_type == "ValueError"
        assert error_event.error_message == "Test error"
        assert error_event.component == "test_component"
        assert len(handler.error_history) == 1
    
    def test_handle_error_with_recovery(self):
        """Test error handling with recovery actions."""
        handler = ErrorHandler()
        
        # Mock a fallback predictor activation
        with patch.object(handler, '_activate_fallback_predictor', return_value=True) as mock_fallback:
            test_error = ConnectionError("Network error")
            context = {"component": "prediction"}
            
            error_event = handler.handle_error(test_error, context)
            
            # Should attempt recovery
            assert len(error_event.recovery_actions_taken) > 0
            assert any("FALLBACK" in action for action in error_event.recovery_actions_taken)
    
    def test_get_circuit_breaker(self):
        """Test circuit breaker creation and retrieval."""
        handler = ErrorHandler()
        
        cb1 = handler.get_circuit_breaker("service1")
        cb2 = handler.get_circuit_breaker("service1")  # Same service
        cb3 = handler.get_circuit_breaker("service2")  # Different service
        
        assert cb1 is cb2  # Same instance
        assert cb1 is not cb3  # Different instance
        assert cb1.name == "service1"
        assert cb3.name == "service2"
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        handler = ErrorHandler()
        
        # Generate some errors
        handler.handle_error(ValueError("Data error"), {"component": "data"})
        handler.handle_error(ConnectionError("Network error"), {"component": "network"})
        handler.handle_error(ValueError("Another data error"), {"component": "data"})
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert "DATA" in stats["category_breakdown"]
        assert "NETWORK" in stats["category_breakdown"]
        assert stats["category_breakdown"]["DATA"] == 2
        assert stats["category_breakdown"]["NETWORK"] == 1
    
    def test_fallback_predictor_activation(self):
        """Test fallback predictor activation."""
        handler = ErrorHandler()
        
        result = handler._activate_fallback_predictor({"component": "prediction"})
        
        assert result is True
        assert "fallback_simple" in handler.fallback_predictors
        
        # Test the fallback predictor
        predictor = handler.fallback_predictors["fallback_simple"]
        
        # Test with data containing MonthlyCharges
        import pandas as pd
        test_data = pd.DataFrame({"MonthlyCharges": [50, 90]})
        predictions = predictor.predict(test_data)
        
        assert len(predictions) == 2
        assert predictions[0] == 0  # Low charges -> no churn
        assert predictions[1] == 1  # High charges -> churn


class TestErrorHandlingDecorators:
    """Tests for error handling decorators."""
    
    def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator."""
        
        @with_error_handling(component="test_component")
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Test successful execution
        result = test_function(5)
        assert result == 10
        
        # Test error handling
        with pytest.raises(ValueError):
            test_function(-1)
    
    def test_with_error_handling_circuit_breaker(self):
        """Test decorator with circuit breaker enabled."""
        
        @with_error_handling(
            component="test_component",
            enable_circuit_breaker=True
        )
        def test_function(fail=False):
            if fail:
                raise ConnectionError("Connection failed")
            return "success"
        
        # Test successful calls
        assert test_function() == "success"
        assert test_function(False) == "success"
        
        # Test failures that should open circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2)
        
        # This would require modifying the global error handler's circuit breaker
        # For now, just test that the decorator works
        with pytest.raises(ConnectionError):
            test_function(True)
    
    def test_with_error_handling_retry(self):
        """Test decorator with retry enabled."""
        attempt_count = 0
        
        @with_error_handling(
            component="test_component",
            enable_retry=True
        )
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return f"success on attempt {attempt_count}"
        
        result = flaky_function()
        assert "success" in result
        assert attempt_count >= 2
    
    def test_error_context_manager(self):
        """Test error context manager."""
        
        with pytest.raises(ValueError):
            with error_context("test_component", operation="test_op"):
                raise ValueError("Test error")
        
        # The error should have been handled by the global error handler
        # Check that it was recorded
        from src.error_handling_recovery import error_handler
        
        assert len(error_handler.error_history) > 0
        latest_error = error_handler.error_history[-1]
        assert latest_error.component == "test_component"


class TestIntegration:
    """Integration tests for error handling system."""
    
    def test_complete_error_handling_flow(self):
        """Test complete error handling flow."""
        handler = ErrorHandler()
        
        # Register a health check
        def model_health_check():
            return True
        
        handler.healthcheck_manager.register_health_check(
            "model_health", model_health_check
        )
        
        # Create a circuit breaker
        cb = handler.get_circuit_breaker("test_service")
        
        # Handle various types of errors
        errors_and_contexts = [
            (ValueError("Data validation failed"), {"component": "data_processor"}),
            (ConnectionError("API timeout"), {"component": "external_api"}),
            (RuntimeError("Model inference failed"), {"component": "model_predictor"})
        ]
        
        for error, context in errors_and_contexts:
            error_event = handler.handle_error(error, context)
            assert error_event is not None
            assert len(error_event.recovery_actions_taken) > 0
        
        # Check statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 3
        assert len(stats["category_breakdown"]) > 0
    
    def test_system_resilience_under_load(self):
        """Test system resilience under error load."""
        handler = ErrorHandler()
        
        # Generate many errors concurrently
        def generate_errors():
            for i in range(50):
                error = ValueError(f"Error {i}")
                context = {"component": f"component_{i % 5}"}
                handler.handle_error(error, context)
        
        threads = [threading.Thread(target=generate_errors) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # System should handle all errors
        assert len(handler.error_history) == 250  # 5 threads * 50 errors
        
        # Statistics should be consistent
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])