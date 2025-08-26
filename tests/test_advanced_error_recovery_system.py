"""
Comprehensive tests for Advanced Error Recovery System.

Tests retry strategies, error pattern analysis, fallback management, and recovery orchestration.
"""

import pytest
import asyncio
import time
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from src.advanced_error_recovery_system import (
    ErrorRecoverySystem, IntelligentRetryStrategy, ErrorPatternAnalyzer, 
    FallbackManager, ErrorContext, RecoveryAction, ErrorSeverity, RecoveryStrategy,
    get_error_recovery_system, with_error_recovery, initialize_default_recovery_actions,
    clear_memory_recovery, restart_service_recovery
)


class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test error context creation."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Test error",
            function_name="test_func",
            module_name="test_module",
            severity=ErrorSeverity.HIGH
        )
        
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert context.function_name == "test_func"
        assert context.module_name == "test_module"
        assert context.severity == ErrorSeverity.HIGH
        assert len(context.error_id) == 8
        assert context.retry_count == 0
        
    def test_error_context_to_dict(self):
        """Test error context serialization."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.CRITICAL
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict['error_type'] == "ValueError"
        assert context_dict['error_message'] == "Test error"
        assert context_dict['severity'] == "critical"
        assert 'error_id' in context_dict
        assert 'timestamp' in context_dict


class TestIntelligentRetryStrategy:
    """Test intelligent retry strategy."""
    
    def setup_method(self):
        """Setup retry strategy."""
        self.retry_strategy = IntelligentRetryStrategy(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
    def test_delay_calculation(self):
        """Test delay calculation."""
        # Test exponential backoff
        delay_0 = self.retry_strategy.calculate_delay(0)
        delay_1 = self.retry_strategy.calculate_delay(1)
        delay_2 = self.retry_strategy.calculate_delay(2)
        
        assert delay_0 == 0.1
        assert delay_1 == 0.2
        assert delay_2 == 0.4
        
    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        delay = self.retry_strategy.calculate_delay(10)  # Very large attempt
        assert delay <= 1.0  # Should be capped at max_delay
        
    def test_jitter_variation(self):
        """Test jitter introduces variation."""
        jittered_strategy = IntelligentRetryStrategy(
            base_delay=1.0,
            jitter=True
        )
        
        # Multiple calls should produce different delays with jitter
        delays = [jittered_strategy.calculate_delay(1) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have variation
        
    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test successful retry after failures."""
        call_count = 0
        error_context = ErrorContext()
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise ValueError("Temporary failure")
            return "success"
            
        result = await self.retry_strategy.execute_with_retry(
            flaky_function, error_context
        )
        
        assert result == "success"
        assert call_count == 3
        assert error_context.retry_count == 2
        
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        error_context = ErrorContext()
        
        def always_fail():
            raise ValueError("Always fails")
            
        with pytest.raises(ValueError, match="Always fails"):
            await self.retry_strategy.execute_with_retry(
                always_fail, error_context
            )
            
        assert error_context.retry_count == 3  # max_retries
        
    @pytest.mark.asyncio
    async def test_async_function_retry(self):
        """Test retry with async functions."""
        call_count = 0
        error_context = ErrorContext()
        
        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise ConnectionError("Network issue")
            return "async_success"
            
        result = await self.retry_strategy.execute_with_retry(
            async_flaky_function, error_context
        )
        
        assert result == "async_success"
        assert call_count == 2


class TestErrorPatternAnalyzer:
    """Test error pattern analysis."""
    
    def setup_method(self):
        """Setup pattern analyzer."""
        self.analyzer = ErrorPatternAnalyzer(pattern_window=50)
        
    def test_insufficient_data_no_patterns(self):
        """Test no patterns detected with insufficient data."""
        # Add only a few errors
        for i in range(5):
            context = ErrorContext(
                error_type="ValueError",
                function_name=f"func_{i}",
                module_name="test_module"
            )
            self.analyzer.record_error(context)
            
        patterns = self.analyzer.detect_error_patterns()
        assert len(patterns) == 0
        
    def test_frequent_function_errors_pattern(self):
        """Test detection of frequent function errors."""
        # Add many errors from same function
        for i in range(10):
            context = ErrorContext(
                error_type="ValueError",
                function_name="problematic_function",
                module_name="test_module",
                severity=ErrorSeverity.HIGH
            )
            self.analyzer.record_error(context)
            
        patterns = self.analyzer.detect_error_patterns()
        
        # Should detect frequent function errors pattern
        function_patterns = [p for p in patterns if p['type'] == 'frequent_function_errors']
        assert len(function_patterns) >= 1
        
        pattern = function_patterns[0]
        assert pattern['function'] == "test_module.problematic_function"
        assert pattern['count'] >= 5
        
    def test_error_cascade_pattern(self):
        """Test detection of error cascades."""
        base_time = time.time()
        
        # Add errors in quick succession (cascade)
        for i in range(5):
            context = ErrorContext(
                error_type=f"Error_{i}",
                function_name=f"func_{i}",
                module_name="test_module"
            )
            context.timestamp = base_time + i * 30  # 30 seconds apart
            self.analyzer.record_error(context)
            
        patterns = self.analyzer.detect_error_patterns()
        
        # Should detect error cascade
        cascade_patterns = [p for p in patterns if p['type'] == 'error_cascade']
        assert len(cascade_patterns) >= 1
        
    def test_failure_probability_prediction(self):
        """Test failure probability prediction."""
        # Add many errors for a specific function
        for i in range(10):
            context = ErrorContext(
                function_name="unreliable_function",
                module_name="test_module"
            )
            self.analyzer.record_error(context)
            
        # Add some successful operations to dilute
        for i in range(5):
            context = ErrorContext(
                function_name="reliable_function",
                module_name="test_module"
            )
            self.analyzer.record_error(context)
            
        # Check predictions
        unreliable_prob = self.analyzer.predict_failure_probability(
            "unreliable_function", "test_module"
        )
        reliable_prob = self.analyzer.predict_failure_probability(
            "reliable_function", "test_module"
        )
        
        assert unreliable_prob > reliable_prob
        assert 0 <= unreliable_prob <= 1
        assert 0 <= reliable_prob <= 1
        
    def test_pattern_severity_calculation(self):
        """Test pattern severity calculation."""
        # High severity errors
        high_severity_errors = [
            ErrorContext(severity=ErrorSeverity.CRITICAL),
            ErrorContext(severity=ErrorSeverity.HIGH),
            ErrorContext(severity=ErrorSeverity.MEDIUM)
        ]
        
        severity = self.analyzer._calculate_pattern_severity(high_severity_errors)
        assert severity == ErrorSeverity.CRITICAL.value
        
        # Medium severity errors
        medium_severity_errors = [
            ErrorContext(severity=ErrorSeverity.MEDIUM),
            ErrorContext(severity=ErrorSeverity.LOW)
        ]
        
        severity = self.analyzer._calculate_pattern_severity(medium_severity_errors)
        assert severity == ErrorSeverity.MEDIUM.value


class TestFallbackManager:
    """Test fallback management."""
    
    def setup_method(self):
        """Setup fallback manager."""
        self.fallback_manager = FallbackManager()
        
    def test_fallback_registration(self):
        """Test fallback strategy registration."""
        def primary_fallback():
            return "primary_fallback"
            
        def secondary_fallback():
            return "secondary_fallback"
            
        self.fallback_manager.register_fallback("test_operation", primary_fallback, priority=1)
        self.fallback_manager.register_fallback("test_operation", secondary_fallback, priority=2)
        
        assert "test_operation" in self.fallback_manager.fallback_strategies
        assert len(self.fallback_manager.fallback_strategies["test_operation"]) == 2
        
        # Check priority ordering (higher priority first)
        strategies = self.fallback_manager.fallback_strategies["test_operation"]
        assert strategies[0][0] == 1  # Lower number = higher priority after sorting
        assert strategies[1][0] == 2
        
    @pytest.mark.asyncio
    async def test_successful_fallback_execution(self):
        """Test successful fallback execution."""
        def working_fallback():
            return "fallback_success"
            
        self.fallback_manager.register_fallback("test_operation", working_fallback)
        
        error_context = ErrorContext()
        result = await self.fallback_manager.execute_fallback(
            "test_operation", error_context
        )
        
        assert result == "fallback_success"
        
    @pytest.mark.asyncio
    async def test_fallback_priority_ordering(self):
        """Test fallbacks executed in priority order."""
        execution_order = []
        
        def high_priority_fallback():
            execution_order.append("high")
            raise ValueError("High priority failed")
            
        def low_priority_fallback():
            execution_order.append("low")
            return "low_priority_success"
            
        self.fallback_manager.register_fallback("test_operation", high_priority_fallback, priority=1)
        self.fallback_manager.register_fallback("test_operation", low_priority_fallback, priority=5)
        
        error_context = ErrorContext()
        result = await self.fallback_manager.execute_fallback(
            "test_operation", error_context
        )
        
        assert result == "low_priority_success"
        assert execution_order == ["high", "low"]
        
    @pytest.mark.asyncio
    async def test_all_fallbacks_fail(self):
        """Test behavior when all fallbacks fail."""
        def failing_fallback():
            raise ValueError("Fallback failed")
            
        self.fallback_manager.register_fallback("test_operation", failing_fallback)
        
        error_context = ErrorContext()
        
        with pytest.raises(ValueError, match="Fallback failed"):
            await self.fallback_manager.execute_fallback("test_operation", error_context)
            
        assert "all_fallbacks_failed" in error_context.recovery_attempts
        
    @pytest.mark.asyncio
    async def test_no_fallbacks_registered(self):
        """Test behavior when no fallbacks are registered."""
        error_context = ErrorContext()
        
        with pytest.raises(Exception, match="No fallback strategies registered"):
            await self.fallback_manager.execute_fallback("unknown_operation", error_context)
            
    @pytest.mark.asyncio
    async def test_async_fallback_execution(self):
        """Test async fallback execution."""
        async def async_fallback():
            return "async_fallback_success"
            
        self.fallback_manager.register_fallback("async_operation", async_fallback)
        
        error_context = ErrorContext()
        result = await self.fallback_manager.execute_fallback(
            "async_operation", error_context
        )
        
        assert result == "async_fallback_success"


class TestErrorRecoverySystem:
    """Test error recovery system."""
    
    def setup_method(self):
        """Setup recovery system."""
        self.recovery_system = ErrorRecoverySystem()
        
    def test_system_initialization(self):
        """Test system initialization."""
        assert isinstance(self.recovery_system.retry_strategy, IntelligentRetryStrategy)
        assert isinstance(self.recovery_system.pattern_analyzer, ErrorPatternAnalyzer)
        assert isinstance(self.recovery_system.fallback_manager, FallbackManager)
        assert isinstance(self.recovery_system.recovery_actions, dict)
        
    def test_error_severity_classification(self):
        """Test error severity classification."""
        # Test critical errors
        memory_error = MemoryError("Out of memory")
        severity = self.recovery_system._classify_error_severity(memory_error)
        assert severity == ErrorSeverity.CRITICAL
        
        connection_error = ConnectionError("Connection refused")
        severity = self.recovery_system._classify_error_severity(connection_error)
        assert severity == ErrorSeverity.HIGH
        
        value_error = ValueError("Invalid value")
        severity = self.recovery_system._classify_error_severity(value_error)
        assert severity == ErrorSeverity.MEDIUM
        
        runtime_error = RuntimeError("General error")
        severity = self.recovery_system._classify_error_severity(runtime_error)
        assert severity == ErrorSeverity.LOW
        
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        error = ValueError("Test error")
        
        error_context = await self.recovery_system.handle_error(
            error, "test_function", "test_module"
        )
        
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error"
        assert error_context.function_name == "test_function"
        assert error_context.module_name == "test_module"
        assert error_context.severity == ErrorSeverity.MEDIUM
        
    @pytest.mark.asyncio
    async def test_execute_with_recovery_success(self):
        """Test successful execution with recovery."""
        def success_function():
            return "success"
            
        result = await self.recovery_system.execute_with_recovery(
            "test_operation", success_function
        )
        
        assert result == "success"
        
    @pytest.mark.asyncio
    async def test_execute_with_recovery_with_retries(self):
        """Test execution with recovery using retries."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Temporary failure")
            return "recovered"
            
        result = await self.recovery_system.execute_with_recovery(
            "test_operation", flaky_function
        )
        
        assert result == "recovered"
        assert call_count == 3
        
    @pytest.mark.asyncio
    async def test_execute_with_recovery_with_fallback(self):
        """Test execution with recovery using fallback."""
        def failing_function():
            raise ValueError("Always fails")
            
        def fallback_function():
            return "fallback_result"
            
        self.recovery_system.register_fallback("test_operation", fallback_function)
        
        result = await self.recovery_system.execute_with_recovery(
            "test_operation", failing_function
        )
        
        assert result == "fallback_result"
        
    def test_recovery_action_registration(self):
        """Test recovery action registration."""
        def test_recovery_action(error_context):
            pass
            
        action = RecoveryAction(
            name="test_action",
            strategy=RecoveryStrategy.RETRY,
            function=test_recovery_action
        )
        
        self.recovery_system.register_recovery_action(action)
        assert "test_action" in self.recovery_system.recovery_actions
        
    def test_error_statistics_generation(self):
        """Test error statistics generation."""
        # Add some errors to history
        for i in range(5):
            error_context = ErrorContext(
                error_type="ValueError",
                severity=ErrorSeverity.HIGH
            )
            self.recovery_system.pattern_analyzer.record_error(error_context)
            
        stats = self.recovery_system.get_error_statistics()
        
        assert 'total_errors' in stats
        assert 'error_patterns' in stats
        assert 'severity_distribution' in stats
        assert 'most_common_errors' in stats
        assert stats['total_errors'] == 5
        
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with default config
        system = ErrorRecoverySystem()
        assert system.config['max_retries'] == 3
        assert system.config['enable_pattern_analysis'] == True
        
        # Test with custom config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {'max_retries': 5, 'base_delay': 2.0}
            json.dump(config, f)
            config_path = f.name
            
        try:
            system = ErrorRecoverySystem(config_path)
            assert system.config['max_retries'] == 5
            assert system.config['base_delay'] == 2.0
        finally:
            Path(config_path).unlink()


class TestRecoveryActions:
    """Test built-in recovery actions."""
    
    def test_clear_memory_recovery(self):
        """Test memory clearing recovery action."""
        error_context = ErrorContext(error_message="memory error")
        
        # Should not raise exception
        clear_memory_recovery(error_context)
        
    @pytest.mark.asyncio
    async def test_restart_service_recovery(self):
        """Test service restart recovery action."""
        error_context = ErrorContext(error_message="service error")
        
        # Should not raise exception
        await restart_service_recovery(error_context)


class TestDecorator:
    """Test error recovery decorator."""
    
    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Test async decorator with successful execution."""
        @with_error_recovery("test_operation")
        async def async_function():
            return "async_success"
            
        result = await async_function()
        assert result == "async_success"
        
    def test_sync_decorator_success(self):
        """Test sync decorator with successful execution."""
        @with_error_recovery("test_operation")
        def sync_function():
            return "sync_success"
            
        result = sync_function()
        assert result == "sync_success"
        
    def test_sync_decorator_with_error(self):
        """Test sync decorator handles errors."""
        @with_error_recovery("test_operation")
        def failing_function():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            failing_function()


class TestGlobalInstance:
    """Test global instance management."""
    
    def test_get_error_recovery_system(self):
        """Test getting global recovery system instance."""
        system1 = get_error_recovery_system()
        system2 = get_error_recovery_system()
        
        # Should be the same instance
        assert system1 is system2
        assert isinstance(system1, ErrorRecoverySystem)
        
    def test_initialize_default_recovery_actions(self):
        """Test initialization of default recovery actions."""
        initialize_default_recovery_actions()
        
        system = get_error_recovery_system()
        assert "clear_memory" in system.recovery_actions
        assert "restart_service" in system.recovery_actions


class TestRecoveryAction:
    """Test recovery action data structure."""
    
    def test_recovery_action_creation(self):
        """Test recovery action creation."""
        def test_function(context):
            pass
            
        action = RecoveryAction(
            name="test_action",
            strategy=RecoveryStrategy.FALLBACK,
            function=test_function,
            conditions=["high_severity"],
            priority=5,
            timeout=30.0
        )
        
        assert action.name == "test_action"
        assert action.strategy == RecoveryStrategy.FALLBACK
        assert action.function == test_function
        assert action.conditions == ["high_severity"]
        assert action.priority == 5
        assert action.timeout == 30.0


class TestIntegration:
    """Integration tests for error recovery system."""
    
    @pytest.mark.asyncio
    async def test_complete_recovery_flow(self):
        """Test complete error recovery flow."""
        system = ErrorRecoverySystem()
        
        # Register fallback
        def fallback_function():
            return "fallback_success"
            
        system.register_fallback("integration_test", fallback_function)
        
        # Register recovery action
        recovery_called = False
        
        def test_recovery_action(error_context):
            nonlocal recovery_called
            recovery_called = True
            
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RETRY,
            function=test_recovery_action,
            conditions=["high_severity"]
        )
        system.register_recovery_action(action)
        
        # Function that always fails
        def failing_function():
            raise ValueError("Integration test error")
            
        # Should use fallback
        result = await system.execute_with_recovery("integration_test", failing_function)
        assert result == "fallback_success"
        
    @pytest.mark.asyncio
    async def test_pattern_analysis_integration(self):
        """Test integration with pattern analysis."""
        system = ErrorRecoverySystem()
        
        # Generate errors to create patterns
        for i in range(10):
            try:
                await system.execute_with_recovery(
                    "pattern_test",
                    lambda: (_ for _ in ()).throw(ValueError("Pattern test"))
                )
            except:
                pass
                
        # Check that patterns are detected
        stats = system.get_error_statistics()
        assert stats['total_errors'] > 0
        
        # Check failure prediction
        prob = system.pattern_analyzer.predict_failure_probability("pattern_test", "unknown")
        assert prob >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])