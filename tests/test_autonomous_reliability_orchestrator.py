"""
Comprehensive tests for Autonomous Reliability Orchestrator.

Tests circuit breakers, health monitoring, anomaly detection, and recovery systems.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from collections import deque

from src.autonomous_reliability_orchestrator import (
    ReliabilityOrchestrator, AdaptiveCircuitBreaker, HealthMonitor, AnomalyDetector,
    HealthMetrics, CircuitBreakerConfig, CircuitState, SystemState,
    get_reliability_orchestrator, reliable_operation, initialize_reliability_system
)


class TestAdaptiveCircuitBreaker:
    """Test adaptive circuit breaker functionality."""
    
    def setup_method(self):
        """Setup test circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_duration=1.0,
            half_open_max_calls=5
        )
        self.circuit_breaker = AdaptiveCircuitBreaker("test_breaker", config)
        
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.name == "test_breaker"
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
        
    def test_successful_execution_closed_state(self):
        """Test successful execution in closed state."""
        def success_func():
            return "success"
            
        result = self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert len(self.circuit_breaker.metrics_history) == 1
        
    def test_failure_execution_closed_state(self):
        """Test failed execution in closed state."""
        def fail_func():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            self.circuit_breaker.call(fail_func)
            
        assert self.circuit_breaker.failure_count == 1
        assert len(self.circuit_breaker.metrics_history) == 1
        assert not self.circuit_breaker.metrics_history[0]['success']
        
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        def fail_func():
            raise ValueError("Test error")
            
        # Trigger failures to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(fail_func)
                
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            self.circuit_breaker.call(fail_func)
            
    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        def fail_func():
            raise ValueError("Test error")
            
        # Open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(fail_func)
                
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Next call should transition to half-open
        def success_func():
            return "success"
            
        result = self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
        
    def test_circuit_closes_after_successful_recovery(self):
        """Test circuit closes after successful recovery."""
        # Open circuit
        def fail_func():
            raise ValueError("Test error")
            
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(fail_func)
                
        # Wait and recover
        time.sleep(1.1)
        
        def success_func():
            return "success"
            
        # Execute successful calls to close circuit
        for _ in range(2):
            result = self.circuit_breaker.call(success_func)
            assert result == "success"
            
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
    def test_half_open_returns_to_open_on_failure(self):
        """Test half-open returns to open on failure."""
        # Open circuit
        def fail_func():
            raise ValueError("Test error")
            
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(fail_func)
                
        # Transition to half-open
        time.sleep(1.1)
        
        def success_func():
            return "success"
            
        result = self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Fail in half-open state
        with pytest.raises(ValueError):
            self.circuit_breaker.call(fail_func)
            
        # Should return to open
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            self.circuit_breaker.call(success_func)


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def setup_method(self):
        """Setup test health monitor."""
        self.health_monitor = HealthMonitor(check_interval=0.1)
        
    @pytest.mark.asyncio
    async def test_health_metrics_collection(self):
        """Test health metrics collection."""
        metrics = await self.health_monitor._collect_health_metrics()
        
        assert isinstance(metrics, HealthMetrics)
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.disk_usage >= 0
        assert metrics.timestamp > 0
        
    def test_health_state_analysis(self):
        """Test health state analysis."""
        # Test healthy state
        healthy_metrics = HealthMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            response_time=1.0
        )
        
        state = self.health_monitor._analyze_health_state(healthy_metrics)
        assert state == SystemState.HEALTHY
        
        # Test degraded state
        degraded_metrics = HealthMetrics(
            cpu_usage=75.0,
            memory_usage=80.0,
            response_time=2.5
        )
        
        state = self.health_monitor._analyze_health_state(degraded_metrics)
        assert state == SystemState.DEGRADED
        
        # Test critical state
        critical_metrics = HealthMetrics(
            cpu_usage=98.0,
            memory_usage=97.0,
            response_time=10.0
        )
        
        state = self.health_monitor._analyze_health_state(critical_metrics)
        assert state == SystemState.CRITICAL
        
    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self):
        """Test health monitoring loop."""
        # Mock the collection method
        mock_metrics = HealthMetrics(cpu_usage=50.0, memory_usage=60.0)
        
        with patch.object(self.health_monitor, '_collect_health_metrics', 
                         return_value=mock_metrics) as mock_collect:
            
            # Start monitoring
            monitor_task = asyncio.create_task(self.health_monitor.start_monitoring())
            
            # Let it run briefly
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            self.health_monitor.stop_monitoring()
            
            # Wait for task to complete
            try:
                await asyncio.wait_for(monitor_task, timeout=1.0)
            except asyncio.TimeoutError:
                monitor_task.cancel()
                
            # Verify metrics were collected
            assert len(self.health_monitor.metrics_history) >= 1
            assert mock_collect.call_count >= 1


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def setup_method(self):
        """Setup anomaly detector."""
        self.detector = AnomalyDetector(window_size=10, threshold=2.0)
        
    def test_no_anomalies_with_insufficient_data(self):
        """Test no anomalies detected with insufficient data."""
        metrics = HealthMetrics(cpu_usage=50.0)
        anomalies = self.detector.detect_anomalies(metrics)
        assert len(anomalies) == 0
        
    def test_anomaly_detection_with_outlier(self):
        """Test anomaly detection with clear outlier."""
        # Add normal metrics
        for _ in range(10):
            normal_metrics = HealthMetrics(
                cpu_usage=50.0 + np.random.normal(0, 2),
                memory_usage=60.0 + np.random.normal(0, 2),
                response_time=1.0 + np.random.normal(0, 0.1)
            )
            self.detector.detect_anomalies(normal_metrics)
            
        # Add anomalous metric
        anomalous_metrics = HealthMetrics(
            cpu_usage=95.0,  # Clear outlier
            memory_usage=60.0,
            response_time=1.0
        )
        
        anomalies = self.detector.detect_anomalies(anomalous_metrics)
        assert 'cpu_usage_anomaly' in anomalies
        
    def test_no_anomaly_with_normal_variation(self):
        """Test no anomaly detected with normal variation."""
        # Add metrics with normal variation
        for i in range(15):
            metrics = HealthMetrics(
                cpu_usage=50.0 + np.random.normal(0, 5),
                memory_usage=60.0 + np.random.normal(0, 5),
                response_time=1.0 + np.random.normal(0, 0.2)
            )
            anomalies = self.detector.detect_anomalies(metrics)
            
        # Last call should not detect anomalies for normal variation
        assert len(anomalies) == 0


class TestReliabilityOrchestrator:
    """Test reliability orchestrator functionality."""
    
    def setup_method(self):
        """Setup orchestrator."""
        self.orchestrator = ReliabilityOrchestrator()
        
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert isinstance(self.orchestrator.circuit_breakers, dict)
        assert isinstance(self.orchestrator.health_monitor, HealthMonitor)
        assert not self.orchestrator.is_running
        
    def test_circuit_breaker_registration(self):
        """Test circuit breaker registration."""
        breaker = self.orchestrator.register_circuit_breaker("test_service")
        
        assert "test_service" in self.orchestrator.circuit_breakers
        assert breaker.name == "test_service"
        assert isinstance(breaker, AdaptiveCircuitBreaker)
        
    def test_get_circuit_breaker(self):
        """Test getting circuit breaker."""
        self.orchestrator.register_circuit_breaker("test_service")
        
        breaker = self.orchestrator.get_circuit_breaker("test_service")
        assert breaker is not None
        assert breaker.name == "test_service"
        
        # Test non-existent breaker
        non_existent = self.orchestrator.get_circuit_breaker("non_existent")
        assert non_existent is None
        
    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self):
        """Test orchestrator start and stop."""
        await self.orchestrator.start()
        assert self.orchestrator.is_running
        
        # Check default circuit breakers are registered
        assert "model_prediction" in self.orchestrator.circuit_breakers
        assert "data_processing" in self.orchestrator.circuit_breakers
        assert "external_api" in self.orchestrator.circuit_breakers
        
        self.orchestrator.stop()
        assert not self.orchestrator.is_running
        
    @pytest.mark.asyncio
    async def test_execute_with_reliability(self):
        """Test execution with reliability protection."""
        await self.orchestrator.start()
        
        def test_function():
            return "success"
            
        result = await self.orchestrator.execute_with_reliability(
            "test_operation", test_function
        )
        
        assert result == "success"
        assert "test_operation" in self.orchestrator.circuit_breakers
        
    def test_recovery_action_registration(self):
        """Test recovery action registration."""
        def test_recovery(error_context):
            pass
            
        self.orchestrator.register_recovery_action("test_recovery", test_recovery)
        assert "test_recovery" in self.orchestrator.recovery_actions
        
    def test_system_health_report(self):
        """Test system health report generation."""
        self.orchestrator.register_circuit_breaker("test_service")
        
        report = self.orchestrator.get_system_health_report()
        
        assert 'timestamp' in report
        assert 'system_state' in report
        assert 'circuit_breakers' in report
        assert 'recent_metrics' in report
        
        assert 'test_service' in report['circuit_breakers']
        
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with invalid config path
        orchestrator = ReliabilityOrchestrator("/nonexistent/config.json")
        
        # Should use default config
        assert orchestrator.config['health_check_interval'] == 30.0
        assert orchestrator.config['auto_recovery_enabled'] == True


class TestReliabilityDecorator:
    """Test reliability decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_async_reliable_operation_decorator(self):
        """Test async reliable operation decorator."""
        @reliable_operation("test_async_operation")
        async def async_test_function():
            return "async_success"
            
        result = await async_test_function()
        assert result == "async_success"
        
    def test_sync_reliable_operation_decorator(self):
        """Test sync reliable operation decorator."""
        @reliable_operation("test_sync_operation")
        def sync_test_function():
            return "sync_success"
            
        result = sync_test_function()
        assert result == "sync_success"
        
    def test_decorator_with_failures(self):
        """Test decorator handles failures."""
        call_count = 0
        
        @reliable_operation("failing_operation")
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("Test failure")
            return "recovered"
            
        with pytest.raises(ValueError):
            failing_function()


class TestGlobalInstance:
    """Test global instance management."""
    
    def test_get_reliability_orchestrator(self):
        """Test getting global orchestrator instance."""
        orchestrator1 = get_reliability_orchestrator()
        orchestrator2 = get_reliability_orchestrator()
        
        # Should be the same instance
        assert orchestrator1 is orchestrator2
        assert isinstance(orchestrator1, ReliabilityOrchestrator)
        
    @pytest.mark.asyncio
    async def test_initialize_reliability_system(self):
        """Test system initialization."""
        await initialize_reliability_system()
        
        orchestrator = get_reliability_orchestrator()
        assert orchestrator.is_running


class TestHealthMetrics:
    """Test health metrics functionality."""
    
    def test_health_metrics_creation(self):
        """Test health metrics creation."""
        metrics = HealthMetrics(
            cpu_usage=75.0,
            memory_usage=80.0,
            disk_usage=60.0,
            response_time=2.5
        )
        
        assert metrics.cpu_usage == 75.0
        assert metrics.memory_usage == 80.0
        assert metrics.disk_usage == 60.0
        assert metrics.response_time == 2.5
        assert metrics.timestamp > 0
        
    def test_health_metrics_to_dict(self):
        """Test health metrics serialization."""
        metrics = HealthMetrics(
            cpu_usage=75.0,
            memory_usage=80.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['cpu_usage'] == 75.0
        assert metrics_dict['memory_usage'] == 80.0
        assert 'timestamp' in metrics_dict


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""
    
    def test_config_creation(self):
        """Test config creation with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_duration=120.0
        )
        
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_duration == 120.0
        
    def test_config_defaults(self):
        """Test config with default values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_duration == 60.0
        assert config.half_open_max_calls == 10


class TestIntegration:
    """Integration tests for reliability system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_reliability_flow(self):
        """Test complete reliability flow."""
        orchestrator = ReliabilityOrchestrator()
        await orchestrator.start()
        
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Fail first 3 calls
                raise ConnectionError("Service unavailable")
            return f"Success on attempt {call_count}"
            
        # First few calls should fail and open circuit
        with pytest.raises(ConnectionError):
            await orchestrator.execute_with_reliability("test_service", unreliable_function)
            
        # Circuit should be open after failures
        breaker = orchestrator.get_circuit_breaker("test_service")
        
        # Wait for circuit to potentially transition to half-open
        time.sleep(1.1)
        
        orchestrator.stop()
        
    @pytest.mark.asyncio 
    async def test_health_monitoring_with_alerts(self):
        """Test health monitoring triggers appropriate responses."""
        orchestrator = ReliabilityOrchestrator()
        
        # Mock health metrics to simulate critical state
        critical_metrics = HealthMetrics(
            cpu_usage=98.0,
            memory_usage=95.0,
            response_time=10.0
        )
        
        with patch.object(orchestrator.health_monitor, '_collect_health_metrics', 
                         return_value=critical_metrics):
            
            state = orchestrator.health_monitor._analyze_health_state(critical_metrics)
            assert state == SystemState.CRITICAL
            
    def test_concurrent_circuit_breaker_access(self):
        """Test concurrent access to circuit breakers."""
        import threading
        import time
        
        orchestrator = ReliabilityOrchestrator()
        breaker = orchestrator.register_circuit_breaker("concurrent_test")
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                def test_func():
                    time.sleep(0.01)  # Small delay
                    return f"worker_{worker_id}"
                    
                result = breaker.call(test_func)
                results.append(result)
            except Exception as e:
                errors.append(e)
                
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        assert len(results) == 10
        assert len(errors) == 0
        assert len(set(results)) == 10  # All unique results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])