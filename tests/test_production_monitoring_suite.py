"""
Comprehensive tests for Production Monitoring Suite.

Tests metrics collection, alerting, performance profiling, and monitoring orchestration.
"""

import pytest
import asyncio
import time
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from collections import deque

from src.production_monitoring_suite import (
    MonitoringSuite, MetricsCollector, AlertManager, PerformanceProfiler,
    SystemMetrics, Alert, AlertSeverity, AlertChannel, MetricThreshold,
    get_monitoring_suite, monitor_function_performance, initialize_monitoring
)


class TestSystemMetrics:
    """Test system metrics functionality."""
    
    def test_system_metrics_creation(self):
        """Test system metrics creation."""
        metrics = SystemMetrics(
            cpu_usage=75.5,
            memory_usage=60.0,
            disk_usage=45.2,
            response_time_avg=1.5
        )
        
        assert metrics.cpu_usage == 75.5
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == 45.2
        assert metrics.response_time_avg == 1.5
        assert metrics.timestamp > 0
        assert isinstance(metrics.hostname, str)
        
    def test_system_metrics_to_dict(self):
        """Test system metrics serialization."""
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=40.0,
            network_io_sent=1024,
            model_prediction_count=100
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['cpu_usage'] == 50.0
        assert metrics_dict['memory_usage'] == 40.0
        assert metrics_dict['network_io_sent'] == 1024
        assert metrics_dict['model_prediction_count'] == 100
        assert 'timestamp' in metrics_dict
        assert 'hostname' in metrics_dict


class TestAlert:
    """Test alert functionality."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="High CPU Usage",
            message="CPU usage exceeded 80%",
            source="monitoring_system",
            tags={'metric': 'cpu_usage', 'threshold': '80'}
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage exceeded 80%"
        assert alert.source == "monitoring_system"
        assert alert.tags['metric'] == 'cpu_usage'
        assert len(alert.id) == 8
        assert not alert.resolved
        
    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="System Down",
            channels=[AlertChannel.LOG, AlertChannel.EMAIL]
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['severity'] == 'critical'
        assert alert_dict['title'] == "System Down"
        assert 'log' in alert_dict['channels']
        assert 'email' in alert_dict['channels']
        assert 'id' in alert_dict
        assert 'timestamp' in alert_dict


class TestMetricThreshold:
    """Test metric threshold functionality."""
    
    def test_threshold_creation(self):
        """Test threshold creation."""
        threshold = MetricThreshold(
            metric_name="cpu_usage",
            operator=">",
            threshold_value=80.0,
            duration=300.0,
            severity=AlertSeverity.WARNING,
            description="High CPU usage detected"
        )
        
        assert threshold.metric_name == "cpu_usage"
        assert threshold.operator == ">"
        assert threshold.threshold_value == 80.0
        assert threshold.duration == 300.0
        assert threshold.severity == AlertSeverity.WARNING
        assert threshold.description == "High CPU usage detected"


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def setup_method(self):
        """Setup metrics collector."""
        self.collector = MetricsCollector(collection_interval=0.1)
        
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test basic metrics collection."""
        metrics = await self.collector._collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.disk_usage >= 0
        assert metrics.timestamp > 0
        
    @pytest.mark.asyncio
    async def test_collection_loop(self):
        """Test metrics collection loop."""
        # Start collection
        collection_task = asyncio.create_task(self.collector.start_collection())
        
        # Let it collect a few samples
        await asyncio.sleep(0.3)
        
        # Stop collection
        self.collector.stop_collection()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(collection_task, timeout=1.0)
        except asyncio.TimeoutError:
            collection_task.cancel()
            
        # Verify metrics were collected
        assert len(self.collector.metrics_history) >= 1
        
    def test_custom_metric_calculator_registration(self):
        """Test custom metric calculator registration."""
        async def custom_calculator():
            return 42.0
            
        self.collector.register_custom_metric_calculator("custom_metric", custom_calculator)
        
        assert "custom_metric" in self.collector.metric_calculators
        assert self.collector.metric_calculators["custom_metric"] == custom_calculator
        
    @pytest.mark.asyncio
    async def test_custom_metrics_calculation(self):
        """Test custom metrics calculation."""
        async def test_metric():
            return 99.5
            
        self.collector.register_custom_metric_calculator("test_metric", test_metric)
        
        metrics = await self.collector._collect_system_metrics()
        
        # Custom metric should be added to metrics
        assert hasattr(metrics, "test_metric")
        assert getattr(metrics, "test_metric") == 99.5
        
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Add some sample metrics
        for i in range(5):
            sample_metrics = SystemMetrics(
                cpu_usage=50.0 + i,
                memory_usage=60.0 + i,
                response_time_avg=1.0 + i * 0.1
            )
            self.collector.metrics_history.append(sample_metrics)
            
        summary = self.collector.get_metrics_summary()
        
        assert 'timestamp' in summary
        assert 'total_samples' in summary
        assert 'recent_samples' in summary
        assert 'avg_cpu_usage' in summary
        assert 'avg_memory_usage' in summary
        assert summary['total_samples'] == 5
        
    def test_metrics_history_limit(self):
        """Test metrics history size limit."""
        # Add more metrics than the limit
        for i in range(1500):  # Exceeds maxlen of 1000
            metrics = SystemMetrics(cpu_usage=float(i))
            self.collector.metrics_history.append(metrics)
            
        assert len(self.collector.metrics_history) <= 1000


class TestAlertManager:
    """Test alert management functionality."""
    
    def setup_method(self):
        """Setup alert manager."""
        self.alert_manager = AlertManager()
        
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        assert isinstance(self.alert_manager.alert_history, deque)
        assert isinstance(self.alert_manager.active_alerts, dict)
        assert len(self.alert_manager.thresholds) > 0  # Should have default thresholds
        
    def test_threshold_evaluation(self):
        """Test threshold evaluation."""
        threshold = MetricThreshold(
            metric_name="cpu_usage",
            operator=">",
            threshold_value=80.0
        )
        
        # Test above threshold
        assert self.alert_manager._evaluate_threshold(85.0, threshold)
        
        # Test below threshold
        assert not self.alert_manager._evaluate_threshold(75.0, threshold)
        
        # Test different operators
        less_than_threshold = MetricThreshold(
            metric_name="memory_free",
            operator="<",
            threshold_value=20.0
        )
        
        assert self.alert_manager._evaluate_threshold(15.0, less_than_threshold)
        assert not self.alert_manager._evaluate_threshold(25.0, less_than_threshold)
        
    @pytest.mark.asyncio
    async def test_threshold_checking(self):
        """Test checking metrics against thresholds."""
        # Create metrics that exceed thresholds
        high_cpu_metrics = SystemMetrics(
            cpu_usage=95.0,  # Should trigger critical threshold
            memory_usage=50.0,
            disk_usage=40.0
        )
        
        # Mock alert sending to avoid actual alerts
        with patch.object(self.alert_manager, '_send_alert') as mock_send:
            await self.alert_manager.check_thresholds(high_cpu_metrics)
            
            # Should have triggered at least one alert
            assert mock_send.call_count >= 1
            
    @pytest.mark.asyncio
    async def test_alert_suppression(self):
        """Test alert suppression functionality."""
        # Add many recent alerts to trigger suppression
        for i in range(110):  # Exceeds max_alerts_per_hour
            alert = Alert(title=f"Test Alert {i}")
            self.alert_manager.alert_history.append(alert)
            
        # Create new alert that should be suppressed
        test_alert = Alert(
            title="Suppressed Alert",
            severity=AlertSeverity.WARNING
        )
        
        should_suppress = self.alert_manager._should_suppress_alert(test_alert)
        assert should_suppress
        
    @pytest.mark.asyncio
    async def test_alert_handler_registration(self):
        """Test alert handler registration."""
        mock_handler = Mock()
        
        self.alert_manager.register_alert_handler(AlertChannel.EMAIL, mock_handler)
        
        assert AlertChannel.EMAIL in self.alert_manager.alert_handlers
        assert self.alert_manager.alert_handlers[AlertChannel.EMAIL] == mock_handler
        
    def test_custom_threshold_addition(self):
        """Test adding custom thresholds."""
        initial_count = len(self.alert_manager.thresholds)
        
        custom_threshold = MetricThreshold(
            metric_name="custom_metric",
            operator=">",
            threshold_value=100.0
        )
        
        self.alert_manager.add_threshold(custom_threshold)
        
        assert len(self.alert_manager.thresholds) == initial_count + 1
        assert custom_threshold in self.alert_manager.thresholds
        
    def test_alert_resolution(self):
        """Test alert resolution."""
        # Create and add active alert
        alert = Alert(
            title="Test Alert",
            severity=AlertSeverity.WARNING
        )
        
        alert_key = f"test_{alert.id}"
        self.alert_manager.active_alerts[alert_key] = alert
        
        # Resolve alert
        self.alert_manager.resolve_alert(alert.id, "test_resolver")
        
        assert alert.resolved
        assert alert.resolved_at is not None
        assert "test_resolver" in alert.acknowledgments
        assert alert_key not in self.alert_manager.active_alerts
        
    def test_alert_statistics(self):
        """Test alert statistics generation."""
        # Add sample alerts
        for i in range(5):
            alert = Alert(
                severity=AlertSeverity.WARNING if i % 2 == 0 else AlertSeverity.ERROR,
                source=f"source_{i % 3}",
                resolved=(i < 3)
            )
            self.alert_manager.alert_history.append(alert)
            
        stats = self.alert_manager.get_alert_statistics()
        
        assert 'total_alerts' in stats
        assert 'severity_distribution' in stats
        assert 'top_sources' in stats
        assert 'resolution_rate' in stats
        assert stats['total_alerts'] == 5


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def setup_method(self):
        """Setup performance profiler."""
        self.profiler = PerformanceProfiler()
        
    def test_function_execution_recording(self):
        """Test function execution recording."""
        self.profiler.record_function_execution("test_function", 1.5, True)
        self.profiler.record_function_execution("test_function", 2.0, False)
        
        assert "test_function" in self.profiler.function_metrics
        assert len(self.profiler.function_metrics["test_function"]) == 2
        
        metrics = self.profiler.function_metrics["test_function"]
        assert metrics[0]['execution_time'] == 1.5
        assert metrics[0]['success'] == True
        assert metrics[1]['execution_time'] == 2.0
        assert metrics[1]['success'] == False
        
    def test_request_metrics_recording(self):
        """Test request metrics recording."""
        self.profiler.record_request_metrics(
            "/api/predict", "POST", 200, 0.5, 1024, 2048
        )
        
        assert len(self.profiler.request_metrics) == 1
        
        request = self.profiler.request_metrics[0]
        assert request['endpoint'] == "/api/predict"
        assert request['method'] == "POST"
        assert request['status_code'] == 200
        assert request['response_time'] == 0.5
        assert request['success'] == True
        
    def test_bottleneck_detection_slow_function(self):
        """Test bottleneck detection for slow functions."""
        # Record slow function executions
        for i in range(20):
            execution_time = 2.0 + (i * 0.1)  # Increasingly slow
            self.profiler.record_function_execution("slow_function", execution_time, True)
            
        bottlenecks = self.profiler.detect_bottlenecks()
        
        # Should detect slow function bottleneck
        slow_bottlenecks = [b for b in bottlenecks if b['type'] == 'slow_function']
        assert len(slow_bottlenecks) >= 1
        
        bottleneck = slow_bottlenecks[0]
        assert bottleneck['function'] == 'slow_function'
        assert bottleneck['avg_execution_time'] > 2.0
        
    def test_bottleneck_detection_failing_function(self):
        """Test bottleneck detection for failing functions."""
        # Record mostly failing function executions
        for i in range(20):
            success = i < 5  # Only first 5 succeed
            self.profiler.record_function_execution("failing_function", 1.0, success)
            
        bottlenecks = self.profiler.detect_bottlenecks()
        
        # Should detect failing function bottleneck
        failing_bottlenecks = [b for b in bottlenecks if b['type'] == 'failing_function']
        assert len(failing_bottlenecks) >= 1
        
        bottleneck = failing_bottlenecks[0]
        assert bottleneck['function'] == 'failing_function'
        assert bottleneck['success_rate'] < 0.8
        
    def test_bottleneck_detection_slow_endpoint(self):
        """Test bottleneck detection for slow endpoints."""
        # Record slow endpoint requests
        for i in range(10):
            response_time = 3.0 + (i * 0.1)  # Slow responses
            self.profiler.record_request_metrics(
                "/slow/endpoint", "GET", 200, response_time
            )
            
        bottlenecks = self.profiler.detect_bottlenecks()
        
        # Should detect slow endpoint bottleneck
        slow_endpoints = [b for b in bottlenecks if b['type'] == 'slow_endpoint']
        assert len(slow_endpoints) >= 1
        
        bottleneck = slow_endpoints[0]
        assert bottleneck['endpoint'] == '/slow/endpoint'
        assert bottleneck['avg_response_time'] > 2.0
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add sample function metrics
        self.profiler.record_function_execution("func1", 1.0, True)
        self.profiler.record_function_execution("func2", 2.0, True)
        
        # Add sample request metrics
        self.profiler.record_request_metrics("/api/test", "GET", 200, 0.5)
        self.profiler.record_request_metrics("/api/test", "GET", 500, 1.0)
        
        summary = self.profiler.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'functions_monitored' in summary
        assert 'requests_monitored' in summary
        assert 'avg_response_time' in summary
        assert 'error_rate' in summary
        assert 'bottlenecks_detected' in summary
        
        assert summary['functions_monitored'] == 2
        assert summary['requests_monitored'] == 2
        assert summary['avg_response_time'] == 0.75  # (0.5 + 1.0) / 2
        assert summary['error_rate'] == 0.5  # 1 error out of 2


class TestMonitoringSuite:
    """Test monitoring suite orchestration."""
    
    def setup_method(self):
        """Setup monitoring suite."""
        self.monitoring_suite = MonitoringSuite()
        
    def test_monitoring_suite_initialization(self):
        """Test monitoring suite initialization."""
        assert isinstance(self.monitoring_suite.metrics_collector, MetricsCollector)
        assert isinstance(self.monitoring_suite.alert_manager, AlertManager)
        assert isinstance(self.monitoring_suite.performance_profiler, PerformanceProfiler)
        assert not self.monitoring_suite.is_running
        
    @pytest.mark.asyncio
    async def test_monitoring_suite_start_stop(self):
        """Test monitoring suite start and stop."""
        await self.monitoring_suite.start()
        assert self.monitoring_suite.is_running
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        self.monitoring_suite.stop()
        assert not self.monitoring_suite.is_running
        
    def test_function_performance_recording(self):
        """Test function performance recording."""
        self.monitoring_suite.record_function_performance("test_func", 1.5, True)
        
        # Should be recorded in performance profiler
        assert "test_func" in self.monitoring_suite.performance_profiler.function_metrics
        
    def test_request_performance_recording(self):
        """Test request performance recording."""
        self.monitoring_suite.record_request_performance("/test", "GET", 200, 0.8)
        
        # Should be recorded in performance profiler
        assert len(self.monitoring_suite.performance_profiler.request_metrics) == 1
        
    def test_custom_threshold_addition(self):
        """Test adding custom thresholds."""
        threshold = MetricThreshold(
            metric_name="custom_metric",
            operator=">",
            threshold_value=50.0
        )
        
        self.monitoring_suite.add_custom_threshold(threshold)
        
        assert threshold in self.monitoring_suite.alert_manager.thresholds
        
    def test_custom_metric_calculator_registration(self):
        """Test custom metric calculator registration."""
        async def custom_calculator():
            return 42.0
            
        self.monitoring_suite.register_custom_metric_calculator("custom", custom_calculator)
        
        assert "custom" in self.monitoring_suite.metrics_collector.metric_calculators
        
    def test_monitoring_dashboard_data(self):
        """Test monitoring dashboard data generation."""
        dashboard_data = self.monitoring_suite.get_monitoring_dashboard_data()
        
        assert 'timestamp' in dashboard_data
        assert 'system_info' in dashboard_data
        assert 'metrics_summary' in dashboard_data
        assert 'alert_statistics' in dashboard_data
        assert 'performance_summary' in dashboard_data
        assert 'active_alerts' in dashboard_data
        assert 'monitoring_status' in dashboard_data
        
        # Check system info
        system_info = dashboard_data['system_info']
        assert 'hostname' in system_info
        assert 'platform' in system_info
        assert 'python_version' in system_info
        
        # Check monitoring status
        status = dashboard_data['monitoring_status']
        assert 'is_running' in status
        assert 'metrics_collection_enabled' in status
        assert 'alerts_enabled' in status


class TestMonitoringDecorator:
    """Test monitoring decorator."""
    
    def test_function_performance_decorator(self):
        """Test function performance monitoring decorator."""
        @monitor_function_performance
        def test_function():
            time.sleep(0.1)
            return "test_result"
            
        result = test_function()
        
        assert result == "test_result"
        
        # Check if performance was recorded
        suite = get_monitoring_suite()
        assert "test_function" in suite.performance_profiler.function_metrics
        
    def test_decorator_with_exception(self):
        """Test decorator handles exceptions."""
        @monitor_function_performance
        def failing_function():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            failing_function()
            
        # Should still record the performance (as failure)
        suite = get_monitoring_suite()
        metrics = suite.performance_profiler.function_metrics["failing_function"]
        assert len(metrics) >= 1
        assert not metrics[-1]['success']


class TestGlobalInstance:
    """Test global instance management."""
    
    def test_get_monitoring_suite(self):
        """Test getting global monitoring suite instance."""
        suite1 = get_monitoring_suite()
        suite2 = get_monitoring_suite()
        
        # Should be the same instance
        assert suite1 is suite2
        assert isinstance(suite1, MonitoringSuite)
        
    @pytest.mark.asyncio
    async def test_initialize_monitoring(self):
        """Test monitoring system initialization."""
        await initialize_monitoring()
        
        suite = get_monitoring_suite()
        assert suite.is_running


class TestConfigurationLoading:
    """Test configuration loading functionality."""
    
    def test_metrics_collector_config(self):
        """Test metrics collector configuration."""
        # Test with custom interval
        collector = MetricsCollector(collection_interval=5.0)
        assert collector.collection_interval == 5.0
        
    def test_alert_manager_config_loading(self):
        """Test alert manager configuration loading."""
        # Test with temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'max_alerts_per_hour': 50,
                'enable_alert_suppression': False
            }
            json.dump(config, f)
            config_path = f.name
            
        try:
            alert_manager = AlertManager(config_path)
            assert alert_manager.config['max_alerts_per_hour'] == 50
            assert alert_manager.config['enable_alert_suppression'] == False
        finally:
            Path(config_path).unlink()
            
    def test_monitoring_suite_config_loading(self):
        """Test monitoring suite configuration loading."""
        # Test with temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'metrics_collection_interval': 5.0,
                'enable_alerts': False,
                'enable_performance_profiling': False
            }
            json.dump(config, f)
            config_path = f.name
            
        try:
            suite = MonitoringSuite(config_path)
            assert suite.config['metrics_collection_interval'] == 5.0
            assert suite.config['enable_alerts'] == False
            assert suite.config['enable_performance_profiling'] == False
        finally:
            Path(config_path).unlink()


class TestIntegration:
    """Integration tests for monitoring suite."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow."""
        suite = MonitoringSuite()
        
        # Start monitoring
        await suite.start()
        
        # Simulate some activity
        suite.record_function_performance("test_function", 1.5, True)
        suite.record_request_performance("/api/test", "GET", 200, 0.8)
        
        # Let monitoring run briefly
        await asyncio.sleep(0.3)
        
        # Get dashboard data
        dashboard = suite.get_monitoring_dashboard_data()
        
        assert dashboard['monitoring_status']['is_running']
        assert dashboard['performance_summary']['functions_monitored'] >= 1
        assert dashboard['performance_summary']['requests_monitored'] >= 1
        
        suite.stop()
        
    @pytest.mark.asyncio
    async def test_alert_generation_integration(self):
        """Test alert generation from metrics."""
        suite = MonitoringSuite()
        
        # Create high-utilization metrics that should trigger alerts
        critical_metrics = SystemMetrics(
            cpu_usage=98.0,
            memory_usage=95.0,
            response_time_avg=8.0
        )
        
        # Mock alert sending
        with patch.object(suite.alert_manager, '_send_alert') as mock_send:
            await suite.alert_manager.check_thresholds(critical_metrics)
            
            # Should trigger multiple alerts
            assert mock_send.call_count >= 1
            
    @pytest.mark.asyncio
    async def test_bottleneck_detection_integration(self):
        """Test bottleneck detection and alerting integration."""
        suite = MonitoringSuite()
        
        # Generate performance data that creates bottlenecks
        for i in range(15):
            execution_time = 5.0 + i * 0.5  # Increasingly slow
            suite.record_function_performance("slow_function", execution_time, True)
            
        # Detect bottlenecks
        bottlenecks = suite.performance_profiler.detect_bottlenecks()
        
        # Should detect slow function
        slow_bottlenecks = [b for b in bottlenecks if b['type'] == 'slow_function']
        assert len(slow_bottlenecks) >= 1
        
        # Test bottleneck alerting (would need to run background task)
        # This is tested in the main monitoring loop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])