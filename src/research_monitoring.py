"""
Advanced Monitoring and Observability for Research Frameworks.

This module provides comprehensive monitoring, metrics collection, and 
observability for all novel research frameworks, enabling proactive
performance management and system health monitoring.

Key Features:
- Real-time performance monitoring for all frameworks
- Custom metrics collection with Prometheus integration
- Distributed tracing for request flow analysis
- Alert management and notification systems
- Performance profiling and optimization recommendations
- Health check endpoints with detailed diagnostics
- Resource utilization monitoring
"""

import os
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, deque
import functools
import psutil

from .logging_config import get_logger
from .research_error_handling import FrameworkType, get_error_handler

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Container for metric values."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    framework: Optional[FrameworkType] = None


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    framework: FrameworkType
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    input_size: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    severity: AlertSeverity
    message: str
    framework: Optional[FrameworkType]
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """
    Metrics collector for research frameworks.
    
    Collects, stores, and exposes metrics for monitoring and alerting.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.metric_registry: Dict[str, MetricType] = {}
        self.alerts: List[Alert] = []
        self.alert_rules: List['AlertRule'] = []
        self.performance_profiles: List[PerformanceProfile] = []
        
        # Configuration
        self.max_metric_history = 1000
        self.retention_hours = 24
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self.start_cleanup_thread()
        
        logger.info("MetricsCollector initialized")
    
    def register_metric(self, name: str, metric_type: MetricType, 
                       description: str = "") -> None:
        """Register a new metric."""
        self.metric_registry[name] = metric_type
        logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None,
                     framework: Optional[FrameworkType] = None) -> None:
        """Record a metric value."""
        if name not in self.metric_registry:
            # Auto-register as gauge
            self.register_metric(name, MetricType.GAUGE)
        
        metric_value = MetricValue(
            name=name,
            value=value,
            metric_type=self.metric_registry[name],
            labels=labels or {},
            timestamp=datetime.now(),
            framework=framework
        )
        
        self.metrics[name].append(metric_value)
        
        # Limit history
        if len(self.metrics[name]) > self.max_metric_history:
            self.metrics[name] = self.metrics[name][-self.max_metric_history:]
        
        # Check alert rules
        self._check_alerts(metric_value)
    
    def increment_counter(self, name: str, value: int = 1,
                         labels: Optional[Dict[str, str]] = None,
                         framework: Optional[FrameworkType] = None) -> None:
        """Increment a counter metric."""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.COUNTER)
        
        # For counters, we track the cumulative value
        current_total = self.get_current_value(name, labels) or 0
        self.record_metric(name, current_total + value, labels, framework)
    
    def set_gauge(self, name: str, value: Union[int, float],
                  labels: Optional[Dict[str, str]] = None,
                  framework: Optional[FrameworkType] = None) -> None:
        """Set a gauge metric value."""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.GAUGE)
        
        self.record_metric(name, value, labels, framework)
    
    def record_histogram(self, name: str, value: float,
                        labels: Optional[Dict[str, str]] = None,
                        framework: Optional[FrameworkType] = None) -> None:
        """Record a histogram value (e.g., request duration)."""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.HISTOGRAM)
        
        self.record_metric(name, value, labels, framework)
    
    def get_current_value(self, name: str, 
                         labels: Optional[Dict[str, str]] = None) -> Optional[Union[int, float]]:
        """Get the current value of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        # Find most recent value with matching labels
        for metric_value in reversed(self.metrics[name]):
            if labels is None or metric_value.labels == labels:
                return metric_value.value
        
        return None
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricValue]:
        """Get metric history for the specified time period."""
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics[name]
            if metric.timestamp > cutoff_time
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            'total_metrics': len(self.metric_registry),
            'active_alerts': len([alert for alert in self.alerts if not alert.resolved]),
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for metric_name in self.metric_registry:
            current_value = self.get_current_value(metric_name)
            history = self.get_metric_history(metric_name, hours=1)
            
            summary['metrics'][metric_name] = {
                'type': self.metric_registry[metric_name].value,
                'current_value': current_value,
                'samples_last_hour': len(history),
                'min_value': min((m.value for m in history), default=None),
                'max_value': max((m.value for m in history), default=None),
                'avg_value': sum(m.value for m in history) / len(history) if history else None
            }
        
        return summary
    
    def add_alert_rule(self, rule: 'AlertRule') -> None:
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def _check_alerts(self, metric_value: MetricValue) -> None:
        """Check if metric value triggers any alerts."""
        for rule in self.alert_rules:
            if rule.matches_metric(metric_value):
                alert = rule.evaluate(metric_value)
                if alert:
                    self.alerts.append(alert)
                    logger.warning(f"Alert triggered: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def record_performance_profile(self, profile: PerformanceProfile) -> None:
        """Record performance profiling data."""
        self.performance_profiles.append(profile)
        
        # Limit history
        if len(self.performance_profiles) > self.max_metric_history:
            self.performance_profiles = self.performance_profiles[-self.max_metric_history:]
        
        # Record as metrics
        labels = {
            'framework': profile.framework.value,
            'operation': profile.operation
        }
        
        self.record_histogram('operation_duration_seconds', profile.duration, labels, profile.framework)
        self.set_gauge('operation_memory_usage_bytes', profile.memory_usage, labels, profile.framework)
        self.set_gauge('operation_cpu_usage_percent', profile.cpu_usage, labels, profile.framework)
    
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            logger.debug("Started metrics cleanup thread")
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.debug("Stopped metrics cleanup thread")
    
    def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old metrics."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_old_metrics()
                self._stop_cleanup.wait(timeout=300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metric data."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for metric_name in list(self.metrics.keys()):
            old_count = len(self.metrics[metric_name])
            self.metrics[metric_name] = [
                metric for metric in self.metrics[metric_name]
                if metric.timestamp > cutoff_time
            ]
            new_count = len(self.metrics[metric_name])
            
            if old_count != new_count:
                logger.debug(f"Cleaned up {old_count - new_count} old values for {metric_name}")
        
        # Cleanup old performance profiles
        old_count = len(self.performance_profiles)
        self.performance_profiles = [
            profile for profile in self.performance_profiles
            if profile.timestamp > cutoff_time
        ]
        new_count = len(self.performance_profiles)
        
        if old_count != new_count:
            logger.debug(f"Cleaned up {old_count - new_count} old performance profiles")


class AlertRule:
    """Rule for triggering alerts based on metric values."""
    
    def __init__(self, name: str, metric_name: str, condition: str,
                 threshold: Union[int, float], severity: AlertSeverity,
                 framework: Optional[FrameworkType] = None):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition  # 'gt', 'lt', 'eq', 'gte', 'lte'
        self.threshold = threshold
        self.severity = severity
        self.framework = framework
        
        # State tracking
        self.last_triggered = None
        self.cooldown_minutes = 5
    
    def matches_metric(self, metric_value: MetricValue) -> bool:
        """Check if this rule applies to the metric."""
        if metric_value.name != self.metric_name:
            return False
        
        if self.framework and metric_value.framework != self.framework:
            return False
        
        return True
    
    def evaluate(self, metric_value: MetricValue) -> Optional[Alert]:
        """Evaluate if metric value triggers an alert."""
        # Check cooldown
        if (self.last_triggered and 
            datetime.now() - self.last_triggered < timedelta(minutes=self.cooldown_minutes)):
            return None
        
        # Evaluate condition
        triggered = False
        value = metric_value.value
        
        if self.condition == 'gt' and value > self.threshold:
            triggered = True
        elif self.condition == 'lt' and value < self.threshold:
            triggered = True
        elif self.condition == 'gte' and value >= self.threshold:
            triggered = True
        elif self.condition == 'lte' and value <= self.threshold:
            triggered = True
        elif self.condition == 'eq' and value == self.threshold:
            triggered = True
        
        if triggered:
            self.last_triggered = datetime.now()
            
            alert = Alert(
                id=f"{self.name}_{int(time.time())}",
                severity=self.severity,
                message=f"{self.name}: {self.metric_name} {self.condition} {self.threshold} (current: {value})",
                framework=self.framework,
                metric_name=self.metric_name,
                current_value=value,
                threshold=self.threshold,
                timestamp=datetime.now()
            )
            
            return alert
        
        return None


class ResearchFrameworkMonitor:
    """
    Comprehensive monitoring for research frameworks.
    
    Provides performance monitoring, health checks, and observability
    for all novel research framework implementations.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.framework_monitors: Dict[FrameworkType, 'FrameworkSpecificMonitor'] = {}
        self.system_monitor = SystemResourceMonitor()
        
        # Register standard metrics
        self._register_standard_metrics()
        
        # Add default alert rules
        self._add_default_alert_rules()
        
        logger.info("ResearchFrameworkMonitor initialized")
    
    def _register_standard_metrics(self) -> None:
        """Register standard metrics for all frameworks."""
        # Performance metrics
        self.metrics_collector.register_metric('operation_duration_seconds', MetricType.HISTOGRAM)
        self.metrics_collector.register_metric('operation_memory_usage_bytes', MetricType.GAUGE)
        self.metrics_collector.register_metric('operation_cpu_usage_percent', MetricType.GAUGE)
        
        # Error metrics
        self.metrics_collector.register_metric('framework_errors_total', MetricType.COUNTER)
        self.metrics_collector.register_metric('framework_operations_total', MetricType.COUNTER)
        
        # Model metrics
        self.metrics_collector.register_metric('model_accuracy', MetricType.GAUGE)
        self.metrics_collector.register_metric('model_training_time_seconds', MetricType.HISTOGRAM)
        self.metrics_collector.register_metric('prediction_latency_seconds', MetricType.HISTOGRAM)
        
        # System metrics
        self.metrics_collector.register_metric('system_cpu_usage_percent', MetricType.GAUGE)
        self.metrics_collector.register_metric('system_memory_usage_percent', MetricType.GAUGE)
        self.metrics_collector.register_metric('system_disk_usage_percent', MetricType.GAUGE)
    
    def _add_default_alert_rules(self) -> None:
        """Add default alert rules for monitoring."""
        # High error rate
        self.metrics_collector.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="framework_errors_total",
            condition="gt",
            threshold=10,
            severity=AlertSeverity.WARNING
        ))
        
        # High memory usage
        self.metrics_collector.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_usage_percent",
            condition="gt",
            threshold=90,
            severity=AlertSeverity.CRITICAL
        ))
        
        # High CPU usage
        self.metrics_collector.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_usage_percent",
            condition="gt",
            threshold=95,
            severity=AlertSeverity.WARNING
        ))
        
        # Slow predictions
        self.metrics_collector.add_alert_rule(AlertRule(
            name="slow_predictions",
            metric_name="prediction_latency_seconds",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING
        ))
    
    def register_framework_monitor(self, framework_type: FrameworkType,
                                 monitor: 'FrameworkSpecificMonitor') -> None:
        """Register a framework-specific monitor."""
        self.framework_monitors[framework_type] = monitor
        logger.info(f"Registered monitor for {framework_type.value} framework")
    
    def record_operation(self, framework: FrameworkType, operation: str,
                        duration: float, success: bool = True,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record framework operation metrics."""
        labels = {
            'framework': framework.value,
            'operation': operation,
            'status': 'success' if success else 'error'
        }
        
        # Record operation
        self.metrics_collector.increment_counter('framework_operations_total', labels=labels, framework=framework)
        
        # Record duration
        self.metrics_collector.record_histogram('operation_duration_seconds', duration, labels, framework)
        
        # Record errors
        if not success:
            error_labels = {
                'framework': framework.value,
                'operation': operation
            }
            self.metrics_collector.increment_counter('framework_errors_total', labels=error_labels, framework=framework)
        
        # Get system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Record performance profile
        profile = PerformanceProfile(
            framework=framework,
            operation=operation,
            duration=duration,
            memory_usage=psutil.Process().memory_info().rss,
            cpu_usage=cpu_percent,
            input_size=metadata.get('input_size', 0) if metadata else 0,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.metrics_collector.record_performance_profile(profile)
    
    def record_model_metrics(self, framework: FrameworkType, 
                           accuracy: Optional[float] = None,
                           training_time: Optional[float] = None) -> None:
        """Record model-specific metrics."""
        labels = {'framework': framework.value}
        
        if accuracy is not None:
            self.metrics_collector.set_gauge('model_accuracy', accuracy, labels, framework)
        
        if training_time is not None:
            self.metrics_collector.record_histogram('model_training_time_seconds', training_time, labels, framework)
    
    def record_prediction_latency(self, framework: FrameworkType, latency: float) -> None:
        """Record prediction latency."""
        labels = {'framework': framework.value}
        self.metrics_collector.record_histogram('prediction_latency_seconds', latency, labels, framework)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        # System metrics
        self.system_monitor.update_metrics(self.metrics_collector)
        
        # Get framework-specific health
        framework_health = {}
        for framework_type, monitor in self.framework_monitors.items():
            framework_health[framework_type.value] = monitor.get_health_status()
        
        # Get error summaries
        from .research_error_handling import get_all_error_summaries
        error_summary = get_all_error_summaries()
        
        # Get active alerts
        active_alerts = self.metrics_collector.get_active_alerts()
        
        # Determine overall status
        overall_status = 'healthy'
        if any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
            overall_status = 'critical'
        elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            overall_status = 'warning'
        elif error_summary.get('overall_status') != 'healthy':
            overall_status = error_summary.get('overall_status', 'unknown')
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'system_resources': self.system_monitor.get_current_usage(),
            'framework_health': framework_health,
            'error_summary': error_summary,
            'active_alerts': len(active_alerts),
            'alert_details': [asdict(alert) for alert in active_alerts],
            'metrics_summary': self.metrics_collector.get_metrics_summary()
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get performance profiles for the period
        profiles = [
            profile for profile in self.metrics_collector.performance_profiles
            if profile.timestamp > cutoff_time
        ]
        
        # Aggregate by framework and operation
        framework_stats = defaultdict(lambda: defaultdict(list))
        for profile in profiles:
            framework_stats[profile.framework][profile.operation].append(profile)
        
        report = {
            'period_hours': hours,
            'total_operations': len(profiles),
            'frameworks': {},
            'top_slowest_operations': [],
            'top_memory_consumers': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Analyze by framework
        for framework, operations in framework_stats.items():
            framework_report = {
                'total_operations': sum(len(op_profiles) for op_profiles in operations.values()),
                'operations': {}
            }
            
            for operation, op_profiles in operations.items():
                durations = [p.duration for p in op_profiles]
                memory_usage = [p.memory_usage for p in op_profiles]
                
                framework_report['operations'][operation] = {
                    'count': len(op_profiles),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'avg_memory': sum(memory_usage) / len(memory_usage),
                    'max_memory': max(memory_usage)
                }
            
            report['frameworks'][framework.value] = framework_report
        
        # Find slowest operations
        all_operations = [(p.framework.value, p.operation, p.duration) for p in profiles]
        slowest = sorted(all_operations, key=lambda x: x[2], reverse=True)[:10]
        report['top_slowest_operations'] = [
            {'framework': fw, 'operation': op, 'duration': dur}
            for fw, op, dur in slowest
        ]
        
        # Find memory consumers
        memory_consumers = [(p.framework.value, p.operation, p.memory_usage) for p in profiles]
        top_memory = sorted(memory_consumers, key=lambda x: x[2], reverse=True)[:10]
        report['top_memory_consumers'] = [
            {'framework': fw, 'operation': op, 'memory_bytes': mem}
            for fw, op, mem in top_memory
        ]
        
        return report
    
    def shutdown(self) -> None:
        """Shutdown monitoring and cleanup resources."""
        self.metrics_collector.stop_cleanup_thread()
        logger.info("ResearchFrameworkMonitor shutdown complete")


class FrameworkSpecificMonitor:
    """Base class for framework-specific monitoring."""
    
    def __init__(self, framework_type: FrameworkType):
        self.framework_type = framework_type
        self.last_health_check = None
        self.health_status = 'unknown'
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for this specific framework."""
        return {
            'framework': self.framework_type.value,
            'status': self.health_status,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None
        }


class SystemResourceMonitor:
    """Monitor system resource utilization."""
    
    def __init__(self):
        self.last_update = None
    
    def update_metrics(self, metrics_collector: MetricsCollector) -> None:
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics_collector.set_gauge('system_cpu_usage_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics_collector.set_gauge('system_memory_usage_percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics_collector.set_gauge('system_disk_usage_percent', disk_percent)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system usage: {e}")
            return {'error': str(e)}


# Performance monitoring decorator
def monitor_performance(framework: FrameworkType, operation: str = None):
    """Decorator to monitor framework operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            success = False
            metadata = {}
            
            try:
                # Get input size if possible
                if args and hasattr(args[0], '__len__'):
                    metadata['input_size'] = len(args[0])
                
                result = func(*args, **kwargs)
                success = True
                return result
                
            except Exception as error:
                # Record error in error handler
                error_handler = get_error_handler(framework)
                error_handler.handle_error(error, op_name)
                raise
                
            finally:
                duration = time.time() - start_time
                
                # Record in global monitor
                global_monitor = get_global_monitor()
                global_monitor.record_operation(framework, op_name, duration, success, metadata)
        
        return wrapper
    return decorator


# Global monitor instance
_global_monitor: Optional[ResearchFrameworkMonitor] = None

def get_global_monitor() -> ResearchFrameworkMonitor:
    """Get global research framework monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResearchFrameworkMonitor()
    return _global_monitor


def shutdown_monitoring() -> None:
    """Shutdown global monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.shutdown()
        _global_monitor = None


# Export main classes and functions
__all__ = [
    'ResearchFrameworkMonitor',
    'MetricsCollector',
    'AlertRule',
    'Alert',
    'MetricValue',
    'PerformanceProfile',
    'FrameworkSpecificMonitor',
    'SystemResourceMonitor',
    'MetricType',
    'AlertSeverity',
    'monitor_performance',
    'get_global_monitor',
    'shutdown_monitoring'
]