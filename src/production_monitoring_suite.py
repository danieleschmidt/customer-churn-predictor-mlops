"""
Production Monitoring Suite for MLOps Platform.

Comprehensive monitoring, alerting, and observability system for production ML workloads.
Includes real-time metrics, distributed tracing, log aggregation, and intelligent alerting.
"""

import asyncio
import time
import logging
import threading
import json
import os
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import hashlib
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import socket
import platform

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .validation import safe_write_json, safe_read_json

logger = get_logger(__name__)
metrics = get_metrics_collector()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class Alert:
    """Alert message structure."""
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    timestamp: float = field(default_factory=time.time)
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[float] = None
    acknowledgments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'tags': self.tags,
            'channels': [c.value for c in self.channels],
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'acknowledgments': self.acknowledgments
        }


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold_value: float
    duration: float = 60.0  # seconds
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""


@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    timestamp: float = field(default_factory=time.time)
    hostname: str = field(default_factory=socket.gethostname)
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    disk_free: float = 0.0
    network_io_sent: int = 0
    network_io_received: int = 0
    
    # Application metrics
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    active_sessions: int = 0
    
    # ML specific metrics
    model_prediction_count: int = 0
    model_prediction_latency: float = 0.0
    model_accuracy: float = 0.0
    data_drift_score: float = 0.0
    feature_importance_changes: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'hostname': self.hostname,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'disk_free': self.disk_free,
            'network_io_sent': self.network_io_sent,
            'network_io_received': self.network_io_received,
            'response_time_avg': self.response_time_avg,
            'response_time_p95': self.response_time_p95,
            'response_time_p99': self.response_time_p99,
            'request_rate': self.request_rate,
            'error_rate': self.error_rate,
            'active_sessions': self.active_sessions,
            'model_prediction_count': self.model_prediction_count,
            'model_prediction_latency': self.model_prediction_latency,
            'model_accuracy': self.model_accuracy,
            'data_drift_score': self.data_drift_score,
            'feature_importance_changes': self.feature_importance_changes
        }


class MetricsCollector:
    """
    Advanced metrics collection and aggregation system.
    """
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.is_running = False
        self.metrics_history = deque(maxlen=1000)
        self.custom_metrics = {}
        self.metric_calculators = {}
        self.lock = threading.RLock()
        
    async def start_collection(self):
        """Start metrics collection."""
        self.is_running = True
        logger.info("Metrics collection started")
        
        while self.is_running:
            try:
                system_metrics = await self._collect_system_metrics()
                
                with self.lock:
                    self.metrics_history.append(system_metrics)
                    
                # Update global metrics
                self._update_global_metrics(system_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
                
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        system_metrics = SystemMetrics()
        
        try:
            # CPU metrics
            system_metrics.cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            system_metrics.memory_usage = memory.percent
            system_metrics.memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            system_metrics.disk_usage = (disk.used / disk.total) * 100
            system_metrics.disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                system_metrics.network_io_sent = net_io.bytes_sent
                system_metrics.network_io_received = net_io.bytes_recv
                
            # Calculate custom metrics
            await self._calculate_custom_metrics(system_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return system_metrics
        
    async def _calculate_custom_metrics(self, system_metrics: SystemMetrics):
        """Calculate custom application-specific metrics."""
        # Response time metrics (would be calculated from actual request data)
        with self.lock:
            if len(self.metrics_history) >= 2:
                recent_metrics = list(self.metrics_history)[-10:]
                response_times = [m.response_time_avg for m in recent_metrics if m.response_time_avg > 0]
                
                if response_times:
                    system_metrics.response_time_avg = np.mean(response_times)
                    system_metrics.response_time_p95 = np.percentile(response_times, 95)
                    system_metrics.response_time_p99 = np.percentile(response_times, 99)
                    
        # Custom metric calculations
        for name, calculator in self.metric_calculators.items():
            try:
                value = await calculator()
                setattr(system_metrics, name, value)
            except Exception as e:
                logger.error(f"Error calculating custom metric {name}: {e}")
                
    def register_custom_metric_calculator(self, name: str, calculator: Callable):
        """Register custom metric calculator."""
        self.metric_calculators[name] = calculator
        logger.info(f"Registered custom metric calculator: {name}")
        
    def _update_global_metrics(self, system_metrics: SystemMetrics):
        """Update global metrics instance."""
        metrics.gauge('system_cpu_usage', system_metrics.cpu_usage)
        metrics.gauge('system_memory_usage', system_metrics.memory_usage)
        metrics.gauge('system_disk_usage', system_metrics.disk_usage)
        metrics.gauge('system_response_time_avg', system_metrics.response_time_avg)
        metrics.gauge('system_error_rate', system_metrics.error_rate)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            if not self.metrics_history:
                return {}
                
            recent_metrics = list(self.metrics_history)[-10:]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.metrics_history),
                'recent_samples': len(recent_metrics),
                'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
                'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
                'avg_response_time': np.mean([m.response_time_avg for m in recent_metrics if m.response_time_avg > 0]),
                'latest_metrics': recent_metrics[-1].to_dict() if recent_metrics else {}
            }
            
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_running = False
        logger.info("Metrics collection stopped")


class AlertManager:
    """
    Intelligent alert management system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.alert_history = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self.thresholds: List[MetricThreshold] = []
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.suppression_rules = {}
        self.escalation_rules = []
        
        # Load default thresholds
        self._setup_default_thresholds()
        self._setup_default_handlers()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load alert manager configuration."""
        default_config = {
            'alert_retention_hours': 24,
            'max_alerts_per_hour': 100,
            'enable_alert_suppression': True,
            'enable_escalation': True,
            'default_channels': [AlertChannel.LOG.value],
            'escalation_timeout_minutes': 15
        }
        
        if config_path and Path(config_path).exists():
            try:
                user_config = safe_read_json(config_path)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading alert config: {e}")
                
        return default_config
        
    def _setup_default_thresholds(self):
        """Setup default metric thresholds."""
        default_thresholds = [
            MetricThreshold("cpu_usage", ">", 80.0, 300.0, AlertSeverity.WARNING, "High CPU usage"),
            MetricThreshold("cpu_usage", ">", 95.0, 60.0, AlertSeverity.CRITICAL, "Critical CPU usage"),
            MetricThreshold("memory_usage", ">", 85.0, 300.0, AlertSeverity.WARNING, "High memory usage"),
            MetricThreshold("memory_usage", ">", 95.0, 60.0, AlertSeverity.CRITICAL, "Critical memory usage"),
            MetricThreshold("disk_usage", ">", 85.0, 600.0, AlertSeverity.WARNING, "High disk usage"),
            MetricThreshold("disk_usage", ">", 95.0, 300.0, AlertSeverity.CRITICAL, "Critical disk usage"),
            MetricThreshold("response_time_avg", ">", 2.0, 300.0, AlertSeverity.WARNING, "High response time"),
            MetricThreshold("response_time_avg", ">", 5.0, 60.0, AlertSeverity.CRITICAL, "Critical response time"),
            MetricThreshold("error_rate", ">", 0.05, 300.0, AlertSeverity.WARNING, "High error rate"),
            MetricThreshold("error_rate", ">", 0.1, 60.0, AlertSeverity.CRITICAL, "Critical error rate")
        ]
        
        self.thresholds.extend(default_thresholds)
        
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        self.alert_handlers[AlertChannel.LOG] = self._log_alert_handler
        
    async def check_thresholds(self, system_metrics: SystemMetrics):
        """Check metrics against defined thresholds."""
        for threshold in self.thresholds:
            try:
                metric_value = getattr(system_metrics, threshold.metric_name, None)
                if metric_value is None:
                    continue
                    
                if self._evaluate_threshold(metric_value, threshold):
                    await self._trigger_alert(threshold, metric_value, system_metrics)
                    
            except Exception as e:
                logger.error(f"Error checking threshold {threshold.metric_name}: {e}")
                
    def _evaluate_threshold(self, value: float, threshold: MetricThreshold) -> bool:
        """Evaluate if metric value breaches threshold."""
        operators = {
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t,
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '==': lambda v, t: v == t,
            '!=': lambda v, t: v != t
        }
        
        op_func = operators.get(threshold.operator)
        if not op_func:
            logger.error(f"Unknown operator: {threshold.operator}")
            return False
            
        return op_func(value, threshold.threshold_value)
        
    async def _trigger_alert(self, threshold: MetricThreshold, value: float, metrics: SystemMetrics):
        """Trigger alert for threshold breach."""
        alert_key = f"{threshold.metric_name}_{threshold.operator}_{threshold.threshold_value}"
        
        # Check if alert is already active (avoid spam)
        if alert_key in self.active_alerts:
            return
            
        # Create alert
        alert = Alert(
            severity=threshold.severity,
            title=f"Threshold breach: {threshold.metric_name}",
            message=f"{threshold.description}. Current value: {value:.2f}, threshold: {threshold.threshold_value}",
            source=f"{metrics.hostname}:{threshold.metric_name}",
            tags={
                'metric': threshold.metric_name,
                'threshold': str(threshold.threshold_value),
                'current_value': str(value),
                'hostname': metrics.hostname
            },
            channels=[AlertChannel.LOG]  # Default channels
        )
        
        # Apply suppression rules
        if self._should_suppress_alert(alert):
            logger.debug(f"Alert suppressed: {alert.title}")
            return
            
        # Send alert
        await self._send_alert(alert)
        
        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        if not self.config['enable_alert_suppression']:
            return False
            
        # Check rate limiting
        recent_alerts = [a for a in self.alert_history 
                        if time.time() - a.timestamp < 3600]  # Last hour
        
        if len(recent_alerts) >= self.config['max_alerts_per_hour']:
            return True
            
        # Check for similar alerts
        for existing_alert in list(self.alert_history)[-10:]:
            if (alert.source == existing_alert.source and 
                alert.title == existing_alert.title and
                time.time() - existing_alert.timestamp < 300):  # 5 minutes
                return True
                
        return False
        
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        logger.info(f"Sending alert: {alert.title}")
        
        for channel in alert.channels:
            handler = self.alert_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")
                    
    async def _log_alert_handler(self, alert: Alert):
        """Default log alert handler."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
        
    def register_alert_handler(self, channel: AlertChannel, handler: Callable):
        """Register custom alert handler."""
        self.alert_handlers[channel] = handler
        logger.info(f"Registered alert handler for channel: {channel.value}")
        
    def add_threshold(self, threshold: MetricThreshold):
        """Add custom threshold."""
        self.thresholds.append(threshold)
        logger.info(f"Added threshold for {threshold.metric_name}")
        
    def resolve_alert(self, alert_id: str, resolver: str = "system"):
        """Resolve active alert."""
        for alert_key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = time.time()
                alert.acknowledgments.append(resolver)
                del self.active_alerts[alert_key]
                logger.info(f"Alert resolved: {alert_id} by {resolver}")
                return
                
        logger.warning(f"Alert not found for resolution: {alert_id}")
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {'total_alerts': 0}
            
        alerts = list(self.alert_history)
        severity_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
            source_counts[alert.source] += 1
            
        return {
            'total_alerts': len(alerts),
            'active_alerts': len(self.active_alerts),
            'severity_distribution': dict(severity_counts),
            'top_sources': dict(list(source_counts.items())[:10]),
            'recent_alert_rate': len([a for a in alerts if time.time() - a.timestamp < 3600]),
            'resolution_rate': len([a for a in alerts if a.resolved]) / len(alerts) if alerts else 0
        }


class PerformanceProfiler:
    """
    Performance profiling and bottleneck detection.
    """
    
    def __init__(self):
        self.function_metrics = defaultdict(list)
        self.request_metrics = deque(maxlen=1000)
        self.bottlenecks = []
        self.lock = threading.RLock()
        
    def record_function_execution(self, function_name: str, execution_time: float, success: bool = True):
        """Record function execution metrics."""
        with self.lock:
            self.function_metrics[function_name].append({
                'timestamp': time.time(),
                'execution_time': execution_time,
                'success': success
            })
            
            # Keep only recent metrics (last 1000 per function)
            if len(self.function_metrics[function_name]) > 1000:
                self.function_metrics[function_name] = self.function_metrics[function_name][-1000:]
                
    def record_request_metrics(self, endpoint: str, method: str, status_code: int, 
                              response_time: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics."""
        with self.lock:
            self.request_metrics.append({
                'timestamp': time.time(),
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time,
                'request_size': request_size,
                'response_size': response_size,
                'success': status_code < 400
            })
            
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        with self.lock:
            # Function bottlenecks
            for func_name, metrics in self.function_metrics.items():
                if len(metrics) < 10:
                    continue
                    
                recent_metrics = metrics[-50:]  # Last 50 executions
                avg_time = np.mean([m['execution_time'] for m in recent_metrics])
                p95_time = np.percentile([m['execution_time'] for m in recent_metrics], 95)
                success_rate = np.mean([m['success'] for m in recent_metrics])
                
                # Detect slow functions
                if avg_time > 1.0 or p95_time > 5.0:
                    bottlenecks.append({
                        'type': 'slow_function',
                        'function': func_name,
                        'avg_execution_time': avg_time,
                        'p95_execution_time': p95_time,
                        'success_rate': success_rate,
                        'severity': 'high' if p95_time > 10.0 else 'medium'
                    })
                    
                # Detect failing functions
                if success_rate < 0.8:
                    bottlenecks.append({
                        'type': 'failing_function',
                        'function': func_name,
                        'success_rate': success_rate,
                        'avg_execution_time': avg_time,
                        'severity': 'critical' if success_rate < 0.5 else 'high'
                    })
                    
            # Request bottlenecks
            if len(self.request_metrics) >= 10:
                recent_requests = list(self.request_metrics)[-100:]
                
                # Group by endpoint
                endpoint_metrics = defaultdict(list)
                for req in recent_requests:
                    endpoint_metrics[req['endpoint']].append(req)
                    
                for endpoint, reqs in endpoint_metrics.items():
                    if len(reqs) < 5:
                        continue
                        
                    avg_response_time = np.mean([r['response_time'] for r in reqs])
                    error_rate = 1 - np.mean([r['success'] for r in reqs])
                    
                    if avg_response_time > 2.0:
                        bottlenecks.append({
                            'type': 'slow_endpoint',
                            'endpoint': endpoint,
                            'avg_response_time': avg_response_time,
                            'error_rate': error_rate,
                            'request_count': len(reqs),
                            'severity': 'high' if avg_response_time > 5.0 else 'medium'
                        })
                        
                    if error_rate > 0.1:
                        bottlenecks.append({
                            'type': 'failing_endpoint',
                            'endpoint': endpoint,
                            'error_rate': error_rate,
                            'avg_response_time': avg_response_time,
                            'request_count': len(reqs),
                            'severity': 'critical' if error_rate > 0.3 else 'high'
                        })
                        
        self.bottlenecks = bottlenecks
        return bottlenecks
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        bottlenecks = self.detect_bottlenecks()
        
        with self.lock:
            function_count = len(self.function_metrics)
            request_count = len(self.request_metrics)
            
            if self.request_metrics:
                recent_requests = list(self.request_metrics)[-100:]
                avg_response_time = np.mean([r['response_time'] for r in recent_requests])
                error_rate = 1 - np.mean([r['success'] for r in recent_requests])
            else:
                avg_response_time = 0
                error_rate = 0
                
        return {
            'timestamp': datetime.now().isoformat(),
            'functions_monitored': function_count,
            'requests_monitored': request_count,
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'bottlenecks_detected': len(bottlenecks),
            'bottlenecks': bottlenecks
        }


class MonitoringSuite:
    """
    Main monitoring suite coordinating all monitoring components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector(
            collection_interval=self.config.get('metrics_collection_interval', 10.0)
        )
        self.alert_manager = AlertManager()
        self.performance_profiler = PerformanceProfiler()
        self.is_running = False
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            'metrics_collection_interval': 10.0,
            'enable_alerts': True,
            'enable_performance_profiling': True,
            'monitoring_port': 9090
        }
        
        if config_path and Path(config_path).exists():
            try:
                user_config = safe_read_json(config_path)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading monitoring config: {e}")
                
        return default_config
        
    async def start(self):
        """Start monitoring suite."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Production monitoring suite started")
        
        # Start metrics collection
        asyncio.create_task(self.metrics_collector.start_collection())
        
        # Start alert checking
        if self.config['enable_alerts']:
            asyncio.create_task(self._alert_checking_loop())
            
        # Start bottleneck detection
        if self.config['enable_performance_profiling']:
            asyncio.create_task(self._bottleneck_detection_loop())
            
    async def _alert_checking_loop(self):
        """Alert checking loop."""
        while self.is_running:
            try:
                if self.metrics_collector.metrics_history:
                    latest_metrics = self.metrics_collector.metrics_history[-1]
                    await self.alert_manager.check_thresholds(latest_metrics)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(30)
                
    async def _bottleneck_detection_loop(self):
        """Bottleneck detection loop."""
        while self.is_running:
            try:
                bottlenecks = self.performance_profiler.detect_bottlenecks()
                
                # Create alerts for severe bottlenecks
                for bottleneck in bottlenecks:
                    if bottleneck['severity'] in ['high', 'critical']:
                        alert_severity = AlertSeverity.CRITICAL if bottleneck['severity'] == 'critical' else AlertSeverity.ERROR
                        
                        alert = Alert(
                            severity=alert_severity,
                            title=f"Performance bottleneck detected: {bottleneck['type']}",
                            message=json.dumps(bottleneck, indent=2),
                            source="performance_profiler",
                            tags=bottleneck
                        )
                        
                        await self.alert_manager._send_alert(alert)
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Bottleneck detection error: {e}")
                await asyncio.sleep(300)
                
    def record_function_performance(self, function_name: str, execution_time: float, success: bool = True):
        """Record function performance metrics."""
        if self.config['enable_performance_profiling']:
            self.performance_profiler.record_function_execution(function_name, execution_time, success)
            
    def record_request_performance(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Record request performance metrics."""
        if self.config['enable_performance_profiling']:
            self.performance_profiler.record_request_metrics(endpoint, method, status_code, response_time)
            
    def add_custom_threshold(self, threshold: MetricThreshold):
        """Add custom alert threshold."""
        self.alert_manager.add_threshold(threshold)
        
    def register_custom_metric_calculator(self, name: str, calculator: Callable):
        """Register custom metric calculator."""
        self.metrics_collector.register_custom_metric_calculator(name, calculator)
        
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            },
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'alert_statistics': self.alert_manager.get_alert_statistics(),
            'performance_summary': self.performance_profiler.get_performance_summary(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'monitoring_status': {
                'is_running': self.is_running,
                'metrics_collection_enabled': self.metrics_collector.is_running,
                'alerts_enabled': self.config['enable_alerts'],
                'performance_profiling_enabled': self.config['enable_performance_profiling']
            }
        }
        
    def stop(self):
        """Stop monitoring suite."""
        self.is_running = False
        self.metrics_collector.stop_collection()
        logger.info("Production monitoring suite stopped")


# Global instance
_monitoring_suite: Optional[MonitoringSuite] = None


def get_monitoring_suite() -> MonitoringSuite:
    """Get global monitoring suite instance."""
    global _monitoring_suite
    if _monitoring_suite is None:
        _monitoring_suite = MonitoringSuite()
    return _monitoring_suite


def monitor_function_performance(func):
    """Decorator to monitor function performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            execution_time = time.time() - start_time
            monitoring_suite = get_monitoring_suite()
            monitoring_suite.record_function_performance(func.__name__, execution_time, success)
            
    return wrapper


async def initialize_monitoring():
    """Initialize monitoring suite."""
    monitoring_suite = get_monitoring_suite()
    await monitoring_suite.start()
    logger.info("Monitoring suite initialized")


if __name__ == "__main__":
    async def main():
        # Initialize monitoring
        await initialize_monitoring()
        
        # Simulate some activity
        monitoring_suite = get_monitoring_suite()
        
        @monitor_function_performance
        def test_function():
            import random
            time.sleep(random.uniform(0.1, 2.0))
            if random.random() < 0.1:
                raise Exception("Random error")
            return "success"
            
        # Run test functions
        for i in range(20):
            try:
                result = test_function()
                print(f"Test {i}: {result}")
            except Exception as e:
                print(f"Test {i}: Failed - {e}")
            await asyncio.sleep(1)
            
        # Get dashboard data
        dashboard_data = monitoring_suite.get_monitoring_dashboard_data()
        print(json.dumps(dashboard_data, indent=2))
        
    asyncio.run(main())