"""
Advanced Monitoring and Alerting System.

This module provides comprehensive monitoring, alerting, and observability including:
- Real-time performance monitoring with custom metrics
- Intelligent alerting with anomaly detection
- Distributed tracing and logging correlation
- SLA monitoring and breach detection
- Automated incident response and escalation
- Performance profiling and bottleneck detection
- Business metrics tracking and reporting
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from enum import Enum, auto
import psutil
import threading
import logging
from urllib.parse import urljoin
import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .logging_config import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class AlertStatus(Enum):
    """Alert status."""
    TRIGGERED = auto()
    ACKNOWLEDGED = auto()
    RESOLVED = auto()
    SUPPRESSED = auto()


class MonitoringMetric:
    """Represents a monitoring metric."""
    
    def __init__(self, name: str, description: str, metric_type: str = "gauge"):
        self.name = name
        self.description = description
        self.metric_type = metric_type  # gauge, counter, histogram
        self.values = deque(maxlen=10000)
        self.timestamps = deque(maxlen=10000)
        self.labels = {}
        self.lock = threading.Lock()
    
    def record(self, value: float, timestamp: Optional[datetime] = None, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self.lock:
            self.values.append(value)
            self.timestamps.append(timestamp or datetime.now())
            if labels:
                self.labels.update(labels)
    
    def get_recent_values(self, window_minutes: int = 5) -> Tuple[List[float], List[datetime]]:
        """Get recent values within time window."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_values = []
            recent_timestamps = []
            
            for value, timestamp in zip(self.values, self.timestamps):
                if timestamp > cutoff_time:
                    recent_values.append(value)
                    recent_timestamps.append(timestamp)
            
            return recent_values, recent_timestamps
    
    def get_statistics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Get statistical summary of recent values."""
        values, _ = self.get_recent_values(window_minutes)
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., ">" for greater than
    threshold: float
    severity: AlertSeverity
    evaluation_window_minutes: int = 5
    min_data_points: int = 3
    suppress_duration_minutes: int = 60
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = []


@dataclass
class Alert:
    """Represents an active alert."""
    alert_id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    trigger_value: Optional[float] = None
    threshold: Optional[float] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class AnomalyDetector:
    """Detects anomalies in metric time series."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_windows = {}
        
    def detect_anomaly(self, metric: MonitoringMetric, window_minutes: int = 60) -> Tuple[bool, float, str]:
        """
        Detect if current metric value is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score, description)
        """
        
        values, timestamps = metric.get_recent_values(window_minutes)
        
        if len(values) < 10:  # Not enough data
            return False, 0.0, "Insufficient data for anomaly detection"
        
        recent_value = values[-1]
        historical_values = values[:-1]
        
        # Statistical anomaly detection using z-score
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:  # No variation
            return False, 0.0, "No variation in metric values"
        
        z_score = abs((recent_value - mean_val) / std_val)
        is_anomaly = z_score > self.sensitivity
        
        if is_anomaly:
            direction = "higher" if recent_value > mean_val else "lower"
            description = f"Value {recent_value:.2f} is {z_score:.2f} standard deviations {direction} than recent average {mean_val:.2f}"
        else:
            description = f"Value {recent_value:.2f} is within normal range (z-score: {z_score:.2f})"
        
        return is_anomaly, z_score, description


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        raise NotImplementedError
        

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Email configuration
            smtp_server = self.config.get('smtp_server', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            from_email = self.config.get('from_email')
            to_emails = self.config.get('to_emails', [])
            
            if not all([from_email, to_emails]):
                logger.error("Email configuration incomplete")
                return False
            
            # Create message
            message = MimeMultipart()
            message['From'] = from_email
            message['To'] = ', '.join(to_emails)
            message['Subject'] = f"[{alert.severity.name}] {alert.name}"
            
            body = f"""
Alert Details:
- Name: {alert.name}
- Description: {alert.description}
- Severity: {alert.severity.name}
- Status: {alert.status.name}
- Triggered At: {alert.triggered_at}
- Trigger Value: {alert.trigger_value}
- Threshold: {alert.threshold}

Context:
{json.dumps(alert.context, indent=2, default=str)}
            """
            
            message.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(message)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            webhook_url = self.config.get('webhook_url')
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            timeout = self.config.get('timeout', 30)
            
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'name': alert.name,
                'description': alert.description,
                'severity': alert.severity.name,
                'status': alert.status.name,
                'triggered_at': alert.triggered_at.isoformat(),
                'trigger_value': alert.trigger_value,
                'threshold': alert.threshold,
                'context': alert.context
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            logger.info(f"Webhook alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class SLAMonitor:
    """Monitors Service Level Agreements."""
    
    def __init__(self):
        self.sla_configs = {}
        self.sla_status = {}
        
    def register_sla(self, sla_id: str, name: str, target_value: float, 
                     metric_name: str, aggregation: str = "average",
                     window_minutes: int = 60, threshold_type: str = "min") -> None:
        """Register an SLA to monitor."""
        
        self.sla_configs[sla_id] = {
            'name': name,
            'target_value': target_value,
            'metric_name': metric_name,
            'aggregation': aggregation,
            'window_minutes': window_minutes,
            'threshold_type': threshold_type  # 'min' or 'max'
        }
        
        self.sla_status[sla_id] = {
            'current_value': None,
            'target_value': target_value,
            'compliance': None,
            'last_check': None,
            'breach_count_24h': 0
        }
        
        logger.info(f"Registered SLA: {name} ({sla_id})")
    
    def check_sla_compliance(self, sla_id: str, metrics_registry: Dict[str, MonitoringMetric]) -> Dict[str, Any]:
        """Check SLA compliance."""
        
        if sla_id not in self.sla_configs:
            raise ValueError(f"SLA {sla_id} not registered")
        
        config = self.sla_configs[sla_id]
        metric_name = config['metric_name']
        
        if metric_name not in metrics_registry:
            logger.warning(f"Metric {metric_name} not found for SLA {sla_id}")
            return self.sla_status[sla_id]
        
        metric = metrics_registry[metric_name]
        values, _ = metric.get_recent_values(config['window_minutes'])
        
        if not values:
            logger.warning(f"No recent data for SLA {sla_id}")
            return self.sla_status[sla_id]
        
        # Calculate aggregated value
        if config['aggregation'] == 'average':
            current_value = np.mean(values)
        elif config['aggregation'] == 'median':
            current_value = np.median(values)
        elif config['aggregation'] == 'p95':
            current_value = np.percentile(values, 95)
        elif config['aggregation'] == 'p99':
            current_value = np.percentile(values, 99)
        elif config['aggregation'] == 'max':
            current_value = np.max(values)
        elif config['aggregation'] == 'min':
            current_value = np.min(values)
        else:
            current_value = np.mean(values)  # Default
        
        # Check compliance
        target_value = config['target_value']
        threshold_type = config['threshold_type']
        
        if threshold_type == 'min':
            compliance = current_value >= target_value
        else:  # 'max'
            compliance = current_value <= target_value
        
        # Update status
        self.sla_status[sla_id].update({
            'current_value': current_value,
            'compliance': compliance,
            'last_check': datetime.now()
        })
        
        # Track breaches
        if not compliance:
            # Check if this is a new breach (within last hour)
            last_24h = datetime.now() - timedelta(hours=24)
            # Simplified breach counting - in production, would track more precisely
            self.sla_status[sla_id]['breach_count_24h'] += 1
        
        return self.sla_status[sla_id]
    
    def get_sla_report(self) -> Dict[str, Any]:
        """Get comprehensive SLA report."""
        report = {
            'total_slas': len(self.sla_configs),
            'compliant_slas': 0,
            'breached_slas': 0,
            'sla_details': {}
        }
        
        for sla_id, status in self.sla_status.items():
            if status['compliance'] is True:
                report['compliant_slas'] += 1
            elif status['compliance'] is False:
                report['breached_slas'] += 1
            
            config = self.sla_configs[sla_id]
            report['sla_details'][sla_id] = {
                'name': config['name'],
                'target_value': config['target_value'],
                'current_value': status['current_value'],
                'compliance': status['compliance'],
                'breach_count_24h': status['breach_count_24h']
            }
        
        return report


class PerformanceProfiler:
    """Profiles system and application performance."""
    
    def __init__(self, profiling_interval: int = 10):
        self.profiling_interval = profiling_interval
        self.system_metrics = deque(maxlen=1000)
        self.application_metrics = deque(maxlen=1000)
        self.profiling_active = False
        self.profiling_thread = None
        
    def start_profiling(self) -> None:
        """Start performance profiling."""
        if self.profiling_active:
            return
        
        self.profiling_active = True
        self.profiling_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profiling_thread.start()
        logger.info("Performance profiling started")
    
    def stop_profiling(self) -> None:
        """Stop performance profiling."""
        self.profiling_active = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5)
        logger.info("Performance profiling stopped")
    
    def _profile_loop(self) -> None:
        """Main profiling loop."""
        while self.profiling_active:
            try:
                # Collect system metrics
                system_snapshot = self._collect_system_metrics()
                self.system_metrics.append(system_snapshot)
                
                # Collect application metrics
                app_snapshot = self._collect_application_metrics()
                self.application_metrics.append(app_snapshot)
                
            except Exception as e:
                logger.error(f"Error in profiling loop: {e}")
            
            time.sleep(self.profiling_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_mb = memory.total / (1024 * 1024)
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network metrics (simplified)
        network = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'total_mb': memory_mb,
                'used_mb': memory_used_mb,
                'available_mb': memory_available_mb,
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk_total_gb,
                'used_gb': disk_used_gb,
                'free_gb': disk_free_gb,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        }
    
    def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        
        # Get current process
        process = psutil.Process()
        
        # Process metrics
        process_cpu = process.cpu_percent()
        process_memory = process.memory_info()
        process_threads = process.num_threads()
        
        # Thread metrics
        active_threads = threading.active_count()
        
        return {
            'timestamp': datetime.now(),
            'process': {
                'cpu_percent': process_cpu,
                'memory_rss_mb': process_memory.rss / (1024 * 1024),
                'memory_vms_mb': process_memory.vms / (1024 * 1024),
                'num_threads': process_threads
            },
            'threading': {
                'active_threads': active_threads
            }
        }
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics if m['timestamp'] > cutoff_time]
        recent_app = [m for m in self.application_metrics if m['timestamp'] > cutoff_time]
        
        if not recent_system or not recent_app:
            return {'error': 'No recent performance data available'}
        
        # Calculate system averages
        cpu_values = [m['cpu']['percent'] for m in recent_system]
        memory_values = [m['memory']['percent'] for m in recent_system]
        disk_values = [m['disk']['percent'] for m in recent_system]
        
        # Calculate application averages
        app_cpu_values = [m['process']['cpu_percent'] for m in recent_app]
        app_memory_values = [m['process']['memory_rss_mb'] for m in recent_app]
        
        return {
            'window_minutes': window_minutes,
            'system': {
                'cpu_percent_avg': np.mean(cpu_values),
                'cpu_percent_max': np.max(cpu_values),
                'memory_percent_avg': np.mean(memory_values),
                'memory_percent_max': np.max(memory_values),
                'disk_percent_avg': np.mean(disk_values),
                'disk_percent_max': np.max(disk_values)
            },
            'application': {
                'cpu_percent_avg': np.mean(app_cpu_values),
                'cpu_percent_max': np.max(app_cpu_values),
                'memory_mb_avg': np.mean(app_memory_values),
                'memory_mb_max': np.max(app_memory_values)
            },
            'data_points': len(recent_system)
        }


class AdvancedMonitoringSystem:
    """Main advanced monitoring and alerting system."""
    
    def __init__(self):
        self.metrics_registry = {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.resolved_alerts = deque(maxlen=10000)
        self.notification_channels = {}
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.sla_monitor = SLAMonitor()
        self.performance_profiler = PerformanceProfiler()
        
        # State
        self.monitoring_active = False
        self.monitoring_thread = None
        self.evaluation_interval = 30  # seconds
        
        # Setup default metrics and alerts
        self._setup_default_monitoring()
    
    def _setup_default_monitoring(self) -> None:
        """Setup default metrics and alert rules."""
        
        # Register default metrics
        self.register_metric('system.cpu_percent', 'System CPU Usage Percentage')
        self.register_metric('system.memory_percent', 'System Memory Usage Percentage')
        self.register_metric('system.disk_percent', 'System Disk Usage Percentage')
        self.register_metric('app.prediction_latency_ms', 'Prediction Latency in Milliseconds')
        self.register_metric('app.prediction_throughput', 'Predictions per Second')
        self.register_metric('app.error_rate', 'Error Rate Percentage')
        self.register_metric('app.model_accuracy', 'Model Accuracy')
        
        # Register default alert rules
        self.register_alert_rule(AlertRule(
            rule_id='high_cpu',
            name='High CPU Usage',
            description='System CPU usage is high',
            metric_name='system.cpu_percent',
            condition='>',
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            notification_channels=['email', 'webhook']
        ))
        
        self.register_alert_rule(AlertRule(
            rule_id='high_memory',
            name='High Memory Usage',
            description='System memory usage is high',
            metric_name='system.memory_percent',
            condition='>',
            threshold=90.0,
            severity=AlertSeverity.ERROR,
            notification_channels=['email', 'webhook']
        ))
        
        self.register_alert_rule(AlertRule(
            rule_id='high_prediction_latency',
            name='High Prediction Latency',
            description='Prediction latency is too high',
            metric_name='app.prediction_latency_ms',
            condition='>',
            threshold=1000.0,
            severity=AlertSeverity.WARNING,
            notification_channels=['webhook']
        ))
        
        self.register_alert_rule(AlertRule(
            rule_id='high_error_rate',
            name='High Error Rate',
            description='Application error rate is high',
            metric_name='app.error_rate',
            condition='>',
            threshold=5.0,
            severity=AlertSeverity.ERROR,
            notification_channels=['email', 'webhook']
        ))
        
        # Register default SLAs
        self.sla_monitor.register_sla(
            'prediction_latency_sla',
            'Prediction Latency SLA',
            target_value=500.0,  # max 500ms
            metric_name='app.prediction_latency_ms',
            aggregation='p95',
            threshold_type='max'
        )
        
        self.sla_monitor.register_sla(
            'system_availability_sla',
            'System Availability SLA',
            target_value=99.9,  # 99.9% uptime
            metric_name='app.availability_percent',
            aggregation='average',
            threshold_type='min'
        )
    
    def register_metric(self, name: str, description: str, metric_type: str = "gauge") -> None:
        """Register a new metric."""
        self.metrics_registry[name] = MonitoringMetric(name, description, metric_type)
        logger.info(f"Registered metric: {name}")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if name in self.metrics_registry:
            self.metrics_registry[name].record(value, labels=labels)
        else:
            logger.warning(f"Metric {name} not registered")
    
    def register_alert_rule(self, alert_rule: AlertRule) -> None:
        """Register an alert rule."""
        self.alert_rules[alert_rule.rule_id] = alert_rule
        logger.info(f"Registered alert rule: {alert_rule.name}")
    
    def register_notification_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self.notification_channels[channel.name] = channel
        logger.info(f"Registered notification channel: {channel.name}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.performance_profiler.start_profiling()
        
        logger.info("Advanced monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.performance_profiler.stop_profiling()
        
        logger.info("Advanced monitoring system stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Evaluate alert rules
                self._evaluate_alert_rules()
                
                # Check SLA compliance
                self._check_sla_compliance()
                
                # Detect anomalies
                self._detect_anomalies()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.evaluation_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('system.cpu_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('system.memory_percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk_percent', disk.percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules."""
        for rule_id, rule in self.alert_rules.items():
            try:
                self._evaluate_single_alert_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    def _evaluate_single_alert_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        
        if rule.metric_name not in self.metrics_registry:
            return
        
        metric = self.metrics_registry[rule.metric_name]
        values, timestamps = metric.get_recent_values(rule.evaluation_window_minutes)
        
        if len(values) < rule.min_data_points:
            return
        
        # Get current value (latest)
        current_value = values[-1]
        
        # Evaluate condition
        triggered = False
        if rule.condition == '>':
            triggered = current_value > rule.threshold
        elif rule.condition == '<':
            triggered = current_value < rule.threshold
        elif rule.condition == '>=':
            triggered = current_value >= rule.threshold
        elif rule.condition == '<=':
            triggered = current_value <= rule.threshold
        elif rule.condition == '==':
            triggered = current_value == rule.threshold
        elif rule.condition == '!=':
            triggered = current_value != rule.threshold
        
        # Handle alert state
        if triggered and rule.rule_id not in self.active_alerts:
            # New alert
            alert = Alert(
                alert_id=f"alert_{rule.rule_id}_{int(time.time())}",
                rule_id=rule.rule_id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.TRIGGERED,
                triggered_at=datetime.now(),
                trigger_value=current_value,
                threshold=rule.threshold,
                context={
                    'metric_name': rule.metric_name,
                    'condition': rule.condition,
                    'evaluation_window': rule.evaluation_window_minutes,
                    'recent_values': values[-5:]  # Last 5 values for context
                }
            )
            
            self.active_alerts[rule.rule_id] = alert
            self._send_alert_notifications(alert, rule)
            logger.warning(f"Alert triggered: {alert.name} (value: {current_value}, threshold: {rule.threshold})")
            
        elif not triggered and rule.rule_id in self.active_alerts:
            # Alert resolved
            alert = self.active_alerts[rule.rule_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            self.resolved_alerts.append(alert)
            del self.active_alerts[rule.rule_id]
            
            logger.info(f"Alert resolved: {alert.name}")
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    channel = self.notification_channels[channel_name]
                    success = channel.send_notification(alert)
                    if not success:
                        logger.error(f"Failed to send notification via {channel_name}")
                except Exception as e:
                    logger.error(f"Error sending notification via {channel_name}: {e}")
            else:
                logger.warning(f"Notification channel {channel_name} not registered")
    
    def _check_sla_compliance(self) -> None:
        """Check SLA compliance."""
        try:
            for sla_id in self.sla_monitor.sla_configs.keys():
                status = self.sla_monitor.check_sla_compliance(sla_id, self.metrics_registry)
                
                # Generate alert for SLA breach
                if status.get('compliance') is False:
                    # Create SLA breach alert
                    sla_config = self.sla_monitor.sla_configs[sla_id]
                    alert_rule_id = f"sla_breach_{sla_id}"
                    
                    if alert_rule_id not in self.active_alerts:
                        alert = Alert(
                            alert_id=f"sla_alert_{sla_id}_{int(time.time())}",
                            rule_id=alert_rule_id,
                            name=f"SLA Breach: {sla_config['name']}",
                            description=f"SLA {sla_config['name']} has been breached",
                            severity=AlertSeverity.ERROR,
                            status=AlertStatus.TRIGGERED,
                            triggered_at=datetime.now(),
                            trigger_value=status['current_value'],
                            threshold=status['target_value'],
                            context={'sla_id': sla_id, 'sla_config': sla_config}
                        )
                        
                        self.active_alerts[alert_rule_id] = alert
                        logger.error(f"SLA breach detected: {sla_config['name']}")
        
        except Exception as e:
            logger.error(f"Error checking SLA compliance: {e}")
    
    def _detect_anomalies(self) -> None:
        """Detect anomalies in metrics."""
        
        try:
            for metric_name, metric in self.metrics_registry.items():
                is_anomaly, score, description = self.anomaly_detector.detect_anomaly(metric)
                
                if is_anomaly:
                    alert_rule_id = f"anomaly_{metric_name}"
                    
                    if alert_rule_id not in self.active_alerts:
                        alert = Alert(
                            alert_id=f"anomaly_alert_{metric_name}_{int(time.time())}",
                            rule_id=alert_rule_id,
                            name=f"Anomaly Detected: {metric_name}",
                            description=f"Anomalous behavior detected in {metric_name}: {description}",
                            severity=AlertSeverity.WARNING,
                            status=AlertStatus.TRIGGERED,
                            triggered_at=datetime.now(),
                            context={
                                'metric_name': metric_name,
                                'anomaly_score': score,
                                'anomaly_description': description
                            }
                        )
                        
                        self.active_alerts[alert_rule_id] = alert
                        logger.warning(f"Anomaly detected in {metric_name}: {description}")
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        
        # Metrics summary
        metrics_summary = {}
        for name, metric in self.metrics_registry.items():
            stats = metric.get_statistics(window_minutes=60)
            if stats:
                metrics_summary[name] = stats
        
        # Active alerts
        active_alerts_summary = []
        for alert in self.active_alerts.values():
            active_alerts_summary.append({
                'alert_id': alert.alert_id,
                'name': alert.name,
                'severity': alert.severity.name,
                'status': alert.status.name,
                'triggered_at': alert.triggered_at.isoformat(),
                'trigger_value': alert.trigger_value,
                'threshold': alert.threshold
            })
        
        # SLA report
        sla_report = self.sla_monitor.get_sla_report()
        
        # Performance summary
        performance_summary = self.performance_profiler.get_performance_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy' if not self.active_alerts else 'alerts_active',
            'metrics_summary': metrics_summary,
            'active_alerts': active_alerts_summary,
            'active_alerts_count': len(self.active_alerts),
            'resolved_alerts_24h': len([a for a in self.resolved_alerts 
                                       if a.resolved_at and a.resolved_at > datetime.now() - timedelta(days=1)]),
            'sla_report': sla_report,
            'performance_summary': performance_summary,
            'monitoring_active': self.monitoring_active
        }
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        
        for rule_id, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.context['acknowledged_by'] = acknowledged_by
                
                logger.info(f"Alert acknowledged: {alert.name} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Manually resolve an active alert."""
        
        for rule_id, alert in list(self.active_alerts.items()):
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.context['resolved_by'] = resolved_by
                
                self.resolved_alerts.append(alert)
                del self.active_alerts[rule_id]
                
                logger.info(f"Alert resolved: {alert.name} by {resolved_by}")
                return True
        
        return False


# Global monitoring instance
monitoring_system = AdvancedMonitoringSystem()


# Decorators for monitoring
def monitor_performance(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function performance."""
    
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"app.{func.__name__}_duration_ms"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                monitoring_system.record_metric(metric_name, duration_ms, labels)
                monitoring_system.record_metric(f"app.{func.__name__}_success_count", 1)
                
                return result
                
            except Exception as e:
                # Record error metrics
                monitoring_system.record_metric(f"app.{func.__name__}_error_count", 1)
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    print("Advanced Monitoring and Alerting System")
    print("This system provides comprehensive monitoring, alerting, and observability.")