"""
Prometheus metrics collection for churn prediction application.

This module provides comprehensive metrics collection for monitoring
application performance, model behavior, and system health.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)


class PrometheusMetric:
    """Base class for Prometheus metrics."""
    
    def __init__(self, name: str, help_text: str, metric_type: str):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self.samples: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def to_prometheus_format(self) -> str:
        """Convert metric to Prometheus exposition format."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.metric_type}"
        ]
        
        with self._lock:
            for labels, value in self.samples.items():
                if labels:
                    lines.append(f"{self.name}{{{labels}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        
        return "\n".join(lines)


class Counter(PrometheusMetric):
    """Prometheus Counter metric."""
    
    def __init__(self, name: str, help_text: str):
        super().__init__(name, help_text, "counter")
    
    def inc(self, amount: float = 1.0, labels: str = ""):
        """Increment counter by amount."""
        with self._lock:
            self.samples[labels] = self.samples.get(labels, 0) + amount


class Gauge(PrometheusMetric):
    """Prometheus Gauge metric."""
    
    def __init__(self, name: str, help_text: str):
        super().__init__(name, help_text, "gauge")
    
    def set(self, value: float, labels: str = ""):
        """Set gauge to value."""
        with self._lock:
            self.samples[labels] = value
    
    def inc(self, amount: float = 1.0, labels: str = ""):
        """Increment gauge by amount."""
        with self._lock:
            self.samples[labels] = self.samples.get(labels, 0) + amount
    
    def dec(self, amount: float = 1.0, labels: str = ""):
        """Decrement gauge by amount."""
        with self._lock:
            self.samples[labels] = self.samples.get(labels, 0) - amount


class Histogram(PrometheusMetric):
    """Simplified Prometheus Histogram metric."""
    
    def __init__(self, name: str, help_text: str, buckets: Optional[List[float]] = None):
        super().__init__(name, help_text, "histogram")
        self.buckets = buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._observations = defaultdict(list)
        self._count = defaultdict(int)
        self._sum = defaultdict(float)
    
    def observe(self, value: float, labels: str = ""):
        """Observe a value."""
        with self._lock:
            self._observations[labels].append(value)
            self._count[labels] += 1
            self._sum[labels] += value
            
            # Keep only recent observations to prevent memory bloat
            if len(self._observations[labels]) > 1000:
                self._observations[labels] = self._observations[labels][-500:]
    
    def to_prometheus_format(self) -> str:
        """Convert histogram to Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram"
        ]
        
        with self._lock:
            for labels in self._count.keys():
                label_suffix = f"{{{labels}}}" if labels else ""
                
                # Bucket counts
                for bucket in self.buckets:
                    count = sum(1 for obs in self._observations[labels] if obs <= bucket)
                    bucket_labels = f"le=\"{bucket}\""
                    if labels:
                        bucket_labels = f"{labels},le=\"{bucket}\""
                    lines.append(f"{self.name}_bucket{{{bucket_labels}}} {count}")
                
                # +Inf bucket
                inf_labels = f"le=\"+Inf\""
                if labels:
                    inf_labels = f"{labels},le=\"+Inf\""
                lines.append(f"{self.name}_bucket{{{inf_labels}}} {self._count[labels]}")
                
                # Count and sum
                lines.append(f"{self.name}_count{label_suffix} {self._count[labels]}")
                lines.append(f"{self.name}_sum{label_suffix} {self._sum[labels]}")
        
        return "\n".join(lines)


class MetricsCollector:
    """
    Comprehensive metrics collector for the churn prediction application.
    
    Collects metrics for:
    - Prediction performance and latency
    - Model accuracy and behavior
    - Cache performance
    - Health check status
    - System resource usage
    """
    
    def __init__(self):
        """Initialize metrics collector with all required metrics."""
        
        # Prediction metrics
        self._prediction_latency = Histogram(
            "prediction_latency_seconds",
            "Time taken to generate predictions",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self._prediction_count = Counter(
            "prediction_count_total",
            "Total number of predictions made"
        )
        
        # Model metrics
        self._model_accuracy = Gauge(
            "model_accuracy",
            "Current model accuracy score"
        )
        
        self._model_load_time = Histogram(
            "model_load_time_seconds",
            "Time taken to load models from disk or cache"
        )
        
        # Cache metrics
        self._cache_hit_rate = Gauge(
            "cache_hit_rate_percent",
            "Cache hit rate percentage"
        )
        
        self._cache_memory_usage = Gauge(
            "cache_memory_usage_bytes",
            "Memory usage of model cache"
        )
        
        self._cache_entries = Gauge(
            "cache_entries_total",
            "Number of entries in model cache"
        )
        
        # Health check metrics
        self._health_check_duration = Histogram(
            "health_check_duration_seconds",
            "Duration of health checks"
        )
        
        self._health_check_status = Gauge(
            "health_check_status",
            "Health check status (1=healthy, 0.5=degraded, 0=unhealthy)"
        )
        
        # System metrics
        self._active_requests = Gauge(
            "active_requests",
            "Number of currently active requests"
        )
        
        self._error_count = Counter(
            "error_count_total",
            "Total number of errors encountered"
        )
        
        # Application info
        self._app_info = Gauge(
            "app_info",
            "Application information"
        )
        
        # Initialize app info
        self._app_info.set(1, 'version="1.0.0",service="churn-predictor"')
        
        logger.info("MetricsCollector initialized with all metrics")
    
    def record_prediction_latency(self, duration: float, prediction_type: str):
        """
        Record prediction latency.
        
        Args:
            duration: Duration in seconds
            prediction_type: Type of prediction (single, batch)
        """
        labels = f'type="{prediction_type}"'
        self._prediction_latency.observe(duration, labels)
        logger.debug(f"Recorded prediction latency: {duration:.3f}s for {prediction_type}")
    
    def record_prediction_count(self, count: int, status: str, prediction_type: str):
        """
        Record prediction count.
        
        Args:
            count: Number of predictions
            status: Status (success, error)
            prediction_type: Type of prediction (single, batch)
        """
        labels = f'status="{status}",type="{prediction_type}"'
        self._prediction_count.inc(count, labels)
        logger.debug(f"Recorded {count} predictions: {status} {prediction_type}")
    
    def record_model_accuracy(self, accuracy: float, dataset: str):
        """
        Record model accuracy.
        
        Args:
            accuracy: Accuracy score (0-1)
            dataset: Dataset name (validation, test, etc.)
        """
        labels = f'dataset="{dataset}"'
        self._model_accuracy.set(accuracy, labels)
        logger.debug(f"Recorded model accuracy: {accuracy:.3f} for {dataset}")
    
    def record_model_load_time(self, duration: float, model_type: str, cache_hit: bool):
        """
        Record model loading time.
        
        Args:
            duration: Load duration in seconds
            model_type: Type of model (model, preprocessor, metadata)
            cache_hit: Whether it was a cache hit
        """
        cache_status = "hit" if cache_hit else "miss"
        labels = f'type="{model_type}",cache="{cache_status}"'
        self._model_load_time.observe(duration, labels)
    
    def record_cache_metrics(self, cache_stats: Dict[str, Any]):
        """
        Record cache performance metrics.
        
        Args:
            cache_stats: Dictionary containing cache statistics
        """
        if "hit_rate" in cache_stats:
            self._cache_hit_rate.set(cache_stats["hit_rate"])
        
        if "memory_used_mb" in cache_stats:
            memory_bytes = cache_stats["memory_used_mb"] * 1024 * 1024
            self._cache_memory_usage.set(memory_bytes)
        
        if "entries" in cache_stats:
            self._cache_entries.set(cache_stats["entries"])
        
        logger.debug(f"Recorded cache metrics: {cache_stats}")
    
    def record_health_check_duration(self, duration: float, check_type: str, status: str):
        """
        Record health check duration and status.
        
        Args:
            duration: Check duration in seconds
            check_type: Type of check (basic, detailed, readiness)
            status: Health status (healthy, degraded, unhealthy)
        """
        # Record duration
        labels = f'type="{check_type}",status="{status}"'
        self._health_check_duration.observe(duration, labels)
        
        # Record status as numeric value
        status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(status, 0.0)
        status_labels = f'type="{check_type}"'
        self._health_check_status.set(status_value, status_labels)
    
    def record_error(self, error_type: str, component: str):
        """
        Record error occurrence.
        
        Args:
            error_type: Type of error (validation, prediction, etc.)
            component: Component where error occurred
        """
        labels = f'type="{error_type}",component="{component}"'
        self._error_count.inc(1, labels)
        logger.debug(f"Recorded error: {error_type} in {component}")
    
    def record_active_request(self, increment: bool = True):
        """
        Record active request count change.
        
        Args:
            increment: True to increment, False to decrement
        """
        if increment:
            self._active_requests.inc(1)
        else:
            self._active_requests.dec(1)
    
    def get_metrics(self) -> List[str]:
        """
        Get all metrics as list of strings.
        
        Returns:
            List of metric lines
        """
        metrics = []
        
        # Get all metric instances
        metric_instances = [
            self._prediction_latency,
            self._prediction_count,
            self._model_accuracy,
            self._model_load_time,
            self._cache_hit_rate,
            self._cache_memory_usage,
            self._cache_entries,
            self._health_check_duration,
            self._health_check_status,
            self._active_requests,
            self._error_count,
            self._app_info
        ]
        
        for metric in metric_instances:
            prometheus_output = metric.to_prometheus_format()
            if prometheus_output.strip():  # Only include non-empty metrics
                metrics.extend(prometheus_output.split('\n'))
        
        return metrics
    
    def get_prometheus_format(self) -> str:
        """
        Get all metrics in Prometheus exposition format.
        
        Returns:
            Prometheus format string
        """
        metrics = self.get_metrics()
        return '\n'.join(metrics) + '\n'


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
    """
    global _global_collector
    
    if _global_collector is None:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = MetricsCollector()
                logger.info("Global MetricsCollector instance created")
    
    return _global_collector


def get_prometheus_metrics() -> str:
    """
    Get all metrics in Prometheus format.
    
    Returns:
        Prometheus exposition format string
    """
    collector = get_metrics_collector()
    return collector.get_prometheus_format()


def record_prediction_latency(start_time: float, prediction_type: str):
    """
    Record prediction latency from start time.
    
    Args:
        start_time: Start timestamp from time.time()
        prediction_type: Type of prediction (single, batch)
    """
    duration = time.time() - start_time
    collector = get_metrics_collector()
    collector.record_prediction_latency(duration, prediction_type)


def record_prediction_count(count: int, status: str, prediction_type: str):
    """
    Record prediction count.
    
    Args:
        count: Number of predictions
        status: Status (success, error)
        prediction_type: Type of prediction (single, batch)
    """
    collector = get_metrics_collector()
    collector.record_prediction_count(count, status, prediction_type)


def record_model_accuracy(accuracy: float, dataset: str):
    """
    Record model accuracy.
    
    Args:
        accuracy: Accuracy score (0-1)
        dataset: Dataset name (validation, test, etc.)
    """
    collector = get_metrics_collector()
    collector.record_model_accuracy(accuracy, dataset)


def record_cache_hit_rate(hit_rate: float):
    """
    Record cache hit rate.
    
    Args:
        hit_rate: Hit rate percentage
    """
    collector = get_metrics_collector()
    collector.record_cache_metrics({"hit_rate": hit_rate})


def record_error(error_type: str, component: str):
    """
    Record error occurrence.
    
    Args:
        error_type: Type of error
        component: Component where error occurred
    """
    collector = get_metrics_collector()
    collector.record_error(error_type, component)


# Context manager for request tracking
class request_tracker:
    """Context manager for tracking active requests."""
    
    def __init__(self, request_type: str = "unknown"):
        self.request_type = request_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        collector = get_metrics_collector()
        collector.record_active_request(increment=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        collector = get_metrics_collector()
        collector.record_active_request(increment=False)
        
        if exc_type is not None:
            # Record error if exception occurred
            error_type = exc_type.__name__ if exc_type else "unknown"
            collector.record_error(error_type, self.request_type)
        
        # Record request duration
        if self.start_time:
            duration = time.time() - self.start_time
            collector.record_prediction_latency(duration, self.request_type)