"""
Advanced Error Handling and Recovery for Research Frameworks.

This module provides comprehensive error handling, graceful degradation,
and automatic recovery mechanisms for all novel research frameworks
to ensure robust production deployment.

Key Features:
- Framework-specific error handling and recovery
- Graceful degradation to simpler models on failure
- Automatic retry mechanisms with exponential backoff
- Comprehensive error logging and monitoring
- Model fallback chains for high availability
- Performance monitoring and alerting
"""

import os
import time
import traceback
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import contextmanager
import warnings

from .logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for research frameworks."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class FrameworkType(Enum):
    """Supported research framework types."""
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    MULTIMODAL = "multimodal"
    UNCERTAINTY = "uncertainty"


@dataclass
class ErrorContext:
    """Context information for research framework errors."""
    framework_type: FrameworkType
    operation: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: datetime
    input_shape: Optional[Tuple[int, ...]] = None
    model_state: Optional[str] = None
    retry_count: int = 0
    additional_info: Optional[Dict[str, Any]] = None


class ResearchFrameworkError(Exception):
    """Base exception for research framework errors."""
    
    def __init__(self, message: str, context: ErrorContext):
        super().__init__(message)
        self.context = context


class ModelNotFittedError(ResearchFrameworkError):
    """Raised when attempting to use an unfitted model."""
    pass


class InsufficientDataError(ResearchFrameworkError):
    """Raised when data is insufficient for model training or prediction."""
    pass


class ModelConvergenceError(ResearchFrameworkError):
    """Raised when model fails to converge during training."""
    pass


class FeatureExtractionError(ResearchFrameworkError):
    """Raised when feature extraction fails."""
    pass


class PredictionError(ResearchFrameworkError):
    """Raised when prediction fails."""
    pass


class ResearchErrorHandler:
    """
    Comprehensive error handler for research frameworks.
    
    Provides automatic error recovery, graceful degradation,
    and comprehensive logging for all research frameworks.
    """
    
    def __init__(self, framework_type: FrameworkType):
        self.framework_type = framework_type
        self.error_history: List[ErrorContext] = []
        self.fallback_models: List[Any] = []
        self.max_retries = 3
        self.retry_delays = [1, 2, 5]  # seconds
        self.performance_tracker = PerformanceTracker(framework_type)
        
        # Framework-specific configurations
        self.config = self._get_framework_config()
        
        logger.info(f"Initialized error handler for {framework_type.value} framework")
    
    def _get_framework_config(self) -> Dict[str, Any]:
        """Get framework-specific error handling configuration."""
        base_config = {
            'max_retries': 3,
            'timeout_seconds': 300,
            'min_samples_required': 10,
            'enable_fallback': True,
            'log_errors': True
        }
        
        framework_configs = {
            FrameworkType.CAUSAL: {
                'min_samples_required': 50,  # Causal discovery needs more data
                'timeout_seconds': 600,      # Longer timeout for graph discovery
                'enable_causal_validation': True
            },
            FrameworkType.TEMPORAL: {
                'min_samples_required': 30,  # Need temporal sequences
                'min_sequence_length': 3,
                'enable_temporal_validation': True
            },
            FrameworkType.MULTIMODAL: {
                'min_samples_required': 20,
                'enable_modality_fallback': True,
                'text_preprocessing_timeout': 60
            },
            FrameworkType.UNCERTAINTY: {
                'min_ensemble_members': 2,
                'enable_calibration_check': True,
                'uncertainty_threshold': 0.5
            }
        }
        
        config = base_config.copy()
        config.update(framework_configs.get(self.framework_type, {}))
        return config
    
    def handle_error(self, error: Exception, operation: str, 
                    input_data: Any = None, **kwargs) -> ErrorContext:
        """Handle and log research framework errors."""
        # Determine error severity and type
        severity = self._determine_severity(error, operation)
        error_type = type(error).__name__
        
        # Create error context
        context = ErrorContext(
            framework_type=self.framework_type,
            operation=operation,
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            timestamp=datetime.now(),
            input_shape=self._get_input_shape(input_data),
            additional_info=kwargs
        )
        
        # Log error
        self._log_error(context)
        
        # Store in error history
        self.error_history.append(context)
        
        # Update performance tracker
        self.performance_tracker.record_error(context)
        
        return context
    
    def _determine_severity(self, error: Exception, operation: str) -> ErrorSeverity:
        """Determine error severity based on error type and operation."""
        critical_errors = [
            SystemError, MemoryError, KeyboardInterrupt
        ]
        
        high_severity_errors = [
            ModelNotFittedError, ModelConvergenceError
        ]
        
        medium_severity_errors = [
            InsufficientDataError, FeatureExtractionError
        ]
        
        if any(isinstance(error, err_type) for err_type in critical_errors):
            return ErrorSeverity.CRITICAL
        elif any(isinstance(error, err_type) for err_type in high_severity_errors):
            return ErrorSeverity.HIGH
        elif any(isinstance(error, err_type) for err_type in medium_severity_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_input_shape(self, input_data: Any) -> Optional[Tuple[int, ...]]:
        """Extract shape information from input data."""
        try:
            if hasattr(input_data, 'shape'):
                return tuple(input_data.shape)
            elif hasattr(input_data, '__len__'):
                if hasattr(input_data, 'iloc'):  # DataFrame-like
                    return (len(input_data), len(input_data.columns) if hasattr(input_data, 'columns') else 0)
                else:
                    return (len(input_data),)
            else:
                return None
        except Exception:
            return None
    
    def _log_error(self, context: ErrorContext) -> None:
        """Log error with appropriate level and details."""
        log_message = (
            f"{self.framework_type.value.upper()} ERROR: {context.operation} failed\n"
            f"Error Type: {context.error_type}\n"
            f"Message: {context.error_message}\n"
            f"Severity: {context.severity.value}\n"
            f"Input Shape: {context.input_shape}\n"
            f"Timestamp: {context.timestamp}"
        )
        
        if context.additional_info:
            log_message += f"\nAdditional Info: {context.additional_info}"
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def with_retry(self, max_retries: Optional[int] = None):
        """Decorator for automatic retry with exponential backoff."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                last_error = None
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as error:
                        last_error = error
                        
                        # Handle error and decide if we should retry
                        context = self.handle_error(error, func.__name__, 
                                                  args[0] if args else None,
                                                  attempt=attempt)
                        
                        # Don't retry on critical errors
                        if context.severity == ErrorSeverity.CRITICAL:
                            raise error
                        
                        # Don't retry on last attempt
                        if attempt == retries:
                            break
                        
                        # Wait before retry
                        if attempt < len(self.retry_delays):
                            delay = self.retry_delays[attempt]
                        else:
                            delay = self.retry_delays[-1] * (2 ** (attempt - len(self.retry_delays) + 1))
                        
                        logger.info(f"Retrying {func.__name__} in {delay} seconds (attempt {attempt + 1}/{retries + 1})")
                        time.sleep(delay)
                
                # All retries exhausted
                logger.error(f"All {retries} retries exhausted for {func.__name__}")
                raise last_error
            
            return wrapper
        return decorator
    
    def with_fallback(self, fallback_func: Callable):
        """Decorator for automatic fallback on failure."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    context = self.handle_error(error, func.__name__, 
                                              args[0] if args else None)
                    
                    # Use fallback for non-critical errors
                    if context.severity != ErrorSeverity.CRITICAL and self.config.get('enable_fallback', True):
                        logger.warning(f"Using fallback for {func.__name__} due to {context.error_type}")
                        try:
                            return fallback_func(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed: {fallback_error}")
                            raise error
                    else:
                        raise error
            
            return wrapper
        return decorator
    
    @contextmanager
    def error_context(self, operation: str, **kwargs):
        """Context manager for operation-level error handling."""
        start_time = time.time()
        
        try:
            logger.debug(f"Starting {self.framework_type.value} operation: {operation}")
            yield
            
            # Record successful operation
            execution_time = time.time() - start_time
            self.performance_tracker.record_success(operation, execution_time)
            logger.debug(f"Completed {operation} in {execution_time:.3f}s")
            
        except Exception as error:
            execution_time = time.time() - start_time
            context = self.handle_error(error, operation, **kwargs)
            context.additional_info = context.additional_info or {}
            context.additional_info['execution_time'] = execution_time
            
            # Re-raise with research framework error if not already
            if not isinstance(error, ResearchFrameworkError):
                raise ResearchFrameworkError(str(error), context) from error
            else:
                raise
    
    def validate_input(self, data: Any, operation: str) -> None:
        """Validate input data for research framework operations."""
        with self.error_context(f"validate_{operation}_input"):
            # Basic validation
            if data is None:
                raise InsufficientDataError(
                    "Input data is None",
                    ErrorContext(
                        framework_type=self.framework_type,
                        operation=operation,
                        error_type="ValidationError",
                        error_message="Input data is None",
                        severity=ErrorSeverity.HIGH,
                        timestamp=datetime.now()
                    )
                )
            
            # Check minimum samples
            min_samples = self.config.get('min_samples_required', 10)
            if hasattr(data, '__len__') and len(data) < min_samples:
                raise InsufficientDataError(
                    f"Insufficient data: {len(data)} samples, minimum {min_samples} required",
                    ErrorContext(
                        framework_type=self.framework_type,
                        operation=operation,
                        error_type="ValidationError", 
                        error_message=f"Only {len(data)} samples available, need {min_samples}",
                        severity=ErrorSeverity.MEDIUM,
                        timestamp=datetime.now(),
                        input_shape=self._get_input_shape(data)
                    )
                )
            
            # Framework-specific validation
            self._framework_specific_validation(data, operation)
    
    def _framework_specific_validation(self, data: Any, operation: str) -> None:
        """Perform framework-specific input validation."""
        if self.framework_type == FrameworkType.TEMPORAL:
            self._validate_temporal_data(data, operation)
        elif self.framework_type == FrameworkType.MULTIMODAL:
            self._validate_multimodal_data(data, operation)
        elif self.framework_type == FrameworkType.CAUSAL:
            self._validate_causal_data(data, operation)
        elif self.framework_type == FrameworkType.UNCERTAINTY:
            self._validate_uncertainty_data(data, operation)
    
    def _validate_temporal_data(self, data: Any, operation: str) -> None:
        """Validate temporal data requirements."""
        # Check for customer_id or timestamp columns if DataFrame-like
        if hasattr(data, 'columns'):
            required_cols = ['customer_id', 'timestamp'] 
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Temporal data missing recommended columns: {missing_cols}")
    
    def _validate_multimodal_data(self, data: Any, operation: str) -> None:
        """Validate multi-modal data requirements."""
        # For multi-modal, we can work with just tabular data
        pass
    
    def _validate_causal_data(self, data: Any, operation: str) -> None:
        """Validate causal discovery data requirements."""
        # Causal discovery needs multiple features
        if hasattr(data, 'shape') and len(data.shape) > 1 and data.shape[1] < 3:
            logger.warning("Causal discovery works best with multiple features (3+)")
    
    def _validate_uncertainty_data(self, data: Any, operation: str) -> None:
        """Validate uncertainty quantification data requirements."""
        # Uncertainty quantification needs sufficient samples for calibration
        min_samples = self.config.get('min_samples_required', 50)
        if hasattr(data, '__len__') and len(data) < min_samples:
            logger.warning(f"Uncertainty quantification may not be well-calibrated with < {min_samples} samples")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for monitoring."""
        if not self.error_history:
            return {'status': 'healthy', 'total_errors': 0}
        
        # Count errors by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(
                1 for error in self.error_history 
                if error.severity == severity
            )
        
        # Recent errors (last hour)
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp > recent_threshold
        ]
        
        # Most common error types
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'framework': self.framework_type.value,
            'status': self._determine_health_status(),
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'severity_counts': severity_counts,
            'most_common_errors': most_common_errors,
            'performance_metrics': self.performance_tracker.get_metrics()
        }
    
    def _determine_health_status(self) -> str:
        """Determine overall health status based on error history."""
        if not self.error_history:
            return 'healthy'
        
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp > recent_threshold
        ]
        
        # Check for critical errors
        critical_errors = [
            error for error in recent_errors 
            if error.severity == ErrorSeverity.CRITICAL
        ]
        
        if critical_errors:
            return 'critical'
        
        # Check error rate
        if len(recent_errors) > 10:
            return 'degraded'
        elif len(recent_errors) > 5:
            return 'warning'
        else:
            return 'healthy'


class PerformanceTracker:
    """Track performance metrics for research frameworks."""
    
    def __init__(self, framework_type: FrameworkType):
        self.framework_type = framework_type
        self.operation_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self.start_time = datetime.now()
    
    def record_success(self, operation: str, execution_time: float) -> None:
        """Record successful operation."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(execution_time)
        self.success_counts[operation] = self.success_counts.get(operation, 0) + 1
    
    def record_error(self, error_context: ErrorContext) -> None:
        """Record error occurrence."""
        operation = error_context.operation
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        metrics = {
            'framework': self.framework_type.value,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'operations': {}
        }
        
        all_operations = set(self.operation_times.keys()) | set(self.error_counts.keys()) | set(self.success_counts.keys())
        
        for operation in all_operations:
            operation_metrics = {
                'total_calls': self.success_counts.get(operation, 0) + self.error_counts.get(operation, 0),
                'success_count': self.success_counts.get(operation, 0),
                'error_count': self.error_counts.get(operation, 0),
                'success_rate': 0.0
            }
            
            if operation_metrics['total_calls'] > 0:
                operation_metrics['success_rate'] = operation_metrics['success_count'] / operation_metrics['total_calls']
            
            if operation in self.operation_times and self.operation_times[operation]:
                times = self.operation_times[operation]
                operation_metrics.update({
                    'avg_execution_time': sum(times) / len(times),
                    'min_execution_time': min(times),
                    'max_execution_time': max(times),
                    'recent_execution_time': times[-1] if times else 0
                })
            
            metrics['operations'][operation] = operation_metrics
        
        return metrics


def create_error_handler(framework_type: FrameworkType) -> ResearchErrorHandler:
    """Factory function to create framework-specific error handlers."""
    return ResearchErrorHandler(framework_type)


# Convenience decorators for each framework type
def causal_error_handler():
    """Decorator for causal discovery framework operations."""
    handler = create_error_handler(FrameworkType.CAUSAL)
    return handler.with_retry()


def temporal_error_handler():
    """Decorator for temporal graph framework operations.""" 
    handler = create_error_handler(FrameworkType.TEMPORAL)
    return handler.with_retry()


def multimodal_error_handler():
    """Decorator for multi-modal fusion framework operations."""
    handler = create_error_handler(FrameworkType.MULTIMODAL)
    return handler.with_retry()


def uncertainty_error_handler():
    """Decorator for uncertainty quantification framework operations."""
    handler = create_error_handler(FrameworkType.UNCERTAINTY)
    return handler.with_retry()


# Global error handlers for each framework
_error_handlers: Dict[FrameworkType, ResearchErrorHandler] = {}

def get_error_handler(framework_type: FrameworkType) -> ResearchErrorHandler:
    """Get global error handler instance for framework type."""
    if framework_type not in _error_handlers:
        _error_handlers[framework_type] = create_error_handler(framework_type)
    return _error_handlers[framework_type]


def get_all_error_summaries() -> Dict[str, Any]:
    """Get error summaries for all active research frameworks."""
    summaries = {}
    for framework_type, handler in _error_handlers.items():
        summaries[framework_type.value] = handler.get_error_summary()
    
    # Overall system health
    all_statuses = [summary.get('status', 'unknown') for summary in summaries.values()]
    
    if 'critical' in all_statuses:
        overall_status = 'critical'
    elif 'degraded' in all_statuses:
        overall_status = 'degraded'
    elif 'warning' in all_statuses:
        overall_status = 'warning'
    else:
        overall_status = 'healthy'
    
    return {
        'overall_status': overall_status,
        'frameworks': summaries,
        'timestamp': datetime.now().isoformat()
    }


# Export main classes and functions
__all__ = [
    'ResearchErrorHandler',
    'ErrorContext',
    'ErrorSeverity',
    'FrameworkType',
    'ResearchFrameworkError',
    'ModelNotFittedError',
    'InsufficientDataError', 
    'ModelConvergenceError',
    'FeatureExtractionError',
    'PredictionError',
    'PerformanceTracker',
    'create_error_handler',
    'get_error_handler',
    'get_all_error_summaries',
    'causal_error_handler',
    'temporal_error_handler',
    'multimodal_error_handler',
    'uncertainty_error_handler'
]