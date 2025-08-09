"""
Advanced Error Handling and Recovery System.

This module provides comprehensive error handling, recovery mechanisms, and resilience patterns
for the ML system including:
- Circuit breaker patterns for external dependencies
- Retry mechanisms with exponential backoff
- Graceful degradation strategies
- Error classification and intelligent recovery
- System health monitoring and automatic healing
- Comprehensive logging and error reporting
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from functools import wraps
import traceback
import pandas as pd
import numpy as np
from enum import Enum, auto
import joblib
import redis
from sklearn.base import BaseEstimator

from .logging_config import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = auto()          # System-level errors
    DATA = auto()           # Data-related errors
    MODEL = auto()          # Model-related errors
    NETWORK = auto()        # Network/connectivity errors
    RESOURCE = auto()       # Resource exhaustion
    CONFIGURATION = auto()  # Configuration errors
    EXTERNAL = auto()       # External service errors
    UNKNOWN = auto()        # Unknown/unclassified


class RecoveryAction(Enum):
    """Possible recovery actions."""
    RETRY = auto()
    FALLBACK = auto()
    DEGRADE = auto()
    RESTART = auto()
    ALERT = auto()
    IGNORE = auto()
    ESCALATE = auto()


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    error_details: Dict[str, Any]
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    resolution_status: str = "unresolved"
    recovery_actions_taken: List[str] = None
    
    def __post_init__(self):
        if self.recovery_actions_taken is None:
            self.recovery_actions_taken = []


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Failing, calls rejected
    HALF_OPEN = auto()  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
        self.lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if we should transition to half-open
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.config.recovery_timeout):
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN"
                    )
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} half-open call limit reached"
                    )
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _record_success(self) -> None:
        """Record successful call."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                # Reset failure count on success
                self.failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self._transition_to_open()
    
    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'name': self.name,
            'state': self.state.name,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'half_open_calls': self.half_open_calls,
            'last_failure_time': self.last_failure_time
        }


class RetryMechanism:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry behavior."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain exception types
                if self._should_not_retry(e):
                    logger.warning(f"Not retrying {func.__name__} due to non-retriable error: {e}")
                    raise e
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), "
                             f"retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        raise last_exception
    
    def _should_not_retry(self, exception: Exception) -> bool:
        """Determine if we should not retry for this exception."""
        non_retriable_types = (
            ValueError,  # Bad input data
            TypeError,   # Programming error
            KeyError,    # Missing key
        )
        return isinstance(exception, non_retriable_types)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            jitter = np.random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay + jitter)
        
        return delay


class ErrorClassifier:
    """Intelligent error classification system."""
    
    def __init__(self):
        self.classification_rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification rules."""
        return {
            'data_errors': {
                'keywords': ['data', 'dataframe', 'column', 'missing', 'nan', 'invalid'],
                'exception_types': ['ValueError', 'KeyError', 'IndexError'],
                'category': ErrorCategory.DATA,
                'typical_severity': ErrorSeverity.MEDIUM
            },
            'model_errors': {
                'keywords': ['model', 'predict', 'fit', 'transform', 'sklearn'],
                'exception_types': ['AttributeError', 'NotFittedError'],
                'category': ErrorCategory.MODEL,
                'typical_severity': ErrorSeverity.HIGH
            },
            'network_errors': {
                'keywords': ['connection', 'timeout', 'network', 'http', 'api'],
                'exception_types': ['ConnectionError', 'TimeoutError', 'HTTPError'],
                'category': ErrorCategory.NETWORK,
                'typical_severity': ErrorSeverity.MEDIUM
            },
            'resource_errors': {
                'keywords': ['memory', 'disk', 'cpu', 'resource', 'quota'],
                'exception_types': ['MemoryError', 'OSError'],
                'category': ErrorCategory.RESOURCE,
                'typical_severity': ErrorSeverity.HIGH
            },
            'system_errors': {
                'keywords': ['system', 'os', 'file', 'permission', 'access'],
                'exception_types': ['OSError', 'PermissionError', 'FileNotFoundError'],
                'category': ErrorCategory.SYSTEM,
                'typical_severity': ErrorSeverity.HIGH
            }
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error into category and severity."""
        
        error_text = str(exception).lower()
        exception_type = type(exception).__name__
        context = context or {}
        
        # Score each category
        category_scores = {}
        
        for category_name, rules in self.classification_rules.items():
            score = 0
            
            # Keyword matching
            for keyword in rules['keywords']:
                if keyword in error_text:
                    score += 1
            
            # Exception type matching
            if exception_type in rules['exception_types']:
                score += 5
            
            # Context matching
            component = context.get('component', '')
            if any(keyword in component.lower() for keyword in rules['keywords']):
                score += 2
            
            category_scores[category_name] = score
        
        # Determine category
        if category_scores:
            best_category_name = max(category_scores.items(), key=lambda x: x[1])[0]
            if category_scores[best_category_name] > 0:
                category = self.classification_rules[best_category_name]['category']
                severity = self.classification_rules[best_category_name]['typical_severity']
            else:
                category = ErrorCategory.UNKNOWN
                severity = ErrorSeverity.MEDIUM
        else:
            category = ErrorCategory.UNKNOWN
            severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on context
        if context.get('is_critical_path', False):
            severity = ErrorSeverity.CRITICAL
        elif context.get('affects_production', False):
            if severity == ErrorSeverity.LOW:
                severity = ErrorSeverity.MEDIUM
            elif severity == ErrorSeverity.MEDIUM:
                severity = ErrorSeverity.HIGH
        
        return category, severity


class RecoveryStrategist:
    """Determines and executes recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[ErrorCategory, Dict[ErrorSeverity, List[RecoveryAction]]]:
        """Initialize recovery strategies by error category and severity."""
        return {
            ErrorCategory.DATA: {
                ErrorSeverity.LOW: [RecoveryAction.RETRY, RecoveryAction.FALLBACK],
                ErrorSeverity.MEDIUM: [RecoveryAction.FALLBACK, RecoveryAction.DEGRADE],
                ErrorSeverity.HIGH: [RecoveryAction.DEGRADE, RecoveryAction.ALERT],
                ErrorSeverity.CRITICAL: [RecoveryAction.ALERT, RecoveryAction.ESCALATE]
            },
            ErrorCategory.MODEL: {
                ErrorSeverity.LOW: [RecoveryAction.RETRY, RecoveryAction.FALLBACK],
                ErrorSeverity.MEDIUM: [RecoveryAction.FALLBACK, RecoveryAction.DEGRADE],
                ErrorSeverity.HIGH: [RecoveryAction.FALLBACK, RecoveryAction.ALERT],
                ErrorSeverity.CRITICAL: [RecoveryAction.RESTART, RecoveryAction.ALERT]
            },
            ErrorCategory.NETWORK: {
                ErrorSeverity.LOW: [RecoveryAction.RETRY],
                ErrorSeverity.MEDIUM: [RecoveryAction.RETRY, RecoveryAction.FALLBACK],
                ErrorSeverity.HIGH: [RecoveryAction.FALLBACK, RecoveryAction.DEGRADE],
                ErrorSeverity.CRITICAL: [RecoveryAction.DEGRADE, RecoveryAction.ALERT]
            },
            ErrorCategory.RESOURCE: {
                ErrorSeverity.LOW: [RecoveryAction.RETRY, RecoveryAction.DEGRADE],
                ErrorSeverity.MEDIUM: [RecoveryAction.DEGRADE, RecoveryAction.ALERT],
                ErrorSeverity.HIGH: [RecoveryAction.RESTART, RecoveryAction.ALERT],
                ErrorSeverity.CRITICAL: [RecoveryAction.RESTART, RecoveryAction.ESCALATE]
            },
            ErrorCategory.SYSTEM: {
                ErrorSeverity.LOW: [RecoveryAction.RETRY],
                ErrorSeverity.MEDIUM: [RecoveryAction.RETRY, RecoveryAction.ALERT],
                ErrorSeverity.HIGH: [RecoveryAction.RESTART, RecoveryAction.ALERT],
                ErrorSeverity.CRITICAL: [RecoveryAction.RESTART, RecoveryAction.ESCALATE]
            }
        }
    
    def get_recovery_actions(self, category: ErrorCategory, severity: ErrorSeverity) -> List[RecoveryAction]:
        """Get recommended recovery actions for error category and severity."""
        
        if category in self.recovery_strategies and severity in self.recovery_strategies[category]:
            return self.recovery_strategies[category][severity].copy()
        else:
            # Default recovery strategy
            return [RecoveryAction.RETRY, RecoveryAction.ALERT]


class HealthcheckManager:
    """Manages system health checks and automatic healing."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_status = defaultdict(dict)
        self.healing_actions = {}
        self.running = False
        self.thread = None
        
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            healing_func: Optional[Callable[[], bool]] = None) -> None:
        """Register a health check with optional healing function."""
        self.health_checks[name] = check_func
        if healing_func:
            self.healing_actions[name] = healing_func
        
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_health, daemon=True)
        self.thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_health(self) -> None:
        """Main health monitoring loop."""
        while self.running:
            self._perform_health_checks()
            time.sleep(self.check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                self.health_status[name]['healthy'] = is_healthy
                self.health_status[name]['last_check'] = datetime.now()
                
                if not is_healthy:
                    logger.warning(f"Health check failed: {name}")
                    
                    # Attempt healing if available
                    if name in self.healing_actions:
                        logger.info(f"Attempting to heal: {name}")
                        try:
                            healed = self.healing_actions[name]()
                            if healed:
                                logger.info(f"Successfully healed: {name}")
                                self.health_status[name]['healed'] = True
                            else:
                                logger.error(f"Failed to heal: {name}")
                                self.health_status[name]['healed'] = False
                        except Exception as e:
                            logger.error(f"Healing function failed for {name}: {e}")
                            self.health_status[name]['healed'] = False
                
            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                self.health_status[name]['healthy'] = False
                self.health_status[name]['error'] = str(e)
                self.health_status[name]['last_check'] = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return dict(self.health_status)


class ErrorHandler:
    """Main error handling and recovery system."""
    
    def __init__(self):
        self.error_history = deque(maxlen=10000)
        self.circuit_breakers = {}
        self.classifier = ErrorClassifier()
        self.recovery_strategist = RecoveryStrategist()
        self.healthcheck_manager = HealthcheckManager()
        
        # Fallback mechanisms
        self.fallback_predictors = {}
        self.degraded_mode_active = False
        
        # Metrics
        self.metrics_collector = get_metrics_collector()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self) -> None:
        """Setup default system health checks."""
        
        # Memory usage check
        def check_memory_usage() -> bool:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        
        # Disk usage check
        def check_disk_usage() -> bool:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 95  # Alert if disk usage > 95%
        
        # Model availability check
        def check_model_availability() -> bool:
            try:
                model_path = "models/churn_model.joblib"
                return os.path.exists(model_path)
            except:
                return False
        
        self.healthcheck_manager.register_health_check('memory', check_memory_usage)
        self.healthcheck_manager.register_health_check('disk', check_disk_usage)
        self.healthcheck_manager.register_health_check('model', check_model_availability)
        
        self.healthcheck_manager.start_monitoring()
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorEvent:
        """Main error handling entry point."""
        
        context = context or {}
        
        # Create error event
        error_event = ErrorEvent(
            error_id=f"err_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            error_details={'args': exception.args if hasattr(exception, 'args') else []},
            severity=ErrorSeverity.MEDIUM,  # Will be updated by classifier
            category=ErrorCategory.UNKNOWN,  # Will be updated by classifier
            component=context.get('component', 'unknown'),
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Classify error
        category, severity = self.classifier.classify_error(exception, context)
        error_event.category = category
        error_event.severity = severity
        
        # Log error
        self._log_error(error_event)
        
        # Get recovery actions
        recovery_actions = self.recovery_strategist.get_recovery_actions(category, severity)
        
        # Execute recovery actions
        for action in recovery_actions:
            try:
                success = self._execute_recovery_action(action, error_event, context)
                if success:
                    error_event.recovery_actions_taken.append(f"{action.name}_SUCCESS")
                    if action in [RecoveryAction.RETRY, RecoveryAction.FALLBACK]:
                        error_event.resolution_status = "recovered"
                        break
                else:
                    error_event.recovery_actions_taken.append(f"{action.name}_FAILED")
            except Exception as recovery_error:
                logger.error(f"Recovery action {action.name} failed: {recovery_error}")
                error_event.recovery_actions_taken.append(f"{action.name}_ERROR")
        
        # Store error event
        self.error_history.append(error_event)
        
        # Update metrics
        self._update_error_metrics(error_event)
        
        return error_event
    
    def _log_error(self, error_event: ErrorEvent) -> None:
        """Log error with appropriate level."""
        
        log_message = (f"Error {error_event.error_id}: {error_event.error_type} - "
                      f"{error_event.error_message} (Category: {error_event.category.name}, "
                      f"Severity: {error_event.severity.name})")
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _execute_recovery_action(self, action: RecoveryAction, error_event: ErrorEvent, 
                               context: Dict[str, Any]) -> bool:
        """Execute a specific recovery action."""
        
        try:
            if action == RecoveryAction.RETRY:
                # Retry is usually handled by decorator, but can trigger cleanup here
                return self._handle_retry_recovery(error_event, context)
                
            elif action == RecoveryAction.FALLBACK:
                return self._handle_fallback_recovery(error_event, context)
                
            elif action == RecoveryAction.DEGRADE:
                return self._handle_degradation_recovery(error_event, context)
                
            elif action == RecoveryAction.RESTART:
                return self._handle_restart_recovery(error_event, context)
                
            elif action == RecoveryAction.ALERT:
                return self._handle_alert_recovery(error_event, context)
                
            elif action == RecoveryAction.ESCALATE:
                return self._handle_escalation_recovery(error_event, context)
                
            elif action == RecoveryAction.IGNORE:
                logger.info(f"Ignoring error {error_event.error_id} as per recovery strategy")
                return True
                
        except Exception as e:
            logger.error(f"Recovery action {action.name} execution failed: {e}")
            return False
        
        return False
    
    def _handle_retry_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle retry recovery."""
        # Clear any cached state that might be causing issues
        component = context.get('component', '')
        if 'model' in component.lower():
            # Clear model cache
            from .model_cache import invalidate_model_cache
            try:
                invalidate_model_cache()
                logger.info("Model cache cleared for retry recovery")
                return True
            except:
                pass
        
        return False  # Actual retry is handled by RetryMechanism decorator
    
    def _handle_fallback_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle fallback recovery."""
        component = context.get('component', '')
        
        if 'prediction' in component.lower() or 'model' in component.lower():
            # Use fallback predictor
            return self._activate_fallback_predictor(context)
        
        return False
    
    def _handle_degradation_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle graceful degradation."""
        self.degraded_mode_active = True
        logger.warning("System entering degraded mode")
        
        # Implement degraded mode behaviors
        # - Reduce feature complexity
        # - Use simpler models
        # - Disable non-essential features
        
        return True
    
    def _handle_restart_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle service restart."""
        component = context.get('component', '')
        
        if 'model' in component.lower():
            # Restart model-related components
            try:
                # Reload model
                from .model_cache import invalidate_model_cache
                invalidate_model_cache()
                logger.info("Model components restarted")
                return True
            except Exception as e:
                logger.error(f"Failed to restart model components: {e}")
        
        return False
    
    def _handle_alert_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle alerting."""
        # In production, this would send alerts to monitoring systems
        alert_message = (f"ALERT: {error_event.severity.name} error in {error_event.component} - "
                        f"{error_event.error_message}")
        
        logger.critical(alert_message)
        
        # Could integrate with PagerDuty, Slack, email, etc.
        # For now, just log the alert
        return True
    
    def _handle_escalation_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Handle error escalation."""
        escalation_message = (f"ESCALATION: Critical error requires immediate attention - "
                             f"Error ID: {error_event.error_id}, Component: {error_event.component}")
        
        logger.critical(escalation_message)
        
        # In production, this would trigger high-priority alerts
        return True
    
    def _activate_fallback_predictor(self, context: Dict[str, Any]) -> bool:
        """Activate fallback prediction mechanism."""
        
        # Use simple fallback predictor (e.g., majority class or simple heuristic)
        if 'fallback_simple' not in self.fallback_predictors:
            
            class SimpleFallbackPredictor:
                """Simple fallback predictor using heuristics."""
                
                def predict(self, X):
                    # Simple heuristic: predict churn based on monthly charges
                    if isinstance(X, pd.DataFrame):
                        if 'MonthlyCharges' in X.columns:
                            return (X['MonthlyCharges'] > 80).astype(int).values
                        elif 'monthly_charges' in X.columns:
                            return (X['monthly_charges'] > 80).astype(int).values
                    
                    # Default to no churn
                    n_samples = len(X) if hasattr(X, '__len__') else 1
                    return np.zeros(n_samples, dtype=int)
                
                def predict_proba(self, X):
                    predictions = self.predict(X)
                    # Convert to probabilities
                    probas = np.column_stack([1 - predictions, predictions])
                    return probas
            
            self.fallback_predictors['fallback_simple'] = SimpleFallbackPredictor()
            logger.info("Activated simple fallback predictor")
        
        return True
    
    def _update_error_metrics(self, error_event: ErrorEvent) -> None:
        """Update error metrics."""
        try:
            self.metrics_collector.record_error(
                error_type=error_event.category.name.lower(),
                component=error_event.component
            )
        except Exception as e:
            logger.warning(f"Failed to update error metrics: {e}")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        
        # Count by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        resolution_counts = defaultdict(int)
        
        for error in self.error_history:
            category_counts[error.category.name] += 1
            severity_counts[error.severity.name] += 1
            resolution_counts[error.resolution_status] += 1
        
        # Recent errors (last hour)
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp > hour_ago]
        
        return {
            'total_errors': total_errors,
            'recent_errors_1h': len(recent_errors),
            'category_breakdown': dict(category_counts),
            'severity_breakdown': dict(severity_counts),
            'resolution_breakdown': dict(resolution_counts),
            'degraded_mode_active': self.degraded_mode_active,
            'circuit_breaker_states': {name: cb.get_state() 
                                     for name, cb in self.circuit_breakers.items()},
            'health_status': self.healthcheck_manager.get_health_status()
        }


# Custom exceptions for error handling system
class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class FallbackModeError(Exception):
    """Exception raised when operating in fallback mode."""
    pass


class DegradedModeError(Exception):
    """Exception raised when operating in degraded mode."""
    pass


# Global error handler instance
error_handler = ErrorHandler()


# Decorators for easy error handling
def with_error_handling(component: str = None, 
                       critical: bool = False,
                       enable_circuit_breaker: bool = False,
                       enable_retry: bool = False):
    """Decorator to add comprehensive error handling to functions."""
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                'component': component or func.__name__,
                'is_critical_path': critical,
                'function_name': func.__name__
            }
            
            # Apply circuit breaker if enabled
            if enable_circuit_breaker:
                circuit_breaker = error_handler.get_circuit_breaker(f"{func.__name__}_cb")
                func_to_call = circuit_breaker.call
            else:
                func_to_call = lambda f, *a, **kw: f(*a, **kw)
            
            # Apply retry if enabled
            if enable_retry:
                retry_mechanism = RetryMechanism()
                final_func = retry_mechanism.execute_with_retry
            else:
                final_func = func_to_call
            
            try:
                if enable_retry and enable_circuit_breaker:
                    # Both retry and circuit breaker
                    return retry_mechanism.execute_with_retry(
                        lambda: circuit_breaker.call(func, *args, **kwargs)
                    )
                elif enable_retry:
                    return final_func(func, *args, **kwargs)
                elif enable_circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error_event = error_handler.handle_error(e, context)
                
                # If recovery was successful, try to return a fallback result
                if error_event.resolution_status == "recovered":
                    if 'fallback' in error_event.recovery_actions_taken:
                        # Return fallback result if available
                        return _get_fallback_result(func.__name__, args, kwargs)
                
                # Re-raise the exception if no recovery
                raise e
        
        return wrapper
    return decorator


def _get_fallback_result(func_name: str, args: tuple, kwargs: dict) -> Any:
    """Get fallback result for a function."""
    
    if 'predict' in func_name.lower():
        # For prediction functions, return default prediction
        return 0, 0.5  # No churn, 50% confidence
    
    return None


# Context manager for error handling
@contextmanager
def error_context(component: str, **context_kwargs):
    """Context manager for error handling."""
    
    context = {'component': component, **context_kwargs}
    
    try:
        yield
    except Exception as e:
        error_handler.handle_error(e, context)
        raise


if __name__ == "__main__":
    print("Advanced Error Handling and Recovery System")
    print("This system provides comprehensive error handling and recovery mechanisms.")