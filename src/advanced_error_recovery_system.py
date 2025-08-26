"""
Advanced Error Recovery System for MLOps Platform.

This module implements sophisticated error handling, recovery strategies,
and fault tolerance mechanisms for production ML systems.
"""

import asyncio
import time
import logging
import traceback
import threading
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .validation import safe_write_json, safe_read_json
from .autonomous_reliability_orchestrator import get_reliability_orchestrator

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    traceback_str: str = ""
    function_name: str = ""
    module_name: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'traceback_str': self.traceback_str,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'severity': self.severity.value,
            'context_data': self.context_data,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'recovery_attempts': self.recovery_attempts
        }


@dataclass 
class RecoveryAction:
    """Recovery action configuration."""
    name: str
    strategy: RecoveryStrategy
    function: Callable
    conditions: List[str] = field(default_factory=list)
    priority: int = 0
    timeout: float = 30.0
    prerequisites: List[str] = field(default_factory=list)


class IntelligentRetryStrategy:
    """
    Intelligent retry mechanism with exponential backoff and jitter.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            import random
            delay *= (0.5 + random.random())
            
        return delay
        
    async def execute_with_retry(self, 
                                func: Callable, 
                                error_context: ErrorContext,
                                *args, **kwargs) -> Any:
        """Execute function with intelligent retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.calculate_delay(attempt - 1)
                    logger.info(f"Retrying {func.__name__} (attempt {attempt}/{self.max_retries}) after {delay:.2f}s delay")
                    await asyncio.sleep(delay)
                    
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 0:
                    logger.info(f"Retry successful for {func.__name__} on attempt {attempt}")
                    metrics.increment('recovery_retry_success', {
                        'function': func.__name__,
                        'attempt': attempt
                    })
                    
                return result
                
            except Exception as e:
                last_exception = e
                error_context.retry_count = attempt
                
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                metrics.increment('recovery_retry_failure', {
                    'function': func.__name__,
                    'attempt': attempt,
                    'error_type': type(e).__name__
                })
                
                if attempt == self.max_retries:
                    break
                    
        error_context.error_message = str(last_exception)
        error_context.error_type = type(last_exception).__name__
        raise last_exception


class ErrorPatternAnalyzer:
    """
    Analyze error patterns to predict and prevent future failures.
    """
    
    def __init__(self, pattern_window: int = 100):
        self.pattern_window = pattern_window
        self.error_history = []
        self.pattern_cache = {}
        
    def record_error(self, error_context: ErrorContext):
        """Record error for pattern analysis."""
        self.error_history.append(error_context)
        
        # Keep only recent errors
        if len(self.error_history) > self.pattern_window:
            self.error_history = self.error_history[-self.pattern_window:]
            
    def detect_error_patterns(self) -> List[Dict[str, Any]]:
        """Detect recurring error patterns."""
        if len(self.error_history) < 10:
            return []
            
        patterns = []
        
        # Pattern 1: Frequent errors from same function
        function_errors = {}
        for error in self.error_history[-50:]:  # Last 50 errors
            func_key = f"{error.module_name}.{error.function_name}"
            if func_key not in function_errors:
                function_errors[func_key] = []
            function_errors[func_key].append(error)
            
        for func_key, errors in function_errors.items():
            if len(errors) >= 5:  # 5+ errors from same function
                patterns.append({
                    'type': 'frequent_function_errors',
                    'function': func_key,
                    'count': len(errors),
                    'severity': self._calculate_pattern_severity(errors),
                    'recommendation': 'Investigate function reliability'
                })
                
        # Pattern 2: Error cascades (errors occurring in quick succession)
        cascade_threshold = 60  # 60 seconds
        cascades = []
        current_cascade = []
        
        for i, error in enumerate(self.error_history):
            if not current_cascade:
                current_cascade = [error]
                continue
                
            time_diff = error.timestamp - current_cascade[-1].timestamp
            if time_diff <= cascade_threshold:
                current_cascade.append(error)
            else:
                if len(current_cascade) >= 3:
                    cascades.append(current_cascade)
                current_cascade = [error]
                
        for cascade in cascades:
            if len(cascade) >= 3:
                patterns.append({
                    'type': 'error_cascade',
                    'count': len(cascade),
                    'duration': cascade[-1].timestamp - cascade[0].timestamp,
                    'severity': ErrorSeverity.HIGH.value,
                    'recommendation': 'Investigate system stability'
                })
                
        return patterns
        
    def _calculate_pattern_severity(self, errors: List[ErrorContext]) -> str:
        """Calculate severity of error pattern."""
        severities = [error.severity for error in errors]
        
        if any(s == ErrorSeverity.CRITICAL for s in severities):
            return ErrorSeverity.CRITICAL.value
        elif any(s == ErrorSeverity.HIGH for s in severities):
            return ErrorSeverity.HIGH.value
        elif len(errors) >= 10:
            return ErrorSeverity.HIGH.value
        else:
            return ErrorSeverity.MEDIUM.value
            
    def predict_failure_probability(self, function_name: str, module_name: str) -> float:
        """Predict probability of failure for given function."""
        func_key = f"{module_name}.{function_name}"
        recent_errors = [e for e in self.error_history[-20:] 
                        if f"{e.module_name}.{e.function_name}" == func_key]
        
        if not recent_errors:
            return 0.0
            
        # Simple prediction based on recent error frequency
        error_rate = len(recent_errors) / min(20, len(self.error_history))
        return min(error_rate * 2, 1.0)  # Cap at 100%


class FallbackManager:
    """
    Manage fallback strategies for different operations.
    """
    
    def __init__(self):
        self.fallback_strategies: Dict[str, List[Callable]] = {}
        self.fallback_results_cache: Dict[str, Any] = {}
        
    def register_fallback(self, operation_name: str, fallback_func: Callable, priority: int = 0):
        """Register fallback strategy for operation."""
        if operation_name not in self.fallback_strategies:
            self.fallback_strategies[operation_name] = []
            
        self.fallback_strategies[operation_name].append((priority, fallback_func))
        self.fallback_strategies[operation_name].sort(key=lambda x: x[0])  # Sort by priority
        
        logger.info(f"Registered fallback for {operation_name}: {fallback_func.__name__}")
        
    async def execute_fallback(self, operation_name: str, error_context: ErrorContext, *args, **kwargs) -> Any:
        """Execute fallback strategies for failed operation."""
        if operation_name not in self.fallback_strategies:
            raise Exception(f"No fallback strategies registered for {operation_name}")
            
        fallbacks = self.fallback_strategies[operation_name]
        last_exception = None
        
        for priority, fallback_func in fallbacks:
            try:
                logger.info(f"Executing fallback {fallback_func.__name__} for {operation_name}")
                
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(*args, **kwargs)
                else:
                    result = fallback_func(*args, **kwargs)
                    
                # Cache successful fallback result
                cache_key = f"{operation_name}_{hashlib.md5(str(args).encode()).hexdigest()[:8]}"
                self.fallback_results_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time(),
                    'fallback_used': fallback_func.__name__
                }
                
                metrics.increment('fallback_success', {
                    'operation': operation_name,
                    'fallback': fallback_func.__name__
                })
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Fallback {fallback_func.__name__} failed: {e}")
                metrics.increment('fallback_failure', {
                    'operation': operation_name,
                    'fallback': fallback_func.__name__,
                    'error_type': type(e).__name__
                })
                continue
                
        # All fallbacks failed
        error_context.recovery_attempts.append("all_fallbacks_failed")
        raise last_exception or Exception("All fallback strategies failed")


class ErrorRecoverySystem:
    """
    Main error recovery system coordinating all recovery mechanisms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.retry_strategy = IntelligentRetryStrategy(
            max_retries=self.config.get('max_retries', 3),
            base_delay=self.config.get('base_delay', 1.0),
            max_delay=self.config.get('max_delay', 60.0)
        )
        self.pattern_analyzer = ErrorPatternAnalyzer()
        self.fallback_manager = FallbackManager()
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.error_log_path = Path("logs/error_recovery.jsonl")
        self.error_log_path.parent.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load recovery system configuration."""
        default_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 60.0,
            'enable_pattern_analysis': True,
            'enable_fallbacks': True,
            'enable_auto_recovery': True,
            'error_log_retention_days': 30
        }
        
        if config_path and Path(config_path).exists():
            try:
                user_config = safe_read_json(config_path)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading recovery config: {e}")
                
        return default_config
        
    def register_recovery_action(self, action: RecoveryAction):
        """Register custom recovery action."""
        self.recovery_actions[action.name] = action
        logger.info(f"Registered recovery action: {action.name}")
        
    def register_fallback(self, operation_name: str, fallback_func: Callable, priority: int = 0):
        """Register fallback strategy."""
        self.fallback_manager.register_fallback(operation_name, fallback_func, priority)
        
    async def handle_error(self, 
                          error: Exception, 
                          function_name: str, 
                          module_name: str,
                          context_data: Optional[Dict[str, Any]] = None,
                          operation_name: Optional[str] = None) -> ErrorContext:
        """Comprehensive error handling and recovery."""
        # Create error context
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            function_name=function_name,
            module_name=module_name,
            context_data=context_data or {},
            severity=self._classify_error_severity(error)
        )
        
        # Log error
        await self._log_error(error_context)
        
        # Record for pattern analysis
        if self.config['enable_pattern_analysis']:
            self.pattern_analyzer.record_error(error_context)
            
        # Update metrics
        self._update_error_metrics(error_context)
        
        # Trigger recovery if enabled
        if self.config['enable_auto_recovery'] and operation_name:
            try:
                await self._attempt_recovery(error_context, operation_name)
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {recovery_error}")
                
        return error_context
        
    async def execute_with_recovery(self,
                                   operation_name: str,
                                   func: Callable,
                                   *args, **kwargs) -> Any:
        """Execute function with comprehensive error recovery."""
        function_name = func.__name__
        module_name = getattr(func, '__module__', 'unknown')
        
        try:
            # Check failure prediction
            failure_prob = self.pattern_analyzer.predict_failure_probability(function_name, module_name)
            if failure_prob > 0.7:  # High probability of failure
                logger.warning(f"High failure probability ({failure_prob:.2f}) for {function_name}")
                
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            return result
            
        except Exception as e:
            # Handle error and attempt recovery
            error_context = await self.handle_error(e, function_name, module_name, 
                                                   operation_name=operation_name)
            
            # Try retry strategy
            if error_context.severity != ErrorSeverity.CRITICAL:
                try:
                    return await self.retry_strategy.execute_with_retry(
                        func, error_context, *args, **kwargs
                    )
                except Exception:
                    pass  # Continue to fallback
                    
            # Try fallbacks if enabled
            if self.config['enable_fallbacks'] and operation_name:
                try:
                    return await self.fallback_manager.execute_fallback(
                        operation_name, error_context, *args, **kwargs
                    )
                except Exception:
                    pass  # No fallback available
                    
            # All recovery attempts failed
            error_context.recovery_attempts.append("all_recovery_failed")
            await self._escalate_error(error_context)
            raise
            
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['memory', 'disk full', 'connection refused', 'timeout']):
            return ErrorSeverity.CRITICAL
        elif error_type in ['SystemError', 'MemoryError', 'OSError']:
            return ErrorSeverity.CRITICAL
            
        # High severity errors
        elif error_type in ['ConnectionError', 'TimeoutError', 'FileNotFoundError']:
            return ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ['permission denied', 'access denied']):
            return ErrorSeverity.HIGH
            
        # Medium severity errors
        elif error_type in ['ValueError', 'KeyError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
            
        # Low severity errors
        else:
            return ErrorSeverity.LOW
            
    async def _log_error(self, error_context: ErrorContext):
        """Log error context to file."""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(error_context.timestamp).isoformat(),
                **error_context.to_dict()
            }
            
            with open(self.error_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
            
    def _update_error_metrics(self, error_context: ErrorContext):
        """Update error metrics."""
        metrics.increment('error_total', {
            'error_type': error_context.error_type,
            'function': error_context.function_name,
            'module': error_context.module_name,
            'severity': error_context.severity.value
        })
        
    async def _attempt_recovery(self, error_context: ErrorContext, operation_name: str):
        """Attempt automatic recovery based on error context."""
        # Select appropriate recovery actions
        applicable_actions = []
        
        for action_name, action in self.recovery_actions.items():
            if self._is_action_applicable(action, error_context):
                applicable_actions.append(action)
                
        # Sort by priority
        applicable_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # Execute recovery actions
        for action in applicable_actions:
            try:
                logger.info(f"Executing recovery action: {action.name}")
                
                if asyncio.iscoroutinefunction(action.function):
                    await asyncio.wait_for(action.function(error_context), timeout=action.timeout)
                else:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, action.function, error_context),
                        timeout=action.timeout
                    )
                    
                error_context.recovery_attempts.append(action.name)
                metrics.increment('recovery_action_success', {'action': action.name})
                
            except Exception as e:
                logger.error(f"Recovery action {action.name} failed: {e}")
                metrics.increment('recovery_action_failure', {'action': action.name})
                
    def _is_action_applicable(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Check if recovery action is applicable to error context."""
        if not action.conditions:
            return True
            
        for condition in action.conditions:
            if condition == "high_severity" and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                return True
            elif condition == "memory_error" and "memory" in error_context.error_message.lower():
                return True
            elif condition == "connection_error" and "connection" in error_context.error_message.lower():
                return True
                
        return False
        
    async def _escalate_error(self, error_context: ErrorContext):
        """Escalate critical errors that couldn't be recovered."""
        logger.critical(f"Escalating unrecoverable error: {error_context.error_id}")
        
        # Send to reliability orchestrator
        try:
            orchestrator = get_reliability_orchestrator()
            # Trigger emergency procedures if critical
            if error_context.severity == ErrorSeverity.CRITICAL:
                await orchestrator.health_monitor._trigger_emergency_recovery(None)
        except Exception as e:
            logger.error(f"Failed to escalate to reliability orchestrator: {e}")
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        patterns = self.pattern_analyzer.detect_error_patterns()
        
        return {
            'total_errors': len(self.pattern_analyzer.error_history),
            'error_patterns': patterns,
            'severity_distribution': self._get_severity_distribution(),
            'most_common_errors': self._get_most_common_errors(),
            'recovery_success_rate': self._calculate_recovery_success_rate()
        }
        
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get error severity distribution."""
        distribution = {severity.value: 0 for severity in ErrorSeverity}
        
        for error in self.pattern_analyzer.error_history:
            distribution[error.severity.value] += 1
            
        return distribution
        
    def _get_most_common_errors(self) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_counts = {}
        
        for error in self.pattern_analyzer.error_history:
            error_type = error.error_type
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
            
        return [
            {'error_type': error_type, 'count': count}
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate overall recovery success rate."""
        total_attempts = len(self.pattern_analyzer.error_history)
        if total_attempts == 0:
            return 1.0
            
        successful_recoveries = sum(
            1 for error in self.pattern_analyzer.error_history 
            if error.recovery_attempts and "all_recovery_failed" not in error.recovery_attempts
        )
        
        return successful_recoveries / total_attempts


# Global instance
_error_recovery_system: Optional[ErrorRecoverySystem] = None


def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system instance."""
    global _error_recovery_system
    if _error_recovery_system is None:
        _error_recovery_system = ErrorRecoverySystem()
    return _error_recovery_system


def with_error_recovery(operation_name: str):
    """Decorator for functions with comprehensive error recovery."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            recovery_system = get_error_recovery_system()
            return await recovery_system.execute_with_recovery(operation_name, func, *args, **kwargs)
            
        def sync_wrapper(*args, **kwargs):
            recovery_system = get_error_recovery_system()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we handle errors but don't do async recovery
                error_context = asyncio.run(recovery_system.handle_error(
                    e, func.__name__, getattr(func, '__module__', 'unknown'), operation_name=operation_name
                ))
                raise
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Built-in recovery actions
def clear_memory_recovery(error_context: ErrorContext):
    """Recovery action to clear memory when memory errors occur."""
    import gc
    gc.collect()
    logger.info("Executed memory cleanup recovery")


async def restart_service_recovery(error_context: ErrorContext):
    """Recovery action to restart services for critical errors."""
    logger.info("Simulated service restart recovery")
    # Implementation would restart necessary services
    
    
def initialize_default_recovery_actions():
    """Initialize default recovery actions."""
    recovery_system = get_error_recovery_system()
    
    # Register default recovery actions
    recovery_system.register_recovery_action(RecoveryAction(
        name="clear_memory",
        strategy=RecoveryStrategy.RETRY,
        function=clear_memory_recovery,
        conditions=["memory_error"],
        priority=1
    ))
    
    recovery_system.register_recovery_action(RecoveryAction(
        name="restart_service",
        strategy=RecoveryStrategy.RESTART,
        function=restart_service_recovery,
        conditions=["high_severity"],
        priority=0
    ))


if __name__ == "__main__":
    async def main():
        # Initialize recovery system
        initialize_default_recovery_actions()
        recovery_system = get_error_recovery_system()
        
        # Register fallback
        def simple_fallback():
            return "fallback_result"
            
        recovery_system.register_fallback("test_operation", simple_fallback)
        
        # Test error recovery
        @with_error_recovery("test_operation")
        async def test_function():
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise ValueError("Test error")
            return "success"
            
        # Run tests
        for i in range(10):
            try:
                result = await test_function()
                print(f"Test {i}: {result}")
            except Exception as e:
                print(f"Test {i}: Failed - {e}")
                
        # Get statistics
        stats = recovery_system.get_error_statistics()
        print(json.dumps(stats, indent=2))
        
    asyncio.run(main())