"""
High-Performance Optimization Engine for ML Systems at Scale.

This module provides comprehensive performance optimization including:
- Multi-level caching with intelligent cache hierarchies
- CPU and GPU acceleration with auto-optimization
- Memory pool management and garbage collection optimization
- Vectorized operations and SIMD acceleration
- Async/await patterns for I/O bound operations
- JIT compilation and hot path optimization
- Model quantization and pruning
- Batch processing optimization
- Resource-aware computation scheduling
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from collections import deque, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import gc
import psutil
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import pickle
import redis
import multiprocessing as mp
from abc import ABC, abstractmethod

# Performance monitoring
import cProfile
import pstats
from memory_profiler import profile as memory_profile
import tracemalloc

# Try to import optional performance libraries
try:
    import numba
    from numba import jit, cuda, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .logging_config import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Caching configuration
    enable_caching: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    max_cpu_threads: int = 0  # 0 = auto-detect
    enable_vectorization: bool = True
    enable_jit_compilation: bool = True
    
    # GPU optimization
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_pool_size_mb: int = 512
    gc_threshold: int = 1000
    enable_memory_mapping: bool = True
    
    # I/O optimization
    enable_async_io: bool = True
    io_thread_pool_size: int = 10
    prefetch_buffer_size: int = 100
    
    # Model optimization
    enable_model_quantization: bool = True
    quantization_bits: int = 8
    enable_model_pruning: bool = True
    pruning_ratio: float = 0.1
    
    # Batch processing
    optimal_batch_size: int = 0  # 0 = auto-detect
    batch_timeout_ms: float = 100.0
    enable_batch_optimization: bool = True
    
    # Profiling
    enable_profiling: bool = False
    profiling_interval_seconds: int = 60


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    timestamp: datetime
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    batch_size: int
    processing_time_ms: float


class MemoryPool:
    """High-performance memory pool for frequently allocated objects."""
    
    def __init__(self, size_mb: int = 512):
        self.size_bytes = size_mb * 1024 * 1024
        self.pool = {}  # size -> deque of available objects
        self.allocated_bytes = 0
        self.allocation_stats = defaultdict(int)
        self.lock = threading.Lock()
        
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get array from pool or allocate new one."""
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        with self.lock:
            if size_bytes in self.pool and self.pool[size_bytes]:
                array = self.pool[size_bytes].popleft()
                array.fill(0)  # Clear previous data
                return array.reshape(shape)
            
            # Allocate new array if pool space available
            if self.allocated_bytes + size_bytes <= self.size_bytes:
                array = np.empty(shape, dtype=dtype)
                self.allocated_bytes += size_bytes
                self.allocation_stats[size_bytes] += 1
                return array
            
            # Pool full, return regular array
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool for reuse."""
        size_bytes = array.nbytes
        
        with self.lock:
            if size_bytes not in self.pool:
                self.pool[size_bytes] = deque()
            
            if len(self.pool[size_bytes]) < 10:  # Limit pool depth
                self.pool[size_bytes].append(array)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            total_objects = sum(len(pool) for pool in self.pool.values())
            return {
                'allocated_bytes': self.allocated_bytes,
                'total_pooled_objects': total_objects,
                'allocation_stats': dict(self.allocation_stats),
                'pool_utilization': self.allocated_bytes / self.size_bytes
            }


class IntelligentCache:
    """Multi-level intelligent caching system."""
    
    def __init__(self, size_mb: int = 1024, ttl_seconds: int = 3600):
        self.size_bytes = size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        # L1 Cache: In-memory LRU
        self.l1_cache = OrderedDict()
        self.l1_size_bytes = 0
        self.l1_max_size = self.size_bytes // 4
        
        # L2 Cache: Redis (if available)
        self.l2_cache = None
        self._setup_l2_cache()
        
        # L3 Cache: Memory-mapped files
        self.l3_cache_dir = "/tmp/ml_cache"
        os.makedirs(self.l3_cache_dir, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        self.lock = threading.RLock()
    
    def _setup_l2_cache(self) -> None:
        """Setup Redis L2 cache if available."""
        try:
            self.l2_cache = redis.Redis(host='localhost', port=6379, db=1, decode_responses=False)
            self.l2_cache.ping()
            logger.info("L2 Redis cache connected")
        except Exception as e:
            logger.warning(f"L2 cache not available: {e}")
            self.l2_cache = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        with self.lock:
            # Try L1 cache first
            if key in self.l1_cache:
                value, timestamp = self.l1_cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (LRU)
                    self.l1_cache.move_to_end(key)
                    self.stats['l1_hits'] += 1
                    return value
                else:
                    # Expired
                    del self.l1_cache[key]
            
            # Try L2 cache (Redis)
            if self.l2_cache:
                try:
                    cached_data = self.l2_cache.get(key)
                    if cached_data:
                        value = pickle.loads(cached_data)
                        self.stats['l2_hits'] += 1
                        # Promote to L1
                        self._put_l1(key, value)
                        return value
                except Exception as e:
                    logger.warning(f"L2 cache error: {e}")
            
            # Try L3 cache (memory-mapped file)
            try:
                cache_file = os.path.join(self.l3_cache_dir, f"{key}.cache")
                if os.path.exists(cache_file):
                    stat = os.stat(cache_file)
                    if time.time() - stat.st_mtime < self.ttl_seconds:
                        with open(cache_file, 'rb') as f:
                            value = pickle.load(f)
                        self.stats['l3_hits'] += 1
                        # Promote to L2 and L1
                        self._put_l2(key, value)
                        self._put_l1(key, value)
                        return value
                    else:
                        os.unlink(cache_file)  # Remove expired file
            except Exception as e:
                logger.warning(f"L3 cache error: {e}")
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache hierarchy."""
        with self.lock:
            self._put_l1(key, value)
            self._put_l2(key, value)
            self._put_l3(key, value)
    
    def _put_l1(self, key: str, value: Any) -> None:
        """Put value in L1 cache."""
        try:
            value_size = len(pickle.dumps(value))
            
            # Evict if necessary
            while (self.l1_size_bytes + value_size > self.l1_max_size and self.l1_cache):
                oldest_key = next(iter(self.l1_cache))
                oldest_value, _ = self.l1_cache[oldest_key]
                oldest_size = len(pickle.dumps(oldest_value))
                del self.l1_cache[oldest_key]
                self.l1_size_bytes -= oldest_size
                self.stats['evictions'] += 1
            
            self.l1_cache[key] = (value, time.time())
            self.l1_size_bytes += value_size
        except Exception as e:
            logger.warning(f"L1 cache put error: {e}")
    
    def _put_l2(self, key: str, value: Any) -> None:
        """Put value in L2 cache."""
        if self.l2_cache:
            try:
                serialized = pickle.dumps(value)
                self.l2_cache.setex(key, self.ttl_seconds, serialized)
            except Exception as e:
                logger.warning(f"L2 cache put error: {e}")
    
    def _put_l3(self, key: str, value: Any) -> None:
        """Put value in L3 cache."""
        try:
            cache_file = os.path.join(self.l3_cache_dir, f"{key}.cache")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"L3 cache put error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = sum(self.stats.values())
        hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']) / max(total_requests, 1)
        
        return {
            'hit_rate': hit_rate,
            'l1_hit_rate': self.stats['l1_hits'] / max(total_requests, 1),
            'l2_hit_rate': self.stats['l2_hits'] / max(total_requests, 1),
            'l3_hit_rate': self.stats['l3_hits'] / max(total_requests, 1),
            'miss_rate': self.stats['misses'] / max(total_requests, 1),
            'l1_size_mb': self.l1_size_bytes / (1024 * 1024),
            'l1_objects': len(self.l1_cache),
            'evictions': self.stats['evictions'],
            **self.stats
        }


class VectorizedOperations:
    """Vectorized operations for high-performance computation."""
    
    def __init__(self):
        self.use_gpu = GPU_AVAILABLE
        self.use_jit = NUMBA_AVAILABLE
        
    @staticmethod
    def vectorized_prediction(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        """Vectorized linear prediction."""
        if GPU_AVAILABLE:
            try:
                X_gpu = cp.asarray(X)
                weights_gpu = cp.asarray(weights)
                result = cp.dot(X_gpu, weights_gpu) + bias
                return cp.asnumpy(result)
            except:
                pass
        
        # CPU vectorized operation
        return np.dot(X, weights) + bias
    
    @staticmethod
    def batch_normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Vectorized batch normalization."""
        if GPU_AVAILABLE:
            try:
                X_gpu = cp.asarray(X)
                mean_gpu = cp.asarray(mean)
                std_gpu = cp.asarray(std)
                result = (X_gpu - mean_gpu) / (std_gpu + 1e-8)
                return cp.asnumpy(result)
            except:
                pass
        
        return (X - mean) / (std + 1e-8)
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def fast_euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            """JIT-compiled Euclidean distance calculation."""
            n_samples_X = X.shape[0]
            n_samples_Y = Y.shape[0]
            distances = np.zeros((n_samples_X, n_samples_Y))
            
            for i in prange(n_samples_X):
                for j in prange(n_samples_Y):
                    dist = 0.0
                    for k in range(X.shape[1]):
                        diff = X[i, k] - Y[j, k]
                        dist += diff * diff
                    distances[i, j] = np.sqrt(dist)
            
            return distances
    else:
        @staticmethod
        def fast_euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            """Fallback Euclidean distance calculation."""
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(X, Y)


class AsyncBatchProcessor:
    """Asynchronous batch processor for I/O bound operations."""
    
    def __init__(self, batch_size: int = 32, timeout_ms: float = 100.0, max_workers: int = 10):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.pending_requests = deque()
        self.batch_buffer = []
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()
        
        # Start batch processing thread
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self.processing_thread.start()
    
    async def process_async(self, data: Any, process_func: Callable) -> Any:
        """Process data asynchronously with batching."""
        future = asyncio.get_event_loop().create_future()
        
        with self.batch_lock:
            self.pending_requests.append((data, process_func, future))
            self.batch_event.set()
        
        return await future
    
    def _batch_processing_loop(self) -> None:
        """Main batch processing loop."""
        while self.processing_active:
            try:
                # Wait for requests or timeout
                if self.batch_event.wait(timeout=self.timeout_seconds):
                    self.batch_event.clear()
                
                # Collect batch
                batch = []
                with self.batch_lock:
                    while self.pending_requests and len(batch) < self.batch_size:
                        batch.append(self.pending_requests.popleft())
                
                if batch:
                    self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _process_batch(self, batch: List[Tuple[Any, Callable, asyncio.Future]]) -> None:
        """Process a batch of requests."""
        try:
            # Group by processing function
            function_batches = defaultdict(list)
            for data, process_func, future in batch:
                function_batches[process_func].append((data, future))
            
            # Process each function batch
            for process_func, items in function_batches.items():
                try:
                    data_batch = [item[0] for item in items]
                    futures_batch = [item[1] for item in items]
                    
                    # Execute batch processing
                    if hasattr(process_func, '__call__'):
                        if len(data_batch) == 1:
                            results = [process_func(data_batch[0])]
                        else:
                            # Try batch processing
                            try:
                                results = process_func(data_batch)
                                if not isinstance(results, (list, tuple)):
                                    results = [results] * len(data_batch)
                            except:
                                # Fall back to individual processing
                                results = [process_func(data) for data in data_batch]
                    else:
                        results = [None] * len(data_batch)
                    
                    # Set results
                    for future, result in zip(futures_batch, results):
                        if not future.cancelled():
                            future.set_result(result)
                            
                except Exception as e:
                    # Set error for all futures in this batch
                    for _, future in items:
                        if not future.cancelled():
                            future.set_exception(e)
                            
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            # Set error for all futures
            for _, _, future in batch:
                if not future.cancelled():
                    future.set_exception(e)
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self.processing_active = False
        self.batch_event.set()
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        self.executor.shutdown(wait=True)


class ModelOptimizer:
    """Model optimization for inference performance."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
    def optimize_model(self, model: BaseEstimator, X_sample: np.ndarray) -> BaseEstimator:
        """Optimize model for inference performance."""
        optimized_model = model
        
        # Model quantization
        if self.config.enable_model_quantization:
            optimized_model = self._quantize_model(optimized_model, X_sample)
        
        # Model pruning
        if self.config.enable_model_pruning:
            optimized_model = self._prune_model(optimized_model)
        
        # Convert to ONNX for faster inference (if available)
        if ONNX_AVAILABLE:
            optimized_model = self._convert_to_onnx(optimized_model, X_sample)
        
        return optimized_model
    
    def _quantize_model(self, model: BaseEstimator, X_sample: np.ndarray) -> BaseEstimator:
        """Quantize model weights and activations."""
        # Simplified quantization - in practice would be more sophisticated
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # Quantize linear model coefficients
            original_coef = model.coef_.copy()
            original_intercept = model.intercept_.copy()
            
            # Quantize to specified bits
            coef_scale = np.max(np.abs(original_coef)) / (2 ** (self.config.quantization_bits - 1) - 1)
            quantized_coef = np.round(original_coef / coef_scale) * coef_scale
            
            intercept_scale = np.max(np.abs(original_intercept)) / (2 ** (self.config.quantization_bits - 1) - 1)
            quantized_intercept = np.round(original_intercept / intercept_scale) * intercept_scale
            
            # Create quantized model
            from sklearn.base import clone
            quantized_model = clone(model)
            quantized_model.coef_ = quantized_coef
            quantized_model.intercept_ = quantized_intercept
            
            logger.info(f"Model quantized to {self.config.quantization_bits} bits")
            return quantized_model
        
        return model
    
    def _prune_model(self, model: BaseEstimator) -> BaseEstimator:
        """Prune model by removing less important weights."""
        if hasattr(model, 'coef_'):
            # Prune coefficients below threshold
            coef = model.coef_.copy()
            threshold = np.percentile(np.abs(coef), self.config.pruning_ratio * 100)
            
            # Zero out small coefficients
            pruned_coef = np.where(np.abs(coef) < threshold, 0, coef)
            
            from sklearn.base import clone
            pruned_model = clone(model)
            pruned_model.coef_ = pruned_coef
            
            pruned_count = np.sum(pruned_coef == 0)
            total_count = np.size(pruned_coef)
            logger.info(f"Model pruned: {pruned_count}/{total_count} weights removed")
            
            return pruned_model
        
        return model
    
    def _convert_to_onnx(self, model: BaseEstimator, X_sample: np.ndarray) -> Any:
        """Convert model to ONNX format for optimized inference."""
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Define input shape
            initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Create ONNX runtime session
            class ONNXModel:
                def __init__(self, onnx_model):
                    self.session = ort.InferenceSession(onnx_model.SerializeToString())
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                
                def predict(self, X):
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    X = X.astype(np.float32)
                    result = self.session.run([self.output_name], {self.input_name: X})
                    return result[0]
                
                def predict_proba(self, X):
                    # For probability predictions (if supported)
                    predictions = self.predict(X)
                    if len(predictions.shape) == 2:
                        return predictions
                    else:
                        # Convert binary predictions to probabilities
                        proba = np.column_stack([1 - predictions, predictions])
                        return proba
            
            logger.info("Model converted to ONNX for optimized inference")
            return ONNXModel(onnx_model)
            
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
            return model


class PerformanceProfiler:
    """Advanced performance profiler for optimization guidance."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.profiling_results = {}
        
        if enable_memory_tracking:
            tracemalloc.start()
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance."""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(func, func_name, *args, **kwargs)
            return wrapper
        return decorator
    
    def _profile_execution(self, func: Callable, func_name: str, *args, **kwargs) -> Any:
        """Profile function execution."""
        start_time = time.perf_counter()
        
        # Memory tracking
        if self.enable_memory_tracking:
            gc.collect()  # Force garbage collection
            memory_before = self._get_memory_usage()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Collect statistics
        stats = pstats.Stats(profiler)
        
        if self.enable_memory_tracking:
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
        else:
            memory_delta = 0
        
        # Store profiling results
        self.profiling_results[func_name] = {
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta / (1024 * 1024),
            'call_count': stats.total_calls,
            'primitive_calls': stats.prim_calls,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            return current
        else:
            process = psutil.Process()
            return process.memory_info().rss
    
    def get_profiling_report(self) -> Dict[str, Any]:
        """Get comprehensive profiling report."""
        if not self.profiling_results:
            return {"message": "No profiling data available"}
        
        report = {
            'total_functions_profiled': len(self.profiling_results),
            'profiling_results': self.profiling_results.copy(),
            'performance_summary': {}
        }
        
        # Calculate summary statistics
        execution_times = [r['execution_time'] for r in self.profiling_results.values()]
        memory_deltas = [r['memory_delta_mb'] for r in self.profiling_results.values()]
        
        report['performance_summary'] = {
            'total_execution_time': sum(execution_times),
            'average_execution_time': np.mean(execution_times),
            'max_execution_time': max(execution_times),
            'total_memory_delta_mb': sum(memory_deltas),
            'average_memory_delta_mb': np.mean(memory_deltas),
            'max_memory_delta_mb': max(memory_deltas)
        }
        
        return report
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for func_name, stats in self.profiling_results.items():
            # Identify functions with high execution time or memory usage
            if stats['execution_time'] > 1.0:  # More than 1 second
                bottlenecks.append({
                    'function': func_name,
                    'type': 'cpu_bound',
                    'severity': 'high' if stats['execution_time'] > 5.0 else 'medium',
                    'execution_time': stats['execution_time'],
                    'recommendation': 'Consider optimization or async execution'
                })
            
            if stats['memory_delta_mb'] > 100:  # More than 100MB
                bottlenecks.append({
                    'function': func_name,
                    'type': 'memory_intensive',
                    'severity': 'high' if stats['memory_delta_mb'] > 500 else 'medium',
                    'memory_delta_mb': stats['memory_delta_mb'],
                    'recommendation': 'Consider memory optimization or streaming'
                })
        
        return bottlenecks


class HighPerformanceOptimizer:
    """Main high-performance optimization system."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.memory_pool = MemoryPool(self.config.memory_pool_size_mb) if self.config.enable_memory_optimization else None
        self.cache = IntelligentCache(self.config.cache_size_mb, self.config.cache_ttl_seconds) if self.config.enable_caching else None
        self.vectorized_ops = VectorizedOperations()
        self.batch_processor = AsyncBatchProcessor() if self.config.enable_async_io else None
        self.model_optimizer = ModelOptimizer(self.config)
        self.profiler = PerformanceProfiler() if self.config.enable_profiling else None
        
        # Performance metrics
        self.performance_history = deque(maxlen=1000)
        self.optimization_stats = {
            'cache_enabled': self.config.enable_caching,
            'vectorization_enabled': self.config.enable_vectorization,
            'gpu_enabled': self.config.enable_gpu and GPU_AVAILABLE,
            'jit_enabled': self.config.enable_jit_compilation and NUMBA_AVAILABLE,
            'async_enabled': self.config.enable_async_io,
            'optimizations_applied': 0
        }
        
        # Auto-configure optimal settings
        self._auto_configure()
        
        logger.info("High-performance optimizer initialized")
    
    def _auto_configure(self) -> None:
        """Auto-configure optimal performance settings."""
        # Detect optimal CPU thread count
        if self.config.max_cpu_threads == 0:
            self.config.max_cpu_threads = min(mp.cpu_count(), 8)  # Cap at 8 for stability
        
        # Detect optimal batch size
        if self.config.optimal_batch_size == 0:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb >= 32:
                self.config.optimal_batch_size = 128
            elif memory_gb >= 16:
                self.config.optimal_batch_size = 64
            else:
                self.config.optimal_batch_size = 32
        
        # Enable GPU if available and sufficient memory
        if GPU_AVAILABLE and self.config.enable_gpu:
            try:
                gpu_memory = cp.cuda.Device().mem_info[1] / (1024**3)  # Total GPU memory in GB
                if gpu_memory >= 4:  # At least 4GB GPU memory
                    logger.info(f"GPU acceleration enabled with {gpu_memory:.1f}GB memory")
                else:
                    self.config.enable_gpu = False
                    logger.warning("GPU memory insufficient, disabling GPU acceleration")
            except:
                self.config.enable_gpu = False
        
        logger.info(f"Auto-configured: {self.config.max_cpu_threads} CPU threads, "
                   f"batch size {self.config.optimal_batch_size}")
    
    def optimize_prediction_pipeline(self, model: BaseEstimator, X_sample: np.ndarray) -> BaseEstimator:
        """Optimize entire prediction pipeline."""
        optimized_model = model
        
        # Apply model optimizations
        optimized_model = self.model_optimizer.optimize_model(model, X_sample)
        
        # Wrap with caching if enabled
        if self.cache:
            optimized_model = self._add_caching_wrapper(optimized_model)
        
        # Add performance monitoring if enabled
        if self.profiler:
            optimized_model = self._add_profiling_wrapper(optimized_model)
        
        self.optimization_stats['optimizations_applied'] += 1
        return optimized_model
    
    def _add_caching_wrapper(self, model: BaseEstimator) -> Any:
        """Add intelligent caching wrapper to model."""
        
        class CachedModel:
            def __init__(self, model, cache):
                self.model = model
                self.cache = cache
            
            def predict(self, X):
                # Create cache key from input hash
                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                else:
                    X_array = X
                
                cache_key = f"pred_{hash(X_array.tobytes())}"
                
                # Try cache first
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache result
                result = self.model.predict(X)
                self.cache.put(cache_key, result)
                
                return result
            
            def predict_proba(self, X):
                if hasattr(self.model, 'predict_proba'):
                    if isinstance(X, pd.DataFrame):
                        X_array = X.values
                    else:
                        X_array = X
                    
                    cache_key = f"proba_{hash(X_array.tobytes())}"
                    
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                    
                    result = self.model.predict_proba(X)
                    self.cache.put(cache_key, result)
                    
                    return result
                else:
                    predictions = self.predict(X)
                    return np.column_stack([1 - predictions, predictions])
        
        return CachedModel(model, self.cache)
    
    def _add_profiling_wrapper(self, model: BaseEstimator) -> Any:
        """Add performance profiling wrapper to model."""
        
        class ProfiledModel:
            def __init__(self, model, profiler):
                self.model = model
                self.profiler = profiler
            
            def predict(self, X):
                @self.profiler.profile_function("model_predict")
                def _predict(X):
                    return self.model.predict(X)
                
                return _predict(X)
            
            def predict_proba(self, X):
                @self.profiler.profile_function("model_predict_proba")
                def _predict_proba(X):
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(X)
                    else:
                        predictions = self.model.predict(X)
                        return np.column_stack([1 - predictions, predictions])
                
                return _predict_proba(X)
        
        return ProfiledModel(model, self.profiler)
    
    async def async_batch_predict(self, model: BaseEstimator, X_batches: List[np.ndarray]) -> List[np.ndarray]:
        """Asynchronous batch prediction processing."""
        if not self.batch_processor:
            # Fallback to synchronous processing
            return [model.predict(X) for X in X_batches]
        
        # Process batches asynchronously
        tasks = []
        for X_batch in X_batches:
            task = self.batch_processor.process_async(X_batch, model.predict)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def optimize_data_loading(self, data_loader: Callable, use_memory_mapping: bool = None) -> Callable:
        """Optimize data loading operations."""
        if use_memory_mapping is None:
            use_memory_mapping = self.config.enable_memory_mapping
        
        @wraps(data_loader)
        def optimized_loader(*args, **kwargs):
            # Add prefetching and memory mapping optimizations
            if use_memory_mapping:
                # Use memory mapping for large files
                kwargs['memory_map'] = True
            
            return data_loader(*args, **kwargs)
        
        return optimized_loader
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        
        # GPU metrics
        gpu_usage = 0.0
        if GPU_AVAILABLE and self.config.enable_gpu:
            try:
                gpu_usage = cp.cuda.Device().memory_info()[0] / cp.cuda.Device().memory_info()[1] * 100
            except:
                gpu_usage = 0.0
        
        # Cache metrics
        cache_hit_rate = 0.0
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            cache_hit_rate = cache_stats['hit_rate']
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_usage=gpu_usage,
            cache_hit_rate=cache_hit_rate,
            throughput_ops_per_sec=0.0,  # Would be calculated from operation history
            latency_p50_ms=0.0,  # Would be calculated from latency history
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            batch_size=self.config.optimal_batch_size,
            processing_time_ms=0.0
        )
        
        self.performance_history.append(metrics)
        return metrics
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        current_metrics = self.get_performance_metrics()
        
        report = {
            'configuration': asdict(self.config),
            'optimization_stats': self.optimization_stats.copy(),
            'current_performance': asdict(current_metrics),
            'system_capabilities': {
                'cpu_cores': mp.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': GPU_AVAILABLE,
                'numba_available': NUMBA_AVAILABLE,
                'onnx_available': ONNX_AVAILABLE
            }
        }
        
        # Add cache statistics
        if self.cache:
            report['cache_stats'] = self.cache.get_cache_stats()
        
        # Add memory pool statistics
        if self.memory_pool:
            report['memory_pool_stats'] = self.memory_pool.get_pool_stats()
        
        # Add profiling results
        if self.profiler:
            report['profiling_report'] = self.profiler.get_profiling_report()
            report['bottlenecks'] = self.profiler.identify_bottlenecks()
        
        return report
    
    def auto_tune_performance(self, workload_sample: Callable = None) -> Dict[str, Any]:
        """Automatically tune performance parameters based on workload."""
        
        tuning_results = {}
        
        if workload_sample:
            # Run performance tests with different configurations
            test_configs = [
                {'batch_size': 16, 'enable_gpu': False},
                {'batch_size': 32, 'enable_gpu': False},
                {'batch_size': 64, 'enable_gpu': False},
                {'batch_size': 128, 'enable_gpu': False},
            ]
            
            if GPU_AVAILABLE:
                test_configs.extend([
                    {'batch_size': 32, 'enable_gpu': True},
                    {'batch_size': 64, 'enable_gpu': True},
                    {'batch_size': 128, 'enable_gpu': True},
                ])
            
            best_config = None
            best_throughput = 0
            
            for config in test_configs:
                try:
                    # Test configuration
                    start_time = time.perf_counter()
                    
                    # Run workload sample
                    workload_sample(**config)
                    
                    end_time = time.perf_counter()
                    throughput = 1.0 / (end_time - start_time)
                    
                    tuning_results[str(config)] = {
                        'throughput': throughput,
                        'execution_time': end_time - start_time
                    }
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_config = config
                        
                except Exception as e:
                    tuning_results[str(config)] = {'error': str(e)}
            
            # Apply best configuration
            if best_config:
                if 'batch_size' in best_config:
                    self.config.optimal_batch_size = best_config['batch_size']
                if 'enable_gpu' in best_config:
                    self.config.enable_gpu = best_config['enable_gpu']
                
                logger.info(f"Auto-tuned to optimal config: {best_config}")
        
        return {
            'tuning_results': tuning_results,
            'optimal_config': asdict(self.config),
            'performance_improvement': 'Configuration optimized for workload'
        }
    
    def cleanup_resources(self) -> None:
        """Clean up optimization resources."""
        if self.batch_processor:
            self.batch_processor.shutdown()
        
        if self.memory_pool:
            # Force garbage collection
            gc.collect()
        
        logger.info("High-performance optimizer resources cleaned up")


# Factory function for easy optimizer creation
def create_high_performance_optimizer(enable_all: bool = True,
                                    cache_size_mb: int = 1024,
                                    enable_gpu: bool = None) -> HighPerformanceOptimizer:
    """Create optimized performance configuration."""
    
    if enable_gpu is None:
        enable_gpu = GPU_AVAILABLE
    
    config = PerformanceConfig(
        enable_caching=enable_all,
        cache_size_mb=cache_size_mb,
        enable_cpu_optimization=enable_all,
        enable_vectorization=enable_all,
        enable_jit_compilation=enable_all and NUMBA_AVAILABLE,
        enable_gpu=enable_gpu and GPU_AVAILABLE,
        enable_memory_optimization=enable_all,
        enable_async_io=enable_all,
        enable_model_quantization=enable_all,
        enable_model_pruning=enable_all,
        enable_batch_optimization=enable_all,
        enable_profiling=True
    )
    
    return HighPerformanceOptimizer(config)


if __name__ == "__main__":
    print("High-Performance Optimization Engine")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"JIT Available: {NUMBA_AVAILABLE}")
    print(f"ONNX Available: {ONNX_AVAILABLE}")
    print("This system provides comprehensive performance optimization for ML workloads.")