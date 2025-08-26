"""
Comprehensive Performance Benchmarking Suite for MLOps Platform.

Tests system performance, scalability, and resource utilization across all components.
"""

import asyncio
import time
import sys
import os
import json
import statistics
import threading
import multiprocessing
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.logging_config import get_logger
from src.validation import safe_write_json
from src.predict_churn import make_prediction
from src.train_model import train_churn_model
from src.metrics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    test_name: str
    duration_seconds: float
    throughput_ops_per_second: float
    memory_peak_mb: float
    cpu_peak_percent: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    timestamp: float = field(default_factory=time.time)
    system_info: Dict[str, Any] = field(default_factory=dict)
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    scalability_results: Dict[str, Any] = field(default_factory=dict)
    resource_utilization: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'benchmark_results': [
                {
                    'test_name': r.test_name,
                    'duration_seconds': r.duration_seconds,
                    'throughput_ops_per_second': r.throughput_ops_per_second,
                    'memory_peak_mb': r.memory_peak_mb,
                    'cpu_peak_percent': r.cpu_peak_percent,
                    'success_rate': r.success_rate,
                    'error_count': r.error_count,
                    'metadata': r.metadata
                }
                for r in self.benchmark_results
            ],
            'scalability_results': self.scalability_results,
            'resource_utilization': self.resource_utilization,
            'performance_score': self.performance_score,
            'recommendations': self.recommendations
        }


class ResourceMonitor:
    """Monitor system resource usage during benchmarks."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.disk_io_samples = []
        self.network_io_samples = []
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.disk_io_samples.clear()
        self.network_io_samples.clear()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_samples.append(memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.disk_io_samples.append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })
                    
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.network_io_samples.append({
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    })
                    
                await asyncio.sleep(0.5)  # Sample every 0.5 seconds
                
            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")
                await asyncio.sleep(1)
                
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage."""
        return {
            'cpu_peak_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'cpu_avg_percent': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'memory_peak_percent': max(self.memory_samples) if self.memory_samples else 0,
            'memory_avg_percent': statistics.mean(self.memory_samples) if self.memory_samples else 0,
            'memory_peak_mb': max(self.memory_samples) * psutil.virtual_memory().total / (1024**3) * 10.24 if self.memory_samples else 0
        }


class PerformanceBenchmarker:
    """Comprehensive performance benchmarker."""
    
    def __init__(self):
        self.report = PerformanceReport()
        self.resource_monitor = ResourceMonitor()
        
    async def run_comprehensive_benchmarks(self) -> PerformanceReport:
        """Run comprehensive performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks...")
        
        # Collect system information
        await self._collect_system_info()
        
        # Core functionality benchmarks
        await self._benchmark_core_functionality()
        
        # Scalability benchmarks
        await self._benchmark_scalability()
        
        # Stress tests
        await self._run_stress_tests()
        
        # Memory efficiency tests
        await self._benchmark_memory_efficiency()
        
        # I/O performance tests
        await self._benchmark_io_performance()
        
        # Concurrent processing tests
        await self._benchmark_concurrency()
        
        # Calculate performance score
        self._calculate_performance_score()
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"Performance benchmarks completed. Score: {self.report.performance_score:.1f}/100")
        return self.report
        
    async def _collect_system_info(self):
        """Collect system information."""
        self.report.system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': sys.version,
            'platform': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'Unknown'
        }
        
    async def _benchmark_core_functionality(self):
        """Benchmark core ML functionality."""
        logger.info("Benchmarking core functionality...")
        
        # Benchmark prediction performance
        await self._benchmark_prediction_performance()
        
        # Benchmark training performance
        await self._benchmark_training_performance()
        
        # Benchmark data processing
        await self._benchmark_data_processing()
        
    async def _benchmark_prediction_performance(self):
        """Benchmark prediction performance."""
        # Prepare test data
        test_data = {
            'gender_Female': 1.0,
            'gender_Male': 0.0,
            'Partner_No': 0.0,
            'Partner_Yes': 1.0,
            'SeniorCitizen': 0.0,
            'tenure': 12.0,
            'MonthlyCharges': 70.0,
            'TotalCharges': 840.0
        }
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        iterations = 1000
        
        try:
            for i in range(iterations):
                try:
                    prediction, probability = make_prediction(test_data)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    if i < 10:  # Log first few errors
                        logger.warning(f"Prediction error: {e}")
                        
        except Exception as e:
            logger.error(f"Prediction benchmark failed: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Prediction Performance",
            duration_seconds=duration,
            throughput_ops_per_second=success_count / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=success_count / iterations if iterations > 0 else 0,
            error_count=error_count,
            metadata={
                'iterations': iterations,
                'avg_latency_ms': (duration / success_count * 1000) if success_count > 0 else 0
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_training_performance(self):
        """Benchmark model training performance."""
        # Skip if training data not available
        features_path = Path('data/processed/processed_features.csv')
        target_path = Path('data/processed/processed_target.csv')
        
        if not (features_path.exists() and target_path.exists()):
            logger.warning("Training data not available, skipping training benchmark")
            return
            
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        try:
            model_path, run_id = train_churn_model(str(features_path), str(target_path))
            success_count = 1
        except Exception as e:
            error_count = 1
            logger.error(f"Training benchmark error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Model Training Performance",
            duration_seconds=duration,
            throughput_ops_per_second=success_count / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0,
            error_count=error_count,
            metadata={
                'training_duration_minutes': duration / 60
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_data_processing(self):
        """Benchmark data processing performance."""
        # Generate test data
        import pandas as pd
        
        # Create synthetic dataset
        n_samples = 10000
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.randint(0, 100, n_samples)
        })
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        try:
            # Simulate data processing operations
            processed_data = test_data.copy()
            
            # Feature engineering
            processed_data['feature_1_squared'] = processed_data['feature_1'] ** 2
            processed_data['feature_interaction'] = processed_data['feature_1'] * processed_data['feature_2']
            
            # Categorical encoding
            processed_data = pd.get_dummies(processed_data, columns=['feature_3'])
            
            # Scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            
            success_count = 1
            
        except Exception as e:
            error_count = 1
            logger.error(f"Data processing benchmark error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Data Processing Performance",
            duration_seconds=duration,
            throughput_ops_per_second=n_samples / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0,
            error_count=error_count,
            metadata={
                'samples_processed': n_samples,
                'processing_rate_samples_per_second': n_samples / duration if duration > 0 else 0
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_scalability(self):
        """Benchmark scalability with increasing load."""
        logger.info("Benchmarking scalability...")
        
        scalability_results = {}
        
        # Test different load levels
        load_levels = [1, 5, 10, 25, 50, 100]
        
        for load in load_levels:
            logger.info(f"Testing scalability with {load} concurrent operations...")
            
            # Prepare test data
            test_data = {
                'gender_Female': 1.0,
                'gender_Male': 0.0,
                'SeniorCitizen': 0.0,
                'tenure': 12.0,
                'MonthlyCharges': 70.0,
                'TotalCharges': 840.0
            }
            
            # Start monitoring
            monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
            
            start_time = time.time()
            success_count = 0
            error_count = 0
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=min(load, 20)) as executor:
                futures = []
                
                for _ in range(load):
                    future = executor.submit(self._safe_prediction, test_data)
                    futures.append(future)
                    
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        
            duration = time.time() - start_time
            
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            try:
                monitor_task.cancel()
            except:
                pass
                
            peak_usage = self.resource_monitor.get_peak_usage()
            
            scalability_results[f"load_{load}"] = {
                'concurrent_operations': load,
                'duration_seconds': duration,
                'throughput_ops_per_second': success_count / duration if duration > 0 else 0,
                'success_rate': success_count / load if load > 0 else 0,
                'error_count': error_count,
                'cpu_peak_percent': peak_usage['cpu_peak_percent'],
                'memory_peak_mb': peak_usage['memory_peak_mb']
            }
            
        self.report.scalability_results = scalability_results
        
    def _safe_prediction(self, test_data: Dict[str, Any]):
        """Make prediction safely for concurrent testing."""
        try:
            return make_prediction(test_data)
        except Exception as e:
            raise e
            
    async def _run_stress_tests(self):
        """Run stress tests to find system limits."""
        logger.info("Running stress tests...")
        
        # Memory stress test
        await self._stress_test_memory()
        
        # CPU stress test
        await self._stress_test_cpu()
        
    async def _stress_test_memory(self):
        """Stress test memory usage."""
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        memory_blocks = []
        max_blocks = 0
        
        try:
            # Allocate memory until we hit a limit
            for i in range(1000):
                # Allocate 10MB blocks
                block = np.random.randn(10 * 1024 * 1024 // 8)  # 10MB of float64
                memory_blocks.append(block)
                max_blocks = i + 1
                
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:  # Stop before system becomes unstable
                    break
                    
                if i % 10 == 0:  # Small delay every 10 allocations
                    await asyncio.sleep(0.01)
                    
        except MemoryError:
            logger.info("Memory limit reached")
        except Exception as e:
            logger.error(f"Memory stress test error: {e}")
            
        duration = time.time() - start_time
        
        # Clean up
        memory_blocks.clear()
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Memory Stress Test",
            duration_seconds=duration,
            throughput_ops_per_second=max_blocks / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=1.0,
            error_count=0,
            metadata={
                'max_memory_blocks': max_blocks,
                'estimated_memory_mb': max_blocks * 10
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _stress_test_cpu(self):
        """Stress test CPU usage."""
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        iterations = 0
        test_duration = 10  # Run for 10 seconds
        
        def cpu_intensive_task():
            """CPU intensive computation."""
            result = 0
            for i in range(1000000):
                result += i * i * 0.1
            return result
            
        try:
            with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
                futures = []
                
                end_time = start_time + test_duration
                while time.time() < end_time:
                    future = executor.submit(cpu_intensive_task)
                    futures.append(future)
                    iterations += 1
                    
                    if len(futures) >= 100:  # Limit futures to prevent memory issues
                        # Wait for some to complete
                        for i, future in enumerate(futures[:50]):
                            try:
                                future.result(timeout=1)
                            except:
                                pass
                        futures = futures[50:]
                        
                # Wait for remaining futures
                for future in futures:
                    try:
                        future.result(timeout=1)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"CPU stress test error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="CPU Stress Test",
            duration_seconds=duration,
            throughput_ops_per_second=iterations / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=1.0,
            error_count=0,
            metadata={
                'cpu_intensive_iterations': iterations,
                'target_duration_seconds': test_duration
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_memory_efficiency(self):
        """Benchmark memory efficiency."""
        logger.info("Benchmarking memory efficiency...")
        
        # Test memory usage patterns
        await self._test_memory_patterns()
        
    async def _test_memory_patterns(self):
        """Test different memory usage patterns."""
        import gc
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        
        # Test 1: Large object creation and cleanup
        large_objects = []
        for i in range(10):
            # Create large object
            obj = np.random.randn(1024 * 1024)  # ~8MB
            large_objects.append(obj)
            
            if i % 3 == 0:  # Cleanup periodically
                large_objects = large_objects[-3:]
                gc.collect()
                
            await asyncio.sleep(0.1)
            
        # Cleanup
        large_objects.clear()
        gc.collect()
        
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Memory Efficiency Test",
            duration_seconds=duration,
            throughput_ops_per_second=10 / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=1.0,
            error_count=0,
            metadata={
                'memory_pattern': 'large_object_lifecycle'
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_io_performance(self):
        """Benchmark I/O performance."""
        logger.info("Benchmarking I/O performance...")
        
        # File I/O benchmark
        await self._benchmark_file_io()
        
    async def _benchmark_file_io(self):
        """Benchmark file I/O performance."""
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        files_created = 0
        
        try:
            test_dir = Path('temp_io_test')
            test_dir.mkdir(exist_ok=True)
            
            # Write test
            test_data = "x" * 1024  # 1KB of data
            
            for i in range(100):
                file_path = test_dir / f"test_file_{i}.txt"
                with open(file_path, 'w') as f:
                    f.write(test_data * 100)  # 100KB file
                files_created += 1
                
            # Read test
            for i in range(files_created):
                file_path = test_dir / f"test_file_{i}.txt"
                with open(file_path, 'r') as f:
                    content = f.read()
                    
            # Cleanup
            for i in range(files_created):
                file_path = test_dir / f"test_file_{i}.txt"
                if file_path.exists():
                    file_path.unlink()
            test_dir.rmdir()
            
        except Exception as e:
            logger.error(f"I/O benchmark error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="File I/O Performance",
            duration_seconds=duration,
            throughput_ops_per_second=(files_created * 2) / duration if duration > 0 else 0,  # Read + Write
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=1.0,
            error_count=0,
            metadata={
                'files_processed': files_created,
                'file_size_kb': 100,
                'operations': 'write_and_read'
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _benchmark_concurrency(self):
        """Benchmark concurrent processing capabilities."""
        logger.info("Benchmarking concurrency...")
        
        # Test different concurrency models
        await self._test_thread_concurrency()
        await self._test_process_concurrency()
        
    async def _test_thread_concurrency(self):
        """Test thread-based concurrency."""
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        tasks_completed = 0
        max_workers = min(20, psutil.cpu_count() * 2)
        
        def worker_task(task_id):
            """Worker task for concurrency test."""
            result = 0
            for i in range(100000):
                result += i * 0.001
            return result
            
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit tasks
                for i in range(100):
                    future = executor.submit(worker_task, i)
                    futures.append(future)
                    
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        tasks_completed += 1
                    except Exception as e:
                        logger.error(f"Thread task error: {e}")
                        
        except Exception as e:
            logger.error(f"Thread concurrency test error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Thread Concurrency Performance",
            duration_seconds=duration,
            throughput_ops_per_second=tasks_completed / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=tasks_completed / 100.0,
            error_count=100 - tasks_completed,
            metadata={
                'max_workers': max_workers,
                'total_tasks': 100,
                'concurrency_model': 'threads'
            }
        )
        
        self.report.benchmark_results.append(result)
        
    async def _test_process_concurrency(self):
        """Test process-based concurrency."""
        # Start monitoring
        monitor_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        start_time = time.time()
        tasks_completed = 0
        max_workers = min(4, psutil.cpu_count())  # Fewer processes
        
        def worker_process(task_id):
            """Worker process for concurrency test."""
            import time
            result = 0
            for i in range(50000):  # Smaller workload for processes
                result += i * 0.001
            return result
            
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit tasks
                for i in range(20):  # Fewer tasks for processes
                    future = executor.submit(worker_process, i)
                    futures.append(future)
                    
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        tasks_completed += 1
                    except Exception as e:
                        logger.error(f"Process task error: {e}")
                        
        except Exception as e:
            logger.error(f"Process concurrency test error: {e}")
            
        duration = time.time() - start_time
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        try:
            monitor_task.cancel()
        except:
            pass
            
        peak_usage = self.resource_monitor.get_peak_usage()
        
        result = BenchmarkResult(
            test_name="Process Concurrency Performance",
            duration_seconds=duration,
            throughput_ops_per_second=tasks_completed / duration if duration > 0 else 0,
            memory_peak_mb=peak_usage['memory_peak_mb'],
            cpu_peak_percent=peak_usage['cpu_peak_percent'],
            success_rate=tasks_completed / 20.0,
            error_count=20 - tasks_completed,
            metadata={
                'max_workers': max_workers,
                'total_tasks': 20,
                'concurrency_model': 'processes'
            }
        )
        
        self.report.benchmark_results.append(result)
        
    def _calculate_performance_score(self):
        """Calculate overall performance score."""
        if not self.report.benchmark_results:
            self.report.performance_score = 0.0
            return
            
        scores = []
        
        # Score individual benchmarks
        for result in self.report.benchmark_results:
            benchmark_score = 100.0
            
            # Success rate impact
            benchmark_score *= result.success_rate
            
            # Throughput bonus (relative)
            if result.throughput_ops_per_second > 0:
                throughput_score = min(100, result.throughput_ops_per_second * 10)
                benchmark_score = (benchmark_score + throughput_score) / 2
                
            # Resource efficiency (lower is better for CPU/memory usage)
            if result.cpu_peak_percent > 0:
                cpu_efficiency = max(0, 100 - result.cpu_peak_percent)
                benchmark_score = (benchmark_score * 2 + cpu_efficiency) / 3
                
            scores.append(benchmark_score)
            
        # Calculate overall score
        self.report.performance_score = statistics.mean(scores) if scores else 0.0
        
    def _generate_recommendations(self):
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze results for recommendations
        for result in self.report.benchmark_results:
            if result.success_rate < 0.9:
                recommendations.append(f"Improve reliability of {result.test_name} - success rate: {result.success_rate:.1%}")
                
            if result.cpu_peak_percent > 80:
                recommendations.append(f"Optimize CPU usage in {result.test_name} - peak usage: {result.cpu_peak_percent:.1f}%")
                
            if result.throughput_ops_per_second < 10:
                recommendations.append(f"Improve throughput of {result.test_name} - current: {result.throughput_ops_per_second:.1f} ops/sec")
                
        # Scalability recommendations
        if self.report.scalability_results:
            high_load_result = self.report.scalability_results.get('load_100')
            if high_load_result and high_load_result['success_rate'] < 0.8:
                recommendations.append("Consider implementing connection pooling or load balancing for high concurrent loads")
                
        # System recommendations
        if self.report.system_info['memory_total_gb'] < 4:
            recommendations.append("Consider increasing system memory for better performance")
            
        if self.report.system_info['cpu_count'] < 4:
            recommendations.append("Consider upgrading to more CPU cores for better parallel processing")
            
        self.report.recommendations = recommendations[:10]  # Top 10 recommendations


async def main():
    """Run comprehensive performance benchmarks."""
    print("🚀 Starting Comprehensive Performance Benchmarks")
    print("=" * 60)
    
    benchmarker = PerformanceBenchmarker()
    
    try:
        # Run benchmarks
        report = await benchmarker.run_comprehensive_benchmarks()
        
        # Save report
        report_data = report.to_dict()
        report_path = Path('performance_benchmark_report.json')
        safe_write_json(str(report_path), report_data)
        
        # Print summary
        print(f"\n📊 Performance Benchmark Summary")
        print(f"System: {report.system_info['cpu_count']} CPU cores, {report.system_info['memory_total_gb']:.1f} GB RAM")
        print(f"Total Benchmarks: {len(report.benchmark_results)}")
        print(f"Performance Score: {report.performance_score:.1f}/100")
        
        # Print top results
        print(f"\n🏆 Top Performance Results:")
        sorted_results = sorted(report.benchmark_results, 
                               key=lambda x: x.throughput_ops_per_second, reverse=True)
        
        for result in sorted_results[:5]:
            print(f"  {result.test_name}:")
            print(f"    Throughput: {result.throughput_ops_per_second:.1f} ops/sec")
            print(f"    Success Rate: {result.success_rate:.1%}")
            print(f"    Duration: {result.duration_seconds:.2f}s")
            
        # Print scalability results
        if report.scalability_results:
            print(f"\n📈 Scalability Results:")
            for load_key, result in report.scalability_results.items():
                load = result['concurrent_operations']
                throughput = result['throughput_ops_per_second']
                print(f"  {load} concurrent: {throughput:.1f} ops/sec")
                
        # Print recommendations
        if report.recommendations:
            print(f"\n💡 Performance Recommendations:")
            for rec in report.recommendations[:5]:
                print(f"  • {rec}")
                
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit code based on performance score
        if report.performance_score >= 80:
            print("✅ Performance benchmarks passed!")
            return 0
        elif report.performance_score >= 60:
            print("⚠️  Performance benchmarks completed with warnings")
            return 1
        else:
            print("❌ Performance benchmarks indicate issues")
            return 2
            
    except Exception as e:
        logger.error(f"Performance benchmarks failed: {e}")
        print(f"❌ Performance benchmarks failed: {e}")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())