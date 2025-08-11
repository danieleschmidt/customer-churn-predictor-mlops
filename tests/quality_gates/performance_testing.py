"""
Comprehensive Performance Testing Framework.

This module provides advanced performance testing capabilities including:
- Benchmark testing for all ML operations with statistical analysis
- Load testing scenarios for distributed systems with realistic traffic patterns
- Memory profiling and performance regression detection with alerts
- Latency and throughput monitoring with percentile analysis
- Automated performance baseline management and drift detection
- Resource utilization tracking and capacity planning insights
"""

import os
import time
import json
import asyncio
import threading
import statistics
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile as memory_profile
import cProfile
import pstats
import tracemalloc
import gc

# Load testing libraries
try:
    import locust
    from locust import HttpUser, task, between, events
    from locust.env import Environment
    from locust.stats import stats_printer
    from locust.log import setup_logging
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

# Async HTTP client for load testing
try:
    import aiohttp
    import httpx
    ASYNC_HTTP_AVAILABLE = True
except ImportError:
    ASYNC_HTTP_AVAILABLE = False

# Machine learning for performance modeling
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PerformanceTestType:
    """Performance test type constants."""
    BENCHMARK = "benchmark"
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    VOLUME = "volume"
    ENDURANCE = "endurance"
    MEMORY = "memory"
    CPU = "cpu"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during testing."""
    test_id: str
    test_type: str
    timestamp: datetime
    duration_seconds: float
    
    # Response time metrics (milliseconds)
    response_time_min: float
    response_time_max: float
    response_time_avg: float
    response_time_p50: float
    response_time_p90: float
    response_time_p95: float
    response_time_p99: float
    
    # Throughput metrics
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    error_rate: float
    
    # Resource utilization
    cpu_usage_avg: float
    cpu_usage_max: float
    memory_usage_avg: float
    memory_usage_max: float
    memory_allocated_mb: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition."""
    name: str
    function: Callable
    iterations: int
    warmup_iterations: int
    timeout_seconds: float
    parameters: Dict[str, Any]
    expected_performance: Optional[Dict[str, float]] = None
    regression_threshold: float = 0.1  # 10% regression tolerance


@dataclass
class LoadTestScenario:
    """Load test scenario definition."""
    name: str
    endpoint: str
    method: str = "GET"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    
    # Load parameters
    users: int = 10
    spawn_rate: float = 1.0  # users per second
    duration_seconds: int = 60
    
    # Success criteria
    max_response_time: float = 1000.0  # milliseconds
    max_error_rate: float = 0.05  # 5%
    min_throughput: float = 10.0  # requests per second


@dataclass
class PerformanceTest:
    """Complete performance test configuration."""
    test_id: str
    name: str
    description: str
    test_type: str
    scenarios: List[Any]  # Benchmarks or Load Test Scenarios
    baseline_metrics: Optional[PerformanceMetrics] = None
    thresholds: Dict[str, float] = field(default_factory=dict)


class PerformanceProfiler:
    """Advanced performance profiler with detailed metrics collection."""
    
    def __init__(self):
        self.current_profile = None
        self.memory_snapshots = deque(maxlen=1000)
        self.cpu_snapshots = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Start memory tracking
        tracemalloc.start()
    
    @contextmanager
    def profile_context(self, test_id: str):
        """Context manager for profiling a test execution."""
        self.start_profiling(test_id)
        try:
            yield self
        finally:
            self.stop_profiling()
    
    def start_profiling(self, test_id: str) -> None:
        """Start performance profiling."""
        self.current_profile = {
            'test_id': test_id,
            'start_time': time.perf_counter(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': psutil.cpu_percent(),
            'profiler': cProfile.Profile()
        }
        
        # Start CPU profiling
        self.current_profile['profiler'].enable()
        
        # Start system monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        
        # Force garbage collection
        gc.collect()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        if not self.current_profile:
            return {}
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        
        # Stop CPU profiling
        self.current_profile['profiler'].disable()
        
        # Calculate metrics
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        duration = end_time - self.current_profile['start_time']
        memory_delta = end_memory - self.current_profile['start_memory']
        
        # Get profiling stats
        stats = pstats.Stats(self.current_profile['profiler'])
        stats.sort_stats('cumulative')
        
        # Collect system metrics
        cpu_usage = self._calculate_cpu_metrics()
        memory_usage = self._calculate_memory_metrics()
        
        profile_result = {
            'test_id': self.current_profile['test_id'],
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta / (1024 * 1024),
            'cpu_metrics': cpu_usage,
            'memory_metrics': memory_usage,
            'function_stats': self._extract_function_stats(stats),
            'system_snapshots': {
                'cpu_snapshots': list(self.cpu_snapshots),
                'memory_snapshots': list(self.memory_snapshots)
            }
        }
        
        self.current_profile = None
        return profile_result
    
    def _monitor_system(self) -> None:
        """Monitor system resources during test execution."""
        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.cpu_snapshots.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'load_avg': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                })
                
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                self.memory_snapshots.append({
                    'timestamp': time.time(),
                    'memory_percent': memory_info.percent,
                    'memory_used_gb': memory_info.used / (1024**3),
                    'memory_available_gb': memory_info.available / (1024**3),
                    'traced_current_mb': current_memory / (1024**2),
                    'traced_peak_mb': peak_memory / (1024**2)
                })
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                print(f"Error monitoring system: {e}")
                break
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _calculate_cpu_metrics(self) -> Dict[str, float]:
        """Calculate CPU usage statistics."""
        if not self.cpu_snapshots:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.cpu_snapshots]
        return {
            'avg': statistics.mean(cpu_values),
            'max': max(cpu_values),
            'min': min(cpu_values),
            'median': statistics.median(cpu_values),
            'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        }
    
    def _calculate_memory_metrics(self) -> Dict[str, float]:
        """Calculate memory usage statistics."""
        if not self.memory_snapshots:
            return {}
        
        memory_values = [s['memory_percent'] for s in self.memory_snapshots]
        traced_values = [s['traced_current_mb'] for s in self.memory_snapshots]
        
        return {
            'system_memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'median': statistics.median(memory_values)
            },
            'traced_memory': {
                'avg': statistics.mean(traced_values),
                'max': max(traced_values),
                'min': min(traced_values),
                'median': statistics.median(traced_values)
            }
        }
    
    def _extract_function_stats(self, stats: pstats.Stats) -> List[Dict[str, Any]]:
        """Extract top function statistics from profiler."""
        function_stats = []
        
        # Get top 20 functions by cumulative time
        stats_data = stats.get_stats_profile()
        sorted_stats = sorted(
            stats_data.func_profiles.items(),
            key=lambda x: x[1].cumtime,
            reverse=True
        )
        
        for (filename, line, function), prof in sorted_stats[:20]:
            function_stats.append({
                'function': function,
                'filename': filename,
                'line': line,
                'calls': prof.ncalls,
                'total_time': prof.tottime,
                'cumulative_time': prof.cumtime,
                'time_per_call': prof.tottime / prof.ncalls if prof.ncalls > 0 else 0
            })
        
        return function_stats


class BenchmarkRunner:
    """Runs performance benchmarks with statistical analysis."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = []
    
    def run_benchmark(self, benchmark: PerformanceBenchmark) -> PerformanceMetrics:
        """Run a single benchmark with multiple iterations."""
        print(f"🏃 Running benchmark: {benchmark.name}")
        
        test_id = f"benchmark_{benchmark.name}_{int(time.time())}"
        
        # Warmup iterations
        print(f"🔥 Warming up with {benchmark.warmup_iterations} iterations...")
        for _ in range(benchmark.warmup_iterations):
            try:
                benchmark.function(**benchmark.parameters)
            except Exception as e:
                print(f"Warmup iteration failed: {e}")
        
        # Collect metrics during actual benchmark
        execution_times = []
        memory_deltas = []
        
        with self.profiler.profile_context(test_id):
            start_time = time.perf_counter()
            
            for i in range(benchmark.iterations):
                iteration_start = time.perf_counter()
                memory_before = self.profiler._get_memory_usage()
                
                try:
                    result = benchmark.function(**benchmark.parameters)
                    
                    iteration_end = time.perf_counter()
                    memory_after = self.profiler._get_memory_usage()
                    
                    execution_time = (iteration_end - iteration_start) * 1000  # milliseconds
                    memory_delta = memory_after - memory_before
                    
                    execution_times.append(execution_time)
                    memory_deltas.append(memory_delta)
                    
                    if i % max(1, benchmark.iterations // 10) == 0:
                        progress = (i + 1) / benchmark.iterations * 100
                        print(f"Progress: {progress:.1f}% - Last iteration: {execution_time:.2f}ms")
                
                except Exception as e:
                    print(f"Benchmark iteration {i} failed: {e}")
                    execution_times.append(float('inf'))
                    memory_deltas.append(0)
            
            total_duration = time.perf_counter() - start_time
        
        # Calculate statistics
        valid_times = [t for t in execution_times if t != float('inf')]
        failed_iterations = len(execution_times) - len(valid_times)
        
        if not valid_times:
            raise RuntimeError(f"All benchmark iterations failed for {benchmark.name}")
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            test_id=test_id,
            test_type=PerformanceTestType.BENCHMARK,
            timestamp=datetime.now(),
            duration_seconds=total_duration,
            
            # Response time metrics
            response_time_min=min(valid_times),
            response_time_max=max(valid_times),
            response_time_avg=statistics.mean(valid_times),
            response_time_p50=np.percentile(valid_times, 50),
            response_time_p90=np.percentile(valid_times, 90),
            response_time_p95=np.percentile(valid_times, 95),
            response_time_p99=np.percentile(valid_times, 99),
            
            # Throughput metrics
            requests_per_second=len(valid_times) / total_duration,
            successful_requests=len(valid_times),
            failed_requests=failed_iterations,
            error_rate=failed_iterations / benchmark.iterations,
            
            # Resource metrics (approximated)
            cpu_usage_avg=0.0,  # Will be filled from profiler
            cpu_usage_max=0.0,
            memory_usage_avg=statistics.mean(memory_deltas) / (1024 * 1024) if memory_deltas else 0,
            memory_usage_max=max(memory_deltas) / (1024 * 1024) if memory_deltas else 0,
            memory_allocated_mb=sum(memory_deltas) / (1024 * 1024) if memory_deltas else 0,
            disk_io_read=0.0,
            disk_io_write=0.0,
            network_io_sent=0.0,
            network_io_recv=0.0,
            
            custom_metrics={
                'iterations': benchmark.iterations,
                'warmup_iterations': benchmark.warmup_iterations,
                'std_deviation': statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                'coefficient_of_variation': (statistics.stdev(valid_times) / statistics.mean(valid_times)) if len(valid_times) > 1 else 0
            }
        )
        
        self.results.append(metrics)
        
        print(f"✅ Benchmark completed: {benchmark.name}")
        print(f"   Average: {metrics.response_time_avg:.2f}ms")
        print(f"   P95: {metrics.response_time_p95:.2f}ms")
        print(f"   Throughput: {metrics.requests_per_second:.2f} ops/sec")
        
        return metrics
    
    def run_benchmark_suite(self, benchmarks: List[PerformanceBenchmark]) -> List[PerformanceMetrics]:
        """Run a suite of benchmarks."""
        results = []
        
        for benchmark in benchmarks:
            try:
                result = self.run_benchmark(benchmark)
                results.append(result)
            except Exception as e:
                print(f"❌ Benchmark {benchmark.name} failed: {e}")
                continue
        
        return results


class LoadTestRunner:
    """Runs load tests with realistic traffic patterns."""
    
    def __init__(self):
        self.results = []
    
    async def run_load_test(self, scenario: LoadTestScenario, 
                           base_url: str = "http://localhost:8000") -> PerformanceMetrics:
        """Run a load test scenario using async HTTP clients."""
        if not ASYNC_HTTP_AVAILABLE:
            print("❌ Async HTTP libraries not available for load testing")
            return None
        
        print(f"🚀 Running load test: {scenario.name}")
        
        test_id = f"load_test_{scenario.name}_{int(time.time())}"
        
        # Collect metrics
        response_times = []
        successful_requests = 0
        failed_requests = 0
        start_time = time.perf_counter()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(scenario.users)
        
        async def make_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
            """Make a single HTTP request."""
            async with semaphore:
                request_start = time.perf_counter()
                
                try:
                    url = f"{base_url.rstrip('/')}/{scenario.endpoint.lstrip('/')}"
                    
                    async with session.request(
                        method=scenario.method,
                        url=url,
                        json=scenario.payload,
                        headers=scenario.headers or {}
                    ) as response:
                        await response.text()  # Read response body
                        
                        request_end = time.perf_counter()
                        response_time = (request_end - request_start) * 1000  # milliseconds
                        
                        return {
                            'request_id': request_id,
                            'status': response.status,
                            'response_time': response_time,
                            'success': 200 <= response.status < 400
                        }
                
                except Exception as e:
                    request_end = time.perf_counter()
                    response_time = (request_end - request_start) * 1000
                    
                    return {
                        'request_id': request_id,
                        'status': 0,
                        'response_time': response_time,
                        'success': False,
                        'error': str(e)
                    }
        
        async def run_requests():
            """Run all requests for the load test."""
            nonlocal successful_requests, failed_requests, response_times
            
            connector = aiohttp.TCPConnector(limit=scenario.users)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = []
                request_count = 0
                
                # Generate requests based on duration and spawn rate
                requests_per_interval = max(1, int(scenario.spawn_rate))
                intervals = int(scenario.duration_seconds * scenario.spawn_rate / requests_per_interval)
                
                for interval in range(intervals):
                    # Create batch of requests
                    batch_tasks = []
                    for _ in range(requests_per_interval):
                        task = make_request(session, request_count)
                        batch_tasks.append(task)
                        request_count += 1
                    
                    # Start batch
                    tasks.extend(batch_tasks)
                    
                    # Wait for spawn rate interval
                    if interval < intervals - 1:  # Don't wait after last batch
                        await asyncio.sleep(1.0 / scenario.spawn_rate)
                
                # Wait for all requests to complete
                print(f"🔄 Executing {len(tasks)} requests...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, dict):
                        if result['success']:
                            successful_requests += 1
                            response_times.append(result['response_time'])
                        else:
                            failed_requests += 1
                            if result.get('error'):
                                print(f"Request failed: {result['error']}")
                    else:
                        failed_requests += 1
                        print(f"Request exception: {result}")
        
        # Run the load test
        await run_requests()
        
        total_duration = time.perf_counter() - start_time
        total_requests = successful_requests + failed_requests
        
        if not response_times:
            print("❌ No successful requests in load test")
            return None
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            test_id=test_id,
            test_type=PerformanceTestType.LOAD,
            timestamp=datetime.now(),
            duration_seconds=total_duration,
            
            # Response time metrics
            response_time_min=min(response_times),
            response_time_max=max(response_times),
            response_time_avg=statistics.mean(response_times),
            response_time_p50=np.percentile(response_times, 50),
            response_time_p90=np.percentile(response_times, 90),
            response_time_p95=np.percentile(response_times, 95),
            response_time_p99=np.percentile(response_times, 99),
            
            # Throughput metrics
            requests_per_second=successful_requests / total_duration,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            
            # Resource metrics (basic)
            cpu_usage_avg=psutil.cpu_percent(),
            cpu_usage_max=psutil.cpu_percent(),
            memory_usage_avg=psutil.virtual_memory().percent,
            memory_usage_max=psutil.virtual_memory().percent,
            memory_allocated_mb=0.0,
            disk_io_read=0.0,
            disk_io_write=0.0,
            network_io_sent=0.0,
            network_io_recv=0.0,
            
            custom_metrics={
                'target_users': scenario.users,
                'spawn_rate': scenario.spawn_rate,
                'duration_seconds': scenario.duration_seconds,
                'total_requests': total_requests
            }
        )
        
        self.results.append(metrics)
        
        print(f"✅ Load test completed: {scenario.name}")
        print(f"   Requests: {successful_requests} success, {failed_requests} failed")
        print(f"   Average response time: {metrics.response_time_avg:.2f}ms")
        print(f"   P95 response time: {metrics.response_time_p95:.2f}ms")
        print(f"   Throughput: {metrics.requests_per_second:.2f} req/sec")
        print(f"   Error rate: {metrics.error_rate:.2%}")
        
        return metrics


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize performance history database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration_seconds REAL,
                response_time_avg REAL,
                response_time_p95 REAL,
                requests_per_second REAL,
                error_rate REAL,
                cpu_usage_avg REAL,
                memory_usage_avg REAL,
                custom_metrics TEXT,
                baseline_version TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT UNIQUE NOT NULL,
                baseline_metrics TEXT NOT NULL,
                created_date TEXT NOT NULL,
                updated_date TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_metrics(self, metrics: PerformanceMetrics, baseline_version: str = None) -> None:
        """Store performance metrics in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_history 
            (test_id, test_type, timestamp, duration_seconds, response_time_avg, response_time_p95,
             requests_per_second, error_rate, cpu_usage_avg, memory_usage_avg, custom_metrics, baseline_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.test_id,
            metrics.test_type,
            metrics.timestamp.isoformat(),
            metrics.duration_seconds,
            metrics.response_time_avg,
            metrics.response_time_p95,
            metrics.requests_per_second,
            metrics.error_rate,
            metrics.cpu_usage_avg,
            metrics.memory_usage_avg,
            json.dumps(metrics.custom_metrics),
            baseline_version
        ))
        
        conn.commit()
        conn.close()
    
    def set_baseline(self, test_name: str, metrics: PerformanceMetrics) -> None:
        """Set performance baseline for a test."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        baseline_data = json.dumps(asdict(metrics))
        current_time = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO performance_baselines 
            (test_name, baseline_metrics, created_date, updated_date)
            VALUES (?, ?, COALESCE((SELECT created_date FROM performance_baselines WHERE test_name = ?), ?), ?)
        """, (test_name, baseline_data, test_name, current_time, current_time))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Baseline set for test: {test_name}")
    
    def detect_regression(self, test_name: str, current_metrics: PerformanceMetrics,
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Detect performance regression against baseline."""
        # Get baseline metrics
        baseline_metrics = self._get_baseline(test_name)
        if not baseline_metrics:
            print(f"⚠️ No baseline found for test: {test_name}")
            return {"status": "no_baseline", "message": "No baseline available for comparison"}
        
        # Compare key metrics
        regressions = []
        improvements = []
        
        # Response time regression (higher is worse)
        if current_metrics.response_time_avg > baseline_metrics.response_time_avg * (1 + threshold):
            regression_pct = ((current_metrics.response_time_avg / baseline_metrics.response_time_avg) - 1) * 100
            regressions.append({
                "metric": "response_time_avg",
                "baseline": baseline_metrics.response_time_avg,
                "current": current_metrics.response_time_avg,
                "regression_percent": regression_pct,
                "severity": "high" if regression_pct > 50 else "medium" if regression_pct > 20 else "low"
            })
        elif current_metrics.response_time_avg < baseline_metrics.response_time_avg * (1 - threshold):
            improvement_pct = ((baseline_metrics.response_time_avg / current_metrics.response_time_avg) - 1) * 100
            improvements.append({
                "metric": "response_time_avg",
                "improvement_percent": improvement_pct
            })
        
        # P95 response time regression
        if current_metrics.response_time_p95 > baseline_metrics.response_time_p95 * (1 + threshold):
            regression_pct = ((current_metrics.response_time_p95 / baseline_metrics.response_time_p95) - 1) * 100
            regressions.append({
                "metric": "response_time_p95",
                "baseline": baseline_metrics.response_time_p95,
                "current": current_metrics.response_time_p95,
                "regression_percent": regression_pct,
                "severity": "high" if regression_pct > 50 else "medium" if regression_pct > 20 else "low"
            })
        
        # Throughput regression (lower is worse)
        if current_metrics.requests_per_second < baseline_metrics.requests_per_second * (1 - threshold):
            regression_pct = ((baseline_metrics.requests_per_second / current_metrics.requests_per_second) - 1) * 100
            regressions.append({
                "metric": "requests_per_second",
                "baseline": baseline_metrics.requests_per_second,
                "current": current_metrics.requests_per_second,
                "regression_percent": regression_pct,
                "severity": "high" if regression_pct > 50 else "medium" if regression_pct > 20 else "low"
            })
        elif current_metrics.requests_per_second > baseline_metrics.requests_per_second * (1 + threshold):
            improvement_pct = ((current_metrics.requests_per_second / baseline_metrics.requests_per_second) - 1) * 100
            improvements.append({
                "metric": "requests_per_second",
                "improvement_percent": improvement_pct
            })
        
        # Error rate regression (higher is worse)
        if current_metrics.error_rate > baseline_metrics.error_rate + threshold:
            regression_pct = ((current_metrics.error_rate - baseline_metrics.error_rate) / max(baseline_metrics.error_rate, 0.01)) * 100
            regressions.append({
                "metric": "error_rate",
                "baseline": baseline_metrics.error_rate,
                "current": current_metrics.error_rate,
                "regression_percent": regression_pct,
                "severity": "critical" if current_metrics.error_rate > 0.1 else "high" if current_metrics.error_rate > 0.05 else "medium"
            })
        
        # Determine overall status
        if regressions:
            high_severity = any(r["severity"] in ["critical", "high"] for r in regressions)
            status = "regression_detected"
            severity = "high" if high_severity else "low"
        else:
            status = "no_regression"
            severity = "none"
        
        result = {
            "status": status,
            "severity": severity,
            "regressions": regressions,
            "improvements": improvements,
            "baseline_timestamp": baseline_metrics.timestamp.isoformat(),
            "current_timestamp": current_metrics.timestamp.isoformat(),
            "comparison_summary": {
                "response_time_change": ((current_metrics.response_time_avg / baseline_metrics.response_time_avg) - 1) * 100,
                "throughput_change": ((current_metrics.requests_per_second / baseline_metrics.requests_per_second) - 1) * 100,
                "error_rate_change": current_metrics.error_rate - baseline_metrics.error_rate
            }
        }
        
        # Print results
        if regressions:
            print(f"🚨 Performance regression detected for {test_name}:")
            for regression in regressions:
                print(f"   • {regression['metric']}: {regression['regression_percent']:.1f}% worse ({regression['severity']} severity)")
        else:
            print(f"✅ No performance regression detected for {test_name}")
        
        if improvements:
            print(f"🚀 Performance improvements detected:")
            for improvement in improvements:
                print(f"   • {improvement['metric']}: {improvement['improvement_percent']:.1f}% better")
        
        return result
    
    def _get_baseline(self, test_name: str) -> Optional[PerformanceMetrics]:
        """Get baseline metrics for a test."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT baseline_metrics FROM performance_baselines WHERE test_name = ?
        """, (test_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            baseline_data = json.loads(result[0])
            # Convert timestamp string back to datetime
            baseline_data['timestamp'] = datetime.fromisoformat(baseline_data['timestamp'])
            return PerformanceMetrics(**baseline_data)
        
        return None
    
    def get_performance_trends(self, test_name_pattern: str = "%", days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT test_id, timestamp, response_time_avg, response_time_p95, 
                   requests_per_second, error_rate, cpu_usage_avg, memory_usage_avg
            FROM performance_history 
            WHERE test_id LIKE ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days), (test_name_pattern,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {"message": "No performance data found"}
        
        # Organize data by test
        test_data = defaultdict(list)
        
        for row in results:
            test_id = row[0].split('_')[0] if '_' in row[0] else row[0]  # Extract test name
            test_data[test_id].append({
                'timestamp': row[1],
                'response_time_avg': row[2],
                'response_time_p95': row[3],
                'requests_per_second': row[4],
                'error_rate': row[5],
                'cpu_usage_avg': row[6],
                'memory_usage_avg': row[7]
            })
        
        # Calculate trends
        trends = {}
        for test_name, data in test_data.items():
            if len(data) < 2:
                continue
            
            # Calculate linear trends
            response_times = [d['response_time_avg'] for d in data]
            throughputs = [d['requests_per_second'] for d in data]
            
            trends[test_name] = {
                'data_points': len(data),
                'date_range': {
                    'start': data[0]['timestamp'],
                    'end': data[-1]['timestamp']
                },
                'response_time_trend': self._calculate_trend(response_times),
                'throughput_trend': self._calculate_trend(throughputs),
                'latest_metrics': data[-1],
                'baseline_metrics': data[0]
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {'direction': 'stable', 'change_percent': 0.0}
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if first_avg == 0:
            return {'direction': 'stable', 'change_percent': 0.0}
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if abs(change_percent) < 5:
            direction = 'stable'
        elif change_percent > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change_percent': abs(change_percent),
            'raw_change_percent': change_percent
        }


class PerformanceTestSuite:
    """Main performance testing suite with comprehensive analysis."""
    
    def __init__(self):
        self.benchmark_runner = BenchmarkRunner()
        self.load_test_runner = LoadTestRunner()
        self.regression_detector = PerformanceRegressionDetector()
        self.results = []
    
    def create_ml_benchmarks(self) -> List[PerformanceBenchmark]:
        """Create performance benchmarks for ML operations."""
        benchmarks = []
        
        # Data preprocessing benchmark
        benchmarks.append(PerformanceBenchmark(
            name="data_preprocessing",
            function=self._benchmark_data_preprocessing,
            iterations=100,
            warmup_iterations=10,
            timeout_seconds=30,
            parameters={"data_size": 10000}
        ))
        
        # Model training benchmark
        benchmarks.append(PerformanceBenchmark(
            name="model_training",
            function=self._benchmark_model_training,
            iterations=10,
            warmup_iterations=2,
            timeout_seconds=120,
            parameters={"data_size": 1000}
        ))
        
        # Model prediction benchmark
        benchmarks.append(PerformanceBenchmark(
            name="model_prediction",
            function=self._benchmark_model_prediction,
            iterations=1000,
            warmup_iterations=50,
            timeout_seconds=60,
            parameters={"batch_size": 100}
        ))
        
        # High-performance optimization benchmark
        benchmarks.append(PerformanceBenchmark(
            name="optimization_pipeline",
            function=self._benchmark_optimization_pipeline,
            iterations=50,
            warmup_iterations=5,
            timeout_seconds=90,
            parameters={"complexity_level": "medium"}
        ))
        
        return benchmarks
    
    def create_api_load_tests(self) -> List[LoadTestScenario]:
        """Create load test scenarios for API endpoints."""
        scenarios = []
        
        # Health check endpoint
        scenarios.append(LoadTestScenario(
            name="health_check",
            endpoint="/health",
            method="GET",
            users=50,
            spawn_rate=5.0,
            duration_seconds=30,
            max_response_time=100.0,
            max_error_rate=0.01,
            min_throughput=40.0
        ))
        
        # Prediction endpoint
        scenarios.append(LoadTestScenario(
            name="prediction_api",
            endpoint="/predict",
            method="POST",
            payload={"features": [0.5, 1.2, 0.8, 2.1, 0.3]},
            headers={"Content-Type": "application/json"},
            users=20,
            spawn_rate=2.0,
            duration_seconds=60,
            max_response_time=500.0,
            max_error_rate=0.05,
            min_throughput=15.0
        ))
        
        # Batch prediction endpoint
        scenarios.append(LoadTestScenario(
            name="batch_prediction",
            endpoint="/predict/batch",
            method="POST",
            payload={"features": [[0.5, 1.2], [0.8, 2.1], [0.3, 1.7]]},
            headers={"Content-Type": "application/json"},
            users=10,
            spawn_rate=1.0,
            duration_seconds=45,
            max_response_time=1000.0,
            max_error_rate=0.03,
            min_throughput=8.0
        ))
        
        return scenarios
    
    async def run_complete_performance_suite(self, set_baselines: bool = False) -> Dict[str, Any]:
        """Run complete performance test suite."""
        print("🏁 Starting comprehensive performance test suite...")
        
        suite_start = time.perf_counter()
        results = {
            "suite_id": f"perf_suite_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "benchmarks": [],
            "load_tests": [],
            "regressions": [],
            "summary": {}
        }
        
        # Run benchmarks
        print("\n🏃 Running performance benchmarks...")
        benchmarks = self.create_ml_benchmarks()
        for benchmark in benchmarks:
            try:
                metrics = self.benchmark_runner.run_benchmark(benchmark)
                results["benchmarks"].append(asdict(metrics))
                
                # Store metrics and check for regressions
                self.regression_detector.store_metrics(metrics)
                
                if set_baselines:
                    self.regression_detector.set_baseline(benchmark.name, metrics)
                else:
                    regression_result = self.regression_detector.detect_regression(benchmark.name, metrics)
                    results["regressions"].append({
                        "test_name": benchmark.name,
                        "regression_analysis": regression_result
                    })
                
            except Exception as e:
                print(f"❌ Benchmark {benchmark.name} failed: {e}")
                continue
        
        # Run load tests
        print("\n🚀 Running load tests...")
        load_scenarios = self.create_api_load_tests()
        for scenario in load_scenarios:
            try:
                metrics = await self.load_test_runner.run_load_test(scenario)
                if metrics:
                    results["load_tests"].append(asdict(metrics))
                    
                    # Store metrics and check for regressions
                    self.regression_detector.store_metrics(metrics)
                    
                    if set_baselines:
                        self.regression_detector.set_baseline(scenario.name, metrics)
                    else:
                        regression_result = self.regression_detector.detect_regression(scenario.name, metrics)
                        results["regressions"].append({
                            "test_name": scenario.name,
                            "regression_analysis": regression_result
                        })
                
            except Exception as e:
                print(f"❌ Load test {scenario.name} failed: {e}")
                continue
        
        suite_duration = time.perf_counter() - suite_start
        
        # Generate summary
        benchmark_count = len(results["benchmarks"])
        load_test_count = len(results["load_tests"])
        regression_count = sum(1 for r in results["regressions"] 
                             if r["regression_analysis"]["status"] == "regression_detected")
        
        results["summary"] = {
            "total_duration_seconds": suite_duration,
            "total_tests": benchmark_count + load_test_count,
            "benchmarks_run": benchmark_count,
            "load_tests_run": load_test_count,
            "regressions_detected": regression_count,
            "baselines_set": set_baselines
        }
        
        results["end_time"] = datetime.now().isoformat()
        
        print(f"\n🎯 Performance test suite completed in {suite_duration:.2f}s")
        print(f"   Tests run: {results['summary']['total_tests']}")
        print(f"   Regressions detected: {regression_count}")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _benchmark_data_preprocessing(self, data_size: int) -> Any:
        """Benchmark data preprocessing operations."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Generate synthetic data
        data = pd.DataFrame({
            'feature1': np.random.randn(data_size),
            'feature2': np.random.randn(data_size),
            'category': np.random.choice(['A', 'B', 'C'], data_size),
            'target': np.random.randint(0, 2, data_size)
        })
        
        # Preprocessing operations
        scaler = StandardScaler()
        encoder = LabelEncoder()
        
        data['feature1_scaled'] = scaler.fit_transform(data[['feature1']])
        data['category_encoded'] = encoder.fit_transform(data['category'])
        
        return len(data)
    
    def _benchmark_model_training(self, data_size: int) -> Any:
        """Benchmark model training operations."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate synthetic data
        X = np.random.randn(data_size, 10)
        y = np.random.randint(0, 2, data_size)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model.score(X_test, y_test)
    
    def _benchmark_model_prediction(self, batch_size: int) -> Any:
        """Benchmark model prediction operations."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create pre-trained model
        X_train = np.random.randn(1000, 10)
        y_train = np.random.randint(0, 2, 1000)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Generate prediction data
        X_pred = np.random.randn(batch_size, 10)
        
        # Make predictions
        predictions = model.predict(X_pred)
        probabilities = model.predict_proba(X_pred)
        
        return len(predictions)
    
    def _benchmark_optimization_pipeline(self, complexity_level: str) -> Any:
        """Benchmark optimization pipeline operations."""
        # Simulate different complexity levels
        if complexity_level == "low":
            iterations = 1000
            operations_per_iter = 100
        elif complexity_level == "medium":
            iterations = 5000
            operations_per_iter = 500
        else:  # high
            iterations = 10000
            operations_per_iter = 1000
        
        # Simulate computational work
        total_ops = 0
        for i in range(iterations):
            # Matrix operations
            matrix = np.random.randn(10, 10)
            result = np.dot(matrix, matrix.T)
            total_ops += operations_per_iter
            
            # Some optimization logic
            if i % 1000 == 0:
                gc.collect()  # Simulate memory optimization
        
        return total_ops
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save performance test results."""
        results_dir = Path("performance_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / f"performance_suite_{results['suite_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved to {results_file}")


async def main():
    """Main function for running performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Testing Suite")
    parser.add_argument("--test-type", choices=["benchmark", "load", "all"], default="all",
                       help="Type of performance test to run")
    parser.add_argument("--set-baselines", action="store_true",
                       help="Set new performance baselines instead of checking regressions")
    parser.add_argument("--trends", type=int, help="Show performance trends for N days")
    
    args = parser.parse_args()
    
    suite = PerformanceTestSuite()
    
    if args.trends:
        trends = suite.regression_detector.get_performance_trends(days=args.trends)
        print("📈 Performance Trends:")
        print(json.dumps(trends, indent=2, default=str))
        return
    
    # Run performance tests
    if args.test_type in ["benchmark", "all"]:
        print("Running benchmark tests...")
        benchmarks = suite.create_ml_benchmarks()
        for benchmark in benchmarks:
            metrics = suite.benchmark_runner.run_benchmark(benchmark)
            if args.set_baselines:
                suite.regression_detector.set_baseline(benchmark.name, metrics)
            else:
                suite.regression_detector.detect_regression(benchmark.name, metrics)
    
    if args.test_type in ["load", "all"]:
        print("Running load tests...")
        scenarios = suite.create_api_load_tests()
        for scenario in scenarios:
            metrics = await suite.load_test_runner.run_load_test(scenario)
            if metrics:
                if args.set_baselines:
                    suite.regression_detector.set_baseline(scenario.name, metrics)
                else:
                    suite.regression_detector.detect_regression(scenario.name, metrics)
    
    if args.test_type == "all":
        print("Running complete performance suite...")
        results = await suite.run_complete_performance_suite(args.set_baselines)
        print("🎯 Complete performance test suite finished!")


if __name__ == "__main__":
    asyncio.run(main())