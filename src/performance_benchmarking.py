"""
Performance Benchmarking and Profiling System.

This module provides comprehensive performance analysis and benchmarking capabilities including:
- Detailed performance profiling with CPU, memory, and I/O analysis
- Automated benchmarking suites with statistical analysis
- A/B testing framework for performance comparisons
- Real-time performance monitoring and alerting
- Bottleneck detection and optimization recommendations
- Load testing and stress testing capabilities
- Performance regression detection
- Resource utilization analysis and optimization suggestions
"""

import os
import json
import time
import psutil
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile
import pstats
import io
import tracemalloc
import gc
import sys
import importlib
import pkgutil
import requests
from contextlib import contextmanager
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .error_handling_recovery import with_error_handling, error_handler

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    test_name: str
    timestamp: datetime
    duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_count: int
    success_count: int
    success_rate: float
    gc_collections: Dict[str, int]
    custom_metrics: Dict[str, float]


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark_name: str
    test_case: str
    metrics: PerformanceMetrics
    statistical_summary: Dict[str, float]
    comparison_baseline: Optional[str] = None
    regression_detected: bool = False
    optimization_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class ProfileResult:
    """Result from code profiling."""
    function_name: str
    total_time: float
    cum_time: float
    call_count: int
    time_per_call: float
    filename: str
    line_number: int
    hotspots: List[Dict[str, Any]]
    memory_profile: Optional[Dict[str, Any]] = None


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int = 10
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    target_rps: Optional[float] = None
    max_response_time_ms: float = 5000
    success_threshold: float = 0.95
    custom_scenarios: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.custom_scenarios is None:
            self.custom_scenarios = []


class SystemResourceMonitor:
    """Monitor system resources during performance tests."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_history = deque(maxlen=10000)
        self.thread = None
        self.process = psutil.Process()
        
        # Track initial state
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu_times = self.process.cpu_times()
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)
        
        return self._calculate_summary()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current resource metrics."""
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            cpu_times = self.process.cpu_times()
            
            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # I/O metrics
            try:
                io_counters = self.process.io_counters()
                disk_read = io_counters.read_bytes
                disk_write = io_counters.write_bytes
            except (AttributeError, OSError):
                disk_read = disk_write = 0
            
            # Network metrics (system-wide)
            try:
                net_io = psutil.net_io_counters()
                network_sent = net_io.bytes_sent
                network_recv = net_io.bytes_recv
            except AttributeError:
                network_sent = network_recv = 0
            
            return {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'memory_percent': memory_percent,
                'disk_read_bytes': disk_read,
                'disk_write_bytes': disk_write,
                'network_sent_bytes': network_sent,
                'network_recv_bytes': network_recv,
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(list(self.metrics_history))
        
        if df.empty:
            return {}
        
        summary = {}
        
        # CPU metrics
        summary['cpu_usage_avg'] = df['cpu_percent'].mean()
        summary['cpu_usage_max'] = df['cpu_percent'].max()
        summary['cpu_usage_p95'] = df['cpu_percent'].quantile(0.95)
        
        # Memory metrics
        summary['memory_avg_mb'] = df['memory_rss_mb'].mean()
        summary['memory_peak_mb'] = df['memory_rss_mb'].max()
        summary['memory_growth_mb'] = df['memory_rss_mb'].iloc[-1] - df['memory_rss_mb'].iloc[0]
        
        # I/O metrics
        if 'disk_read_bytes' in df.columns:
            summary['disk_read_mb'] = (df['disk_read_bytes'].iloc[-1] - df['disk_read_bytes'].iloc[0]) / (1024 * 1024)
            summary['disk_write_mb'] = (df['disk_write_bytes'].iloc[-1] - df['disk_write_bytes'].iloc[0]) / (1024 * 1024)
        
        if 'network_sent_bytes' in df.columns:
            summary['network_sent_mb'] = (df['network_sent_bytes'].iloc[-1] - df['network_sent_bytes'].iloc[0]) / (1024 * 1024)
            summary['network_recv_mb'] = (df['network_recv_bytes'].iloc[-1] - df['network_recv_bytes'].iloc[0]) / (1024 * 1024)
        
        # Thread and FD metrics
        summary['threads_avg'] = df['num_threads'].mean()
        summary['threads_max'] = df['num_threads'].max()
        
        return summary


class CodeProfiler:
    """Advanced code profiling with CPU and memory analysis."""
    
    def __init__(self):
        self.profilers = {}
        self.memory_trackers = {}
        
    @contextmanager
    def profile(self, name: str, include_memory: bool = True):
        """Context manager for profiling code blocks."""
        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Start memory tracking
        if include_memory:
            tracemalloc.start()
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # Stop CPU profiling
            profiler.disable()
            
            # Process CPU profile
            cpu_stats = self._process_cpu_profile(profiler)
            
            # Process memory profile
            memory_stats = None
            if include_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_stats = {
                    'current_mb': current / (1024 * 1024),
                    'peak_mb': peak / (1024 * 1024)
                }
            
            # Store results
            self.profilers[name] = {
                'duration': end_time - start_time,
                'cpu_stats': cpu_stats,
                'memory_stats': memory_stats,
                'timestamp': datetime.now()
            }
    
    def _process_cpu_profile(self, profiler: cProfile.Profile) -> List[ProfileResult]:
        """Process CPU profiling results."""
        # Create string buffer for stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        
        # Get top functions
        results = []
        for func_info, (call_count, total_time, cum_time, callers) in stats.stats.items():
            filename, line_number, function_name = func_info
            
            if call_count > 0 and cum_time > 0.001:  # Filter small functions
                result = ProfileResult(
                    function_name=function_name,
                    total_time=total_time,
                    cum_time=cum_time,
                    call_count=call_count,
                    time_per_call=total_time / call_count,
                    filename=filename,
                    line_number=line_number,
                    hotspots=[]  # Would be populated with line-level analysis
                )
                results.append(result)
        
        # Sort by cumulative time and return top 20
        results.sort(key=lambda x: x.cum_time, reverse=True)
        return results[:20]
    
    def get_profile_results(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profiling results for a named profile."""
        return self.profilers.get(name)
    
    def generate_profile_report(self, name: str, output_file: Optional[str] = None) -> str:
        """Generate detailed profiling report."""
        if name not in self.profilers:
            return f"No profiling data found for '{name}'"
        
        data = self.profilers[name]
        report = []
        
        report.append(f"Performance Profile Report: {name}")
        report.append("=" * 50)
        report.append(f"Total Duration: {data['duration']:.4f} seconds")
        report.append(f"Timestamp: {data['timestamp']}")
        report.append("")
        
        if data['memory_stats']:
            report.append("Memory Usage:")
            report.append(f"  Current: {data['memory_stats']['current_mb']:.2f} MB")
            report.append(f"  Peak: {data['memory_stats']['peak_mb']:.2f} MB")
            report.append("")
        
        report.append("Top CPU-consuming functions:")
        report.append("")
        
        for i, result in enumerate(data['cpu_stats'][:10], 1):
            report.append(f"{i:2d}. {result.function_name}")
            report.append(f"    File: {os.path.basename(result.filename)}:{result.line_number}")
            report.append(f"    Calls: {result.call_count}")
            report.append(f"    Total Time: {result.total_time:.4f}s")
            report.append(f"    Cumulative Time: {result.cum_time:.4f}s")
            report.append(f"    Time per Call: {result.time_per_call:.6f}s")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""
    
    def __init__(self):
        self.benchmarks = {}
        self.baseline_results = {}
        self.profiler = CodeProfiler()
        self.results_history = []
        
    def register_benchmark(self, name: str, func: Callable, 
                         setup_func: Optional[Callable] = None,
                         teardown_func: Optional[Callable] = None) -> None:
        """Register a benchmark function."""
        self.benchmarks[name] = {
            'func': func,
            'setup': setup_func,
            'teardown': teardown_func
        }
    
    def run_benchmark(self, name: str, iterations: int = 10, 
                     warmup_iterations: int = 2,
                     profile: bool = True) -> BenchmarkResult:
        """Run a single benchmark with statistical analysis."""
        
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not registered")
        
        benchmark = self.benchmarks[name]
        
        # Setup
        if benchmark['setup']:
            benchmark['setup']()
        
        try:
            # Warmup runs
            logger.info(f"Running {warmup_iterations} warmup iterations for {name}")
            for _ in range(warmup_iterations):
                benchmark['func']()
            
            # Actual benchmark runs
            logger.info(f"Running {iterations} benchmark iterations for {name}")
            
            durations = []
            resource_summaries = []
            latencies = []
            
            for i in range(iterations):
                # Start resource monitoring
                monitor = SystemResourceMonitor(sampling_interval=0.01)
                monitor.start_monitoring()
                
                # Run with profiling on first iteration
                if profile and i == 0:
                    with self.profiler.profile(f"{name}_detailed"):
                        start_time = time.time()
                        result = benchmark['func']()
                        end_time = time.time()
                else:
                    start_time = time.time()
                    result = benchmark['func']()
                    end_time = time.time()
                
                duration = end_time - start_time
                durations.append(duration)
                
                # Stop resource monitoring
                resource_summary = monitor.stop_monitoring()
                resource_summaries.append(resource_summary)
                
                # If benchmark returns latency data
                if isinstance(result, dict) and 'latencies' in result:
                    latencies.extend(result['latencies'])
                
                logger.debug(f"Iteration {i+1}/{iterations}: {duration:.4f}s")
            
            # Calculate statistics
            stats_summary = self._calculate_statistics(durations)
            resource_stats = self._aggregate_resource_stats(resource_summaries)
            
            # Calculate latency percentiles
            if latencies:
                latency_p50 = np.percentile(latencies, 50)
                latency_p95 = np.percentile(latencies, 95)
                latency_p99 = np.percentile(latencies, 99)
            else:
                latency_p50 = latency_p95 = latency_p99 = stats_summary['mean'] * 1000
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                test_name=name,
                timestamp=datetime.now(),
                duration_seconds=stats_summary['mean'],
                cpu_usage_percent=resource_stats.get('cpu_usage_avg', 0),
                memory_usage_mb=resource_stats.get('memory_avg_mb', 0),
                peak_memory_mb=resource_stats.get('memory_peak_mb', 0),
                disk_read_mb=resource_stats.get('disk_read_mb', 0),
                disk_write_mb=resource_stats.get('disk_write_mb', 0),
                network_sent_mb=resource_stats.get('network_sent_mb', 0),
                network_recv_mb=resource_stats.get('network_recv_mb', 0),
                throughput_ops_per_sec=1.0 / stats_summary['mean'] if stats_summary['mean'] > 0 else 0,
                latency_p50_ms=latency_p50,
                latency_p95_ms=latency_p95,
                latency_p99_ms=latency_p99,
                error_count=0,  # Would be tracked by benchmark function
                success_count=iterations,
                success_rate=1.0,
                gc_collections=self._get_gc_stats(),
                custom_metrics={}
            )
            
            # Check for performance regression
            regression_detected = self._check_regression(name, metrics)
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(metrics, resource_stats)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(metrics)
            
            result = BenchmarkResult(
                benchmark_name=name,
                test_case="standard",
                metrics=metrics,
                statistical_summary=stats_summary,
                regression_detected=regression_detected,
                optimization_score=optimization_score,
                recommendations=recommendations
            )
            
            self.results_history.append(result)
            return result
            
        finally:
            # Teardown
            if benchmark['teardown']:
                benchmark['teardown']()
    
    def _calculate_statistics(self, durations: List[float]) -> Dict[str, float]:
        """Calculate statistical summary of benchmark durations."""
        durations_array = np.array(durations)
        
        return {
            'mean': float(np.mean(durations_array)),
            'median': float(np.median(durations_array)),
            'std': float(np.std(durations_array)),
            'min': float(np.min(durations_array)),
            'max': float(np.max(durations_array)),
            'p25': float(np.percentile(durations_array, 25)),
            'p75': float(np.percentile(durations_array, 75)),
            'p95': float(np.percentile(durations_array, 95)),
            'p99': float(np.percentile(durations_array, 99)),
            'coefficient_of_variation': float(np.std(durations_array) / np.mean(durations_array)) if np.mean(durations_array) > 0 else 0
        }
    
    def _aggregate_resource_stats(self, resource_summaries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate resource statistics across iterations."""
        if not resource_summaries:
            return {}
        
        # Filter out empty summaries
        valid_summaries = [s for s in resource_summaries if s]
        if not valid_summaries:
            return {}
        
        aggregated = {}
        
        # Get all keys from first valid summary
        keys = valid_summaries[0].keys()
        
        for key in keys:
            values = [s.get(key, 0) for s in valid_summaries if key in s]
            if values:
                aggregated[f"{key}"] = np.mean(values)
                aggregated[f"{key}_max"] = np.max(values)
                aggregated[f"{key}_min"] = np.min(values)
        
        return aggregated
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collection statistics."""
        return {f"gen_{i}": gc.get_count()[i] for i in range(len(gc.get_count()))}
    
    def _check_regression(self, name: str, metrics: PerformanceMetrics) -> bool:
        """Check if performance has regressed compared to baseline."""
        if name not in self.baseline_results:
            return False
        
        baseline = self.baseline_results[name]
        
        # Check if current performance is significantly worse
        duration_regression = metrics.duration_seconds > baseline.duration_seconds * 1.2  # 20% slower
        memory_regression = metrics.peak_memory_mb > baseline.peak_memory_mb * 1.3  # 30% more memory
        
        return duration_regression or memory_regression
    
    def _generate_recommendations(self, metrics: PerformanceMetrics, 
                                resource_stats: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        # CPU recommendations
        if metrics.cpu_usage_percent > 80:
            recommendations.append("High CPU usage detected. Consider optimizing computational algorithms or using multiprocessing.")
        
        # Memory recommendations
        if metrics.peak_memory_mb > 1000:  # > 1GB
            recommendations.append("High memory usage detected. Consider implementing memory pooling or streaming data processing.")
        
        # I/O recommendations
        if metrics.disk_read_mb + metrics.disk_write_mb > 100:  # > 100MB I/O
            recommendations.append("High disk I/O detected. Consider implementing caching or optimizing data access patterns.")
        
        # Latency recommendations
        if metrics.latency_p95_ms > 1000:  # > 1 second
            recommendations.append("High latency detected. Consider implementing asynchronous processing or request batching.")
        
        # Throughput recommendations
        if metrics.throughput_ops_per_sec < 10:  # < 10 ops/sec
            recommendations.append("Low throughput detected. Consider implementing parallel processing or connection pooling.")
        
        return recommendations
    
    def _calculate_optimization_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate an overall optimization score (0-100)."""
        score = 100.0
        
        # Deduct points for performance issues
        if metrics.cpu_usage_percent > 80:
            score -= 20
        elif metrics.cpu_usage_percent > 60:
            score -= 10
        
        if metrics.peak_memory_mb > 1000:
            score -= 15
        elif metrics.peak_memory_mb > 500:
            score -= 8
        
        if metrics.latency_p95_ms > 1000:
            score -= 20
        elif metrics.latency_p95_ms > 500:
            score -= 10
        
        if metrics.throughput_ops_per_sec < 10:
            score -= 15
        elif metrics.throughput_ops_per_sec < 50:
            score -= 8
        
        return max(0.0, score)
    
    def set_baseline(self, name: str, metrics: PerformanceMetrics) -> None:
        """Set baseline performance metrics for regression detection."""
        self.baseline_results[name] = metrics
        logger.info(f"Baseline set for benchmark '{name}'")
    
    def compare_benchmarks(self, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two benchmark results."""
        results1 = [r for r in self.results_history if r.benchmark_name == name1]
        results2 = [r for r in self.results_history if r.benchmark_name == name2]
        
        if not results1 or not results2:
            return {"error": "Not enough data for comparison"}
        
        latest1 = results1[-1].metrics
        latest2 = results2[-1].metrics
        
        comparison = {
            "benchmark_1": name1,
            "benchmark_2": name2,
            "duration_ratio": latest1.duration_seconds / latest2.duration_seconds,
            "memory_ratio": latest1.peak_memory_mb / latest2.peak_memory_mb if latest2.peak_memory_mb > 0 else float('inf'),
            "throughput_ratio": latest1.throughput_ops_per_sec / latest2.throughput_ops_per_sec if latest2.throughput_ops_per_sec > 0 else float('inf'),
            "winner": name1 if latest1.duration_seconds < latest2.duration_seconds else name2,
            "improvement_percent": abs(1 - latest1.duration_seconds / latest2.duration_seconds) * 100 if latest2.duration_seconds > 0 else 0
        }
        
        return comparison
    
    def generate_benchmark_report(self, output_dir: str = "benchmark_reports") -> str:
        """Generate comprehensive benchmark report."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results_history:
            return "No benchmark results available"
        
        # Generate text report
        report_path = os.path.join(output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_path, 'w') as f:
            f.write("Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Benchmarks: {len(set(r.benchmark_name for r in self.results_history))}\n")
            f.write(f"Total Runs: {len(self.results_history)}\n\n")
            
            # Summary by benchmark
            for benchmark_name in set(r.benchmark_name for r in self.results_history):
                benchmark_results = [r for r in self.results_history if r.benchmark_name == benchmark_name]
                latest = benchmark_results[-1]
                
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Latest Run: {latest.metrics.timestamp}\n")
                f.write(f"Duration: {latest.metrics.duration_seconds:.4f} seconds\n")
                f.write(f"Throughput: {latest.metrics.throughput_ops_per_sec:.2f} ops/sec\n")
                f.write(f"Peak Memory: {latest.metrics.peak_memory_mb:.2f} MB\n")
                f.write(f"CPU Usage: {latest.metrics.cpu_usage_percent:.1f}%\n")
                f.write(f"Optimization Score: {latest.optimization_score:.1f}/100\n")
                
                if latest.recommendations:
                    f.write("Recommendations:\n")
                    for rec in latest.recommendations:
                        f.write(f"  - {rec}\n")
                
                f.write("\n")
        
        # Generate performance plots
        self._generate_performance_plots(output_dir)
        
        return report_path
    
    def _generate_performance_plots(self, output_dir: str) -> None:
        """Generate performance visualization plots."""
        if not self.results_history:
            return
        
        # Prepare data
        df_data = []
        for result in self.results_history:
            df_data.append({
                'benchmark': result.benchmark_name,
                'timestamp': result.metrics.timestamp,
                'duration': result.metrics.duration_seconds,
                'throughput': result.metrics.throughput_ops_per_sec,
                'memory': result.metrics.peak_memory_mb,
                'cpu': result.metrics.cpu_usage_percent,
                'optimization_score': result.optimization_score
            })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Duration over time
        for benchmark in df['benchmark'].unique():
            benchmark_data = df[df['benchmark'] == benchmark]
            axes[0, 0].plot(benchmark_data['timestamp'], benchmark_data['duration'], 
                           marker='o', label=benchmark)
        axes[0, 0].set_title('Execution Time Over Time')
        axes[0, 0].set_ylabel('Duration (seconds)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput over time
        for benchmark in df['benchmark'].unique():
            benchmark_data = df[df['benchmark'] == benchmark]
            axes[0, 1].plot(benchmark_data['timestamp'], benchmark_data['throughput'], 
                           marker='s', label=benchmark)
        axes[0, 1].set_title('Throughput Over Time')
        axes[0, 1].set_ylabel('Operations per Second')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage over time
        for benchmark in df['benchmark'].unique():
            benchmark_data = df[df['benchmark'] == benchmark]
            axes[1, 0].plot(benchmark_data['timestamp'], benchmark_data['memory'], 
                           marker='^', label=benchmark)
        axes[1, 0].set_title('Peak Memory Usage Over Time')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Optimization score over time
        for benchmark in df['benchmark'].unique():
            benchmark_data = df[df['benchmark'] == benchmark]
            axes[1, 1].plot(benchmark_data['timestamp'], benchmark_data['optimization_score'], 
                           marker='d', label=benchmark)
        axes[1, 1].set_title('Optimization Score Over Time')
        axes[1, 1].set_ylabel('Score (0-100)')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance distribution
        if len(df['benchmark'].unique()) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Duration distribution
            sns.boxplot(data=df, x='benchmark', y='duration', ax=axes[0, 0])
            axes[0, 0].set_title('Duration Distribution by Benchmark')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Throughput distribution
            sns.boxplot(data=df, x='benchmark', y='throughput', ax=axes[0, 1])
            axes[0, 1].set_title('Throughput Distribution by Benchmark')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Memory distribution
            sns.boxplot(data=df, x='benchmark', y='memory', ax=axes[1, 0])
            axes[1, 0].set_title('Memory Usage Distribution by Benchmark')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Optimization score distribution
            sns.boxplot(data=df, x='benchmark', y='optimization_score', ax=axes[1, 1])
            axes[1, 1].set_title('Optimization Score Distribution by Benchmark')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()


class LoadTester:
    """Load testing and stress testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.active_tests = {}
        
    def run_load_test(self, target_function: Callable, config: LoadTestConfig,
                     test_name: str = "load_test") -> Dict[str, Any]:
        """Run load test against a target function."""
        
        logger.info(f"Starting load test: {test_name}")
        logger.info(f"Config: {config.concurrent_users} users, {config.duration_seconds}s duration")
        
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        # Shared metrics
        results_queue = []
        errors = []
        
        def worker_thread(worker_id: int):
            """Worker thread function."""
            local_results = []
            local_errors = []
            
            # Ramp-up delay
            ramp_delay = (config.ramp_up_seconds / config.concurrent_users) * worker_id
            time.sleep(ramp_delay)
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    result = target_function()
                    request_end = time.time()
                    
                    response_time = request_end - request_start
                    
                    local_results.append({
                        'timestamp': request_start,
                        'response_time_ms': response_time * 1000,
                        'success': True,
                        'worker_id': worker_id
                    })
                    
                    # Rate limiting
                    if config.target_rps:
                        expected_interval = config.concurrent_users / config.target_rps
                        time.sleep(max(0, expected_interval - response_time))
                    
                except Exception as e:
                    local_errors.append({
                        'timestamp': time.time(),
                        'error': str(e),
                        'worker_id': worker_id
                    })
            
            results_queue.extend(local_results)
            errors.extend(local_errors)
        
        # Start worker threads
        threads = []
        for i in range(config.concurrent_users):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze results
        total_requests = len(results_queue)
        total_errors = len(errors)
        success_rate = total_requests / (total_requests + total_errors) if (total_requests + total_errors) > 0 else 0
        
        if results_queue:
            response_times = [r['response_time_ms'] for r in results_queue]
            
            avg_response_time = np.mean(response_times)
            p50_response_time = np.percentile(response_times, 50)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            
            # Calculate throughput
            actual_duration = time.time() - start_time
            throughput = total_requests / actual_duration
            
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            throughput = 0
            actual_duration = config.duration_seconds
        
        # Check if test passed
        test_passed = (
            success_rate >= config.success_threshold and
            (p95_response_time <= config.max_response_time_ms if results_queue else False)
        )
        
        result = {
            'test_name': test_name,
            'config': asdict(config),
            'results': {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'success_rate': success_rate,
                'avg_response_time_ms': avg_response_time,
                'p50_response_time_ms': p50_response_time,
                'p95_response_time_ms': p95_response_time,
                'p99_response_time_ms': p99_response_time,
                'throughput_rps': throughput,
                'actual_duration_seconds': actual_duration,
                'test_passed': test_passed
            },
            'timestamp': datetime.now()
        }
        
        self.test_results.append(result)
        
        logger.info(f"Load test completed: {test_name}")
        logger.info(f"Results: {total_requests} requests, {success_rate:.2%} success rate, "
                   f"{throughput:.2f} RPS, {p95_response_time:.1f}ms P95")
        
        return result


class PerformanceTestSuite:
    """Complete performance testing suite."""
    
    def __init__(self):
        self.benchmark_suite = BenchmarkSuite()
        self.load_tester = LoadTester()
        self.profiler = CodeProfiler()
        
        # Register common benchmarks
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self) -> None:
        """Register default ML system benchmarks."""
        
        def model_prediction_benchmark():
            """Benchmark model prediction performance."""
            # This would be implemented with actual model
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            
            # Create sample data
            X = np.random.random((1000, 10))
            y = np.random.randint(0, 2, 1000)
            
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Prediction benchmark
            test_X = np.random.random((100, 10))
            start_time = time.time()
            predictions = model.predict(test_X)
            end_time = time.time()
            
            return {
                'predictions': len(predictions),
                'latencies': [(end_time - start_time) * 1000 / len(predictions)] * len(predictions)
            }
        
        def data_preprocessing_benchmark():
            """Benchmark data preprocessing performance."""
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            # Generate sample data
            data = pd.DataFrame(np.random.random((10000, 20)))
            
            # Preprocessing operations
            start_time = time.time()
            
            # Standardization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Feature engineering
            data['feature_sum'] = data.sum(axis=1)
            data['feature_mean'] = data.mean(axis=1)
            
            end_time = time.time()
            
            return {
                'rows_processed': len(data),
                'latencies': [(end_time - start_time) * 1000]
            }
        
        self.benchmark_suite.register_benchmark(
            'model_prediction',
            model_prediction_benchmark
        )
        
        self.benchmark_suite.register_benchmark(
            'data_preprocessing',
            data_preprocessing_benchmark
        )
    
    @with_error_handling(component="performance_testing", enable_retry=True)
    def run_comprehensive_test(self, test_name: str = "comprehensive") -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        
        logger.info("Starting comprehensive performance test suite")
        
        results = {
            'test_name': test_name,
            'timestamp': datetime.now(),
            'benchmark_results': {},
            'load_test_results': {},
            'profile_results': {},
            'summary': {}
        }
        
        # Run benchmarks
        for benchmark_name in self.benchmark_suite.benchmarks.keys():
            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                result = self.benchmark_suite.run_benchmark(benchmark_name, iterations=5)
                results['benchmark_results'][benchmark_name] = asdict(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {e}")
                results['benchmark_results'][benchmark_name] = {'error': str(e)}
        
        # Run load tests (simplified)
        try:
            def simple_load_target():
                time.sleep(0.01)  # Simulate 10ms operation
                return "success"
            
            load_config = LoadTestConfig(
                concurrent_users=5,
                duration_seconds=30,
                ramp_up_seconds=10
            )
            
            load_result = self.load_tester.run_load_test(
                simple_load_target, load_config, "basic_load_test"
            )
            results['load_test_results']['basic_load_test'] = load_result
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            results['load_test_results']['basic_load_test'] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_test_summary(results)
        
        logger.info("Comprehensive performance test suite completed")
        
        return results
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        
        summary = {
            'total_benchmarks': len(results['benchmark_results']),
            'successful_benchmarks': len([r for r in results['benchmark_results'].values() if 'error' not in r]),
            'total_load_tests': len(results['load_test_results']),
            'successful_load_tests': len([r for r in results['load_test_results'].values() if 'error' not in r]),
            'overall_performance_score': 0,
            'key_findings': []
        }
        
        # Calculate overall performance score
        scores = []
        for benchmark_result in results['benchmark_results'].values():
            if 'error' not in benchmark_result and 'optimization_score' in benchmark_result:
                scores.append(benchmark_result['optimization_score'])
        
        if scores:
            summary['overall_performance_score'] = np.mean(scores)
        
        # Generate key findings
        findings = []
        
        for name, benchmark_result in results['benchmark_results'].items():
            if 'error' not in benchmark_result and 'recommendations' in benchmark_result:
                if benchmark_result['recommendations']:
                    findings.append(f"{name}: {len(benchmark_result['recommendations'])} optimization opportunities")
        
        for name, load_result in results['load_test_results'].items():
            if 'error' not in load_result:
                if load_result['results']['test_passed']:
                    findings.append(f"{name}: Load test passed with {load_result['results']['success_rate']:.1%} success rate")
                else:
                    findings.append(f"{name}: Load test failed - investigate performance issues")
        
        summary['key_findings'] = findings
        
        return summary
    
    def generate_performance_report(self, output_dir: str = "performance_reports") -> str:
        """Generate comprehensive performance report."""
        
        # Generate benchmark report
        benchmark_report = self.benchmark_suite.generate_benchmark_report(output_dir)
        
        # Generate load test report
        load_report_path = os.path.join(output_dir, f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(load_report_path, 'w') as f:
            json.dump(self.load_tester.test_results, f, indent=2, default=str)
        
        return benchmark_report


# Factory functions
def create_benchmark_suite() -> BenchmarkSuite:
    """Create and configure benchmark suite."""
    return BenchmarkSuite()


def create_performance_test_suite() -> PerformanceTestSuite:
    """Create comprehensive performance test suite."""
    return PerformanceTestSuite()


if __name__ == "__main__":
    print("Performance Benchmarking and Profiling System")
    print("This system provides comprehensive performance analysis and optimization tools.")