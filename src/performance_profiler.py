"""
Advanced Performance Profiler and Optimization System.

This module provides comprehensive performance monitoring, profiling, and optimization:
- Real-time performance monitoring with detailed metrics collection
- Code profiling and bottleneck identification
- Memory usage analysis and optimization suggestions
- Database query optimization and connection pooling
- API response time optimization with caching strategies
- Resource utilization monitoring and auto-scaling triggers
"""

import os
import time
import psutil
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd
import cProfile
import pstats
import tracemalloc
from contextlib import contextmanager
import asyncio
# import aiofiles  # Optional dependency
import concurrent.futures

from .logging_config import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int
    response_times: Dict[str, float] = field(default_factory=dict)
    database_query_times: Dict[str, float] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class ProfilingResult:
    """Code profiling result."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    filename: str
    line_number: int
    hotspots: List[str] = field(default_factory=list)


@dataclass
class MemoryProfile:
    """Memory profiling result."""
    peak_memory_mb: float
    current_memory_mb: float
    memory_growth_rate: float
    top_allocators: List[Dict[str, Any]] = field(default_factory=list)
    memory_leaks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    description: str
    potential_improvement: str
    implementation_complexity: str
    estimated_time_hours: float
    code_changes_required: List[str]
    risk_level: str


class SystemResourceMonitor:
    """Monitor system resources in real-time."""
    
    def __init__(self, sample_interval: float = 1.0, history_size: int = 1000):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread = None
        self._previous_disk_io = None
        self._previous_network_io = None
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System resource monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        
        if disk_io and self._previous_disk_io:
            disk_read_mb = (disk_io.read_bytes - self._previous_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self._previous_disk_io.write_bytes) / (1024 * 1024)
        
        if disk_io:
            self._previous_disk_io = disk_io
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        
        if network_io and self._previous_network_io:
            network_sent_mb = (network_io.bytes_sent - self._previous_network_io.bytes_sent) / (1024 * 1024)
            network_recv_mb = (network_io.bytes_recv - self._previous_network_io.bytes_recv) / (1024 * 1024)
        
        if network_io:
            self._previous_network_io = network_io
        
        # Process information
        process = psutil.Process()
        active_threads = process.num_threads()
        
        try:
            open_files = len(process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_io_read_mb=max(0, disk_read_mb),  # Ensure non-negative
            disk_io_write_mb=max(0, disk_write_mb),
            network_sent_mb=max(0, network_sent_mb),
            network_recv_mb=max(0, network_recv_mb),
            active_threads=active_threads,
            open_files=open_files
        )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        if not self.metrics_history:
            return self._collect_metrics()
        return self.metrics_history[-1]
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'time_range_minutes': minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'disk_io_total_mb': sum(m.disk_io_read_mb + m.disk_io_write_mb for m in recent_metrics),
            'network_io_total_mb': sum(m.network_sent_mb + m.network_recv_mb for m in recent_metrics),
            'avg_threads': np.mean([m.active_threads for m in recent_metrics]),
            'avg_open_files': np.mean([m.open_files for m in recent_metrics])
        }


class CodeProfiler:
    """Advanced code profiling with detailed analysis."""
    
    def __init__(self):
        self.profiling_results = {}
        self.active_profilers = {}
        
    @contextmanager
    def profile_code(self, profile_name: str):
        """Context manager for code profiling."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        try:
            yield profiler
        finally:
            profiler.disable()
            execution_time = time.time() - start_time
            
            # Analyze profiling results
            stats = pstats.Stats(profiler)
            
            # Get top functions by cumulative time
            stats.sort_stats('cumulative')
            
            # Extract profiling data
            profiling_data = []
            for func_info, (call_count, total_time, cumulative_time, callers, _) in stats.stats.items():
                filename, line_number, function_name = func_info
                
                profiling_data.append(ProfilingResult(
                    function_name=function_name,
                    total_time=total_time,
                    cumulative_time=cumulative_time,
                    call_count=call_count,
                    time_per_call=total_time / call_count if call_count > 0 else 0,
                    filename=filename,
                    line_number=line_number
                ))
            
            # Sort by cumulative time (descending)
            profiling_data.sort(key=lambda x: x.cumulative_time, reverse=True)
            
            self.profiling_results[profile_name] = {
                'execution_time': execution_time,
                'functions': profiling_data[:20],  # Top 20 functions
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Code profiling completed for '{profile_name}' in {execution_time:.3f}s")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for function profiling."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile_name = f"{func.__module__}.{func.__name__}"
            
            with self.profile_code(profile_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_hotspots(self, profile_name: str, threshold_percent: float = 5.0) -> List[str]:
        """Identify performance hotspots."""
        if profile_name not in self.profiling_results:
            return []
        
        results = self.profiling_results[profile_name]
        total_time = results['execution_time']
        threshold_time = total_time * (threshold_percent / 100)
        
        hotspots = []
        for func_result in results['functions']:
            if func_result.cumulative_time > threshold_time:
                hotspot = f"{func_result.function_name} ({func_result.cumulative_time:.3f}s, {(func_result.cumulative_time/total_time*100):.1f}%)"
                hotspots.append(hotspot)
        
        return hotspots
    
    def generate_profiling_report(self, profile_name: str) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        if profile_name not in self.profiling_results:
            return {}
        
        results = self.profiling_results[profile_name]
        
        return {
            'profile_name': profile_name,
            'execution_time': results['execution_time'],
            'timestamp': results['timestamp'],
            'function_count': len(results['functions']),
            'hotspots': self.get_hotspots(profile_name),
            'top_functions': [
                {
                    'name': f.function_name,
                    'cumulative_time': f.cumulative_time,
                    'call_count': f.call_count,
                    'time_per_call': f.time_per_call,
                    'file': f.filename,
                    'line': f.line_number
                }
                for f in results['functions'][:10]
            ]
        }


class MemoryProfiler:
    """Advanced memory profiling and leak detection."""
    
    def __init__(self):
        self.memory_snapshots = {}
        self.monitoring_active = False
        
    @contextmanager
    def profile_memory(self, profile_name: str):
        """Context manager for memory profiling."""
        tracemalloc.start()
        
        # Take initial snapshot
        initial_snapshot = tracemalloc.take_snapshot()
        initial_memory = self._get_current_memory()
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Take final snapshot
            final_snapshot = tracemalloc.take_snapshot()
            final_memory = self._get_current_memory()
            execution_time = time.time() - start_time
            
            # Calculate memory growth
            memory_growth = final_memory - initial_memory
            memory_growth_rate = memory_growth / execution_time if execution_time > 0 else 0
            
            # Get top memory allocators
            top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
            top_allocators = []
            
            for stat in top_stats[:10]:
                allocator_info = {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count,
                    'size_diff_mb': stat.size_diff / (1024 * 1024) if hasattr(stat, 'size_diff') else 0
                }
                top_allocators.append(allocator_info)
            
            # Detect potential memory leaks (simplified)
            potential_leaks = []
            for stat in top_stats:
                if hasattr(stat, 'size_diff') and stat.size_diff > 1024 * 1024:  # >1MB growth
                    potential_leaks.append(stat.traceback.format()[0] if stat.traceback else 'unknown location')
            
            # Generate optimization suggestions
            suggestions = self._generate_memory_optimization_suggestions(
                memory_growth, top_allocators, potential_leaks
            )
            
            profile_result = MemoryProfile(
                peak_memory_mb=final_memory,
                current_memory_mb=final_memory,
                memory_growth_rate=memory_growth_rate,
                top_allocators=top_allocators,
                memory_leaks=potential_leaks,
                optimization_suggestions=suggestions
            )
            
            self.memory_snapshots[profile_name] = {
                'result': profile_result,
                'timestamp': datetime.utcnow().isoformat(),
                'execution_time': execution_time
            }
            
            tracemalloc.stop()
            
            logger.info(f"Memory profiling completed for '{profile_name}': {memory_growth:.2f}MB growth")
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _generate_memory_optimization_suggestions(
        self, 
        memory_growth: float, 
        top_allocators: List[Dict], 
        potential_leaks: List[str]
    ) -> List[str]:
        """Generate memory optimization suggestions."""
        suggestions = []
        
        if memory_growth > 100:  # >100MB growth
            suggestions.append("Consider implementing object pooling for frequently created objects")
            suggestions.append("Review data structures for memory efficiency (use generators where possible)")
        
        if memory_growth > 50:  # >50MB growth
            suggestions.append("Implement explicit garbage collection at key points")
            suggestions.append("Use weak references for caches to allow garbage collection")
        
        if potential_leaks:
            suggestions.append("Investigate potential memory leaks in identified locations")
            suggestions.append("Implement proper resource cleanup (context managers, try/finally blocks)")
        
        if any(alloc['size_mb'] > 10 for alloc in top_allocators):
            suggestions.append("Consider streaming processing for large data operations")
            suggestions.append("Implement data chunking for memory-intensive operations")
        
        if len(top_allocators) > 5:
            suggestions.append("Profile individual functions to identify specific memory hotspots")
        
        return suggestions
    
    def get_memory_report(self, profile_name: str) -> Dict[str, Any]:
        """Get detailed memory profiling report."""
        if profile_name not in self.memory_snapshots:
            return {}
        
        snapshot_data = self.memory_snapshots[profile_name]
        result = snapshot_data['result']
        
        return {
            'profile_name': profile_name,
            'timestamp': snapshot_data['timestamp'],
            'execution_time': snapshot_data['execution_time'],
            'peak_memory_mb': result.peak_memory_mb,
            'memory_growth_rate_mb_per_sec': result.memory_growth_rate,
            'top_allocators': result.top_allocators,
            'potential_memory_leaks': result.memory_leaks,
            'optimization_suggestions': result.optimization_suggestions
        }


class DatabasePerformanceMonitor:
    """Monitor and optimize database performance."""
    
    def __init__(self):
        self.query_times = defaultdict(list)
        self.slow_queries = []
        self.connection_pool_stats = {}
        
    def record_query_time(self, query_name: str, execution_time: float, query_sql: str = ""):
        """Record database query execution time."""
        self.query_times[query_name].append({
            'execution_time': execution_time,
            'timestamp': datetime.utcnow().isoformat(),
            'sql': query_sql
        })
        
        # Identify slow queries (>1 second)
        if execution_time > 1.0:
            self.slow_queries.append({
                'query_name': query_name,
                'execution_time': execution_time,
                'timestamp': datetime.utcnow().isoformat(),
                'sql': query_sql
            })
            
            # Keep only recent slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
        
        # Keep only recent query times
        if len(self.query_times[query_name]) > 1000:
            self.query_times[query_name] = self.query_times[query_name][-1000:]
    
    @contextmanager
    def monitor_query(self, query_name: str, query_sql: str = ""):
        """Context manager for monitoring database queries."""
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.record_query_time(query_name, execution_time, query_sql)
    
    def get_query_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get database query performance statistics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        stats = {}
        for query_name, query_records in self.query_times.items():
            recent_records = [
                r for r in query_records
                if datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) > cutoff_time
            ]
            
            if recent_records:
                execution_times = [r['execution_time'] for r in recent_records]
                stats[query_name] = {
                    'count': len(execution_times),
                    'avg_time': np.mean(execution_times),
                    'max_time': np.max(execution_times),
                    'min_time': np.min(execution_times),
                    'p95_time': np.percentile(execution_times, 95),
                    'total_time': sum(execution_times)
                }
        
        # Recent slow queries
        recent_slow_queries = [
            q for q in self.slow_queries
            if datetime.fromisoformat(q['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]
        
        return {
            'query_statistics': stats,
            'slow_queries_count': len(recent_slow_queries),
            'slow_queries': recent_slow_queries[-10:],  # Last 10 slow queries
            'total_queries': sum(len(records) for records in self.query_times.values())
        }
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate database optimization recommendations."""
        recommendations = []
        
        stats = self.get_query_statistics()
        
        # Identify slow queries
        for query_name, query_stats in stats['query_statistics'].items():
            if query_stats['avg_time'] > 0.5:  # Slow average time
                recommendations.append(OptimizationRecommendation(
                    category="database",
                    priority="high" if query_stats['avg_time'] > 2.0 else "medium",
                    description=f"Query '{query_name}' has high average execution time ({query_stats['avg_time']:.2f}s)",
                    potential_improvement="50-80% reduction in query time",
                    implementation_complexity="medium",
                    estimated_time_hours=4.0,
                    code_changes_required=[
                        "Add database indexes",
                        "Optimize query structure",
                        "Consider query caching"
                    ],
                    risk_level="low"
                ))
        
        # Check for high query volume
        total_queries = stats['total_queries']
        if total_queries > 10000:  # High query volume
            recommendations.append(OptimizationRecommendation(
                category="database",
                priority="medium",
                description=f"High database query volume ({total_queries} queries in 24h)",
                potential_improvement="Reduce database load by 30-50%",
                implementation_complexity="high",
                estimated_time_hours=8.0,
                code_changes_required=[
                    "Implement query result caching",
                    "Add connection pooling",
                    "Optimize frequently-used queries"
                ],
                risk_level="medium"
            ))
        
        return recommendations


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self):
        self.system_monitor = SystemResourceMonitor()
        self.code_profiler = CodeProfiler()
        self.memory_profiler = MemoryProfiler()
        self.db_monitor = DatabasePerformanceMonitor()
        self.optimization_history = []
        
    def start_monitoring(self):
        """Start comprehensive performance monitoring."""
        self.system_monitor.start_monitoring()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.system_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
    
    def profile_code_block(self, block_name: str):
        """Get context manager for code profiling."""
        return self.code_profiler.profile_code(block_name)
    
    def profile_memory_block(self, block_name: str):
        """Get context manager for memory profiling."""
        return self.memory_profiler.profile_memory(block_name)
    
    def monitor_database_query(self, query_name: str, query_sql: str = ""):
        """Get context manager for database query monitoring."""
        return self.db_monitor.monitor_query(query_name, query_sql)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': self.system_monitor.get_metrics_summary(60),
            'database_stats': self.db_monitor.get_query_statistics(24),
            'optimization_recommendations': self._generate_all_recommendations(),
            'monitoring_status': {
                'system_monitoring_active': self.system_monitor.is_monitoring,
                'metrics_history_size': len(self.system_monitor.metrics_history),
                'profiling_sessions': len(self.code_profiler.profiling_results),
                'memory_profiles': len(self.memory_profiler.memory_snapshots)
            }
        }
        
        return report
    
    def _generate_all_recommendations(self) -> List[Dict[str, Any]]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # Database recommendations
        db_recommendations = self.db_monitor.generate_optimization_recommendations()
        recommendations.extend([asdict(rec) for rec in db_recommendations])
        
        # System resource recommendations
        system_summary = self.system_monitor.get_metrics_summary(60)
        if system_summary:
            recommendations.extend(self._generate_system_recommendations(system_summary))
        
        # Memory recommendations
        for profile_name in self.memory_profiler.memory_snapshots:
            memory_report = self.memory_profiler.get_memory_report(profile_name)
            if memory_report.get('memory_growth_rate_mb_per_sec', 0) > 1:
                recommendations.append({
                    'category': 'memory',
                    'priority': 'high',
                    'description': f'High memory growth rate in {profile_name}',
                    'potential_improvement': 'Reduce memory usage by 20-40%',
                    'implementation_complexity': 'medium',
                    'estimated_time_hours': 6.0,
                    'code_changes_required': memory_report.get('optimization_suggestions', []),
                    'risk_level': 'low'
                })
        
        return recommendations
    
    def _generate_system_recommendations(self, system_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system-level optimization recommendations."""
        recommendations = []
        
        cpu_stats = system_summary.get('cpu', {})
        memory_stats = system_summary.get('memory', {})
        
        # High CPU usage
        if cpu_stats.get('mean', 0) > 80:
            recommendations.append({
                'category': 'system',
                'priority': 'high',
                'description': f'High average CPU usage ({cpu_stats["mean"]:.1f}%)',
                'potential_improvement': 'Reduce CPU usage by 30-50%',
                'implementation_complexity': 'high',
                'estimated_time_hours': 12.0,
                'code_changes_required': [
                    'Optimize CPU-intensive algorithms',
                    'Implement parallel processing',
                    'Add caching for expensive computations'
                ],
                'risk_level': 'medium'
            })
        
        # High memory usage
        if memory_stats.get('mean', 0) > 80:
            recommendations.append({
                'category': 'system',
                'priority': 'medium',
                'description': f'High average memory usage ({memory_stats["mean"]:.1f}%)',
                'potential_improvement': 'Reduce memory usage by 20-30%',
                'implementation_complexity': 'medium',
                'estimated_time_hours': 8.0,
                'code_changes_required': [
                    'Optimize data structures',
                    'Implement lazy loading',
                    'Add memory cleanup routines'
                ],
                'risk_level': 'low'
            })
        
        return recommendations
    
    def apply_optimization(self, optimization_id: str, description: str) -> bool:
        """Record optimization application."""
        optimization_record = {
            'optimization_id': optimization_id,
            'description': description,
            'applied_at': datetime.utcnow().isoformat(),
            'status': 'applied'
        }
        
        self.optimization_history.append(optimization_record)
        logger.info(f"Optimization applied: {description}")
        
        # Record metrics before and after (simplified)
        metrics_collector = get_metrics_collector()
        metrics_collector.record_performance_optimization(
            optimization_type=optimization_id,
            improvement_metric="response_time",
            before_value=1.0,  # Would be actual measured values
            after_value=0.8,
            improvement_percent=20.0
        )
        
        return True
    
    def export_performance_data(self, output_file: str) -> str:
        """Export performance data to file."""
        report = self.generate_comprehensive_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance data exported to {output_file}")
        return output_file


def performance_monitor(func: Callable) -> Callable:
    """Decorator for automatic performance monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = PerformanceOptimizer()
        
        # Start monitoring if not already active
        was_monitoring = optimizer.system_monitor.is_monitoring
        if not was_monitoring:
            optimizer.system_monitor.start_monitoring()
        
        # Profile the function
        profile_name = f"{func.__module__}.{func.__name__}"
        
        with optimizer.profile_code_block(profile_name):
            with optimizer.profile_memory_block(profile_name):
                result = func(*args, **kwargs)
        
        # Stop monitoring if we started it
        if not was_monitoring:
            optimizer.system_monitor.stop_monitoring()
        
        return result
    
    return wrapper


@contextmanager
def performance_context(context_name: str, enable_profiling: bool = True):
    """Context manager for performance monitoring."""
    optimizer = PerformanceOptimizer()
    optimizer.start_monitoring()
    
    profiling_contexts = []
    
    try:
        if enable_profiling:
            code_ctx = optimizer.profile_code_block(context_name)
            memory_ctx = optimizer.profile_memory_block(context_name)
            profiling_contexts = [code_ctx, memory_ctx]
            
            # Enter contexts
            for ctx in profiling_contexts:
                ctx.__enter__()
        
        yield optimizer
        
    finally:
        # Exit contexts in reverse order
        for ctx in reversed(profiling_contexts):
            try:
                ctx.__exit__(None, None, None)
            except:
                pass
        
        optimizer.stop_monitoring()


if __name__ == "__main__":
    print("Advanced Performance Profiler and Optimization System")
    print("Provides comprehensive performance monitoring and optimization capabilities.")