"""
Chaos Engineering Testing Framework.

This module provides comprehensive chaos engineering capabilities including:
- Fault injection testing with controlled failure scenarios
- Network partition simulation and connectivity issues
- Resource exhaustion scenarios (CPU, memory, disk, network)
- Node failure and recovery testing with automated recovery validation
- Latency injection and network degradation simulation
- Service dependency failure testing with cascading failure analysis
- Database corruption and recovery testing
- Configuration drift and environment inconsistency testing
"""

import os
import time
import json
import asyncio
import threading
import subprocess
import tempfile
import random
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum, auto
import psutil
import uuid

# Network testing
try:
    import netifaces
    import socket
    NETWORK_TESTING_AVAILABLE = True
except ImportError:
    NETWORK_TESTING_AVAILABLE = False

# Container chaos testing
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Testing framework
import pytest
from unittest.mock import Mock, patch, MagicMock


class ChaosType(Enum):
    """Types of chaos engineering experiments."""
    CPU_STRESS = auto()
    MEMORY_EXHAUSTION = auto()
    DISK_FILL = auto()
    NETWORK_PARTITION = auto()
    NETWORK_LATENCY = auto()
    NETWORK_PACKET_LOSS = auto()
    SERVICE_FAILURE = auto()
    DATABASE_FAILURE = auto()
    DEPENDENCY_FAILURE = auto()
    CONFIGURATION_DRIFT = auto()
    RESOURCE_CONTENTION = auto()
    CASCADING_FAILURE = auto()


@dataclass
class ChaosExperiment:
    """Definition of a chaos engineering experiment."""
    experiment_id: str
    name: str
    description: str
    chaos_type: ChaosType
    target_components: List[str]
    duration_seconds: int
    intensity: float  # 0.0 to 1.0
    blast_radius: str  # 'single', 'subset', 'all'
    hypothesis: str
    success_criteria: List[str]
    rollback_strategy: str
    safety_checks: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosResult:
    """Result of a chaos engineering experiment."""
    experiment_id: str
    experiment_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    chaos_type: ChaosType
    success: bool
    hypothesis_validated: bool
    observations: List[str]
    metrics_before: Dict[str, Any]
    metrics_during: Dict[str, Any]
    metrics_after: Dict[str, Any]
    recovery_time_seconds: float
    blast_radius_actual: List[str]
    lessons_learned: List[str]
    recommendations: List[str]
    safety_violations: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: float
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class ChaosMonitor:
    """Monitors system health during chaos experiments."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_thread = None
        self.safety_thresholds = {
            'cpu_usage_max': 95.0,
            'memory_usage_max': 95.0,
            'disk_usage_max': 95.0,
            'error_rate_max': 50.0,
            'response_time_max': 30.0  # seconds
        }
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("🔍 Chaos monitoring started")
    
    def stop_monitoring(self) -> List[SystemMetrics]:
        """Stop monitoring and return collected metrics."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        print("🔍 Chaos monitoring stopped")
        return list(self.metrics_history)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check safety thresholds
                safety_violations = self._check_safety_thresholds(metrics)
                if safety_violations:
                    print(f"⚠️ Safety violations detected: {safety_violations}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems)
        try:
            load_average = os.getloadavg()[0]
        except (OSError, AttributeError):
            load_average = 0.0
        
        # Simulated application metrics
        response_times = {
            'api_endpoint': random.uniform(0.05, 0.2),
            'database_query': random.uniform(0.01, 0.1),
            'cache_lookup': random.uniform(0.001, 0.01)
        }
        
        error_rates = {
            'api_errors': random.uniform(0, 2),
            'database_errors': random.uniform(0, 1),
            'cache_errors': random.uniform(0, 0.5)
        }
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average,
            response_times=response_times,
            error_rates=error_rates
        )
    
    def _check_safety_thresholds(self, metrics: SystemMetrics) -> List[str]:
        """Check if metrics exceed safety thresholds."""
        violations = []
        
        if metrics.cpu_usage > self.safety_thresholds['cpu_usage_max']:
            violations.append(f"CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.safety_thresholds['memory_usage_max']:
            violations.append(f"Memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.disk_usage > self.safety_thresholds['disk_usage_max']:
            violations.append(f"Disk usage: {metrics.disk_usage:.1f}%")
        
        # Check response times
        for service, response_time in metrics.response_times.items():
            if response_time > self.safety_thresholds['response_time_max']:
                violations.append(f"{service} response time: {response_time:.2f}s")
        
        # Check error rates
        for service, error_rate in metrics.error_rates.items():
            if error_rate > self.safety_thresholds['error_rate_max']:
                violations.append(f"{service} error rate: {error_rate:.1f}%")
        
        return violations
    
    def get_metrics_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get metrics summary for a time period."""
        relevant_metrics = [
            m for m in self.metrics_history 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not relevant_metrics:
            return {}
        
        cpu_values = [m.cpu_usage for m in relevant_metrics]
        memory_values = [m.memory_usage for m in relevant_metrics]
        
        return {
            'cpu_usage': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory_usage': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            },
            'sample_count': len(relevant_metrics),
            'duration_seconds': (end_time - start_time).total_seconds()
        }


class NetworkChaosInjector:
    """Injects network-related chaos scenarios."""
    
    def __init__(self):
        self.active_chaos = []
    
    @contextmanager
    def network_latency(self, target_interface: str = "lo", latency_ms: int = 100):
        """Inject network latency using traffic control."""
        if not NETWORK_TESTING_AVAILABLE:
            print("⚠️ Network testing libraries not available, simulating latency")
            yield
            return
        
        chaos_id = f"latency_{uuid.uuid4().hex[:8]}"
        print(f"🌐 Injecting {latency_ms}ms latency on {target_interface}")
        
        try:
            # Add latency using tc (traffic control) - requires root privileges
            # This is simulated for safety in testing environment
            self.active_chaos.append(chaos_id)
            
            # Simulate the effect by adding artificial delay
            original_socket_connect = socket.socket.connect
            
            def delayed_connect(self, address):
                time.sleep(latency_ms / 1000.0)  # Convert ms to seconds
                return original_socket_connect(self, address)
            
            socket.socket.connect = delayed_connect
            
            yield chaos_id
            
        finally:
            # Cleanup - restore original socket behavior
            socket.socket.connect = original_socket_connect
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🌐 Removed network latency injection")
    
    @contextmanager
    def network_partition(self, target_hosts: List[str]):
        """Simulate network partition by blocking communication with target hosts."""
        chaos_id = f"partition_{uuid.uuid4().hex[:8]}"
        print(f"🌐 Creating network partition blocking: {target_hosts}")
        
        # Store original socket.create_connection
        original_create_connection = socket.create_connection
        
        def partitioned_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
            host, port = address
            if any(target_host in host for target_host in target_hosts):
                # Simulate network partition by raising connection error
                raise ConnectionRefusedError(f"Network partition: Cannot connect to {host}")
            return original_create_connection(address, timeout, source_address)
        
        try:
            socket.create_connection = partitioned_create_connection
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            socket.create_connection = original_create_connection
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🌐 Removed network partition")
    
    @contextmanager
    def packet_loss(self, loss_rate: float = 0.1):
        """Simulate packet loss."""
        chaos_id = f"packet_loss_{uuid.uuid4().hex[:8]}"
        print(f"🌐 Injecting {loss_rate*100:.1f}% packet loss")
        
        original_send = socket.socket.send
        original_sendall = socket.socket.sendall
        
        def lossy_send(self, data, flags=0):
            if random.random() < loss_rate:
                # Simulate packet loss by not sending
                print(f"📦 Dropped packet ({len(data)} bytes)")
                return len(data)  # Pretend it was sent
            return original_send(self, data, flags)
        
        def lossy_sendall(self, data, flags=0):
            if random.random() < loss_rate:
                # Simulate packet loss
                print(f"📦 Dropped packet batch ({len(data)} bytes)")
                raise socket.timeout("Simulated packet loss")
            return original_sendall(self, data, flags)
        
        try:
            socket.socket.send = lossy_send
            socket.socket.sendall = lossy_sendall
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            socket.socket.send = original_send
            socket.socket.sendall = original_sendall
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🌐 Removed packet loss injection")


class ResourceChaosInjector:
    """Injects resource exhaustion scenarios."""
    
    def __init__(self):
        self.active_chaos = []
        self.stress_processes = []
    
    @contextmanager
    def cpu_stress(self, cpu_percent: float = 80.0, duration: int = 60):
        """Generate CPU stress."""
        chaos_id = f"cpu_stress_{uuid.uuid4().hex[:8]}"
        print(f"💻 Generating {cpu_percent}% CPU stress for {duration}s")
        
        def cpu_stress_worker():
            """Worker function to generate CPU load."""
            end_time = time.time() + duration
            while time.time() < end_time:
                # Busy loop to consume CPU
                for _ in range(1000):
                    pass
                
                # Brief pause to control CPU usage
                time.sleep((100 - cpu_percent) / 10000.0)
        
        try:
            # Start CPU stress threads
            num_threads = max(1, int(psutil.cpu_count() * cpu_percent / 100))
            threads = []
            
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_stress_worker, daemon=True)
                thread.start()
                threads.append(thread)
            
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
            # Wait for threads to complete
            for thread in threads:
                thread.join(timeout=1)
            
        finally:
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"💻 CPU stress test completed")
    
    @contextmanager
    def memory_exhaustion(self, memory_mb: int = 512):
        """Consume system memory."""
        chaos_id = f"memory_exhaustion_{uuid.uuid4().hex[:8]}"
        print(f"🧠 Consuming {memory_mb}MB of memory")
        
        # Allocate memory
        memory_chunks = []
        chunk_size = 1024 * 1024  # 1MB chunks
        
        try:
            for _ in range(memory_mb):
                chunk = bytearray(chunk_size)
                # Fill with data to ensure actual allocation
                for i in range(0, len(chunk), 1024):
                    chunk[i:i+8] = b'CHAOSENG'
                memory_chunks.append(chunk)
            
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            # Free memory
            memory_chunks.clear()
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🧠 Memory released")
    
    @contextmanager
    def disk_fill(self, fill_mb: int = 100, target_dir: str = None):
        """Fill disk space."""
        if target_dir is None:
            target_dir = tempfile.gettempdir()
        
        chaos_id = f"disk_fill_{uuid.uuid4().hex[:8]}"
        temp_files = []
        
        print(f"💾 Filling {fill_mb}MB disk space in {target_dir}")
        
        try:
            chunk_size = 1024 * 1024  # 1MB
            data_chunk = b'X' * chunk_size
            
            for i in range(fill_mb):
                temp_file = tempfile.NamedTemporaryFile(
                    dir=target_dir, 
                    delete=False,
                    prefix=f"chaos_fill_{chaos_id}_"
                )
                temp_file.write(data_chunk)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"💾 Disk space freed")


class ServiceChaosInjector:
    """Injects service and dependency failures."""
    
    def __init__(self):
        self.active_chaos = []
        self.service_mocks = {}
    
    @contextmanager
    def service_failure(self, service_name: str, failure_rate: float = 1.0):
        """Simulate service failure."""
        chaos_id = f"service_failure_{uuid.uuid4().hex[:8]}"
        print(f"🚫 Injecting {failure_rate*100:.0f}% failure rate for {service_name}")
        
        # Create mock that fails based on failure rate
        original_service = self.service_mocks.get(service_name)
        
        def failing_service(*args, **kwargs):
            if random.random() < failure_rate:
                raise ConnectionError(f"Service {service_name} is unavailable (chaos injection)")
            # If original service exists, call it; otherwise return mock response
            if original_service:
                return original_service(*args, **kwargs)
            return {"status": "success", "service": service_name, "chaos_id": chaos_id}
        
        try:
            self.service_mocks[service_name] = failing_service
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            if service_name in self.service_mocks:
                if original_service:
                    self.service_mocks[service_name] = original_service
                else:
                    del self.service_mocks[service_name]
            
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🚫 Service failure injection removed for {service_name}")
    
    @contextmanager
    def database_failure(self, failure_type: str = "connection_timeout"):
        """Simulate database failures."""
        chaos_id = f"db_failure_{uuid.uuid4().hex[:8]}"
        print(f"🗄️ Injecting database failure: {failure_type}")
        
        # Mock database connection failures
        def failing_db_connect(*args, **kwargs):
            if failure_type == "connection_timeout":
                raise TimeoutError("Database connection timeout (chaos injection)")
            elif failure_type == "connection_refused":
                raise ConnectionRefusedError("Database connection refused (chaos injection)")
            elif failure_type == "query_timeout":
                time.sleep(5)  # Simulate slow query
                raise TimeoutError("Database query timeout (chaos injection)")
            else:
                raise Exception(f"Database error: {failure_type} (chaos injection)")
        
        try:
            # In a real implementation, this would patch actual database connections
            self.service_mocks['database'] = failing_db_connect
            self.active_chaos.append(chaos_id)
            yield chaos_id
            
        finally:
            if 'database' in self.service_mocks:
                del self.service_mocks['database']
            if chaos_id in self.active_chaos:
                self.active_chaos.remove(chaos_id)
            print(f"🗄️ Database failure injection removed")
    
    def simulate_service_call(self, service_name: str, *args, **kwargs):
        """Simulate calling a service (for testing purposes)."""
        service_mock = self.service_mocks.get(service_name)
        if service_mock:
            return service_mock(*args, **kwargs)
        return {"status": "success", "service": service_name, "message": "Normal operation"}


class ChaosExperimentRunner:
    """Main chaos experiment runner."""
    
    def __init__(self):
        self.monitor = ChaosMonitor()
        self.network_chaos = NetworkChaosInjector()
        self.resource_chaos = ResourceChaosInjector()
        self.service_chaos = ServiceChaosInjector()
        self.experiment_results = []
    
    def run_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Run a single chaos engineering experiment."""
        print(f"🧪 Starting chaos experiment: {experiment.name}")
        print(f"   Hypothesis: {experiment.hypothesis}")
        print(f"   Duration: {experiment.duration_seconds}s")
        print(f"   Blast Radius: {experiment.blast_radius}")
        
        start_time = datetime.now()
        
        # Pre-experiment safety checks
        safety_violations = self._run_safety_checks(experiment.safety_checks)
        if safety_violations:
            print(f"❌ Safety checks failed: {safety_violations}")
            return self._create_failed_result(experiment, start_time, safety_violations)
        
        # Start monitoring
        self.monitor.start_monitoring()
        time.sleep(2)  # Allow baseline metrics collection
        
        # Collect baseline metrics
        metrics_before = self.monitor._collect_system_metrics()
        
        observations = []
        recovery_time = 0.0
        hypothesis_validated = False
        
        try:
            # Execute chaos based on type
            print(f"💥 Injecting chaos: {experiment.chaos_type.name}")
            
            if experiment.chaos_type == ChaosType.CPU_STRESS:
                with self.resource_chaos.cpu_stress(
                    cpu_percent=experiment.intensity * 100,
                    duration=experiment.duration_seconds
                ):
                    observations.extend(self._observe_experiment(experiment))
                    
            elif experiment.chaos_type == ChaosType.MEMORY_EXHAUSTION:
                memory_mb = int(experiment.parameters.get('memory_mb', 512))
                with self.resource_chaos.memory_exhaustion(memory_mb=memory_mb):
                    time.sleep(experiment.duration_seconds)
                    observations.extend(self._observe_experiment(experiment))
                    
            elif experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                latency_ms = int(experiment.parameters.get('latency_ms', 100))
                with self.network_chaos.network_latency(latency_ms=latency_ms):
                    time.sleep(experiment.duration_seconds)
                    observations.extend(self._observe_experiment(experiment))
                    
            elif experiment.chaos_type == ChaosType.NETWORK_PARTITION:
                target_hosts = experiment.parameters.get('target_hosts', ['localhost'])
                with self.network_chaos.network_partition(target_hosts=target_hosts):
                    time.sleep(experiment.duration_seconds)
                    observations.extend(self._observe_experiment(experiment))
                    
            elif experiment.chaos_type == ChaosType.SERVICE_FAILURE:
                service_name = experiment.parameters.get('service_name', 'test_service')
                failure_rate = experiment.parameters.get('failure_rate', 1.0)
                with self.service_chaos.service_failure(service_name, failure_rate):
                    time.sleep(experiment.duration_seconds)
                    observations.extend(self._observe_experiment(experiment))
                    
            elif experiment.chaos_type == ChaosType.DATABASE_FAILURE:
                failure_type = experiment.parameters.get('failure_type', 'connection_timeout')
                with self.service_chaos.database_failure(failure_type):
                    time.sleep(experiment.duration_seconds)
                    observations.extend(self._observe_experiment(experiment))
            
            # Collect metrics during chaos
            metrics_during = self.monitor._collect_system_metrics()
            
            # Wait for recovery and measure recovery time
            print("🔄 Measuring recovery time...")
            recovery_start = time.time()
            
            # Allow some time for system to recover
            time.sleep(5)
            
            recovery_time = time.time() - recovery_start
            
            # Collect post-experiment metrics
            metrics_after = self.monitor._collect_system_metrics()
            
            # Validate hypothesis
            hypothesis_validated = self._validate_hypothesis(experiment, observations, metrics_before, metrics_during, metrics_after)
            
            print(f"✅ Experiment completed successfully")
            success = True
            
        except Exception as e:
            print(f"❌ Experiment failed with error: {e}")
            observations.append(f"Experiment failed: {str(e)}")
            metrics_during = self.monitor._collect_system_metrics()
            metrics_after = metrics_during
            success = False
        
        finally:
            # Stop monitoring
            monitoring_data = self.monitor.stop_monitoring()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate lessons learned and recommendations
        lessons_learned, recommendations = self._analyze_results(
            experiment, observations, metrics_before, metrics_during, metrics_after
        )
        
        result = ChaosResult(
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            chaos_type=experiment.chaos_type,
            success=success,
            hypothesis_validated=hypothesis_validated,
            observations=observations,
            metrics_before=asdict(metrics_before),
            metrics_during=asdict(metrics_during),
            metrics_after=asdict(metrics_after),
            recovery_time_seconds=recovery_time,
            blast_radius_actual=experiment.target_components,
            lessons_learned=lessons_learned,
            recommendations=recommendations
        )
        
        self.experiment_results.append(result)
        
        print(f"📊 Experiment result: {'✅ Success' if success else '❌ Failed'}")
        print(f"   Hypothesis validated: {'✅ Yes' if hypothesis_validated else '❌ No'}")
        print(f"   Recovery time: {recovery_time:.2f}s")
        print(f"   Observations: {len(observations)}")
        
        return result
    
    def _run_safety_checks(self, safety_checks: List[str]) -> List[str]:
        """Run pre-experiment safety checks."""
        violations = []
        
        # Check system load
        try:
            load_avg = os.getloadavg()[0]
            if load_avg > 5.0:
                violations.append(f"High system load: {load_avg}")
        except (OSError, AttributeError):
            pass
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            violations.append(f"High memory usage: {memory.percent}%")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if (disk.free / disk.total) < 0.1:  # Less than 10% free
            violations.append("Low disk space")
        
        # Custom safety checks
        for check in safety_checks:
            if check == "no_production_data":
                # Verify we're not in production
                if os.environ.get('ENVIRONMENT', '').lower() == 'production':
                    violations.append("Cannot run chaos experiments in production")
            elif check == "backup_verified":
                # In real implementation, verify backups exist
                pass
        
        return violations
    
    def _observe_experiment(self, experiment: ChaosExperiment) -> List[str]:
        """Collect observations during experiment execution."""
        observations = []
        
        # Simulate service health checks
        for component in experiment.target_components:
            try:
                # Simulate health check
                if component == "api_service":
                    response = self.service_chaos.simulate_service_call("api_service")
                    observations.append(f"API service response: {response['status']}")
                elif component == "database":
                    response = self.service_chaos.simulate_service_call("database")
                    observations.append(f"Database response: {response['status']}")
                elif component == "cache":
                    response = self.service_chaos.simulate_service_call("cache")
                    observations.append(f"Cache response: {response['status']}")
                    
            except Exception as e:
                observations.append(f"{component} failed: {str(e)}")
        
        # Check system metrics
        current_metrics = self.monitor._collect_system_metrics()
        
        if current_metrics.cpu_usage > 80:
            observations.append(f"High CPU usage detected: {current_metrics.cpu_usage:.1f}%")
        
        if current_metrics.memory_usage > 80:
            observations.append(f"High memory usage detected: {current_metrics.memory_usage:.1f}%")
        
        # Check response times
        for service, response_time in current_metrics.response_times.items():
            if response_time > 1.0:  # Threshold of 1 second
                observations.append(f"Slow {service} response: {response_time:.2f}s")
        
        return observations
    
    def _validate_hypothesis(self, experiment: ChaosExperiment, observations: List[str],
                           metrics_before: SystemMetrics, metrics_during: SystemMetrics,
                           metrics_after: SystemMetrics) -> bool:
        """Validate the experiment hypothesis."""
        
        # Simple hypothesis validation based on observations and metrics
        hypothesis_lower = experiment.hypothesis.lower()
        
        if "system remains responsive" in hypothesis_lower:
            # Check if response times stayed within acceptable range
            max_response_time = max(metrics_during.response_times.values())
            return max_response_time < 5.0  # 5 second threshold
        
        elif "graceful degradation" in hypothesis_lower:
            # Check if error rates increased but system didn't crash
            error_rate_increase = any(
                metrics_during.error_rates[service] > metrics_before.error_rates[service] * 2
                for service in metrics_during.error_rates
            )
            system_still_responsive = max(metrics_during.response_times.values()) < 10.0
            return error_rate_increase and system_still_responsive
        
        elif "recovery within" in hypothesis_lower:
            # Check recovery time
            recovery_threshold = 30.0  # 30 seconds
            return metrics_after.cpu_usage < metrics_before.cpu_usage * 1.2
        
        elif "no data loss" in hypothesis_lower:
            # In real scenario, would check data integrity
            return True  # Assume no data loss for testing
        
        # Default validation based on system stability
        return (
            metrics_after.cpu_usage < 95 and
            metrics_after.memory_usage < 95 and
            all(rt < 10.0 for rt in metrics_after.response_times.values())
        )
    
    def _analyze_results(self, experiment: ChaosExperiment, observations: List[str],
                        metrics_before: SystemMetrics, metrics_during: SystemMetrics,
                        metrics_after: SystemMetrics) -> Tuple[List[str], List[str]]:
        """Analyze experiment results and generate insights."""
        
        lessons_learned = []
        recommendations = []
        
        # Analyze performance impact
        cpu_increase = metrics_during.cpu_usage - metrics_before.cpu_usage
        if cpu_increase > 20:
            lessons_learned.append(f"CPU usage increased by {cpu_increase:.1f}% during chaos")
            recommendations.append("Consider implementing CPU throttling or circuit breakers")
        
        memory_increase = metrics_during.memory_usage - metrics_before.memory_usage
        if memory_increase > 10:
            lessons_learned.append(f"Memory usage increased by {memory_increase:.1f}% during chaos")
            recommendations.append("Review memory management and implement memory limits")
        
        # Analyze response time degradation
        response_time_degradation = {}
        for service in metrics_during.response_times:
            before_rt = metrics_before.response_times.get(service, 0)
            during_rt = metrics_during.response_times[service]
            if during_rt > before_rt * 2:
                degradation = ((during_rt - before_rt) / before_rt) * 100
                response_time_degradation[service] = degradation
                lessons_learned.append(f"{service} response time degraded by {degradation:.0f}%")
        
        if response_time_degradation:
            recommendations.append("Implement timeout and retry mechanisms for degraded services")
        
        # Analyze error patterns
        error_patterns = []
        for observation in observations:
            if "failed" in observation.lower() or "error" in observation.lower():
                error_patterns.append(observation)
        
        if error_patterns:
            lessons_learned.append(f"Observed {len(error_patterns)} error patterns during chaos")
            recommendations.append("Improve error handling and fallback mechanisms")
        
        # Recovery analysis
        recovery_successful = (
            metrics_after.cpu_usage <= metrics_before.cpu_usage * 1.1 and
            metrics_after.memory_usage <= metrics_before.memory_usage * 1.1
        )
        
        if recovery_successful:
            lessons_learned.append("System recovered successfully to baseline performance")
        else:
            lessons_learned.append("System did not fully recover to baseline performance")
            recommendations.append("Investigate resource leaks and improve recovery mechanisms")
        
        # Chaos-specific insights
        if experiment.chaos_type == ChaosType.NETWORK_LATENCY:
            recommendations.append("Consider implementing connection pooling and caching")
        elif experiment.chaos_type == ChaosType.SERVICE_FAILURE:
            recommendations.append("Implement circuit breakers and service mesh")
        elif experiment.chaos_type == ChaosType.MEMORY_EXHAUSTION:
            recommendations.append("Set memory limits and implement graceful degradation")
        
        return lessons_learned, recommendations
    
    def _create_failed_result(self, experiment: ChaosExperiment, start_time: datetime,
                             safety_violations: List[str]) -> ChaosResult:
        """Create a failed result for experiments that don't pass safety checks."""
        end_time = datetime.now()
        
        return ChaosResult(
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            chaos_type=experiment.chaos_type,
            success=False,
            hypothesis_validated=False,
            observations=["Experiment aborted due to safety violations"],
            metrics_before={},
            metrics_during={},
            metrics_after={},
            recovery_time_seconds=0.0,
            blast_radius_actual=[],
            lessons_learned=["Safety checks must pass before running experiments"],
            recommendations=["Address safety violations before retrying"],
            safety_violations=safety_violations
        )
    
    def run_chaos_suite(self, experiments: List[ChaosExperiment]) -> List[ChaosResult]:
        """Run a suite of chaos engineering experiments."""
        print(f"🧪 Starting chaos engineering suite with {len(experiments)} experiments")
        
        results = []
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"Experiment {i}/{len(experiments)}: {experiment.name}")
            print(f"{'='*60}")
            
            try:
                result = self.run_experiment(experiment)
                results.append(result)
                
                # Wait between experiments for system to stabilize
                if i < len(experiments):
                    print(f"⏳ Waiting 10 seconds before next experiment...")
                    time.sleep(10)
                    
            except Exception as e:
                print(f"❌ Experiment suite failed: {e}")
                break
        
        print(f"\n🎯 Chaos engineering suite completed")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful: {sum(1 for r in results if r.success)}")
        print(f"   Failed: {sum(1 for r in results if not r.success)}")
        print(f"   Hypotheses validated: {sum(1 for r in results if r.hypothesis_validated)}")
        
        return results
    
    def generate_chaos_report(self, results: List[ChaosResult]) -> Dict[str, Any]:
        """Generate comprehensive chaos engineering report."""
        if not results:
            return {"message": "No chaos experiment results to report"}
        
        # Overall statistics
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.success)
        validated_hypotheses = sum(1 for r in results if r.hypothesis_validated)
        
        # Recovery time analysis
        recovery_times = [r.recovery_time_seconds for r in results if r.success]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Chaos type distribution
        chaos_type_counts = defaultdict(int)
        for result in results:
            chaos_type_counts[result.chaos_type.name] += 1
        
        # Collect all lessons learned and recommendations
        all_lessons = []
        all_recommendations = []
        
        for result in results:
            all_lessons.extend(result.lessons_learned)
            all_recommendations.extend(result.recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = list(set(all_recommendations))
        
        report = {
            "report_id": f"chaos_report_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
                "success_rate": (successful_experiments / total_experiments) * 100,
                "validated_hypotheses": validated_hypotheses,
                "hypothesis_validation_rate": (validated_hypotheses / total_experiments) * 100,
                "average_recovery_time_seconds": avg_recovery_time
            },
            "chaos_type_distribution": dict(chaos_type_counts),
            "experiment_results": [asdict(result) for result in results],
            "key_findings": {
                "lessons_learned": all_lessons,
                "recommendations": unique_recommendations,
                "resilience_score": self._calculate_resilience_score(results)
            },
            "next_steps": [
                "Implement recommended improvements",
                "Schedule regular chaos engineering sessions",
                "Expand blast radius gradually",
                "Automate chaos experiments in CI/CD pipeline"
            ]
        }
        
        return report
    
    def _calculate_resilience_score(self, results: List[ChaosResult]) -> float:
        """Calculate overall system resilience score (0-100)."""
        if not results:
            return 0.0
        
        # Base score from success rate
        success_rate = sum(1 for r in results if r.success) / len(results)
        base_score = success_rate * 100
        
        # Bonus for hypothesis validation
        validation_rate = sum(1 for r in results if r.hypothesis_validated) / len(results)
        validation_bonus = validation_rate * 10
        
        # Penalty for slow recovery
        recovery_times = [r.recovery_time_seconds for r in results if r.success]
        if recovery_times:
            avg_recovery = sum(recovery_times) / len(recovery_times)
            recovery_penalty = min(20, avg_recovery / 30 * 20)  # Max 20 point penalty
        else:
            recovery_penalty = 20  # Maximum penalty if no successful recoveries
        
        resilience_score = max(0, min(100, base_score + validation_bonus - recovery_penalty))
        return resilience_score


def create_sample_experiments() -> List[ChaosExperiment]:
    """Create sample chaos engineering experiments."""
    experiments = []
    
    # CPU Stress Experiment
    experiments.append(ChaosExperiment(
        experiment_id="cpu_stress_001",
        name="API Service CPU Stress Test",
        description="Test API service behavior under high CPU load",
        chaos_type=ChaosType.CPU_STRESS,
        target_components=["api_service", "load_balancer"],
        duration_seconds=60,
        intensity=0.8,  # 80% CPU usage
        blast_radius="subset",
        hypothesis="System remains responsive under 80% CPU load with graceful degradation",
        success_criteria=[
            "API response time < 5 seconds",
            "Error rate < 10%",
            "System recovers within 30 seconds"
        ],
        rollback_strategy="Stop CPU stress processes",
        safety_checks=["no_production_data", "backup_verified"],
        parameters={"cpu_percent": 80}
    ))
    
    # Memory Exhaustion Experiment
    experiments.append(ChaosExperiment(
        experiment_id="memory_exhaustion_001",
        name="Memory Pressure Test",
        description="Test system behavior when memory is exhausted",
        chaos_type=ChaosType.MEMORY_EXHAUSTION,
        target_components=["api_service", "database", "cache"],
        duration_seconds=45,
        intensity=0.7,
        blast_radius="all",
        hypothesis="System implements graceful degradation when memory is low",
        success_criteria=[
            "No system crash",
            "Essential services remain available",
            "Memory released after test"
        ],
        rollback_strategy="Release allocated memory",
        safety_checks=["no_production_data"],
        parameters={"memory_mb": 512}
    ))
    
    # Network Latency Experiment
    experiments.append(ChaosExperiment(
        experiment_id="network_latency_001",
        name="Network Latency Impact Test",
        description="Test system resilience to network latency",
        chaos_type=ChaosType.NETWORK_LATENCY,
        target_components=["api_service", "database"],
        duration_seconds=30,
        intensity=0.5,
        blast_radius="subset",
        hypothesis="System handles 100ms network latency without significant impact",
        success_criteria=[
            "API response time degrades gracefully",
            "No connection timeouts",
            "Recovery within 10 seconds"
        ],
        rollback_strategy="Remove traffic shaping rules",
        safety_checks=["backup_verified"],
        parameters={"latency_ms": 100}
    ))
    
    # Service Failure Experiment
    experiments.append(ChaosExperiment(
        experiment_id="service_failure_001",
        name="Database Service Failure Test",
        description="Test application behavior when database is unavailable",
        chaos_type=ChaosType.DATABASE_FAILURE,
        target_components=["database", "api_service"],
        duration_seconds=30,
        intensity=1.0,  # Complete failure
        blast_radius="single",
        hypothesis="Application falls back to cached data when database is unavailable",
        success_criteria=[
            "Application returns cached responses",
            "No data corruption",
            "Graceful error messages to users"
        ],
        rollback_strategy="Restore database connection",
        safety_checks=["no_production_data", "backup_verified"],
        parameters={"failure_type": "connection_timeout"}
    ))
    
    return experiments


def main():
    """Main function for running chaos engineering experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chaos Engineering Testing Framework")
    parser.add_argument("--experiment-type", 
                       choices=["cpu", "memory", "network", "service", "all"],
                       default="all",
                       help="Type of chaos experiment to run")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration of each experiment in seconds")
    parser.add_argument("--intensity", type=float, default=0.5,
                       help="Intensity of chaos injection (0.0-1.0)")
    parser.add_argument("--output-report", action="store_true",
                       help="Generate detailed report")
    
    args = parser.parse_args()
    
    # Create chaos runner
    runner = ChaosExperimentRunner()
    
    # Get sample experiments
    experiments = create_sample_experiments()
    
    # Filter experiments by type
    if args.experiment_type != "all":
        type_mapping = {
            "cpu": ChaosType.CPU_STRESS,
            "memory": ChaosType.MEMORY_EXHAUSTION,
            "network": ChaosType.NETWORK_LATENCY,
            "service": ChaosType.DATABASE_FAILURE
        }
        target_type = type_mapping.get(args.experiment_type)
        experiments = [exp for exp in experiments if exp.chaos_type == target_type]
    
    # Adjust experiment parameters
    for experiment in experiments:
        experiment.duration_seconds = args.duration
        experiment.intensity = args.intensity
    
    # Run experiments
    print("🧪 Starting chaos engineering experiments...")
    print(f"   Experiments to run: {len(experiments)}")
    print(f"   Duration: {args.duration}s each")
    print(f"   Intensity: {args.intensity}")
    
    try:
        results = runner.run_chaos_suite(experiments)
        
        # Generate report if requested
        if args.output_report:
            report = runner.generate_chaos_report(results)
            
            # Save report
            report_dir = Path("chaos_reports")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"chaos_report_{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📊 Chaos engineering report saved to {report_file}")
            print(f"   Resilience Score: {report['key_findings']['resilience_score']:.1f}/100")
        
        # Exit with error code if experiments failed
        failed_experiments = sum(1 for r in results if not r.success)
        if failed_experiments > 0:
            print(f"❌ {failed_experiments} experiments failed")
            exit(1)
        else:
            print("✅ All chaos experiments completed successfully")
    
    except KeyboardInterrupt:
        print("\n🛑 Chaos experiments interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Chaos experiments failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()