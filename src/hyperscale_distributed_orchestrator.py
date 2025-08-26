"""
HyperScale Distributed Orchestrator for MLOps Platform.

Advanced distributed computing, auto-scaling, load balancing, and resource optimization
for production ML workloads at enterprise scale. Supports multi-cloud, multi-region
deployments with intelligent workload distribution.
"""

import asyncio
import time
import logging
import threading
import json
import os
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import socket
import uuid

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .validation import safe_write_json, safe_read_json
from .production_monitoring_suite import get_monitoring_suite

logger = get_logger(__name__)
metrics = get_metrics_collector()


class WorkloadType(Enum):
    """Types of ML workloads."""
    TRAINING = "training"
    INFERENCE = "inference" 
    PREPROCESSING = "preprocessing"
    BATCH_PREDICTION = "batch_prediction"
    REAL_TIME_PREDICTION = "real_time_prediction"
    MODEL_EVALUATION = "model_evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    CUSTOM = "custom"


@dataclass
class ResourceRequirements:
    """Resource requirements for workloads."""
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 10.0
    network_bandwidth_mbps: float = 100.0
    max_execution_time_minutes: int = 60
    priority: int = 5  # 1-10, higher = more important
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'storage_gb': self.storage_gb,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'max_execution_time_minutes': self.max_execution_time_minutes,
            'priority': self.priority
        }


@dataclass
class WorkloadRequest:
    """Distributed workload request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workload_type: WorkloadType = WorkloadType.INFERENCE
    function: Optional[Callable] = None
    function_name: str = ""
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    deadline: Optional[float] = None  # Unix timestamp
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'workload_type': self.workload_type.value,
            'function_name': self.function_name,
            'resource_requirements': self.resource_requirements.to_dict(),
            'deadline': self.deadline,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'metadata': self.metadata
        }


@dataclass
class ComputeNode:
    """Distributed compute node information."""
    id: str
    hostname: str
    ip_address: str
    port: int
    available_resources: ResourceRequirements
    total_resources: ResourceRequirements
    current_workloads: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    health_status: str = "healthy"
    location: str = "default"
    node_type: str = "standard"
    cost_per_hour: float = 1.0
    
    def get_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization."""
        if self.total_resources.cpu_cores == 0:
            return {'cpu': 0, 'memory': 0, 'gpu': 0, 'storage': 0}
            
        cpu_util = 1.0 - (self.available_resources.cpu_cores / self.total_resources.cpu_cores)
        memory_util = 1.0 - (self.available_resources.memory_gb / self.total_resources.memory_gb)
        gpu_util = 1.0 - (self.available_resources.gpu_count / max(self.total_resources.gpu_count, 1))
        storage_util = 1.0 - (self.available_resources.storage_gb / self.total_resources.storage_gb)
        
        return {
            'cpu': max(0, min(1, cpu_util)),
            'memory': max(0, min(1, memory_util)), 
            'gpu': max(0, min(1, gpu_util)),
            'storage': max(0, min(1, storage_util))
        }
        
    def can_execute_workload(self, requirements: ResourceRequirements) -> bool:
        """Check if node can execute workload."""
        return (
            self.available_resources.cpu_cores >= requirements.cpu_cores and
            self.available_resources.memory_gb >= requirements.memory_gb and
            self.available_resources.gpu_count >= requirements.gpu_count and
            self.available_resources.storage_gb >= requirements.storage_gb and
            self.health_status == "healthy"
        )


class LoadBalancer:
    """
    Intelligent load balancer for distributed workloads.
    """
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.node_weights = {}
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.last_selection = 0
        
    def select_node(self, nodes: List[ComputeNode], requirements: ResourceRequirements) -> Optional[ComputeNode]:
        """Select optimal node for workload execution."""
        # Filter nodes that can handle the workload
        capable_nodes = [node for node in nodes if node.can_execute_workload(requirements)]
        
        if not capable_nodes:
            return None
            
        if self.strategy == "round_robin":
            return self._round_robin_selection(capable_nodes)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(capable_nodes)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(capable_nodes)
        elif self.strategy == "resource_aware":
            return self._resource_aware_selection(capable_nodes, requirements)
        elif self.strategy == "cost_optimized":
            return self._cost_optimized_selection(capable_nodes, requirements)
        else:
            return capable_nodes[0]  # Fallback
            
    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Simple round-robin selection."""
        self.last_selection = (self.last_selection + 1) % len(nodes)
        return nodes[self.last_selection]
        
    def _weighted_round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Weighted round-robin based on node capacity."""
        weights = []
        for node in nodes:
            utilization = node.get_utilization()
            # Higher weight for less utilized nodes
            weight = 1.0 - np.mean(list(utilization.values()))
            weights.append(max(0.1, weight))
            
        # Weighted random selection
        total_weight = sum(weights)
        r = np.random.uniform(0, total_weight)
        
        current_weight = 0
        for i, weight in enumerate(weights):
            current_weight += weight
            if r <= current_weight:
                return nodes[i]
                
        return nodes[-1]
        
    def _least_connections_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node with least active workloads."""
        return min(nodes, key=lambda n: len(n.current_workloads))
        
    def _resource_aware_selection(self, nodes: List[ComputeNode], requirements: ResourceRequirements) -> ComputeNode:
        """Select node based on resource requirements and availability."""
        scores = []
        
        for node in nodes:
            # Calculate resource match score
            cpu_ratio = node.available_resources.cpu_cores / requirements.cpu_cores
            memory_ratio = node.available_resources.memory_gb / requirements.memory_gb
            gpu_ratio = (node.available_resources.gpu_count / max(requirements.gpu_count, 1)) if requirements.gpu_count > 0 else 1.0
            
            # Prefer nodes with sufficient but not excessive resources
            cpu_score = 1.0 / (1.0 + abs(cpu_ratio - 2.0))  # Prefer 2x requirements
            memory_score = 1.0 / (1.0 + abs(memory_ratio - 2.0))
            gpu_score = 1.0 / (1.0 + abs(gpu_ratio - 1.5)) if requirements.gpu_count > 0 else 1.0
            
            utilization = node.get_utilization()
            util_score = 1.0 - np.mean(list(utilization.values()))
            
            total_score = np.mean([cpu_score, memory_score, gpu_score, util_score])
            scores.append(total_score)
            
        best_idx = np.argmax(scores)
        return nodes[best_idx]
        
    def _cost_optimized_selection(self, nodes: List[ComputeNode], requirements: ResourceRequirements) -> ComputeNode:
        """Select most cost-effective node."""
        cost_scores = []
        
        for node in nodes:
            # Calculate cost efficiency
            utilization = node.get_utilization()
            efficiency = 1.0 - np.mean(list(utilization.values()))
            
            # Cost per unit of resource
            cost_score = efficiency / node.cost_per_hour
            cost_scores.append(cost_score)
            
        best_idx = np.argmax(cost_scores)
        return nodes[best_idx]
        
    def update_node_performance(self, node_id: str, response_time: float, success: bool):
        """Update node performance metrics."""
        self.request_counts[node_id] += 1
        
        if node_id not in self.response_times:
            self.response_times[node_id] = deque(maxlen=100)
            
        self.response_times[node_id].append(response_time if success else float('inf'))


class AutoScaler:
    """
    Predictive auto-scaling system for compute resources.
    """
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.REACTIVE):
        self.strategy = strategy
        self.scaling_history = deque(maxlen=1000)
        self.workload_predictions = {}
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.min_nodes = 1
        self.max_nodes = 100
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0
        
    def should_scale(self, nodes: List[ComputeNode], workload_queue_size: int) -> Tuple[str, int]:
        """Determine if scaling action is needed."""
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return "none", 0
            
        if self.strategy == ScalingStrategy.REACTIVE:
            return self._reactive_scaling_decision(nodes, workload_queue_size)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._predictive_scaling_decision(nodes, workload_queue_size)
        elif self.strategy == ScalingStrategy.SCHEDULED:
            return self._scheduled_scaling_decision(nodes, workload_queue_size)
        else:
            return "none", 0
            
    def _reactive_scaling_decision(self, nodes: List[ComputeNode], queue_size: int) -> Tuple[str, int]:
        """Reactive scaling based on current load."""
        if not nodes:
            return "scale_up", max(1, min(queue_size // 10, 5))
            
        # Calculate average utilization
        total_utilization = 0
        healthy_nodes = [n for n in nodes if n.health_status == "healthy"]
        
        if not healthy_nodes:
            return "scale_up", 1
            
        for node in healthy_nodes:
            utilization = node.get_utilization()
            total_utilization += np.mean(list(utilization.values()))
            
        avg_utilization = total_utilization / len(healthy_nodes)
        
        # Scale up if high utilization or large queue
        if avg_utilization > self.scale_up_threshold or queue_size > len(healthy_nodes) * 5:
            scale_amount = max(1, min(queue_size // 10, 5))
            if len(nodes) + scale_amount <= self.max_nodes:
                return "scale_up", scale_amount
                
        # Scale down if low utilization and small queue
        elif avg_utilization < self.scale_down_threshold and queue_size < len(healthy_nodes) * 2:
            scale_amount = max(1, min(len(healthy_nodes) // 3, 3))
            if len(nodes) - scale_amount >= self.min_nodes:
                return "scale_down", scale_amount
                
        return "none", 0
        
    def _predictive_scaling_decision(self, nodes: List[ComputeNode], queue_size: int) -> Tuple[str, int]:
        """Predictive scaling based on workload forecasts."""
        # This would use historical data to predict future load
        # For now, implement a simple trend-based approach
        
        if len(self.scaling_history) < 10:
            return self._reactive_scaling_decision(nodes, queue_size)
            
        # Analyze recent scaling history
        recent_utilizations = [entry['avg_utilization'] for entry in list(self.scaling_history)[-10:]]
        trend = np.polyfit(range(len(recent_utilizations)), recent_utilizations, 1)[0]
        
        # If utilization is trending up, scale proactively
        if trend > 0.05:  # 5% increase per measurement
            return "scale_up", 2
        elif trend < -0.05:  # 5% decrease per measurement
            return "scale_down", 1
            
        return self._reactive_scaling_decision(nodes, queue_size)
        
    def _scheduled_scaling_decision(self, nodes: List[ComputeNode], queue_size: int) -> Tuple[str, int]:
        """Scheduled scaling based on time patterns."""
        current_hour = datetime.now().hour
        
        # Example: Scale up during business hours
        if 9 <= current_hour <= 17:  # Business hours
            target_nodes = max(self.min_nodes, len(nodes) + 2)
        else:
            target_nodes = self.min_nodes
            
        current_nodes = len([n for n in nodes if n.health_status == "healthy"])
        
        if current_nodes < target_nodes:
            return "scale_up", target_nodes - current_nodes
        elif current_nodes > target_nodes:
            return "scale_down", current_nodes - target_nodes
            
        return "none", 0
        
    def record_scaling_event(self, action: str, nodes_changed: int, nodes_before: int, nodes_after: int, 
                            avg_utilization: float):
        """Record scaling event for analysis."""
        event = {
            'timestamp': time.time(),
            'action': action,
            'nodes_changed': nodes_changed,
            'nodes_before': nodes_before,
            'nodes_after': nodes_after,
            'avg_utilization': avg_utilization
        }
        
        self.scaling_history.append(event)
        self.last_scaling_action = time.time()
        
        logger.info(f"Scaling event: {action} {nodes_changed} nodes. Before: {nodes_before}, After: {nodes_after}")


class WorkloadScheduler:
    """
    Advanced workload scheduler with priority queues and deadline-aware scheduling.
    """
    
    def __init__(self):
        self.workload_queues = {
            1: deque(),  # Critical priority
            2: deque(),  # High priority
            3: deque(),  # Normal priority
            4: deque(),  # Low priority
            5: deque()   # Background priority
        }
        self.executing_workloads = {}
        self.completed_workloads = deque(maxlen=1000)
        self.failed_workloads = deque(maxlen=1000)
        self.lock = threading.RLock()
        
    def submit_workload(self, workload: WorkloadRequest):
        """Submit workload for execution."""
        with self.lock:
            priority = workload.resource_requirements.priority
            priority_queue = min(5, max(1, priority))  # Ensure priority is 1-5
            
            self.workload_queues[priority_queue].append(workload)
            
        logger.info(f"Workload {workload.id} submitted with priority {priority_queue}")
        metrics.increment('workload_submitted', {
            'workload_type': workload.workload_type.value,
            'priority': str(priority_queue)
        })
        
    def get_next_workload(self, node_capabilities: ResourceRequirements) -> Optional[WorkloadRequest]:
        """Get next workload for execution based on priority and node capabilities."""
        with self.lock:
            # Check queues in priority order
            for priority in range(1, 6):
                queue = self.workload_queues[priority]
                
                # Look for workload that matches node capabilities
                for i, workload in enumerate(queue):
                    # Check deadline first
                    if workload.deadline and time.time() > workload.deadline:
                        # Move expired workload to failed
                        expired = queue[i]
                        del queue[i]
                        self.failed_workloads.append(expired)
                        logger.warning(f"Workload {expired.id} expired")
                        continue
                        
                    # Check resource requirements
                    req = workload.resource_requirements
                    if (node_capabilities.cpu_cores >= req.cpu_cores and
                        node_capabilities.memory_gb >= req.memory_gb and
                        node_capabilities.gpu_count >= req.gpu_count):
                        
                        # Remove from queue and return
                        selected_workload = queue[i]
                        del queue[i]
                        self.executing_workloads[selected_workload.id] = selected_workload
                        return selected_workload
                        
            return None
            
    def complete_workload(self, workload_id: str, success: bool, result: Any = None, error: str = ""):
        """Mark workload as completed."""
        with self.lock:
            if workload_id in self.executing_workloads:
                workload = self.executing_workloads.pop(workload_id)
                workload.metadata['completed_at'] = time.time()
                workload.metadata['success'] = success
                workload.metadata['result'] = result
                workload.metadata['error'] = error
                
                if success:
                    self.completed_workloads.append(workload)
                    metrics.increment('workload_completed', {
                        'workload_type': workload.workload_type.value
                    })
                else:
                    if workload.retry_count < workload.max_retries:
                        # Retry workload
                        workload.retry_count += 1
                        self.submit_workload(workload)
                        logger.info(f"Retrying workload {workload_id} (attempt {workload.retry_count})")
                    else:
                        self.failed_workloads.append(workload)
                        metrics.increment('workload_failed', {
                            'workload_type': workload.workload_type.value
                        })
                        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.lock:
            return {
                'queued_by_priority': {str(p): len(q) for p, q in self.workload_queues.items()},
                'total_queued': sum(len(q) for q in self.workload_queues.values()),
                'executing': len(self.executing_workloads),
                'completed': len(self.completed_workloads),
                'failed': len(self.failed_workloads)
            }


class DistributedOrchestrator:
    """
    Main distributed orchestrator coordinating all scaling and distribution components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.load_balancer = LoadBalancer(self.config.get('load_balancing_strategy', 'resource_aware'))
        self.auto_scaler = AutoScaler(ScalingStrategy(self.config.get('scaling_strategy', 'reactive')))
        self.scheduler = WorkloadScheduler()
        self.is_running = False
        self.node_discovery_port = self.config.get('node_discovery_port', 8080)
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)
        self.executor_pool = ThreadPoolExecutor(max_workers=self.config.get('max_concurrent_workloads', 50))
        
        # Register local node
        self._register_local_node()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        default_config = {
            'load_balancing_strategy': 'resource_aware',
            'scaling_strategy': 'reactive',
            'node_discovery_port': 8080,
            'heartbeat_interval': 30,
            'max_concurrent_workloads': 50,
            'enable_auto_scaling': True,
            'enable_cost_optimization': False,
            'preferred_regions': ['us-east-1'],
            'max_nodes_per_region': 20
        }
        
        if config_path and Path(config_path).exists():
            try:
                user_config = safe_read_json(config_path)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading orchestrator config: {e}")
                
        return default_config
        
    def _register_local_node(self):
        """Register local compute node."""
        hostname = socket.gethostname()
        node_id = f"local_{hostname}_{int(time.time())}"
        
        # Get system resources
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        total_resources = ResourceRequirements(
            cpu_cores=float(cpu_count),
            memory_gb=memory.total / (1024**3),
            gpu_count=0,  # Would detect GPUs if available
            storage_gb=disk.total / (1024**3),
            network_bandwidth_mbps=1000.0
        )
        
        local_node = ComputeNode(
            id=node_id,
            hostname=hostname,
            ip_address="127.0.0.1",
            port=self.node_discovery_port,
            available_resources=total_resources,
            total_resources=total_resources,
            location="local",
            node_type="local"
        )
        
        self.compute_nodes[node_id] = local_node
        logger.info(f"Registered local node: {node_id}")
        
    async def start(self):
        """Start the distributed orchestrator."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Distributed orchestrator started")
        
        # Start background tasks
        asyncio.create_task(self._workload_execution_loop())
        asyncio.create_task(self._node_health_monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        
    async def _workload_execution_loop(self):
        """Main workload execution loop."""
        while self.is_running:
            try:
                # Process workloads from scheduler
                for node_id, node in list(self.compute_nodes.items()):
                    if node.health_status != "healthy":
                        continue
                        
                    # Check if node has capacity
                    if len(node.current_workloads) >= 5:  # Max 5 concurrent workloads per node
                        continue
                        
                    # Get next workload
                    workload = self.scheduler.get_next_workload(node.available_resources)
                    if workload:
                        await self._execute_workload_on_node(workload, node)
                        
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Workload execution loop error: {e}")
                await asyncio.sleep(5)
                
    async def _execute_workload_on_node(self, workload: WorkloadRequest, node: ComputeNode):
        """Execute workload on specified node."""
        node.current_workloads.append(workload.id)
        
        # Reserve resources
        req = workload.resource_requirements
        node.available_resources.cpu_cores -= req.cpu_cores
        node.available_resources.memory_gb -= req.memory_gb
        node.available_resources.gpu_count -= req.gpu_count
        node.available_resources.storage_gb -= req.storage_gb
        
        logger.info(f"Executing workload {workload.id} on node {node.id}")
        
        # Submit to executor pool
        future = self.executor_pool.submit(self._execute_workload_function, workload)
        
        # Handle completion asynchronously
        asyncio.create_task(self._handle_workload_completion(future, workload, node))
        
    def _execute_workload_function(self, workload: WorkloadRequest) -> Any:
        """Execute workload function."""
        try:
            if workload.function:
                return workload.function(*workload.args, **workload.kwargs)
            else:
                # If no function provided, simulate work
                time.sleep(np.random.uniform(0.5, 3.0))
                return f"Simulated result for {workload.id}"
                
        except Exception as e:
            logger.error(f"Workload {workload.id} execution failed: {e}")
            raise
            
    async def _handle_workload_completion(self, future, workload: WorkloadRequest, node: ComputeNode):
        """Handle workload completion."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            success = True
            error = ""
        except Exception as e:
            result = None
            success = False
            error = str(e)
            
        # Release resources
        req = workload.resource_requirements
        node.available_resources.cpu_cores += req.cpu_cores
        node.available_resources.memory_gb += req.memory_gb
        node.available_resources.gpu_count += req.gpu_count
        node.available_resources.storage_gb += req.storage_gb
        
        # Remove from current workloads
        if workload.id in node.current_workloads:
            node.current_workloads.remove(workload.id)
            
        # Update scheduler
        self.scheduler.complete_workload(workload.id, success, result, error)
        
        # Update load balancer metrics
        execution_time = time.time() - workload.created_at
        self.load_balancer.update_node_performance(node.id, execution_time, success)
        
        logger.info(f"Workload {workload.id} completed on node {node.id}: {'success' if success else 'failed'}")
        
    async def _node_health_monitoring_loop(self):
        """Monitor node health and remove unhealthy nodes."""
        while self.is_running:
            try:
                current_time = time.time()
                
                for node_id, node in list(self.compute_nodes.items()):
                    # Check heartbeat
                    if current_time - node.last_heartbeat > self.heartbeat_interval * 3:
                        node.health_status = "unhealthy"
                        logger.warning(f"Node {node_id} marked as unhealthy")
                        
                    # Remove very old unhealthy nodes
                    if (node.health_status == "unhealthy" and 
                        current_time - node.last_heartbeat > 3600):  # 1 hour
                        del self.compute_nodes[node_id]
                        logger.info(f"Removed unhealthy node: {node_id}")
                        
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Node health monitoring error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
                
    async def _auto_scaling_loop(self):
        """Auto-scaling loop."""
        if not self.config.get('enable_auto_scaling', True):
            return
            
        while self.is_running:
            try:
                # Get current state
                healthy_nodes = [n for n in self.compute_nodes.values() if n.health_status == "healthy"]
                queue_status = self.scheduler.get_queue_status()
                total_queued = queue_status['total_queued']
                
                # Determine scaling action
                action, amount = self.auto_scaler.should_scale(healthy_nodes, total_queued)
                
                if action == "scale_up":
                    await self._scale_up_nodes(amount)
                elif action == "scale_down":
                    await self._scale_down_nodes(amount)
                    
                # Record scaling metrics
                if healthy_nodes:
                    avg_utilization = np.mean([
                        np.mean(list(node.get_utilization().values())) 
                        for node in healthy_nodes
                    ])
                else:
                    avg_utilization = 0
                    
                if action != "none":
                    self.auto_scaler.record_scaling_event(
                        action, amount, len(healthy_nodes), 
                        len(healthy_nodes) + (amount if action == "scale_up" else -amount),
                        avg_utilization
                    )
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)
                
    async def _scale_up_nodes(self, count: int):
        """Scale up compute nodes."""
        logger.info(f"Scaling up {count} nodes")
        
        for i in range(count):
            # In a real implementation, this would provision new cloud instances
            # For now, simulate new nodes
            node_id = f"scaled_{uuid.uuid4().hex[:8]}"
            
            # Simulate varied node types
            node_types = ["standard", "cpu_optimized", "memory_optimized", "gpu_enabled"]
            node_type = np.random.choice(node_types)
            
            if node_type == "cpu_optimized":
                resources = ResourceRequirements(cpu_cores=8, memory_gb=16, gpu_count=0, storage_gb=100)
            elif node_type == "memory_optimized":
                resources = ResourceRequirements(cpu_cores=4, memory_gb=32, gpu_count=0, storage_gb=100)
            elif node_type == "gpu_enabled":
                resources = ResourceRequirements(cpu_cores=4, memory_gb=16, gpu_count=2, storage_gb=100)
            else:
                resources = ResourceRequirements(cpu_cores=4, memory_gb=8, gpu_count=0, storage_gb=50)
                
            new_node = ComputeNode(
                id=node_id,
                hostname=f"scaled-node-{i}",
                ip_address=f"10.0.{i//256}.{i%256}",
                port=8080 + i,
                available_resources=resources,
                total_resources=resources,
                location=np.random.choice(self.config['preferred_regions']),
                node_type=node_type,
                cost_per_hour=1.0 + (i * 0.1)
            )
            
            self.compute_nodes[node_id] = new_node
            
        metrics.increment('nodes_scaled_up', {'count': count})
        
    async def _scale_down_nodes(self, count: int):
        """Scale down compute nodes."""
        logger.info(f"Scaling down {count} nodes")
        
        # Select nodes to remove (prefer least utilized, non-local nodes)
        removable_nodes = [
            (node_id, node) for node_id, node in self.compute_nodes.items()
            if node.node_type != "local" and not node.current_workloads
        ]
        
        # Sort by utilization (remove least utilized first)
        removable_nodes.sort(key=lambda x: np.mean(list(x[1].get_utilization().values())))
        
        removed_count = 0
        for node_id, node in removable_nodes[:count]:
            del self.compute_nodes[node_id]
            removed_count += 1
            logger.info(f"Removed node: {node_id}")
            
        metrics.increment('nodes_scaled_down', {'count': removed_count})
        
    def submit_workload(self, workload_type: WorkloadType, function: Callable = None, 
                       args: Tuple = (), kwargs: Dict[str, Any] = None,
                       resource_requirements: ResourceRequirements = None,
                       deadline: Optional[float] = None) -> str:
        """Submit workload for distributed execution."""
        if kwargs is None:
            kwargs = {}
        if resource_requirements is None:
            resource_requirements = ResourceRequirements()
            
        workload = WorkloadRequest(
            workload_type=workload_type,
            function=function,
            function_name=function.__name__ if function else "unknown",
            args=args,
            kwargs=kwargs,
            resource_requirements=resource_requirements,
            deadline=deadline
        )
        
        self.scheduler.submit_workload(workload)
        return workload.id
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        healthy_nodes = [n for n in self.compute_nodes.values() if n.health_status == "healthy"]
        
        if healthy_nodes:
            avg_cpu_util = np.mean([node.get_utilization()['cpu'] for node in healthy_nodes])
            avg_memory_util = np.mean([node.get_utilization()['memory'] for node in healthy_nodes])
            avg_gpu_util = np.mean([node.get_utilization()['gpu'] for node in healthy_nodes])
        else:
            avg_cpu_util = avg_memory_util = avg_gpu_util = 0
            
        return {
            'timestamp': datetime.now().isoformat(),
            'cluster_health': 'healthy' if len(healthy_nodes) > 0 else 'degraded',
            'total_nodes': len(self.compute_nodes),
            'healthy_nodes': len(healthy_nodes),
            'node_types': list(set(node.node_type for node in self.compute_nodes.values())),
            'resource_utilization': {
                'cpu': avg_cpu_util,
                'memory': avg_memory_util,
                'gpu': avg_gpu_util
            },
            'workload_status': self.scheduler.get_queue_status(),
            'auto_scaling': {
                'enabled': self.config.get('enable_auto_scaling', True),
                'strategy': self.auto_scaler.strategy.value,
                'recent_actions': len(self.auto_scaler.scaling_history)
            },
            'cost_optimization': {
                'enabled': self.config.get('enable_cost_optimization', False),
                'estimated_hourly_cost': sum(node.cost_per_hour for node in healthy_nodes)
            }
        }
        
    def stop(self):
        """Stop the distributed orchestrator."""
        self.is_running = False
        self.executor_pool.shutdown(wait=True)
        logger.info("Distributed orchestrator stopped")


# Global instance
_distributed_orchestrator: Optional[DistributedOrchestrator] = None


def get_distributed_orchestrator() -> DistributedOrchestrator:
    """Get global distributed orchestrator instance."""
    global _distributed_orchestrator
    if _distributed_orchestrator is None:
        _distributed_orchestrator = DistributedOrchestrator()
    return _distributed_orchestrator


def distribute_workload(workload_type: WorkloadType, resource_requirements: ResourceRequirements = None):
    """Decorator for distributed workload execution."""
    if resource_requirements is None:
        resource_requirements = ResourceRequirements()
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            orchestrator = get_distributed_orchestrator()
            workload_id = orchestrator.submit_workload(
                workload_type=workload_type,
                function=func,
                args=args,
                kwargs=kwargs,
                resource_requirements=resource_requirements
            )
            return workload_id
        return wrapper
    return decorator


async def initialize_distributed_system():
    """Initialize distributed orchestrator."""
    orchestrator = get_distributed_orchestrator()
    await orchestrator.start()
    logger.info("Distributed system initialized")


if __name__ == "__main__":
    async def main():
        # Initialize distributed system
        await initialize_distributed_system()
        
        orchestrator = get_distributed_orchestrator()
        
        # Submit test workloads
        @distribute_workload(WorkloadType.INFERENCE, ResourceRequirements(cpu_cores=2, memory_gb=4))
        def cpu_intensive_task(n: int):
            import math
            result = 0
            for i in range(n * 100000):
                result += math.sqrt(i)
            return result
            
        # Submit multiple workloads
        workload_ids = []
        for i in range(10):
            workload_id = cpu_intensive_task(100)
            workload_ids.append(workload_id)
            
        # Wait and monitor
        for _ in range(30):
            status = orchestrator.get_cluster_status()
            print(f"Cluster Status: {json.dumps(status, indent=2)}")
            await asyncio.sleep(2)
            
    asyncio.run(main())