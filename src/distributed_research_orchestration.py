"""
Distributed Research Orchestration System for Large-Scale ML Experimentation.

This module implements advanced distributed computing capabilities for orchestrating
massive research campaigns across multiple nodes, with fault tolerance, load balancing,
and real-time coordination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import json
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from enum import Enum
import random
import hashlib
import copy
import time
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import pickle
import socket
import uuid
from collections import defaultdict, deque
import queue
import multiprocessing as mp
from multiprocessing import Manager
import signal
import statistics

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .config import load_config
from .adaptive_intelligence_framework import AdaptiveResearchOrchestrator

logger = get_logger(__name__)


class NodeStatus(Enum):
    """Status of distributed computing nodes."""
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ComputeNode:
    """Represents a distributed compute node."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    status: NodeStatus = NodeStatus.IDLE
    current_task_id: Optional[str] = None
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    last_heartbeat: Optional[datetime] = None
    failure_count: int = 0
    max_concurrent_tasks: int = 4
    current_task_count: int = 0
    
    def __post_init__(self):
        """Initialize default resource usage."""
        for resource_type in ResourceType:
            if resource_type not in self.resource_usage:
                self.resource_usage[resource_type] = 0.0
    
    def is_available(self, resource_requirements: Dict[ResourceType, float] = None) -> bool:
        """Check if node is available for new tasks."""
        if self.status != NodeStatus.IDLE:
            return False
        
        if self.current_task_count >= self.max_concurrent_tasks:
            return False
        
        if resource_requirements:
            for resource_type, required in resource_requirements.items():
                current_usage = self.resource_usage.get(resource_type, 0.0)
                if current_usage + required > 1.0:  # Assuming normalized resource usage
                    return False
        
        return True
    
    def calculate_performance_score(self) -> float:
        """Calculate node performance score based on history."""
        if not self.performance_history:
            return 0.5  # Default score
        
        recent_tasks = self.performance_history[-10:]  # Last 10 tasks
        
        # Calculate metrics
        completion_times = [task.get('duration', float('inf')) for task in recent_tasks]
        success_rate = sum(1 for task in recent_tasks if task.get('success', False)) / len(recent_tasks)
        
        avg_completion_time = statistics.mean(completion_times) if completion_times else float('inf')
        
        # Normalize performance score (lower time = higher score)
        time_score = max(0.0, 1.0 - min(avg_completion_time / 300, 1.0))  # 5 min baseline
        
        # Composite score
        performance_score = 0.6 * success_rate + 0.4 * time_score
        
        # Penalty for recent failures
        if self.failure_count > 0:
            performance_score *= (0.9 ** self.failure_count)
        
        return min(1.0, max(0.0, performance_score))


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    priority: TaskPriority
    data_path: str
    configuration: Dict[str, Any]
    resource_requirements: Dict[ResourceType, float]
    estimated_duration: float
    created_at: datetime
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if task is completed (successfully or failed)."""
        return self.result is not None or (self.error is not None and self.retry_count >= self.max_retries)
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to be executed (dependencies met)."""
        return len(self.dependencies) == 0  # Simplified - in real system would check dependency completion
    
    @property
    def actual_duration(self) -> Optional[float]:
        """Get actual execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskScheduler:
    """Advanced task scheduler for distributed research."""
    
    def __init__(self):
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        self.scheduling_history = []
        
    def submit_task(self, task: DistributedTask):
        """Submit a task for scheduling."""
        # Priority queue uses (priority, task) tuples
        # Lower priority number = higher priority
        priority_value = -task.priority.value  # Negative for max-heap behavior
        self.task_queue.put((priority_value, time.time(), task.task_id, task))
        
        logger.debug(f"Task {task.task_id} submitted with priority {task.priority.value}")
    
    def get_next_task(self, node_capabilities: Dict[str, Any] = None) -> Optional[DistributedTask]:
        """Get the next task for execution."""
        candidates = []
        
        # Extract candidates from queue
        while not self.task_queue.empty() and len(candidates) < 10:  # Check up to 10 candidates
            try:
                priority, submit_time, task_id, task = self.task_queue.get_nowait()
                if task.is_ready:
                    candidates.append((priority, submit_time, task_id, task))
                else:
                    # Put back tasks that aren't ready
                    self.task_queue.put((priority, submit_time, task_id, task))
                    break
            except queue.Empty:
                break
        
        if not candidates:
            return None
        
        # Sort by priority, then by submission time
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        # Return highest priority task and put others back
        best_task = candidates[0][3]
        for priority, submit_time, task_id, task in candidates[1:]:
            self.task_queue.put((priority, submit_time, task_id, task))
        
        return best_task
    
    def mark_task_active(self, task: DistributedTask, node_id: str):
        """Mark task as active on a node."""
        task.assigned_node = node_id
        task.started_at = datetime.utcnow()
        self.active_tasks[task.task_id] = task
        
        logger.info(f"Task {task.task_id} started on node {node_id}")
    
    def complete_task(self, task_id: str, result: Dict[str, Any] = None, error: str = None):
        """Mark task as completed."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempt to complete unknown task: {task_id}")
            return
        
        task = self.active_tasks.pop(task_id)
        task.completed_at = datetime.utcnow()
        
        if result:
            task.result = result
            self.completed_tasks[task_id] = task
            logger.info(f"Task {task_id} completed successfully")
        elif error:
            task.error = error
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # Requeue for retry
                task.assigned_node = None
                task.started_at = None
                task.completed_at = None
                self.submit_task(task)
                logger.warning(f"Task {task_id} failed, requeuing (retry {task.retry_count}/{task.max_retries})")
            else:
                self.failed_tasks[task_id] = task
                logger.error(f"Task {task_id} failed permanently after {task.retry_count} retries")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get scheduling queue statistics."""
        return {
            'queued_tasks': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_tasks_processed': len(self.completed_tasks) + len(self.failed_tasks)
        }


class NodeManager:
    """Manages distributed compute nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.node_health_monitor = None
        self.load_balancer = LoadBalancer()
        self.heartbeat_interval = 30  # seconds
        self.node_timeout = 90  # seconds
        
    def register_node(self, node: ComputeNode) -> str:
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        logger.info(f"Registered compute node: {node.node_id} at {node.host}:{node.port}")
        return node.node_id
    
    def deregister_node(self, node_id: str) -> bool:
        """Deregister a compute node."""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            logger.info(f"Deregistered compute node: {node_id}")
            return True
        return False
    
    def get_available_nodes(self, resource_requirements: Dict[ResourceType, float] = None) -> List[ComputeNode]:
        """Get list of available nodes matching resource requirements."""
        available = []
        
        for node in self.nodes.values():
            if node.is_available(resource_requirements):
                available.append(node)
        
        # Sort by performance score
        available.sort(key=lambda n: n.calculate_performance_score(), reverse=True)
        return available
    
    def select_best_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select the best node for a given task."""
        available_nodes = self.get_available_nodes(task.resource_requirements)
        
        if not available_nodes:
            return None
        
        # Use load balancer to select optimal node
        selected_node = self.load_balancer.select_node(available_nodes, task)
        return selected_node
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            self.nodes[node_id].last_heartbeat = datetime.utcnow()
    
    def update_node_resources(self, node_id: str, resource_usage: Dict[ResourceType, float]):
        """Update node resource usage."""
        if node_id in self.nodes:
            self.nodes[node_id].resource_usage.update(resource_usage)
    
    def record_task_completion(self, node_id: str, task_result: Dict[str, Any]):
        """Record task completion on node."""
        if node_id in self.nodes:
            self.nodes[node_id].performance_history.append(task_result)
            self.nodes[node_id].current_task_count -= 1
    
    def check_node_health(self):
        """Check health of all nodes."""
        current_time = datetime.utcnow()
        unhealthy_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.last_heartbeat:
                time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.node_timeout:
                    node.status = NodeStatus.FAILED
                    node.failure_count += 1
                    unhealthy_nodes.append(node_id)
        
        if unhealthy_nodes:
            logger.warning(f"Detected unhealthy nodes: {unhealthy_nodes}")
        
        return unhealthy_nodes
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics."""
        total_nodes = len(self.nodes)
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.IDLE or n.status == NodeStatus.BUSY])
        
        # Resource utilization
        total_cpu = sum(node.resource_usage.get(ResourceType.CPU, 0.0) for node in self.nodes.values())
        total_memory = sum(node.resource_usage.get(ResourceType.MEMORY, 0.0) for node in self.nodes.values())
        
        # Performance metrics
        performance_scores = [node.calculate_performance_score() for node in self.nodes.values()]
        avg_performance = statistics.mean(performance_scores) if performance_scores else 0.0
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'node_utilization': active_nodes / max(total_nodes, 1),
            'average_cpu_usage': total_cpu / max(total_nodes, 1),
            'average_memory_usage': total_memory / max(total_nodes, 1),
            'average_performance_score': avg_performance,
            'failed_nodes': len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
        }


class LoadBalancer:
    """Advanced load balancer for distributed tasks."""
    
    def __init__(self, strategy: str = "weighted_performance"):
        self.strategy = strategy
        self.node_loads: Dict[str, float] = defaultdict(float)
        self.selection_history = []
        
    def select_node(self, available_nodes: List[ComputeNode], task: DistributedTask) -> Optional[ComputeNode]:
        """Select optimal node based on load balancing strategy."""
        if not available_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(available_nodes)
        elif self.strategy == "weighted_performance":
            return self._weighted_performance_selection(available_nodes, task)
        elif self.strategy == "random":
            return random.choice(available_nodes)
        else:
            return available_nodes[0]  # Default to first available
    
    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Round-robin node selection."""
        if not self.selection_history:
            return nodes[0]
        
        last_selected = self.selection_history[-1]
        node_ids = [node.node_id for node in nodes]
        
        try:
            last_index = node_ids.index(last_selected)
            next_index = (last_index + 1) % len(nodes)
        except ValueError:
            next_index = 0
        
        selected = nodes[next_index]
        self.selection_history.append(selected.node_id)
        return selected
    
    def _least_loaded_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node with lowest current load."""
        def node_load_score(node):
            return (
                node.current_task_count / max(node.max_concurrent_tasks, 1) +
                node.resource_usage.get(ResourceType.CPU, 0.0) +
                node.resource_usage.get(ResourceType.MEMORY, 0.0)
            )
        
        selected = min(nodes, key=node_load_score)
        self.selection_history.append(selected.node_id)
        return selected
    
    def _weighted_performance_selection(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node based on weighted performance metrics."""
        weights = []
        
        for node in nodes:
            performance_score = node.calculate_performance_score()
            
            # Calculate load penalty
            load_factor = node.current_task_count / max(node.max_concurrent_tasks, 1)
            load_penalty = 1.0 - load_factor
            
            # Resource availability bonus
            cpu_available = 1.0 - node.resource_usage.get(ResourceType.CPU, 0.0)
            memory_available = 1.0 - node.resource_usage.get(ResourceType.MEMORY, 0.0)
            resource_bonus = (cpu_available + memory_available) / 2
            
            # Task type compatibility (simplified)
            compatibility_score = 1.0  # Would be based on node.capabilities vs task requirements
            
            # Composite weight
            weight = (
                performance_score * 0.4 +
                load_penalty * 0.3 +
                resource_bonus * 0.2 +
                compatibility_score * 0.1
            )
            
            weights.append(max(0.01, weight))  # Ensure minimum weight
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0]
        
        normalized_weights = [w / total_weight for w in weights]
        selected = np.random.choice(nodes, p=normalized_weights)
        
        self.selection_history.append(selected.node_id)
        return selected


class DistributedResearchOrchestrator:
    """Main orchestrator for distributed research campaigns."""
    
    def __init__(self, max_concurrent_experiments: int = 50):
        self.node_manager = NodeManager()
        self.task_scheduler = TaskScheduler()
        self.max_concurrent_experiments = max_concurrent_experiments
        
        # Coordination and state
        self.active_campaigns: Dict[str, Dict[str, Any]] = {}
        self.completed_campaigns: Dict[str, Dict[str, Any]] = {}
        self.orchestration_metrics = defaultdict(list)
        
        # Distributed coordination
        self.coordinator_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.worker_pool = ThreadPoolExecutor(max_workers=10)
        
        # Start background services
        self._start_background_services()
    
    def _start_background_services(self):
        """Start background services for orchestration."""
        # Health monitoring
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, 
            daemon=True
        )
        self.health_monitor_thread.start()
        
        # Task dispatcher
        self.task_dispatcher_thread = threading.Thread(
            target=self._task_dispatcher_loop,
            daemon=True
        )
        self.task_dispatcher_thread.start()
        
        logger.info("Distributed orchestration services started")
    
    def register_compute_nodes(self, node_configs: List[Dict[str, Any]]) -> List[str]:
        """Register multiple compute nodes."""
        registered_nodes = []
        
        for config in node_configs:
            node = ComputeNode(
                node_id=config.get('node_id', str(uuid.uuid4())),
                host=config.get('host', 'localhost'),
                port=config.get('port', 8080),
                capabilities=config.get('capabilities', {}),
                max_concurrent_tasks=config.get('max_concurrent_tasks', 4)
            )
            
            node_id = self.node_manager.register_node(node)
            registered_nodes.append(node_id)
        
        logger.info(f"Registered {len(registered_nodes)} compute nodes")
        return registered_nodes
    
    async def launch_distributed_campaign(self,
                                        campaign_config: Dict[str, Any],
                                        data_paths: List[str],
                                        experiment_types: List[str],
                                        resource_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Launch a distributed research campaign across multiple nodes.
        
        Args:
            campaign_config: Campaign configuration
            data_paths: List of dataset paths
            experiment_types: Types of experiments to run
            resource_requirements: Resource requirements per experiment
            
        Returns:
            Campaign results and orchestration metrics
        """
        campaign_id = f"campaign_{int(time.time())}_{random.randint(1000, 9999)}"
        logger.info(f"Launching distributed campaign: {campaign_id}")
        
        campaign_start = time.time()
        
        # Create campaign state
        campaign_state = {
            'campaign_id': campaign_id,
            'config': campaign_config,
            'start_time': campaign_start,
            'status': 'running',
            'tasks': {},
            'results': {},
            'orchestration_metrics': {}
        }
        
        self.active_campaigns[campaign_id] = campaign_state
        
        try:
            # Generate distributed tasks
            tasks = self._generate_distributed_tasks(
                campaign_id, data_paths, experiment_types, resource_requirements or {}
            )
            
            # Submit tasks to scheduler
            for task in tasks:
                self.task_scheduler.submit_task(task)
                campaign_state['tasks'][task.task_id] = task
            
            logger.info(f"Submitted {len(tasks)} tasks for campaign {campaign_id}")
            
            # Wait for campaign completion
            completion_result = await self._wait_for_campaign_completion(
                campaign_id, 
                timeout_minutes=campaign_config.get('timeout_minutes', 120)
            )
            
            # Compile results
            campaign_results = await self._compile_campaign_results(campaign_id)
            
            # Move to completed campaigns
            campaign_state['status'] = 'completed'
            campaign_state['end_time'] = time.time()
            campaign_state['duration'] = campaign_state['end_time'] - campaign_state['start_time']
            campaign_state['results'] = campaign_results
            
            self.completed_campaigns[campaign_id] = campaign_state
            del self.active_campaigns[campaign_id]
            
            logger.info(f"Campaign {campaign_id} completed in {campaign_state['duration']:.2f}s")
            
            return {
                'campaign_id': campaign_id,
                'success': True,
                'duration_seconds': campaign_state['duration'],
                'tasks_completed': len([t for t in tasks if t.is_complete]),
                'orchestration_efficiency': self._calculate_orchestration_efficiency(campaign_state),
                'distributed_results': campaign_results,
                'cluster_utilization': self._calculate_cluster_utilization(),
                'scaling_metrics': self._calculate_scaling_metrics(campaign_state)
            }
            
        except Exception as e:
            logger.error(f"Campaign {campaign_id} failed: {e}")
            
            # Mark campaign as failed
            campaign_state['status'] = 'failed'
            campaign_state['error'] = str(e)
            campaign_state['end_time'] = time.time()
            
            return {
                'campaign_id': campaign_id,
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - campaign_start
            }
    
    def _generate_distributed_tasks(self, 
                                  campaign_id: str,
                                  data_paths: List[str], 
                                  experiment_types: List[str],
                                  resource_requirements: Dict[str, Any]) -> List[DistributedTask]:
        """Generate tasks for distributed execution."""
        tasks = []
        
        # Create task combinations
        task_combinations = []
        for data_path in data_paths:
            for exp_type in experiment_types:
                task_combinations.append((data_path, exp_type))
        
        # Generate tasks
        for i, (data_path, exp_type) in enumerate(task_combinations):
            task_id = f"{campaign_id}_task_{i:04d}_{exp_type}"
            
            # Determine resource requirements
            base_requirements = {
                ResourceType.CPU: 0.25,
                ResourceType.MEMORY: 0.30,
                ResourceType.GPU: 0.0,  # Most experiments don't need GPU
                ResourceType.STORAGE: 0.10,
                ResourceType.NETWORK: 0.05
            }
            
            # Adjust based on experiment type
            if exp_type == 'neural_architecture_search':
                base_requirements[ResourceType.CPU] = 0.50
                base_requirements[ResourceType.MEMORY] = 0.60
            elif exp_type == 'quantum_research':
                base_requirements[ResourceType.CPU] = 0.40
                base_requirements[ResourceType.MEMORY] = 0.45
            elif exp_type == 'evolutionary_learning':
                base_requirements[ResourceType.CPU] = 0.35
                base_requirements[ResourceType.MEMORY] = 0.40
            
            # Create task
            task = DistributedTask(
                task_id=task_id,
                task_type=exp_type,
                priority=TaskPriority.MEDIUM,
                data_path=data_path,
                configuration={'experiment_type': exp_type, 'timeout_minutes': 30},
                resource_requirements=base_requirements,
                estimated_duration=1800,  # 30 minutes
                created_at=datetime.utcnow()
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _wait_for_campaign_completion(self, campaign_id: str, timeout_minutes: int = 120) -> Dict[str, Any]:
        """Wait for campaign completion with timeout."""
        campaign_state = self.active_campaigns.get(campaign_id)
        if not campaign_state:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 30  # Check every 30 seconds
        
        while time.time() - start_time < timeout_seconds:
            # Check completion status
            tasks = campaign_state['tasks']
            completed_tasks = [t for t in tasks.values() if t.is_complete]
            
            completion_rate = len(completed_tasks) / len(tasks) if tasks else 0.0
            
            logger.debug(f"Campaign {campaign_id} completion: {completion_rate:.2%}")
            
            # Check if campaign is complete
            if completion_rate >= 0.95:  # 95% completion threshold
                logger.info(f"Campaign {campaign_id} reached completion threshold")
                break
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        # Final status
        final_completion_rate = len([t for t in campaign_state['tasks'].values() if t.is_complete]) / len(campaign_state['tasks'])
        
        return {
            'completion_rate': final_completion_rate,
            'total_tasks': len(campaign_state['tasks']),
            'completed_tasks': len([t for t in campaign_state['tasks'].values() if t.is_complete]),
            'duration_seconds': time.time() - start_time
        }
    
    async def _compile_campaign_results(self, campaign_id: str) -> Dict[str, Any]:
        """Compile results from distributed campaign."""
        campaign_state = self.active_campaigns.get(campaign_id)
        if not campaign_state:
            return {}
        
        tasks = campaign_state['tasks']
        completed_tasks = [t for t in tasks.values() if t.is_complete and t.result]
        failed_tasks = [t for t in tasks.values() if t.error and t.retry_count >= t.max_retries]
        
        # Aggregate results by experiment type
        results_by_type = defaultdict(list)
        for task in completed_tasks:
            exp_type = task.task_type
            results_by_type[exp_type].append(task.result)
        
        # Calculate best results
        best_results = {}
        for exp_type, results in results_by_type.items():
            if results:
                # Find best result based on performance
                best_result = max(results, key=lambda r: self._extract_performance_score(r))
                best_results[exp_type] = best_result
        
        # Performance statistics
        all_performances = []
        for task in completed_tasks:
            if task.result:
                perf_score = self._extract_performance_score(task.result)
                if perf_score > 0:
                    all_performances.append(perf_score)
        
        campaign_metrics = {
            'total_tasks': len(tasks),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / len(tasks) if tasks else 0.0,
            'experiment_types': list(results_by_type.keys()),
            'best_results': best_results,
            'performance_statistics': {
                'mean': statistics.mean(all_performances) if all_performances else 0.0,
                'median': statistics.median(all_performances) if all_performances else 0.0,
                'std': statistics.stdev(all_performances) if len(all_performances) > 1 else 0.0,
                'min': min(all_performances) if all_performances else 0.0,
                'max': max(all_performances) if all_performances else 0.0
            }
        }
        
        return campaign_metrics
    
    def _extract_performance_score(self, result: Dict[str, Any]) -> float:
        """Extract performance score from task result."""
        # Try different performance metrics
        performance = result.get('performance', {})
        
        if 'accuracy' in performance:
            return performance['accuracy']
        elif 'best_score' in result:
            return result['best_score']
        elif 'fitness_score' in result:
            return result['fitness_score']
        elif 'score' in result:
            return result['score']
        
        return 0.0
    
    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Check node health
                unhealthy_nodes = self.node_manager.check_node_health()
                
                if unhealthy_nodes:
                    logger.warning(f"Health monitor detected unhealthy nodes: {unhealthy_nodes}")
                    # Could implement node recovery logic here
                
                # Sleep for next check
                self.shutdown_event.wait(self.node_manager.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _task_dispatcher_loop(self):
        """Background task dispatcher loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get next task
                task = self.task_scheduler.get_next_task()
                
                if task:
                    # Find suitable node
                    node = self.node_manager.select_best_node(task)
                    
                    if node:
                        # Submit task to worker pool for execution
                        future = self.worker_pool.submit(self._execute_task_on_node, task, node)
                        
                        # Mark task as active
                        self.task_scheduler.mark_task_active(task, node.node_id)
                        node.current_task_count += 1
                        
                        # Handle completion in background
                        def task_completion_handler(fut):
                            try:
                                result = fut.result()
                                self.task_scheduler.complete_task(task.task_id, result=result)
                                self.node_manager.record_task_completion(node.node_id, {
                                    'task_id': task.task_id,
                                    'success': True,
                                    'duration': task.actual_duration or 0
                                })
                            except Exception as e:
                                self.task_scheduler.complete_task(task.task_id, error=str(e))
                                self.node_manager.record_task_completion(node.node_id, {
                                    'task_id': task.task_id,
                                    'success': False,
                                    'error': str(e)
                                })
                        
                        future.add_done_callback(task_completion_handler)
                    else:
                        # No available nodes, wait a bit
                        time.sleep(5)
                else:
                    # No tasks available, wait
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                time.sleep(5)
    
    def _execute_task_on_node(self, task: DistributedTask, node: ComputeNode) -> Dict[str, Any]:
        """Execute a task on a specific node."""
        logger.info(f"Executing task {task.task_id} on node {node.node_id}")
        
        try:
            # Simulate task execution based on type
            if task.task_type == 'quantum_research':
                from .quantum_enhanced_research import run_quantum_research_experiment
                result = asyncio.run(run_quantum_research_experiment(task.data_path))
            elif task.task_type == 'evolutionary_learning':
                from .autonomous_learning_evolution import run_autonomous_evolution_experiment
                result = asyncio.run(run_autonomous_evolution_experiment(task.data_path))
            elif task.task_type == 'neural_architecture_search':
                from .neural_architecture_search import run_nas_experiment
                result = asyncio.run(run_nas_experiment(task.data_path, 15))  # 15 minute budget
            else:
                # Default/unknown task type
                result = {
                    'task_id': task.task_id,
                    'success': False,
                    'error': f'Unknown task type: {task.task_type}'
                }
            
            # Add execution metadata
            result['execution_metadata'] = {
                'node_id': node.node_id,
                'task_id': task.task_id,
                'execution_time': datetime.utcnow().isoformat(),
                'resource_usage': node.resource_usage.copy()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed on node {node.node_id}: {e}")
            return {
                'task_id': task.task_id,
                'success': False,
                'error': str(e),
                'node_id': node.node_id
            }
    
    def _calculate_orchestration_efficiency(self, campaign_state: Dict[str, Any]) -> float:
        """Calculate orchestration efficiency metrics."""
        tasks = campaign_state['tasks']
        if not tasks:
            return 0.0
        
        # Task completion efficiency
        completed_tasks = [t for t in tasks.values() if t.is_complete and t.result]
        completion_rate = len(completed_tasks) / len(tasks)
        
        # Resource utilization efficiency
        cluster_stats = self.node_manager.get_cluster_stats()
        resource_efficiency = cluster_stats.get('node_utilization', 0.0)
        
        # Time efficiency (tasks completed within estimated duration)
        time_efficient_tasks = 0
        for task in completed_tasks:
            if task.actual_duration and task.actual_duration <= task.estimated_duration:
                time_efficient_tasks += 1
        
        time_efficiency = time_efficient_tasks / max(len(completed_tasks), 1)
        
        # Composite efficiency
        efficiency = (
            completion_rate * 0.4 +
            resource_efficiency * 0.3 +
            time_efficiency * 0.3
        )
        
        return efficiency
    
    def _calculate_cluster_utilization(self) -> Dict[str, float]:
        """Calculate current cluster utilization."""
        cluster_stats = self.node_manager.get_cluster_stats()
        queue_stats = self.task_scheduler.get_queue_stats()
        
        return {
            'node_utilization': cluster_stats.get('node_utilization', 0.0),
            'cpu_utilization': cluster_stats.get('average_cpu_usage', 0.0),
            'memory_utilization': cluster_stats.get('average_memory_usage', 0.0),
            'task_queue_depth': queue_stats.get('queued_tasks', 0),
            'active_task_ratio': queue_stats.get('active_tasks', 0) / max(cluster_stats.get('active_nodes', 1), 1)
        }
    
    def _calculate_scaling_metrics(self, campaign_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scaling performance metrics."""
        total_tasks = len(campaign_state['tasks'])
        duration = campaign_state.get('duration', 1)
        
        # Throughput
        throughput = total_tasks / duration if duration > 0 else 0.0
        
        # Parallelization efficiency
        cluster_stats = self.node_manager.get_cluster_stats()
        active_nodes = cluster_stats.get('active_nodes', 1)
        theoretical_max_throughput = active_nodes * (3600 / 1800)  # Assuming 30min avg task time
        parallelization_efficiency = throughput / max(theoretical_max_throughput, 0.1)
        
        return {
            'task_throughput': throughput,
            'parallelization_efficiency': parallelization_efficiency,
            'active_nodes': active_nodes,
            'tasks_per_node': total_tasks / max(active_nodes, 1),
            'scaling_factor': min(active_nodes, total_tasks) / max(total_tasks, 1)
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        logger.info("Shutting down distributed research orchestrator")
        
        self.shutdown_event.set()
        
        # Wait for threads to finish
        if hasattr(self, 'health_monitor_thread'):
            self.health_monitor_thread.join(timeout=5)
        
        if hasattr(self, 'task_dispatcher_thread'):
            self.task_dispatcher_thread.join(timeout=5)
        
        # Shutdown worker pool
        self.worker_pool.shutdown(wait=True)
        
        logger.info("Distributed orchestrator shutdown complete")


async def run_distributed_research_campaign(data_paths: List[str] = None,
                                          experiment_types: List[str] = None,
                                          num_nodes: int = 4,
                                          timeout_minutes: int = 90) -> Dict[str, Any]:
    """
    Run a distributed research campaign.
    
    Args:
        data_paths: List of dataset paths
        experiment_types: Types of experiments to run
        num_nodes: Number of compute nodes to simulate
        timeout_minutes: Campaign timeout
        
    Returns:
        Campaign results
    """
    if data_paths is None:
        data_paths = ["data/processed/processed_features.csv"]
    
    if experiment_types is None:
        experiment_types = ["quantum_research", "evolutionary_learning", "neural_architecture_search"]
    
    # Initialize orchestrator
    orchestrator = DistributedResearchOrchestrator()
    
    try:
        # Register compute nodes
        node_configs = []
        for i in range(num_nodes):
            node_configs.append({
                'node_id': f'compute_node_{i:02d}',
                'host': f'node-{i}.cluster.local',
                'port': 8080 + i,
                'capabilities': {'cpu_cores': 8, 'memory_gb': 16, 'gpu_count': 0},
                'max_concurrent_tasks': 2
            })
        
        registered_nodes = orchestrator.register_compute_nodes(node_configs)
        logger.info(f"Registered {len(registered_nodes)} compute nodes for distributed campaign")
        
        # Campaign configuration
        campaign_config = {
            'timeout_minutes': timeout_minutes,
            'max_concurrent_experiments': num_nodes * 2,
            'resource_constraints': {
                'cpu_per_task': 0.25,
                'memory_per_task': 0.30
            }
        }
        
        # Launch campaign
        results = await orchestrator.launch_distributed_campaign(
            campaign_config=campaign_config,
            data_paths=data_paths,
            experiment_types=experiment_types
        )
        
        logger.info(f"Distributed campaign completed: {results.get('success', False)}")
        return results
        
    finally:
        # Cleanup
        orchestrator.shutdown()


if __name__ == "__main__":
    async def main():
        results = await run_distributed_research_campaign(
            data_paths=["data/processed/processed_features.csv"],
            experiment_types=["quantum_research", "evolutionary_learning"],
            num_nodes=3,
            timeout_minutes=45
        )
        
        print(f"Distributed Research Campaign Results:")
        print(f"Success: {results.get('success', False)}")
        print(f"Tasks Completed: {results.get('tasks_completed', 'N/A')}")
        print(f"Orchestration Efficiency: {results.get('orchestration_efficiency', 'N/A'):.3f}")
        print(f"Duration: {results.get('duration_seconds', 'N/A'):.2f}s")
    
    asyncio.run(main())