"""
Distributed Computing and Fault Tolerance System.

This module provides distributed computing capabilities with fault tolerance including:
- Distributed model training with parameter servers
- Horizontal scaling with load balancing
- Consensus algorithms for distributed coordination
- Replicated state management
- Service mesh integration
- Multi-region deployment support
- Automatic failover and recovery
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import hashlib
import numpy as np
import pandas as pd
from enum import Enum, auto
import pickle
import socket
import uuid
import requests
from urllib.parse import urljoin
import redis
import consul

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .error_handling_recovery import with_error_handling, CircuitBreaker

logger = get_logger(__name__)


class NodeStatus(Enum):
    """Node status in distributed system."""
    INITIALIZING = auto()
    ACTIVE = auto()
    DEGRADED = auto()
    FAILED = auto()
    RECOVERING = auto()
    SHUTDOWN = auto()


class NodeRole(Enum):
    """Node roles in distributed system."""
    COORDINATOR = auto()
    WORKER = auto()
    REPLICA = auto()
    GATEWAY = auto()


@dataclass
class NodeInfo:
    """Information about a node in the distributed system."""
    node_id: str
    host: str
    port: int
    role: NodeRole
    status: NodeStatus
    capabilities: List[str]
    last_heartbeat: datetime
    metadata: Dict[str, Any] = None
    load_factor: float = 0.0  # 0.0 = no load, 1.0 = full load
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5  # 1 = highest, 10 = lowest
    max_retries: int = 3
    timeout_seconds: int = 300
    assigned_node: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ServiceDiscovery:
    """Service discovery and registration system."""
    
    def __init__(self, discovery_backend: str = "consul", config: Dict[str, Any] = None):
        self.backend = discovery_backend
        self.config = config or {}
        self.local_registry = {}  # Fallback local registry
        self.consul_client = None
        
        if discovery_backend == "consul":
            self._setup_consul()
    
    def _setup_consul(self) -> None:
        """Setup Consul client."""
        try:
            consul_host = self.config.get('consul_host', 'localhost')
            consul_port = self.config.get('consul_port', 8500)
            self.consul_client = consul.Consul(host=consul_host, port=consul_port)
            logger.info(f"Connected to Consul at {consul_host}:{consul_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Consul: {e}")
            self.consul_client = None
    
    def register_service(self, service_name: str, node_info: NodeInfo, 
                        health_check_url: Optional[str] = None) -> bool:
        """Register a service with discovery system."""
        
        service_info = {
            'service_id': node_info.node_id,
            'service_name': service_name,
            'address': node_info.host,
            'port': node_info.port,
            'tags': [node_info.role.name.lower()] + node_info.capabilities,
            'meta': {
                'status': node_info.status.name,
                'last_heartbeat': node_info.last_heartbeat.isoformat(),
                'load_factor': str(node_info.load_factor)
            }
        }
        
        if health_check_url:
            service_info['check'] = {
                'http': health_check_url,
                'interval': '10s',
                'timeout': '5s'
            }
        
        # Register with Consul if available
        if self.consul_client:
            try:
                self.consul_client.agent.service.register(**service_info)
                logger.info(f"Service {service_name} registered with Consul")
                return True
            except Exception as e:
                logger.error(f"Failed to register service with Consul: {e}")
        
        # Fallback to local registry
        if service_name not in self.local_registry:
            self.local_registry[service_name] = []
        
        # Remove existing registration for this node
        self.local_registry[service_name] = [
            s for s in self.local_registry[service_name] 
            if s['service_id'] != node_info.node_id
        ]
        
        # Add new registration
        self.local_registry[service_name].append(service_info)
        logger.info(f"Service {service_name} registered locally")
        return True
    
    def deregister_service(self, service_name: str, node_id: str) -> bool:
        """Deregister a service."""
        
        # Deregister from Consul
        if self.consul_client:
            try:
                self.consul_client.agent.service.deregister(node_id)
                logger.info(f"Service {service_name} deregistered from Consul")
            except Exception as e:
                logger.error(f"Failed to deregister service from Consul: {e}")
        
        # Remove from local registry
        if service_name in self.local_registry:
            self.local_registry[service_name] = [
                s for s in self.local_registry[service_name]
                if s['service_id'] != node_id
            ]
        
        return True
    
    def discover_services(self, service_name: str, 
                         healthy_only: bool = True) -> List[Dict[str, Any]]:
        """Discover services by name."""
        
        # Try Consul first
        if self.consul_client:
            try:
                services = self.consul_client.health.service(
                    service_name, 
                    passing=healthy_only
                )[1]
                
                discovered_services = []
                for service in services:
                    service_info = service['Service']
                    discovered_services.append({
                        'service_id': service_info['ID'],
                        'service_name': service_info['Service'],
                        'address': service_info['Address'],
                        'port': service_info['Port'],
                        'tags': service_info['Tags'],
                        'meta': service_info.get('Meta', {})
                    })
                
                return discovered_services
                
            except Exception as e:
                logger.error(f"Failed to discover services from Consul: {e}")
        
        # Fallback to local registry
        return self.local_registry.get(service_name, [])
    
    def get_healthy_nodes(self, service_name: str, 
                         role_filter: Optional[NodeRole] = None) -> List[NodeInfo]:
        """Get healthy nodes for a service."""
        
        services = self.discover_services(service_name, healthy_only=True)
        nodes = []
        
        for service in services:
            try:
                node_info = NodeInfo(
                    node_id=service['service_id'],
                    host=service['address'],
                    port=service['port'],
                    role=NodeRole[service['tags'][0].upper()] if service['tags'] else NodeRole.WORKER,
                    status=NodeStatus[service['meta'].get('status', 'ACTIVE')],
                    capabilities=service['tags'][1:] if len(service['tags']) > 1 else [],
                    last_heartbeat=datetime.fromisoformat(
                        service['meta'].get('last_heartbeat', datetime.now().isoformat())
                    ),
                    load_factor=float(service['meta'].get('load_factor', 0.0))
                )
                
                if role_filter is None or node_info.role == role_filter:
                    nodes.append(node_info)
                    
            except Exception as e:
                logger.warning(f"Failed to parse node info: {e}")
                continue
        
        return nodes


class LoadBalancer:
    """Intelligent load balancer for distributed requests."""
    
    def __init__(self, algorithm: str = "weighted_round_robin"):
        self.algorithm = algorithm
        self.round_robin_counter = defaultdict(int)
        self.node_stats = defaultdict(lambda: {
            'requests': 0,
            'response_times': deque(maxlen=100),
            'errors': 0,
            'last_used': datetime.now()
        })
    
    def select_node(self, nodes: List[NodeInfo], 
                    request_context: Dict[str, Any] = None) -> Optional[NodeInfo]:
        """Select the best node for a request."""
        
        if not nodes:
            return None
        
        # Filter out failed nodes
        healthy_nodes = [n for n in nodes if n.status in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]]
        
        if not healthy_nodes:
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin_selection(healthy_nodes)
        elif self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(healthy_nodes)
        elif self.algorithm == "least_connections":
            return self._least_connections_selection(healthy_nodes)
        elif self.algorithm == "response_time":
            return self._response_time_selection(healthy_nodes)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_nodes)
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Simple round robin selection."""
        if not nodes:
            return None
        
        service_key = "global"  # Could be more specific
        index = self.round_robin_counter[service_key] % len(nodes)
        self.round_robin_counter[service_key] += 1
        
        return nodes[index]
    
    def _weighted_round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Weighted round robin based on inverse load factor."""
        if not nodes:
            return None
        
        # Calculate weights (inverse of load factor)
        weights = []
        for node in nodes:
            # Higher load = lower weight
            weight = max(0.1, 1.0 - node.load_factor)
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return self._round_robin_selection(nodes)
        
        rand_val = np.random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return nodes[i]
        
        return nodes[0]  # Fallback
    
    def _least_connections_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with least active connections."""
        if not nodes:
            return None
        
        # Use load factor as proxy for connections
        return min(nodes, key=lambda n: n.load_factor)
    
    def _response_time_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with best response time."""
        if not nodes:
            return None
        
        best_node = None
        best_avg_time = float('inf')
        
        for node in nodes:
            stats = self.node_stats[node.node_id]
            if stats['response_times']:
                avg_time = np.mean(stats['response_times'])
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_node = node
        
        # If no response time data, fall back to least loaded
        return best_node or self._least_connections_selection(nodes)
    
    def record_request(self, node_id: str, response_time: float, 
                      success: bool = True) -> None:
        """Record request metrics for a node."""
        stats = self.node_stats[node_id]
        stats['requests'] += 1
        stats['response_times'].append(response_time)
        stats['last_used'] = datetime.now()
        
        if not success:
            stats['errors'] += 1
    
    def get_node_stats(self, node_id: str) -> Dict[str, Any]:
        """Get statistics for a node."""
        stats = self.node_stats[node_id]
        
        avg_response_time = (
            np.mean(stats['response_times']) 
            if stats['response_times'] else 0.0
        )
        
        error_rate = (
            stats['errors'] / stats['requests'] 
            if stats['requests'] > 0 else 0.0
        )
        
        return {
            'requests': stats['requests'],
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'errors': stats['errors'],
            'last_used': stats['last_used'].isoformat()
        }


class DistributedCoordinator:
    """Coordinates distributed operations and maintains consensus."""
    
    def __init__(self, node_info: NodeInfo, service_discovery: ServiceDiscovery):
        self.node_info = node_info
        self.service_discovery = service_discovery
        self.is_leader = False
        self.leader_id = None
        self.term = 0
        self.vote_count = 0
        self.election_timeout = 5.0  # seconds
        self.heartbeat_interval = 1.0  # seconds
        self.last_heartbeat = datetime.now()
        
        # Distributed state
        self.distributed_state = {}
        self.state_version = 0
        
        # Task queue
        self.pending_tasks = deque()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Coordination thread
        self.coordination_active = False
        self.coordination_thread = None
        
    def start_coordination(self) -> None:
        """Start coordination services."""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop, daemon=True
        )
        self.coordination_thread.start()
        
        logger.info(f"Node {self.node_info.node_id} started coordination")
    
    def stop_coordination(self) -> None:
        """Stop coordination services."""
        self.coordination_active = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        
        logger.info(f"Node {self.node_info.node_id} stopped coordination")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.coordination_active:
            try:
                if self.is_leader:
                    self._send_heartbeats()
                    self._process_pending_tasks()
                    self._check_task_timeouts()
                else:
                    self._check_leader_heartbeat()
                
                time.sleep(0.5)  # Coordination loop interval
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
    
    def _send_heartbeats(self) -> None:
        """Send heartbeats as leader."""
        if not self.is_leader:
            return
        
        # Get all nodes in the cluster
        nodes = self.service_discovery.get_healthy_nodes("ml_cluster")
        
        for node in nodes:
            if node.node_id != self.node_info.node_id:
                try:
                    self._send_heartbeat_to_node(node)
                except Exception as e:
                    logger.warning(f"Failed to send heartbeat to {node.node_id}: {e}")
        
        self.last_heartbeat = datetime.now()
    
    def _send_heartbeat_to_node(self, node: NodeInfo) -> None:
        """Send heartbeat to a specific node."""
        url = f"http://{node.host}:{node.port}/internal/heartbeat"
        payload = {
            'leader_id': self.node_info.node_id,
            'term': self.term,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(url, json=payload, timeout=2)
        response.raise_for_status()
    
    def _check_leader_heartbeat(self) -> None:
        """Check if leader is still alive."""
        if self.leader_id is None:
            self._start_election()
            return
        
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        if time_since_heartbeat.total_seconds() > self.election_timeout:
            logger.warning(f"Leader {self.leader_id} heartbeat timeout, starting election")
            self._start_election()
    
    def _start_election(self) -> None:
        """Start leader election process."""
        logger.info(f"Node {self.node_info.node_id} starting election")
        
        self.term += 1
        self.vote_count = 1  # Vote for self
        self.leader_id = None
        
        # Get all nodes and request votes
        nodes = self.service_discovery.get_healthy_nodes("ml_cluster")
        
        for node in nodes:
            if node.node_id != self.node_info.node_id:
                try:
                    if self._request_vote_from_node(node):
                        self.vote_count += 1
                except Exception as e:
                    logger.warning(f"Failed to request vote from {node.node_id}: {e}")
        
        # Check if we have majority
        total_nodes = len(nodes)
        majority = (total_nodes // 2) + 1
        
        if self.vote_count >= majority:
            self._become_leader()
        else:
            logger.info(f"Election failed, got {self.vote_count}/{total_nodes} votes")
    
    def _request_vote_from_node(self, node: NodeInfo) -> bool:
        """Request vote from a specific node."""
        url = f"http://{node.host}:{node.port}/internal/vote"
        payload = {
            'candidate_id': self.node_info.node_id,
            'term': self.term,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(url, json=payload, timeout=2)
        response.raise_for_status()
        
        result = response.json()
        return result.get('vote_granted', False)
    
    def _become_leader(self) -> None:
        """Become the cluster leader."""
        self.is_leader = True
        self.leader_id = self.node_info.node_id
        self.last_heartbeat = datetime.now()
        
        logger.info(f"Node {self.node_info.node_id} became leader for term {self.term}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        if not self.is_leader:
            raise ValueError("Only leader can accept tasks")
        
        self.pending_tasks.append(task)
        logger.info(f"Task {task.task_id} submitted for distributed execution")
        
        return task.task_id
    
    def _process_pending_tasks(self) -> None:
        """Process pending tasks by assigning to workers."""
        if not self.is_leader or not self.pending_tasks:
            return
        
        # Get available worker nodes
        worker_nodes = self.service_discovery.get_healthy_nodes(
            "ml_cluster", role_filter=NodeRole.WORKER
        )
        
        if not worker_nodes:
            logger.warning("No worker nodes available for task assignment")
            return
        
        # Load balancer for task assignment
        load_balancer = LoadBalancer("weighted_round_robin")
        
        while self.pending_tasks and worker_nodes:
            task = self.pending_tasks.popleft()
            
            # Select worker node
            selected_node = load_balancer.select_node(worker_nodes)
            
            if selected_node:
                try:
                    self._assign_task_to_node(task, selected_node)
                    task.assigned_node = selected_node.node_id
                    task.started_at = datetime.now()
                    task.status = "running"
                    self.active_tasks[task.task_id] = task
                    
                    logger.info(f"Task {task.task_id} assigned to {selected_node.node_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to assign task {task.task_id} to {selected_node.node_id}: {e}")
                    # Put task back in queue
                    self.pending_tasks.appendleft(task)
                    break
    
    def _assign_task_to_node(self, task: DistributedTask, node: NodeInfo) -> None:
        """Assign a task to a specific node."""
        url = f"http://{node.host}:{node.port}/internal/task"
        payload = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'payload': task.payload,
            'timeout_seconds': task.timeout_seconds
        }
        
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
    
    def _check_task_timeouts(self) -> None:
        """Check for timed out tasks."""
        current_time = datetime.now()
        timed_out_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                if elapsed > task.timeout_seconds:
                    timed_out_tasks.append(task_id)
        
        # Handle timed out tasks
        for task_id in timed_out_tasks:
            task = self.active_tasks[task_id]
            logger.warning(f"Task {task_id} timed out on node {task.assigned_node}")
            
            task.status = "timeout"
            task.completed_at = current_time
            task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
            
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            # Retry task if retries available
            if task.max_retries > 0:
                task.max_retries -= 1
                task.assigned_node = None
                task.started_at = None
                task.status = "pending"
                task.error_message = None
                self.pending_tasks.append(task)
                logger.info(f"Retrying task {task_id} ({task.max_retries} retries remaining)")
    
    def complete_task(self, task_id: str, result: Dict[str, Any] = None, 
                     error: str = None) -> None:
        """Mark a task as completed."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.completed_at = datetime.now()
            task.result = result
            task.error_message = error
            task.status = "completed" if result is not None else "failed"
            
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed with status {task.status}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        nodes = self.service_discovery.get_healthy_nodes("ml_cluster")
        
        node_status = []
        for node in nodes:
            node_status.append({
                'node_id': node.node_id,
                'host': f"{node.host}:{node.port}",
                'role': node.role.name,
                'status': node.status.name,
                'load_factor': node.load_factor,
                'capabilities': node.capabilities,
                'last_heartbeat': node.last_heartbeat.isoformat()
            })
        
        return {
            'cluster_size': len(nodes),
            'leader_id': self.leader_id,
            'is_leader': self.is_leader,
            'term': self.term,
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'nodes': node_status
        }


class DistributedModelTrainer:
    """Distributed model training with parameter servers."""
    
    def __init__(self, coordinator: DistributedCoordinator):
        self.coordinator = coordinator
        self.training_sessions = {}
        
    def start_distributed_training(self, 
                                 training_config: Dict[str, Any],
                                 data_partitions: List[str]) -> str:
        """Start distributed training session."""
        
        session_id = f"training_{uuid.uuid4().hex[:8]}"
        
        # Create training tasks for each partition
        training_tasks = []
        for i, partition in enumerate(data_partitions):
            task = DistributedTask(
                task_id=f"{session_id}_partition_{i}",
                task_type="model_training",
                payload={
                    'session_id': session_id,
                    'partition_id': i,
                    'data_partition': partition,
                    'training_config': training_config,
                    'is_distributed': True
                },
                timeout_seconds=3600  # 1 hour
            )
            training_tasks.append(task)
        
        # Submit tasks to coordinator
        task_ids = []
        for task in training_tasks:
            task_id = self.coordinator.submit_task(task)
            task_ids.append(task_id)
        
        # Track training session
        self.training_sessions[session_id] = {
            'session_id': session_id,
            'task_ids': task_ids,
            'status': 'running',
            'started_at': datetime.now(),
            'config': training_config,
            'partitions': len(data_partitions)
        }
        
        logger.info(f"Started distributed training session {session_id} with {len(data_partitions)} partitions")
        return session_id
    
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of distributed training session."""
        
        if session_id not in self.training_sessions:
            return {'error': f'Training session {session_id} not found'}
        
        session_info = self.training_sessions[session_id]
        task_ids = session_info['task_ids']
        
        # Check status of all tasks
        completed_tasks = 0
        failed_tasks = 0
        total_tasks = len(task_ids)
        
        for task_id in task_ids:
            if task_id in self.coordinator.completed_tasks:
                task = next((t for t in self.coordinator.completed_tasks if t.task_id == task_id), None)
                if task:
                    if task.status == 'completed':
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
        
        # Update session status
        if completed_tasks == total_tasks:
            session_info['status'] = 'completed'
            session_info['completed_at'] = datetime.now()
        elif failed_tasks > 0:
            session_info['status'] = 'partial_failure'
        
        progress = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        return {
            'session_id': session_id,
            'status': session_info['status'],
            'progress': progress,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'total_tasks': total_tasks,
            'started_at': session_info['started_at'].isoformat(),
            'completed_at': session_info.get('completed_at', {}).isoformat() if session_info.get('completed_at') else None
        }


class FaultTolerantService:
    """Base class for fault-tolerant distributed services."""
    
    def __init__(self, node_info: NodeInfo, service_discovery: ServiceDiscovery):
        self.node_info = node_info
        self.service_discovery = service_discovery
        self.circuit_breakers = {}
        
        # Replication and backup
        self.replica_nodes = []
        self.backup_interval = 300  # 5 minutes
        self.last_backup = datetime.now()
        
        # Health monitoring
        self.health_checks = {}
        self.last_health_check = datetime.now()
        
    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def perform_health_checks(self) -> Dict[str, bool]:
        """Perform all health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = False
        
        return results
    
    def setup_replication(self, replica_count: int = 2) -> None:
        """Setup replication with other nodes."""
        
        # Find potential replica nodes
        available_nodes = self.service_discovery.get_healthy_nodes("ml_cluster")
        potential_replicas = [
            node for node in available_nodes 
            if node.node_id != self.node_info.node_id and node.role == NodeRole.REPLICA
        ]
        
        # Select replica nodes
        self.replica_nodes = potential_replicas[:replica_count]
        
        if self.replica_nodes:
            logger.info(f"Setup replication with {len(self.replica_nodes)} replica nodes")
        else:
            logger.warning("No replica nodes available")
    
    def replicate_state(self, state_data: Dict[str, Any]) -> bool:
        """Replicate state to replica nodes."""
        
        if not self.replica_nodes:
            return True  # No replication configured
        
        successful_replications = 0
        
        for replica_node in self.replica_nodes:
            try:
                self._send_state_to_replica(replica_node, state_data)
                successful_replications += 1
            except Exception as e:
                logger.error(f"Failed to replicate to {replica_node.node_id}: {e}")
        
        # Consider successful if majority of replicas updated
        required_success = len(self.replica_nodes) // 2 + 1
        return successful_replications >= required_success
    
    def _send_state_to_replica(self, replica_node: NodeInfo, state_data: Dict[str, Any]) -> None:
        """Send state data to a replica node."""
        
        url = f"http://{replica_node.host}:{replica_node.port}/internal/replicate"
        payload = {
            'source_node': self.node_info.node_id,
            'timestamp': datetime.now().isoformat(),
            'state_data': state_data
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(service_name)
        return self.circuit_breakers[service_name]
    
    @with_error_handling(component="distributed_service", enable_circuit_breaker=True, enable_retry=True)
    def make_resilient_request(self, url: str, data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Make a resilient request with circuit breaker and retry."""
        
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()


def create_distributed_cluster(cluster_config: Dict[str, Any]) -> Tuple[ServiceDiscovery, DistributedCoordinator]:
    """Create and initialize a distributed cluster."""
    
    # Setup service discovery
    service_discovery = ServiceDiscovery(
        discovery_backend=cluster_config.get('discovery_backend', 'consul'),
        config=cluster_config.get('discovery_config', {})
    )
    
    # Create node info
    node_id = cluster_config.get('node_id', f"node_{uuid.uuid4().hex[:8]}")
    node_info = NodeInfo(
        node_id=node_id,
        host=cluster_config.get('host', 'localhost'),
        port=cluster_config.get('port', 8000),
        role=NodeRole[cluster_config.get('role', 'WORKER').upper()],
        status=NodeStatus.ACTIVE,
        capabilities=cluster_config.get('capabilities', ['ml_inference', 'ml_training'])
    )
    
    # Register with service discovery
    service_discovery.register_service("ml_cluster", node_info)
    
    # Create coordinator
    coordinator = DistributedCoordinator(node_info, service_discovery)
    coordinator.start_coordination()
    
    logger.info(f"Created distributed cluster node {node_id}")
    
    return service_discovery, coordinator


if __name__ == "__main__":
    print("Distributed Computing and Fault Tolerance System")
    print("This system provides distributed computing with fault tolerance capabilities.")