"""
Auto-Scaling and Resource Optimization System.

This module provides intelligent auto-scaling and resource optimization capabilities including:
- Dynamic resource allocation based on demand prediction
- Multi-tier auto-scaling (compute, memory, storage)
- Cost-aware resource optimization
- Load balancing and traffic distribution
- Resource prediction and capacity planning
- Kubernetes and cloud-native scaling
- Performance-cost optimization algorithms
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import psutil
import redis
import requests
from kubernetes import client, config
import boto3
from azure.mgmt.compute import ComputeManagementClient
from google.cloud import compute_v1

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .error_handling_recovery import with_error_handling, error_handler

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    active_requests: int = 0
    queue_length: int = 0
    response_time_avg: float = 0.0
    throughput_rps: float = 0.0


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    decision_id: str
    timestamp: datetime
    action: str  # scale_up, scale_down, maintain
    resource_type: str  # cpu, memory, gpu, storage, replicas
    current_value: float
    target_value: float
    confidence: float
    reasoning: str
    estimated_cost_impact: float
    estimated_performance_impact: float


@dataclass
class ResourcePrediction:
    """Resource demand prediction."""
    timestamp: datetime
    horizon_minutes: int
    predicted_cpu: float
    predicted_memory: float
    predicted_requests: float
    confidence_interval: Tuple[float, float]
    seasonal_component: float
    trend_component: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    resource_type: str
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_step_size: float = 1.2  # multiply by this factor
    enable_predictive_scaling: bool = True
    cost_optimization_weight: float = 0.3


class ResourceMonitor:
    """Comprehensive resource monitoring system."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Cloud provider clients (initialized on demand)
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None
        self.k8s_client = None
        
        # Request tracking
        self.request_counter = 0
        self.response_times = deque(maxlen=1000)
        self.request_timestamps = deque(maxlen=1000)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # GPU metrics (if available)
        gpu_percent, gpu_memory_percent = self._get_gpu_metrics()
        
        # Application metrics
        active_requests, queue_length = self._get_application_metrics()
        response_time_avg = self._calculate_average_response_time()
        throughput_rps = self._calculate_throughput()
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_io_sent=network_io.bytes_sent if network_io else 0,
            network_io_recv=network_io.bytes_recv if network_io else 0,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            active_requests=active_requests,
            queue_length=queue_length,
            response_time_avg=response_time_avg,
            throughput_rps=throughput_rps
        )
    
    def _get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU utilization metrics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_percent = gpu_util.gpu
            gpu_memory_percent = (memory_info.used / memory_info.total) * 100
            
            return gpu_percent, gpu_memory_percent
        except:
            return 0.0, 0.0
    
    def _get_application_metrics(self) -> Tuple[int, int]:
        """Get application-specific metrics."""
        # This would be implemented based on your application
        # For now, return simulated values
        return self.request_counter, len(self.response_times)
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        with self.lock:
            if not self.response_times:
                return 0.0
            return sum(self.response_times) / len(self.response_times)
    
    def _calculate_throughput(self) -> float:
        """Calculate requests per second."""
        with self.lock:
            if len(self.request_timestamps) < 2:
                return 0.0
            
            # Count requests in last 60 seconds
            now = datetime.now()
            recent_requests = [ts for ts in self.request_timestamps 
                             if (now - ts).total_seconds() <= 60]
            
            return len(recent_requests) / 60.0
    
    def record_request(self, response_time: float) -> None:
        """Record a request and its response time."""
        with self.lock:
            self.request_counter += 1
            self.response_times.append(response_time)
            self.request_timestamps.append(datetime.now())
    
    def get_recent_metrics(self, minutes: int = 10) -> List[ResourceMetrics]:
        """Get metrics from the last N minutes."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_current_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'gpu_percent': latest.gpu_percent,
            'gpu_memory_percent': latest.gpu_memory_percent,
            'response_time_avg': latest.response_time_avg,
            'throughput_rps': latest.throughput_rps
        }


class DemandPredictor:
    """Predict future resource demand using ML."""
    
    def __init__(self):
        self.cpu_model = None
        self.memory_model = None
        self.request_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.feature_window = 60  # minutes of features
        self.prediction_horizon = 30  # predict 30 minutes ahead
    
    def train_models(self, metrics_history: List[ResourceMetrics]) -> None:
        """Train prediction models on historical data."""
        if len(metrics_history) < 100:
            logger.warning("Not enough data to train prediction models")
            return
        
        try:
            # Prepare features and targets
            X, y_cpu, y_memory, y_requests = self._prepare_training_data(metrics_history)
            
            if len(X) < 50:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.cpu_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.memory_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.request_model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            self.cpu_model.fit(X_scaled, y_cpu)
            self.memory_model.fit(X_scaled, y_memory)
            self.request_model.fit(X_scaled, y_requests)
            
            self.is_trained = True
            logger.info("Demand prediction models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train prediction models: {e}")
    
    def _prepare_training_data(self, metrics_history: List[ResourceMetrics]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from metrics history."""
        # Convert to DataFrame for easier processing
        data = []
        for m in metrics_history:
            data.append({
                'timestamp': m.timestamp,
                'cpu_percent': m.cpu_percent,
                'memory_percent': m.memory_percent,
                'active_requests': m.active_requests,
                'response_time': m.response_time_avg,
                'throughput': m.throughput_rps
            })
        
        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        # Create time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Create lag features
        for lag in [1, 5, 15, 30]:  # 1, 5, 15, 30 periods ago
            df[f'cpu_lag_{lag}'] = df['cpu_percent'].shift(lag)
            df[f'memory_lag_{lag}'] = df['memory_percent'].shift(lag)
            df[f'requests_lag_{lag}'] = df['active_requests'].shift(lag)
        
        # Rolling statistics
        for window in [5, 15, 30]:
            df[f'cpu_mean_{window}'] = df['cpu_percent'].rolling(window).mean()
            df[f'memory_mean_{window}'] = df['memory_percent'].rolling(window).mean()
            df[f'cpu_std_{window}'] = df['cpu_percent'].rolling(window).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Features (excluding targets and timestamp)
        feature_columns = [col for col in df.columns 
                          if col not in ['timestamp', 'cpu_percent', 'memory_percent', 'active_requests']]
        X = df[feature_columns].values
        
        # Targets (shifted forward for prediction)
        future_steps = 6  # 30 minutes / 5 minute intervals
        y_cpu = df['cpu_percent'].shift(-future_steps).fillna(method='ffill').values
        y_memory = df['memory_percent'].shift(-future_steps).fillna(method='ffill').values
        y_requests = df['active_requests'].shift(-future_steps).fillna(method='ffill').values
        
        # Remove last few rows that don't have future targets
        X = X[:-future_steps]
        y_cpu = y_cpu[:-future_steps]
        y_memory = y_memory[:-future_steps]
        y_requests = y_requests[:-future_steps]
        
        return X, y_cpu, y_memory, y_requests
    
    def predict_demand(self, recent_metrics: List[ResourceMetrics]) -> Optional[ResourcePrediction]:
        """Predict future resource demand."""
        if not self.is_trained or len(recent_metrics) < 30:
            return None
        
        try:
            # Prepare features from recent metrics
            features = self._extract_prediction_features(recent_metrics)
            if features is None:
                return None
            
            features_scaled = self.scaler.transform([features])
            
            # Make predictions
            cpu_pred = self.cpu_model.predict(features_scaled)[0]
            memory_pred = self.memory_model.predict(features_scaled)[0]
            requests_pred = self.request_model.predict(features_scaled)[0]
            
            # Calculate confidence intervals (simplified)
            cpu_std = np.std([m.cpu_percent for m in recent_metrics[-30:]])
            confidence_interval = (cpu_pred - 1.96 * cpu_std, cpu_pred + 1.96 * cpu_std)
            
            # Extract seasonal and trend components (simplified)
            recent_cpu = [m.cpu_percent for m in recent_metrics[-10:]]
            trend = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
            seasonal = cpu_pred - np.mean(recent_cpu) - trend
            
            return ResourcePrediction(
                timestamp=datetime.now(),
                horizon_minutes=self.prediction_horizon,
                predicted_cpu=cpu_pred,
                predicted_memory=memory_pred,
                predicted_requests=requests_pred,
                confidence_interval=confidence_interval,
                seasonal_component=seasonal,
                trend_component=trend
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _extract_prediction_features(self, recent_metrics: List[ResourceMetrics]) -> Optional[np.ndarray]:
        """Extract features for prediction from recent metrics."""
        if len(recent_metrics) < 30:
            return None
        
        latest = recent_metrics[-1]
        
        # Time-based features
        hour = latest.timestamp.hour
        dow = latest.timestamp.dayofweek
        
        features = [
            latest.response_time_avg,
            latest.throughput_rps,
            # Time features
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
        
        # Lag features
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        request_values = [m.active_requests for m in recent_metrics]
        
        for lag in [1, 5, 15, 30]:
            if len(cpu_values) > lag:
                features.extend([
                    cpu_values[-lag-1],
                    memory_values[-lag-1],
                    request_values[-lag-1]
                ])
            else:
                features.extend([0, 0, 0])
        
        # Rolling statistics
        for window in [5, 15, 30]:
            if len(cpu_values) >= window:
                window_cpu = cpu_values[-window:]
                window_memory = memory_values[-window:]
                features.extend([
                    np.mean(window_cpu),
                    np.mean(window_memory),
                    np.std(window_cpu)
                ])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features)


class AutoScalingEngine:
    """Intelligent auto-scaling decision engine."""
    
    def __init__(self, policies: Dict[str, ScalingPolicy] = None):
        self.policies = policies or self._get_default_policies()
        self.scaling_history = deque(maxlen=1000)
        self.cooldown_timers = {}
        
        # Cost optimization
        self.cost_models = {}
        self.performance_models = {}
        
    def _get_default_policies(self) -> Dict[str, ScalingPolicy]:
        """Get default scaling policies."""
        return {
            'cpu': ScalingPolicy(
                resource_type='cpu',
                target_cpu_utilization=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                scale_up_cooldown=300,
                scale_down_cooldown=600
            ),
            'memory': ScalingPolicy(
                resource_type='memory',
                target_cpu_utilization=75.0,
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                scale_up_cooldown=300,
                scale_down_cooldown=600
            ),
            'replicas': ScalingPolicy(
                resource_type='replicas',
                min_replicas=1,
                max_replicas=20,
                target_cpu_utilization=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                scale_up_cooldown=300,
                scale_down_cooldown=900
            )
        }
    
    def make_scaling_decision(self, 
                            current_metrics: ResourceMetrics,
                            recent_metrics: List[ResourceMetrics],
                            prediction: Optional[ResourcePrediction] = None) -> List[ScalingDecision]:
        """Make auto-scaling decisions based on current state and predictions."""
        
        decisions = []
        
        for resource_type, policy in self.policies.items():
            decision = self._evaluate_scaling_policy(
                policy, current_metrics, recent_metrics, prediction
            )
            
            if decision and self._is_scaling_allowed(decision, policy):
                decisions.append(decision)
                
                # Update cooldown timer
                self.cooldown_timers[resource_type] = datetime.now()
                
                # Record decision
                self.scaling_history.append(decision)
        
        return decisions
    
    def _evaluate_scaling_policy(self,
                                policy: ScalingPolicy,
                                current_metrics: ResourceMetrics,
                                recent_metrics: List[ResourceMetrics],
                                prediction: Optional[ResourcePrediction]) -> Optional[ScalingDecision]:
        """Evaluate a specific scaling policy."""
        
        # Get current utilization for this resource type
        if policy.resource_type == 'cpu':
            current_utilization = current_metrics.cpu_percent
            predicted_utilization = prediction.predicted_cpu if prediction else current_utilization
        elif policy.resource_type == 'memory':
            current_utilization = current_metrics.memory_percent
            predicted_utilization = prediction.predicted_memory if prediction else current_utilization
        elif policy.resource_type == 'replicas':
            # For replicas, we consider overall system load
            current_utilization = max(current_metrics.cpu_percent, current_metrics.memory_percent)
            predicted_utilization = max(
                prediction.predicted_cpu if prediction else current_utilization,
                prediction.predicted_memory if prediction else current_utilization
            )
        else:
            return None
        
        # Determine action
        action = "maintain"
        target_value = current_utilization
        reasoning = "No scaling needed"
        
        # Predictive scaling consideration
        utilization_to_check = predicted_utilization if (prediction and policy.enable_predictive_scaling) else current_utilization
        
        if utilization_to_check > policy.scale_up_threshold:
            action = "scale_up"
            if policy.resource_type == 'replicas':
                target_value = min(int(current_utilization * policy.scale_step_size), policy.max_replicas)
                reasoning = f"High utilization predicted ({utilization_to_check:.1f}% > {policy.scale_up_threshold}%)"
            else:
                target_value = current_utilization / policy.target_cpu_utilization * 100
                reasoning = f"Scale up to handle predicted load ({utilization_to_check:.1f}%)"
                
        elif utilization_to_check < policy.scale_down_threshold:
            action = "scale_down"
            if policy.resource_type == 'replicas':
                target_value = max(int(current_utilization / policy.scale_step_size), policy.min_replicas)
                reasoning = f"Low utilization detected ({utilization_to_check:.1f}% < {policy.scale_down_threshold}%)"
            else:
                target_value = current_utilization / policy.target_cpu_utilization * 100
                reasoning = f"Scale down to save resources ({utilization_to_check:.1f}%)"
        
        if action == "maintain":
            return None
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(
            policy, current_metrics, recent_metrics, prediction
        )
        
        # Estimate cost and performance impact
        cost_impact = self._estimate_cost_impact(policy, action, current_utilization, target_value)
        performance_impact = self._estimate_performance_impact(
            policy, action, current_utilization, target_value
        )
        
        # Apply cost optimization
        if policy.cost_optimization_weight > 0:
            cost_adjusted_confidence = confidence * (1 - policy.cost_optimization_weight * abs(cost_impact))
            confidence = max(0.1, cost_adjusted_confidence)
        
        return ScalingDecision(
            decision_id=f"{policy.resource_type}_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            action=action,
            resource_type=policy.resource_type,
            current_value=current_utilization,
            target_value=target_value,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact
        )
    
    def _calculate_decision_confidence(self,
                                     policy: ScalingPolicy,
                                     current_metrics: ResourceMetrics,
                                     recent_metrics: List[ResourceMetrics],
                                     prediction: Optional[ResourcePrediction]) -> float:
        """Calculate confidence in scaling decision."""
        
        base_confidence = 0.7
        
        # Increase confidence if trend is consistent
        if len(recent_metrics) >= 5:
            if policy.resource_type == 'cpu':
                recent_values = [m.cpu_percent for m in recent_metrics[-5:]]
            elif policy.resource_type == 'memory':
                recent_values = [m.memory_percent for m in recent_metrics[-5:]]
            else:
                recent_values = [max(m.cpu_percent, m.memory_percent) for m in recent_metrics[-5:]]
            
            # Check if trend is consistent
            if len(recent_values) >= 3:
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                if abs(trend) > 2:  # Strong trend
                    base_confidence += 0.2
        
        # Increase confidence if prediction is available
        if prediction:
            base_confidence += 0.1
            
            # Higher confidence if prediction confidence interval is narrow
            if prediction.confidence_interval:
                ci_width = prediction.confidence_interval[1] - prediction.confidence_interval[0]
                if ci_width < 20:  # Narrow confidence interval
                    base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _estimate_cost_impact(self,
                            policy: ScalingPolicy,
                            action: str,
                            current_value: float,
                            target_value: float) -> float:
        """Estimate cost impact of scaling decision."""
        
        # Simplified cost model (in practice, use actual cloud pricing)
        if policy.resource_type == 'replicas':
            if action == "scale_up":
                return 0.3  # 30% cost increase for scale up
            else:
                return -0.2  # 20% cost savings for scale down
        else:
            # Resource scaling
            change_ratio = target_value / current_value if current_value > 0 else 1
            return (change_ratio - 1) * 0.5  # 50% of change is cost impact
    
    def _estimate_performance_impact(self,
                                   policy: ScalingPolicy,
                                   action: str,
                                   current_value: float,
                                   target_value: float) -> float:
        """Estimate performance impact of scaling decision."""
        
        if action == "scale_up":
            return 0.2  # 20% performance improvement
        elif action == "scale_down":
            return -0.1  # 10% performance reduction
        else:
            return 0.0
    
    def _is_scaling_allowed(self, decision: ScalingDecision, policy: ScalingPolicy) -> bool:
        """Check if scaling is allowed based on cooldown periods."""
        
        if decision.resource_type in self.cooldown_timers:
            last_scaling = self.cooldown_timers[decision.resource_type]
            elapsed = (datetime.now() - last_scaling).total_seconds()
            
            if decision.action == "scale_up":
                return elapsed >= policy.scale_up_cooldown
            else:
                return elapsed >= policy.scale_down_cooldown
        
        return True
    
    def get_scaling_history(self, hours: int = 24) -> List[ScalingDecision]:
        """Get scaling decisions from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [d for d in self.scaling_history if d.timestamp >= cutoff_time]


class CloudResourceManager:
    """Manage cloud resources across different providers."""
    
    def __init__(self):
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None
        self.k8s_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize cloud provider clients."""
        try:
            # Kubernetes
            if os.path.exists(os.path.expanduser("~/.kube/config")):
                config.load_kube_config()
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
        
        try:
            # AWS
            self.aws_client = boto3.client('ec2')
            logger.info("AWS client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client: {e}")
    
    @with_error_handling(component="cloud_scaling", enable_retry=True)
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision on cloud infrastructure."""
        
        if decision.resource_type == 'replicas':
            return self._scale_replicas(decision)
        elif decision.resource_type in ['cpu', 'memory']:
            return self._scale_resources(decision)
        else:
            logger.warning(f"Unsupported scaling resource type: {decision.resource_type}")
            return False
    
    def _scale_replicas(self, decision: ScalingDecision) -> bool:
        """Scale the number of replicas/instances."""
        
        if self.k8s_client:
            return self._scale_kubernetes_replicas(decision)
        elif self.aws_client:
            return self._scale_aws_instances(decision)
        else:
            logger.error("No cloud client available for replica scaling")
            return False
    
    def _scale_kubernetes_replicas(self, decision: ScalingDecision) -> bool:
        """Scale Kubernetes deployment replicas."""
        try:
            # This is a simplified example - in practice, you'd specify the deployment name
            deployment_name = os.getenv('K8S_DEPLOYMENT_NAME', 'ml-service')
            namespace = os.getenv('K8S_NAMESPACE', 'default')
            
            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            # Update replica count
            new_replicas = int(decision.target_value)
            deployment.spec.replicas = new_replicas
            
            # Apply update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled Kubernetes deployment {deployment_name} to {new_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale Kubernetes replicas: {e}")
            return False
    
    def _scale_aws_instances(self, decision: ScalingDecision) -> bool:
        """Scale AWS Auto Scaling Group."""
        try:
            autoscaling_client = boto3.client('autoscaling')
            asg_name = os.getenv('AWS_ASG_NAME', 'ml-service-asg')
            
            new_capacity = int(decision.target_value)
            
            autoscaling_client.update_auto_scaling_group(
                AutoScalingGroupName=asg_name,
                DesiredCapacity=new_capacity
            )
            
            logger.info(f"Scaled AWS ASG {asg_name} to {new_capacity} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale AWS instances: {e}")
            return False
    
    def _scale_resources(self, decision: ScalingDecision) -> bool:
        """Scale compute resources (CPU/Memory)."""
        
        if self.k8s_client:
            return self._scale_kubernetes_resources(decision)
        else:
            logger.warning("Resource scaling only supported on Kubernetes currently")
            return False
    
    def _scale_kubernetes_resources(self, decision: ScalingDecision) -> bool:
        """Scale Kubernetes resource limits."""
        try:
            deployment_name = os.getenv('K8S_DEPLOYMENT_NAME', 'ml-service')
            namespace = os.getenv('K8S_NAMESPACE', 'default')
            
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            # Update resource limits (simplified)
            container = deployment.spec.template.spec.containers[0]
            
            if decision.resource_type == 'cpu':
                # Scale CPU (e.g., from "500m" to "1000m")
                current_cpu = container.resources.limits.get('cpu', '500m')
                cpu_millicores = int(current_cpu.replace('m', ''))
                new_cpu_millicores = int(cpu_millicores * (decision.target_value / decision.current_value))
                container.resources.limits['cpu'] = f"{new_cpu_millicores}m"
                
            elif decision.resource_type == 'memory':
                # Scale Memory (e.g., from "1Gi" to "2Gi")
                current_memory = container.resources.limits.get('memory', '1Gi')
                memory_gi = float(current_memory.replace('Gi', ''))
                new_memory_gi = memory_gi * (decision.target_value / decision.current_value)
                container.resources.limits['memory'] = f"{new_memory_gi:.1f}Gi"
            
            # Apply update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled Kubernetes {decision.resource_type} resources for {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale Kubernetes resources: {e}")
            return False
    
    def get_current_scale(self) -> Dict[str, Any]:
        """Get current scaling configuration."""
        
        result = {}
        
        if self.k8s_client:
            try:
                deployment_name = os.getenv('K8S_DEPLOYMENT_NAME', 'ml-service')
                namespace = os.getenv('K8S_NAMESPACE', 'default')
                
                deployment = self.k8s_client.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace
                )
                
                container = deployment.spec.template.spec.containers[0]
                
                result['kubernetes'] = {
                    'replicas': deployment.spec.replicas,
                    'cpu_limit': container.resources.limits.get('cpu'),
                    'memory_limit': container.resources.limits.get('memory'),
                    'ready_replicas': deployment.status.ready_replicas
                }
                
            except Exception as e:
                logger.error(f"Failed to get Kubernetes scale info: {e}")
        
        return result


class AutoScalingSystem:
    """Complete auto-scaling and resource optimization system."""
    
    def __init__(self, policies: Dict[str, ScalingPolicy] = None):
        self.monitor = ResourceMonitor(collection_interval=30)
        self.predictor = DemandPredictor()
        self.scaler = AutoScalingEngine(policies)
        self.cloud_manager = CloudResourceManager()
        
        self.running = False
        self.thread = None
        
        # Performance tracking
        self.optimization_history = []
        self.cost_savings = 0.0
        self.performance_improvements = 0.0
    
    def start(self) -> None:
        """Start the auto-scaling system."""
        if self.running:
            return
        
        self.running = True
        self.monitor.start_monitoring()
        
        # Start scaling decision loop
        self.thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.thread.start()
        
        logger.info("Auto-scaling system started")
    
    def stop(self) -> None:
        """Stop the auto-scaling system."""
        self.running = False
        self.monitor.stop_monitoring()
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Auto-scaling system stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        
        train_counter = 0
        
        while self.running:
            try:
                # Get current metrics
                current_utilization = self.monitor.get_current_utilization()
                recent_metrics = self.monitor.get_recent_metrics(minutes=30)
                
                if not recent_metrics:
                    time.sleep(60)
                    continue
                
                current_metrics = recent_metrics[-1]
                
                # Periodically retrain prediction models
                train_counter += 1
                if train_counter >= 10:  # Every ~10 minutes
                    all_metrics = self.monitor.get_recent_metrics(minutes=1440)  # 24 hours
                    self.predictor.train_models(all_metrics)
                    train_counter = 0
                
                # Make prediction
                prediction = self.predictor.predict_demand(recent_metrics)
                
                # Make scaling decisions
                decisions = self.scaler.make_scaling_decision(
                    current_metrics, recent_metrics, prediction
                )
                
                # Execute decisions
                for decision in decisions:
                    success = self.cloud_manager.execute_scaling_decision(decision)
                    
                    if success:
                        logger.info(f"Executed scaling decision: {decision.action} {decision.resource_type} "
                                  f"(confidence: {decision.confidence:.2f})")
                        
                        # Track optimization benefits
                        self._track_optimization_benefits(decision)
                    else:
                        logger.error(f"Failed to execute scaling decision: {decision.decision_id}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)
    
    def _track_optimization_benefits(self, decision: ScalingDecision) -> None:
        """Track cost savings and performance improvements."""
        
        # Update cumulative tracking
        self.cost_savings += decision.estimated_cost_impact
        self.performance_improvements += decision.estimated_performance_impact
        
        # Record in history
        self.optimization_history.append({
            'timestamp': decision.timestamp,
            'action': decision.action,
            'resource_type': decision.resource_type,
            'cost_impact': decision.estimated_cost_impact,
            'performance_impact': decision.estimated_performance_impact,
            'confidence': decision.confidence
        })
        
        # Keep only last 7 days of history
        cutoff = datetime.now() - timedelta(days=7)
        self.optimization_history = [
            h for h in self.optimization_history 
            if h['timestamp'] >= cutoff
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling system status."""
        
        current_utilization = self.monitor.get_current_utilization()
        recent_decisions = self.scaler.get_scaling_history(hours=24)
        cloud_status = self.cloud_manager.get_current_scale()
        
        return {
            'system_status': {
                'running': self.running,
                'predictor_trained': self.predictor.is_trained,
                'cloud_providers_available': len([
                    client for client in [
                        self.cloud_manager.k8s_client,
                        self.cloud_manager.aws_client,
                        self.cloud_manager.azure_client
                    ] if client is not None
                ])
            },
            'current_utilization': current_utilization,
            'cloud_configuration': cloud_status,
            'recent_decisions': len(recent_decisions),
            'optimization_benefits': {
                'estimated_cost_savings': self.cost_savings,
                'estimated_performance_improvements': self.performance_improvements,
                'total_optimizations': len(self.optimization_history)
            },
            'scaling_policies': {
                policy_name: {
                    'resource_type': policy.resource_type,
                    'scale_up_threshold': policy.scale_up_threshold,
                    'scale_down_threshold': policy.scale_down_threshold,
                    'predictive_scaling': policy.enable_predictive_scaling
                }
                for policy_name, policy in self.scaler.policies.items()
            }
        }
    
    def force_scaling_decision(self, resource_type: str, action: str) -> bool:
        """Force a manual scaling decision."""
        
        current_metrics = self.monitor.get_recent_metrics(minutes=1)
        if not current_metrics:
            return False
        
        latest = current_metrics[-1]
        
        if resource_type == 'cpu':
            current_value = latest.cpu_percent
        elif resource_type == 'memory':
            current_value = latest.memory_percent
        elif resource_type == 'replicas':
            current_value = max(latest.cpu_percent, latest.memory_percent)
        else:
            return False
        
        # Create manual decision
        decision = ScalingDecision(
            decision_id=f"manual_{resource_type}_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            action=action,
            resource_type=resource_type,
            current_value=current_value,
            target_value=current_value * 1.5 if action == "scale_up" else current_value * 0.7,
            confidence=1.0,
            reasoning="Manual scaling request",
            estimated_cost_impact=0.1 if action == "scale_up" else -0.1,
            estimated_performance_impact=0.1 if action == "scale_up" else -0.1
        )
        
        return self.cloud_manager.execute_scaling_decision(decision)
    
    def update_scaling_policy(self, resource_type: str, policy: ScalingPolicy) -> None:
        """Update a scaling policy."""
        self.scaler.policies[resource_type] = policy
        logger.info(f"Updated scaling policy for {resource_type}")
    
    def get_cost_optimization_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate cost optimization report."""
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_optimizations = [
            h for h in self.optimization_history 
            if h['timestamp'] >= cutoff
        ]
        
        if not recent_optimizations:
            return {'message': 'No optimization data available'}
        
        total_cost_impact = sum(h['cost_impact'] for h in recent_optimizations)
        total_performance_impact = sum(h['performance_impact'] for h in recent_optimizations)
        
        scale_up_count = len([h for h in recent_optimizations if h['action'] == 'scale_up'])
        scale_down_count = len([h for h in recent_optimizations if h['action'] == 'scale_down'])
        
        return {
            'reporting_period_days': days,
            'total_optimizations': len(recent_optimizations),
            'scale_up_actions': scale_up_count,
            'scale_down_actions': scale_down_count,
            'estimated_cost_impact': total_cost_impact,
            'estimated_performance_impact': total_performance_impact,
            'average_confidence': np.mean([h['confidence'] for h in recent_optimizations]),
            'resource_types_optimized': list(set([h['resource_type'] for h in recent_optimizations])),
            'cost_savings_trend': self._calculate_trend([h['cost_impact'] for h in recent_optimizations])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


# Factory function
def create_auto_scaling_system(policies: Dict[str, ScalingPolicy] = None) -> AutoScalingSystem:
    """Create and configure auto-scaling system."""
    return AutoScalingSystem(policies)


if __name__ == "__main__":
    print("Auto-Scaling and Resource Optimization System")
    print("This system provides intelligent auto-scaling and cost optimization.")