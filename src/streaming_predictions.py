"""
Real-time Streaming Prediction System with Event Processing.

This module provides comprehensive streaming capabilities including:
- Real-time data ingestion from multiple sources
- Stream processing with windowing and aggregations
- Low-latency prediction serving
- Event-driven model updates
- Streaming analytics and monitoring
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field
import websockets
import kafka
from kafka import KafkaProducer, KafkaConsumer
import aioredis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .predict_churn import make_prediction
from .model_cache import get_model_from_cache

logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming system."""
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Kafka configuration
    kafka_bootstrap_servers: List[str] = None
    kafka_topic_input: str = "customer_data_stream"
    kafka_topic_predictions: str = "churn_predictions_stream"
    kafka_group_id: str = "churn_predictor_group"
    
    # Stream processing
    batch_size: int = 100
    batch_timeout_seconds: float = 1.0
    max_queue_size: int = 10000
    window_size_seconds: int = 60
    
    # Performance settings
    num_worker_threads: int = 4
    prediction_cache_ttl: int = 300  # 5 minutes
    enable_batching: bool = True
    enable_windowing: bool = True
    
    # Monitoring
    metrics_interval_seconds: int = 10
    enable_detailed_logging: bool = False


@dataclass
class StreamingEvent:
    """Represents a streaming event."""
    event_id: str
    customer_id: str
    timestamp: datetime
    data: Dict[str, Any]
    event_type: str = "prediction_request"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Result of a streaming prediction."""
    event_id: str
    customer_id: str
    prediction: int
    probability: float
    confidence: float
    model_version: str
    processing_time_ms: float
    timestamp: datetime
    features_used: List[str]
    metadata: Optional[Dict[str, Any]] = None


class StreamingEventBuffer:
    """Thread-safe event buffer with windowing support."""
    
    def __init__(self, max_size: int = 10000, window_size_seconds: int = 60):
        self.max_size = max_size
        self.window_size = timedelta(seconds=window_size_seconds)
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.event_index = defaultdict(list)  # customer_id -> [events]
        
    def add_event(self, event: StreamingEvent) -> None:
        """Add event to buffer."""
        with self.lock:
            self.buffer.append(event)
            self.event_index[event.customer_id].append(event)
            
            # Clean old events from index
            cutoff_time = datetime.now() - self.window_size
            self.event_index[event.customer_id] = [
                e for e in self.event_index[event.customer_id] 
                if e.timestamp > cutoff_time
            ]
    
    def get_events_for_customer(self, customer_id: str, 
                              window_seconds: Optional[int] = None) -> List[StreamingEvent]:
        """Get recent events for a specific customer."""
        with self.lock:
            if customer_id not in self.event_index:
                return []
            
            if window_seconds is None:
                return list(self.event_index[customer_id])
            
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            return [e for e in self.event_index[customer_id] if e.timestamp > cutoff_time]
    
    def get_batch(self, max_size: int) -> List[StreamingEvent]:
        """Get a batch of events from buffer."""
        with self.lock:
            batch = []
            for _ in range(min(max_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class StreamingPredictor:
    """Core streaming prediction engine."""
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.model_version = "unknown"
        
        # Event processing
        self.event_buffer = StreamingEventBuffer(
            max_size=self.config.max_queue_size,
            window_size_seconds=self.config.window_size_seconds
        )
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_worker_threads)
        self.processing_active = False
        
        # Caching
        self.prediction_cache = {}
        self.cache_timestamps = {}
        
        # Metrics
        self.metrics_collector = get_metrics_collector()
        self.processing_stats = {
            'total_processed': 0,
            'total_predictions': 0,
            'average_latency_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load prediction model and preprocessor."""
        try:
            # Try to load from cache first
            cached_model = get_model_from_cache()
            if cached_model:
                self.model = cached_model['model']
                self.preprocessor = cached_model.get('preprocessor')
                self.model_version = cached_model.get('version', 'cached')
                logger.info(f"Loaded model from cache: {self.model_version}")
                return
            
            # Load from files
            model_path = "models/churn_model.joblib"
            preprocessor_path = "models/preprocessor.joblib"
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Loaded model from file")
            
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor from file")
            
            # Load feature columns
            feature_columns_path = "models/feature_columns.json"
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    async def process_event(self, event: StreamingEvent) -> Optional[PredictionResult]:
        """Process a single streaming event."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{event.customer_id}_{hash(str(sorted(event.data.items())))}"
            
            if self._is_cached_prediction_valid(cache_key):
                cached_result = self.prediction_cache[cache_key]
                self.processing_stats['cache_hits'] += 1
                
                # Update timestamp and return cached result
                cached_result.timestamp = datetime.now()
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            self.processing_stats['cache_misses'] += 1
            
            # Make prediction
            if self.model is None:
                logger.warning("No model available for prediction")
                return None
            
            # Prepare data
            customer_data = event.data.copy()
            
            # Add windowed features if enabled
            if self.config.enable_windowing:
                windowed_features = self._extract_windowed_features(event)
                customer_data.update(windowed_features)
            
            # Make prediction
            prediction, probability = make_prediction(customer_data)
            
            if prediction is None:
                logger.warning(f"Prediction failed for event {event.event_id}")
                return None
            
            # Calculate confidence (simplified)
            confidence = abs(probability - 0.5) * 2  # Distance from decision boundary
            
            # Create result
            result = PredictionResult(
                event_id=event.event_id,
                customer_id=event.customer_id,
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                model_version=self.model_version,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                features_used=list(customer_data.keys()),
                metadata=event.metadata
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
            
            # Update metrics
            self.processing_stats['total_predictions'] += 1
            self._update_latency_metrics(result.processing_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.processing_stats['errors'] += 1
            return None
    
    def _is_cached_prediction_valid(self, cache_key: str) -> bool:
        """Check if cached prediction is still valid."""
        if cache_key not in self.prediction_cache:
            return False
        
        cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
        return cache_age < self.config.prediction_cache_ttl
    
    def _extract_windowed_features(self, event: StreamingEvent) -> Dict[str, float]:
        """Extract features from windowed customer events."""
        customer_events = self.event_buffer.get_events_for_customer(
            event.customer_id, 
            window_seconds=self.config.window_size_seconds
        )
        
        if not customer_events:
            return {}
        
        windowed_features = {}
        
        # Count of events in window
        windowed_features['events_in_window'] = len(customer_events)
        
        # Time since last event
        if len(customer_events) > 1:
            last_event_time = max(e.timestamp for e in customer_events[:-1])
            time_since_last = (event.timestamp - last_event_time).total_seconds()
            windowed_features['time_since_last_event_seconds'] = time_since_last
        
        # Aggregated numerical features
        numerical_values = defaultdict(list)
        for e in customer_events:
            for key, value in e.data.items():
                if isinstance(value, (int, float)):
                    numerical_values[key].append(value)
        
        for feature, values in numerical_values.items():
            if len(values) > 1:
                windowed_features[f'{feature}_window_mean'] = np.mean(values)
                windowed_features[f'{feature}_window_std'] = np.std(values)
                windowed_features[f'{feature}_window_trend'] = values[-1] - values[0]
        
        return windowed_features
    
    def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics."""
        total = self.processing_stats['total_predictions']
        current_avg = self.processing_stats['average_latency_ms']
        
        # Exponential moving average
        alpha = 0.1
        self.processing_stats['average_latency_ms'] = (
            alpha * latency_ms + (1 - alpha) * current_avg
        )
        
        # Record to metrics collector
        try:
            self.metrics_collector.record_prediction_latency(latency_ms / 1000, "streaming")
        except Exception as e:
            logger.warning(f"Failed to record latency metric: {e}")
    
    async def process_batch(self, events: List[StreamingEvent]) -> List[PredictionResult]:
        """Process a batch of events."""
        if not events:
            return []
        
        start_time = time.time()
        
        # Process events concurrently
        tasks = [self.process_event(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, PredictionResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Exception in batch processing: {result}")
        
        batch_time = (time.time() - start_time) * 1000
        logger.info(f"Processed batch of {len(events)} events in {batch_time:.2f}ms")
        
        return valid_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.processing_stats,
            'buffer_size': self.event_buffer.size(),
            'cache_size': len(self.prediction_cache),
            'model_loaded': self.model is not None,
            'model_version': self.model_version
        }


class StreamingServer:
    """FastAPI-based streaming server."""
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.predictor = StreamingPredictor(config)
        self.app = self._create_app()
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Kafka integration (optional)
        self.kafka_producer = None
        self.kafka_consumer = None
        
        if self.config.kafka_bootstrap_servers:
            self._initialize_kafka()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Streaming Churn Predictions",
            description="Real-time customer churn prediction streaming service",
            version="1.0.0"
        )
        
        # Prediction endpoint
        @app.post("/stream/predict")
        async def stream_predict(request: Dict[str, Any]):
            """Single streaming prediction."""
            try:
                event = StreamingEvent(
                    event_id=request.get("event_id", f"evt_{int(time.time() * 1000)}"),
                    customer_id=request.get("customer_id", "unknown"),
                    timestamp=datetime.now(),
                    data=request.get("data", {}),
                    metadata=request.get("metadata")
                )
                
                self.predictor.event_buffer.add_event(event)
                result = await self.predictor.process_event(event)
                
                if result:
                    return asdict(result)
                else:
                    raise HTTPException(status_code=500, detail="Prediction failed")
                    
            except Exception as e:
                logger.error(f"Stream prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch prediction endpoint
        @app.post("/stream/predict/batch")
        async def stream_predict_batch(requests: List[Dict[str, Any]]):
            """Batch streaming predictions."""
            try:
                events = []
                for req in requests:
                    event = StreamingEvent(
                        event_id=req.get("event_id", f"evt_{int(time.time() * 1000)}_{len(events)}"),
                        customer_id=req.get("customer_id", "unknown"),
                        timestamp=datetime.now(),
                        data=req.get("data", {}),
                        metadata=req.get("metadata")
                    )
                    events.append(event)
                    self.predictor.event_buffer.add_event(event)
                
                results = await self.predictor.process_batch(events)
                return [asdict(r) for r in results]
                
            except Exception as e:
                logger.error(f"Batch stream prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint
        @app.websocket("/stream/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket streaming endpoint."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Receive data from client
                    data = await websocket.receive_json()
                    
                    # Process as streaming event
                    event = StreamingEvent(
                        event_id=data.get("event_id", f"ws_{int(time.time() * 1000)}"),
                        customer_id=data.get("customer_id", "unknown"),
                        timestamp=datetime.now(),
                        data=data.get("data", {}),
                        metadata=data.get("metadata")
                    )
                    
                    self.predictor.event_buffer.add_event(event)
                    result = await self.predictor.process_event(event)
                    
                    if result:
                        await websocket.send_json(asdict(result))
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_connections.remove(websocket)
        
        # Statistics endpoint
        @app.get("/stream/stats")
        async def get_streaming_stats():
            """Get streaming statistics."""
            return self.predictor.get_statistics()
        
        # Health endpoint
        @app.get("/stream/health")
        async def streaming_health():
            """Streaming service health check."""
            return {
                "status": "healthy",
                "model_loaded": self.predictor.model is not None,
                "buffer_size": self.predictor.event_buffer.size(),
                "active_connections": len(self.websocket_connections),
                "timestamp": datetime.now().isoformat()
            }
        
        return app
    
    def _initialize_kafka(self) -> None:
        """Initialize Kafka producer and consumer."""
        try:
            # Producer for sending predictions
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
            )
            
            # Consumer for receiving events (will be started in background)
            self.kafka_consumer = KafkaConsumer(
                self.config.kafka_topic_input,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            logger.info("Kafka integration initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            self.kafka_producer = None
            self.kafka_consumer = None
    
    async def start_kafka_consumer(self) -> None:
        """Start Kafka consumer in background."""
        if not self.kafka_consumer:
            return
        
        def consume_messages():
            """Consume Kafka messages."""
            for message in self.kafka_consumer:
                try:
                    data = message.value
                    event = StreamingEvent(
                        event_id=data.get("event_id", f"kafka_{int(time.time() * 1000)}"),
                        customer_id=data.get("customer_id", "unknown"),
                        timestamp=datetime.now(),
                        data=data.get("data", {}),
                        metadata=data.get("metadata")
                    )
                    
                    # Add to buffer for processing
                    self.predictor.event_buffer.add_event(event)
                    
                    # Process event asynchronously
                    asyncio.create_task(self._process_kafka_event(event))
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
        
        # Start consumer in thread
        consumer_thread = threading.Thread(target=consume_messages, daemon=True)
        consumer_thread.start()
        
        logger.info("Kafka consumer started")
    
    async def _process_kafka_event(self, event: StreamingEvent) -> None:
        """Process event from Kafka and send result back."""
        try:
            result = await self.predictor.process_event(event)
            
            if result and self.kafka_producer:
                # Send prediction result to output topic
                self.kafka_producer.send(
                    self.config.kafka_topic_predictions,
                    value=asdict(result)
                )
                
            # Also send to WebSocket clients
            await self._broadcast_to_websockets(result)
            
        except Exception as e:
            logger.error(f"Error processing Kafka event: {e}")
    
    async def _broadcast_to_websockets(self, result: PredictionResult) -> None:
        """Broadcast result to all WebSocket connections."""
        if not self.websocket_connections or not result:
            return
        
        message = asdict(result)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def run(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        """Run the streaming server."""
        logger.info(f"Starting streaming server on {host}:{port}")
        
        # Start Kafka consumer if available
        if self.kafka_consumer:
            asyncio.create_task(self.start_kafka_consumer())
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


class StreamingClient:
    """Client for sending streaming requests."""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.websocket_url = server_url.replace("http", "ws") + "/stream/ws"
    
    async def send_single_prediction(self, customer_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send single prediction request."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/stream/predict",
                json={
                    "customer_id": customer_id,
                    "data": data,
                    "event_id": f"client_{int(time.time() * 1000)}"
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def send_batch_predictions(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send batch prediction requests."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/stream/predict/batch",
                json=requests
            )
            response.raise_for_status()
            return response.json()
    
    async def stream_websocket(self, data_generator: AsyncGenerator[Dict[str, Any], None]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream data via WebSocket."""
        async with websockets.connect(self.websocket_url) as websocket:
            
            # Start receiving responses
            async def receive_responses():
                while True:
                    try:
                        response = await websocket.recv()
                        yield json.loads(response)
                    except websockets.exceptions.ConnectionClosed:
                        break
            
            # Start sending requests
            async def send_requests():
                async for data in data_generator:
                    await websocket.send(json.dumps(data))
            
            # Run both concurrently
            receive_task = asyncio.create_task(receive_responses())
            send_task = asyncio.create_task(send_requests())
            
            async for response in receive_task:
                yield response


def create_streaming_server(config: StreamingConfig = None) -> StreamingServer:
    """Create and configure streaming server."""
    return StreamingServer(config)


def run_streaming_server(host: str = "0.0.0.0", 
                        port: int = 8001,
                        config: StreamingConfig = None) -> None:
    """Run streaming prediction server."""
    server = create_streaming_server(config)
    server.run(host, port)


async def simulate_streaming_data(client: StreamingClient, 
                                duration_seconds: int = 60,
                                events_per_second: float = 10) -> None:
    """Simulate streaming data for testing."""
    logger.info(f"Starting streaming simulation for {duration_seconds}s at {events_per_second} events/sec")
    
    end_time = time.time() + duration_seconds
    event_interval = 1.0 / events_per_second
    
    while time.time() < end_time:
        # Generate random customer data
        customer_id = f"customer_{np.random.randint(1000, 9999)}"
        data = {
            "tenure": np.random.randint(1, 72),
            "MonthlyCharges": np.random.uniform(20, 120),
            "TotalCharges": np.random.uniform(100, 8000),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"]),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"])
        }
        
        try:
            result = await client.send_single_prediction(customer_id, data)
            logger.info(f"Prediction for {customer_id}: {result['prediction']} (prob: {result['probability']:.3f})")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        
        await asyncio.sleep(event_interval)
    
    logger.info("Streaming simulation completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Prediction Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--simulate", action="store_true", help="Run simulation after starting server")
    
    args = parser.parse_args()
    
    if args.simulate:
        # Run simulation in background
        async def run_with_simulation():
            server = create_streaming_server()
            
            # Start server in background
            server_task = asyncio.create_task(
                server.app({"type": "http", "method": "GET", "path": "/stream/health"})
            )
            
            # Wait a bit for server to start
            await asyncio.sleep(2)
            
            # Run simulation
            client = StreamingClient()
            await simulate_streaming_data(client)
        
        asyncio.run(run_with_simulation())
    else:
        run_streaming_server(args.host, args.port)