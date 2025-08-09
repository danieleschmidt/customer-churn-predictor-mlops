"""
Tests for Streaming Predictions System.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import threading
import time

from src.streaming_predictions import (
    StreamingConfig, StreamingEvent, PredictionResult,
    StreamingEventBuffer, StreamingPredictor, StreamingServer,
    StreamingClient, create_streaming_server
)


@pytest.fixture
def streaming_config():
    """Create streaming configuration for testing."""
    return StreamingConfig(
        batch_size=10,
        batch_timeout_seconds=1.0,
        max_queue_size=100,
        window_size_seconds=60,
        num_worker_threads=2,
        enable_batching=True,
        enable_windowing=True
    )


@pytest.fixture
def sample_event():
    """Create sample streaming event."""
    return StreamingEvent(
        event_id="test_event_001",
        customer_id="customer_123",
        timestamp=datetime.now(),
        data={
            "tenure": 12,
            "MonthlyCharges": 75.0,
            "TotalCharges": 900.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic"
        },
        event_type="prediction_request"
    )


@pytest.fixture
def sample_events():
    """Create multiple sample streaming events."""
    events = []
    for i in range(5):
        event = StreamingEvent(
            event_id=f"test_event_{i:03d}",
            customer_id=f"customer_{i}",
            timestamp=datetime.now(),
            data={
                "tenure": 10 + i,
                "MonthlyCharges": 50.0 + i * 10,
                "TotalCharges": 500.0 + i * 100,
                "Contract": "Month-to-month",
                "InternetService": "DSL"
            }
        )
        events.append(event)
    return events


class TestStreamingEventBuffer:
    """Tests for StreamingEventBuffer."""
    
    def test_add_event(self, sample_event):
        """Test adding events to buffer."""
        buffer = StreamingEventBuffer(max_size=100)
        
        buffer.add_event(sample_event)
        
        assert buffer.size() == 1
        assert sample_event.customer_id in buffer.event_index
        assert len(buffer.event_index[sample_event.customer_id]) == 1
    
    def test_buffer_max_size(self, sample_events):
        """Test buffer maximum size limit."""
        buffer = StreamingEventBuffer(max_size=3)
        
        # Add more events than max size
        for event in sample_events:
            buffer.add_event(event)
        
        # Buffer should not exceed max size
        assert buffer.size() <= 3
    
    def test_get_events_for_customer(self, sample_events):
        """Test retrieving events for specific customer."""
        buffer = StreamingEventBuffer()
        
        # Add events for multiple customers
        for event in sample_events:
            buffer.add_event(event)
        
        # Get events for specific customer
        customer_events = buffer.get_events_for_customer("customer_0")
        
        assert len(customer_events) == 1
        assert customer_events[0].customer_id == "customer_0"
    
    def test_get_events_with_time_window(self, sample_events):
        """Test retrieving events within time window."""
        buffer = StreamingEventBuffer()
        
        # Add events with different timestamps
        old_event = StreamingEvent(
            event_id="old_event",
            customer_id="customer_0",
            timestamp=datetime.now() - timedelta(hours=2),
            data={}
        )
        buffer.add_event(old_event)
        buffer.add_event(sample_events[0])
        
        # Get recent events (last hour)
        recent_events = buffer.get_events_for_customer("customer_0", window_seconds=3600)
        
        # Should only include recent event
        assert len(recent_events) == 1
        assert recent_events[0].event_id == "test_event_000"
    
    def test_get_batch(self, sample_events):
        """Test getting batch of events."""
        buffer = StreamingEventBuffer()
        
        for event in sample_events:
            buffer.add_event(event)
        
        # Get batch of 3 events
        batch = buffer.get_batch(3)
        
        assert len(batch) == 3
        assert buffer.size() == 2  # Remaining events
    
    def test_thread_safety(self, sample_events):
        """Test thread safety of buffer operations."""
        buffer = StreamingEventBuffer()
        
        def add_events():
            for event in sample_events[:3]:
                buffer.add_event(event)
                time.sleep(0.01)
        
        def get_batches():
            for _ in range(3):
                buffer.get_batch(1)
                time.sleep(0.01)
        
        # Run operations concurrently
        thread1 = threading.Thread(target=add_events)
        thread2 = threading.Thread(target=get_batches)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Should complete without errors


class TestStreamingPredictor:
    """Tests for StreamingPredictor."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model
    
    @pytest.fixture
    def streaming_predictor(self, streaming_config, mock_model):
        """Create StreamingPredictor with mocked model."""
        with patch('src.streaming_predictions.get_model_from_cache') as mock_cache:
            mock_cache.return_value = {
                'model': mock_model,
                'version': 'test_v1'
            }
            
            predictor = StreamingPredictor(streaming_config)
            predictor.model = mock_model
            predictor.model_version = 'test_v1'
            
            return predictor
    
    def test_load_model(self, streaming_config):
        """Test model loading."""
        with patch('src.streaming_predictions.get_model_from_cache') as mock_cache, \
             patch('os.path.exists') as mock_exists, \
             patch('joblib.load') as mock_load:
            
            # Mock cache miss, file exists
            mock_cache.return_value = None
            mock_exists.return_value = True
            mock_load.return_value = Mock()
            
            predictor = StreamingPredictor(streaming_config)
            
            assert predictor.model is not None
            mock_load.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_event(self, streaming_predictor, sample_event):
        """Test processing single event."""
        result = await streaming_predictor.process_event(sample_event)
        
        assert isinstance(result, PredictionResult)
        assert result.event_id == sample_event.event_id
        assert result.customer_id == sample_event.customer_id
        assert result.prediction in [0, 1]
        assert 0 <= result.probability <= 1
        assert result.model_version == 'test_v1'
    
    @pytest.mark.asyncio
    async def test_process_event_caching(self, streaming_predictor, sample_event):
        """Test event processing with caching."""
        # Process same event twice
        result1 = await streaming_predictor.process_event(sample_event)
        result2 = await streaming_predictor.process_event(sample_event)
        
        # Should get cached result
        assert result1.prediction == result2.prediction
        assert streaming_predictor.processing_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_process_batch(self, streaming_predictor, sample_events):
        """Test batch processing."""
        results = await streaming_predictor.process_batch(sample_events)
        
        assert len(results) == len(sample_events)
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_windowed_features(self, streaming_predictor, sample_events):
        """Test windowed feature extraction."""
        # Add events to buffer
        for event in sample_events:
            streaming_predictor.event_buffer.add_event(event)
        
        # Extract windowed features for last event
        windowed_features = streaming_predictor._extract_windowed_features(sample_events[-1])
        
        assert isinstance(windowed_features, dict)
        assert 'events_in_window' in windowed_features
    
    def test_get_statistics(self, streaming_predictor):
        """Test getting processing statistics."""
        stats = streaming_predictor.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'samples_processed' in stats or 'total_predictions' in stats
        assert 'buffer_size' in stats
        assert 'model_loaded' in stats
    
    @pytest.mark.asyncio
    async def test_process_event_error_handling(self, streaming_config):
        """Test error handling in event processing."""
        # Create predictor with no model
        predictor = StreamingPredictor(streaming_config)
        predictor.model = None
        
        sample_event = StreamingEvent(
            event_id="test",
            customer_id="test",
            timestamp=datetime.now(),
            data={}
        )
        
        result = await predictor.process_event(sample_event)
        
        # Should handle gracefully
        assert result is None


class TestStreamingServer:
    """Tests for StreamingServer."""
    
    @pytest.fixture
    def streaming_server(self, streaming_config):
        """Create StreamingServer for testing."""
        with patch('src.streaming_predictions.StreamingPredictor'):
            server = StreamingServer(streaming_config)
            server.predictor = Mock()
            return server
    
    def test_server_initialization(self, streaming_config):
        """Test server initialization."""
        with patch('src.streaming_predictions.StreamingPredictor'):
            server = StreamingServer(streaming_config)
            
            assert server.app is not None
            assert server.predictor is not None
            assert server.websocket_connections == []
    
    @pytest.mark.asyncio
    async def test_kafka_consumer_initialization(self, streaming_config):
        """Test Kafka consumer initialization."""
        config_with_kafka = streaming_config
        config_with_kafka.kafka_bootstrap_servers = ['localhost:9092']
        
        with patch('kafka.KafkaConsumer') as mock_consumer, \
             patch('kafka.KafkaProducer') as mock_producer:
            
            server = StreamingServer(config_with_kafka)
            
            assert server.kafka_consumer is not None
            assert server.kafka_producer is not None
    
    def test_create_app_endpoints(self, streaming_server):
        """Test that FastAPI app has required endpoints."""
        app = streaming_server.app
        
        # Check that routes exist
        route_paths = [route.path for route in app.routes]
        
        expected_paths = [
            '/stream/predict',
            '/stream/predict/batch',
            '/stream/ws',
            '/stream/stats',
            '/stream/health'
        ]
        
        for path in expected_paths:
            assert path in route_paths


class TestStreamingClient:
    """Tests for StreamingClient."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = StreamingClient("http://localhost:8001")
        
        assert client.server_url == "http://localhost:8001"
        assert client.websocket_url == "ws://localhost:8001/stream/ws"
    
    @pytest.mark.asyncio
    async def test_send_single_prediction(self):
        """Test sending single prediction request."""
        client = StreamingClient("http://localhost:8001")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "prediction": 1,
                "probability": 0.75,
                "timestamp": datetime.now().isoformat()
            }
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await client.send_single_prediction(
                "customer_123",
                {"tenure": 12, "MonthlyCharges": 75.0}
            )
            
            assert result["prediction"] == 1
            assert result["probability"] == 0.75
    
    @pytest.mark.asyncio
    async def test_send_batch_predictions(self):
        """Test sending batch prediction requests."""
        client = StreamingClient("http://localhost:8001")
        
        requests_data = [
            {"customer_id": "customer_1", "data": {"tenure": 12}},
            {"customer_id": "customer_2", "data": {"tenure": 24}}
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = [
                {"prediction": 1, "probability": 0.75},
                {"prediction": 0, "probability": 0.25}
            ]
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            results = await client.send_batch_predictions(requests_data)
            
            assert len(results) == 2
            assert results[0]["prediction"] == 1
            assert results[1]["prediction"] == 0


class TestStreamingIntegration:
    """Integration tests for streaming system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self, sample_event):
        """Test end-to-end streaming pipeline."""
        config = StreamingConfig(
            batch_size=1,
            batch_timeout_seconds=0.1,
            enable_batching=False
        )
        
        with patch('src.streaming_predictions.get_model_from_cache') as mock_cache, \
             patch('src.streaming_predictions.make_prediction') as mock_predict:
            
            # Mock model and prediction
            mock_cache.return_value = {'model': Mock(), 'version': 'test'}
            mock_predict.return_value = (1, 0.75)
            
            predictor = StreamingPredictor(config)
            
            # Process event
            result = await predictor.process_event(sample_event)
            
            # Verify result
            assert result is not None
            assert result.prediction == 1
            assert result.probability == 0.75
    
    def test_streaming_server_creation(self):
        """Test streaming server creation function."""
        config = StreamingConfig()
        
        server = create_streaming_server(config)
        
        assert isinstance(server, StreamingServer)
        assert server.config == config
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, sample_events):
        """Test concurrent processing of multiple events."""
        config = StreamingConfig(num_worker_threads=2)
        
        with patch('src.streaming_predictions.make_prediction') as mock_predict:
            mock_predict.return_value = (1, 0.75)
            
            predictor = StreamingPredictor(config)
            predictor.model = Mock()
            
            # Process events concurrently
            tasks = [predictor.process_event(event) for event in sample_events]
            results = await asyncio.gather(*tasks)
            
            # All events should be processed
            assert len(results) == len(sample_events)
            assert all(r is not None for r in results)
    
    def test_buffer_windowing_integration(self, sample_events):
        """Test integration between event buffer and windowing."""
        config = StreamingConfig(
            window_size_seconds=60,
            enable_windowing=True
        )
        
        predictor = StreamingPredictor(config)
        
        # Add events to buffer
        for event in sample_events:
            predictor.event_buffer.add_event(event)
        
        # Test windowed feature extraction
        windowed_features = predictor._extract_windowed_features(sample_events[-1])
        
        assert isinstance(windowed_features, dict)
        if windowed_features:
            assert 'events_in_window' in windowed_features


class TestStreamingPerformance:
    """Performance tests for streaming system."""
    
    @pytest.mark.slow
    def test_high_throughput_processing(self):
        """Test system performance under high load."""
        config = StreamingConfig(
            batch_size=50,
            batch_timeout_seconds=1.0,
            max_queue_size=1000
        )
        
        with patch('src.streaming_predictions.make_prediction') as mock_predict:
            mock_predict.return_value = (1, 0.75)
            
            predictor = StreamingPredictor(config)
            predictor.model = Mock()
            
            # Generate many events
            events = []
            for i in range(100):
                event = StreamingEvent(
                    event_id=f"perf_test_{i}",
                    customer_id=f"customer_{i % 10}",  # 10 different customers
                    timestamp=datetime.now(),
                    data={"feature": i}
                )
                events.append(event)
            
            # Measure processing time
            start_time = time.time()
            
            async def process_all_events():
                tasks = [predictor.process_event(event) for event in events]
                return await asyncio.gather(*tasks)
            
            results = asyncio.run(process_all_events())
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify results
            assert len(results) == 100
            assert all(r is not None for r in results)
            
            # Performance should be reasonable (less than 10 seconds for 100 events)
            assert processing_time < 10.0
            
            # Calculate throughput
            throughput = len(events) / processing_time
            assert throughput > 10  # At least 10 events per second
    
    @pytest.mark.slow
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        
        config = StreamingConfig(max_queue_size=100)
        predictor = StreamingPredictor(config)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Add many events
        for i in range(1000):
            event = StreamingEvent(
                event_id=f"mem_test_{i}",
                customer_id=f"customer_{i % 50}",
                timestamp=datetime.now(),
                data={"feature": i}
            )
            predictor.event_buffer.add_event(event)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])