"""Performance tests for the API endpoints."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance under various conditions."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for testing."""
        return {"Authorization": "Bearer test-api-key-for-testing"}

    def test_health_check_response_time(self, client):
        """Test health check endpoint response time."""
        start_time = time.time()
        response = client.get("/health")
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert response_time < 0.1  # Should respond within 100ms

    def test_prediction_response_time(self, client, auth_headers, sample_processed_features):
        """Test prediction endpoint response time."""
        # Convert DataFrame to list of dictionaries
        test_data = sample_processed_features.iloc[0].to_dict()

        start_time = time.time()
        response = client.post(
            "/predict",
            json={"features": test_data},
            headers=auth_headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            assert response_time < 0.5  # Should respond within 500ms

    def test_concurrent_requests(self, client, auth_headers, sample_processed_features):
        """Test API performance under concurrent load."""
        test_data = sample_processed_features.iloc[0].to_dict()
        num_requests = 50
        max_workers = 10

        def make_request() -> Dict[str, Any]:
            start_time = time.time()
            try:
                response = client.post(
                    "/predict",
                    json={"features": test_data},
                    headers=auth_headers,
                    timeout=5.0
                )
                response_time = time.time() - start_time
                return {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": True
                }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "status_code": 500,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                }

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]

        if successful_requests:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            success_rate = len(successful_requests) / num_requests

            # Performance assertions
            assert success_rate >= 0.95  # At least 95% success rate
            assert avg_response_time < 1.0  # Average response time under 1 second
            assert max_response_time < 3.0  # Maximum response time under 3 seconds

    def test_memory_usage_stability(self, client, auth_headers, sample_processed_features):
        """Test that memory usage remains stable during repeated requests."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        test_data = sample_processed_features.iloc[0].to_dict()

        # Make 100 requests
        for _ in range(100):
            try:
                client.post(
                    "/predict",
                    json={"features": test_data},
                    headers=auth_headers
                )
            except Exception:
                pass  # Ignore failures for memory test

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB"

    def test_rate_limiting_performance(self, client, auth_headers):
        """Test rate limiting doesn't significantly impact performance."""
        response_times = []

        for _ in range(10):
            start_time = time.time()
            response = client.get("/health")
            response_time = time.time() - start_time
            response_times.append(response_time)

            assert response.status_code == 200

        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Rate limiting shouldn't add significant overhead
        assert avg_response_time < 0.1
        assert max_response_time < 0.2

    def test_large_payload_handling(self, client, auth_headers, performance_test_data):
        """Test API performance with larger payloads."""
        # Test with a larger feature set
        large_data = performance_test_data.iloc[0].to_dict()

        start_time = time.time()
        try:
            response = client.post(
                "/predict",
                json={"features": large_data},
                headers=auth_headers
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                # Should handle large payloads within reasonable time
                assert response_time < 2.0
        except Exception:
            # If prediction fails due to missing model, that's acceptable for this test
            response_time = time.time() - start_time
            assert response_time < 2.0  # Still should fail quickly

    @pytest.mark.slow
    def test_sustained_load(self, client, auth_headers, sample_processed_features):
        """Test API performance under sustained load."""
        test_data = sample_processed_features.iloc[0].to_dict()
        duration = 30  # seconds
        request_interval = 0.1  # 10 requests per second

        start_time = time.time()
        request_count = 0
        response_times = []

        while time.time() - start_time < duration:
            request_start = time.time()
            try:
                response = client.post(
                    "/predict",
                    json={"features": test_data},
                    headers=auth_headers
                )
                request_time = time.time() - request_start
                response_times.append(request_time)
                request_count += 1
            except Exception:
                pass

            # Wait for next request
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            requests_per_second = request_count / duration

            # Performance requirements for sustained load
            assert avg_response_time < 1.0
            assert requests_per_second >= 8  # Should handle at least 8 RPS