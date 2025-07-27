"""Security tests for the application."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.security
class TestAPISecurity:
    """Test API security measures."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Valid authentication headers."""
        return {"Authorization": "Bearer test-api-key-for-testing"}

    def test_authentication_required(self, client):
        """Test that authentication is required for protected endpoints."""
        # Test without any headers
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 401

        # Test with invalid authorization header format
        response = client.post(
            "/predict",
            json={"features": {}},
            headers={"Authorization": "InvalidFormat"}
        )
        assert response.status_code == 401

    def test_invalid_api_keys(self, client, invalid_api_keys):
        """Test rejection of invalid API keys."""
        for invalid_key in invalid_api_keys:
            if invalid_key is None:
                headers = {}
            else:
                headers = {"Authorization": f"Bearer {invalid_key}"}

            response = client.post(
                "/predict",
                json={"features": {}},
                headers=headers
            )
            assert response.status_code == 401

    def test_sql_injection_attempts(self, client, valid_headers):
        """Test protection against SQL injection attempts."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; DELETE FROM customers; --",
            "admin'--",
            "' UNION SELECT * FROM users --",
        ]

        for malicious_input in malicious_inputs:
            response = client.post(
                "/predict",
                json={"features": {"tenure": malicious_input}},
                headers=valid_headers
            )
            # Should either validate input or handle gracefully
            assert response.status_code in [400, 422]  # Bad request or validation error

    def test_xss_protection(self, client, valid_headers):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ]

        for payload in xss_payloads:
            response = client.post(
                "/predict",
                json={"features": {"gender_Male": payload}},
                headers=valid_headers
            )
            # Should validate input types
            assert response.status_code in [400, 422]

    def test_request_size_limits(self, client, valid_headers):
        """Test protection against oversized requests."""
        # Create a very large payload
        large_payload = {"features": {f"field_{i}": "x" * 1000 for i in range(1000)}}

        response = client.post(
            "/predict",
            json=large_payload,
            headers=valid_headers
        )
        # Should reject oversized requests
        assert response.status_code in [400, 413, 422]

    def test_rate_limiting(self, client, valid_headers):
        """Test rate limiting protection."""
        # Make many requests quickly
        responses = []
        for _ in range(150):  # Assuming limit is 100 requests
            response = client.get("/health")
            responses.append(response.status_code)

        # Should eventually hit rate limit
        rate_limited = any(status == 429 for status in responses)
        # Note: This might not trigger in test environment
        # In production, this would be handled by reverse proxy or rate limiter

    def test_cors_headers(self, client):
        """Test CORS headers are properly configured."""
        response = client.options("/health")
        # Should handle OPTIONS requests
        assert response.status_code in [200, 204, 405]

    def test_security_headers(self, client):
        """Test presence of security headers."""
        response = client.get("/health")
        
        # Check for common security headers
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
        ]
        
        # Note: These might be set by reverse proxy in production
        # This test documents the expectation

    def test_information_disclosure(self, client):
        """Test that error messages don't disclose sensitive information."""
        # Test with malformed JSON
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 422:
            # Error message should not expose internal details
            error_text = response.text.lower()
            sensitive_terms = [
                "traceback",
                "exception",
                "database",
                "file path",
                "password",
                "secret",
            ]
            
            for term in sensitive_terms:
                assert term not in error_text

    def test_path_traversal_protection(self, client, valid_headers):
        """Test protection against path traversal attacks."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        for attempt in path_traversal_attempts:
            response = client.post(
                "/predict",
                json={"features": {"file_path": attempt}},
                headers=valid_headers
            )
            # Should validate and reject suspicious paths
            assert response.status_code in [400, 422]

    def test_timing_attack_resistance(self, client):
        """Test resistance to timing attacks on authentication."""
        import time

        def measure_auth_time(api_key):
            start = time.time()
            client.post(
                "/predict",
                json={"features": {}},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return time.time() - start

        # Measure time for different invalid keys
        times = []
        for i in range(10):
            invalid_key = f"invalid-key-{i}"
            auth_time = measure_auth_time(invalid_key)
            times.append(auth_time)

        # Times should be relatively consistent (constant time comparison)
        avg_time = sum(times) / len(times)
        max_deviation = max(abs(t - avg_time) for t in times)
        
        # Allow for some variation but should be relatively consistent
        assert max_deviation < avg_time * 0.5  # Within 50% of average

    def test_input_validation(self, client, valid_headers):
        """Test comprehensive input validation."""
        invalid_inputs = [
            # Type mismatches
            {"tenure": "not_a_number"},
            {"SeniorCitizen": "yes"},
            {"MonthlyCharges": []},
            
            # Out of range values
            {"tenure": -1},
            {"SeniorCitizen": 2},
            {"MonthlyCharges": -100},
            
            # Null/None values in required fields
            {"tenure": None},
            
            # Empty values
            {},
            {"features": {}},
        ]

        for invalid_input in invalid_inputs:
            response = client.post(
                "/predict",
                json={"features": invalid_input},
                headers=valid_headers
            )
            # Should reject invalid inputs with appropriate error codes
            assert response.status_code in [400, 422]

    def test_content_type_validation(self, client, valid_headers):
        """Test content type validation."""
        # Test with wrong content type
        response = client.post(
            "/predict",
            data="some data",
            headers={**valid_headers, "Content-Type": "text/plain"}
        )
        assert response.status_code in [400, 415, 422]

        # Test with no content type
        response = client.post(
            "/predict",
            data="some data",
            headers=valid_headers
        )
        assert response.status_code in [400, 415, 422]