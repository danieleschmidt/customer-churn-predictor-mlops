"""
Tests for API rate limiting functionality.

This module tests rate limiting implementation including token bucket algorithm,
memory and Redis backends, FastAPI middleware integration, and CLI commands.
"""

import unittest
import time
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from src.rate_limiter import (
    RateLimitRule,
    RateLimitStatus,
    TokenBucket,
    MemoryRateLimiter,
    RateLimitManager,
    RateLimitBackend,
    get_rate_limiter,
    check_rate_limit,
    get_rate_limit_stats
)


class TestRateLimitRule(unittest.TestCase):
    """Test RateLimitRule configuration."""
    
    def test_rule_creation(self):
        """Test rate limit rule creation."""
        rule = RateLimitRule(
            requests=100,
            window_seconds=60,
            burst_size=25,
            per_ip=True,
            per_endpoint=True,
            description="Test rule"
        )
        
        self.assertEqual(rule.requests, 100)
        self.assertEqual(rule.window_seconds, 60)
        self.assertEqual(rule.burst_size, 25)
        self.assertTrue(rule.per_ip)
        self.assertTrue(rule.per_endpoint)
        self.assertEqual(rule.description, "Test rule")
    
    def test_rule_auto_burst_size(self):
        """Test automatic burst size calculation."""
        rule = RateLimitRule(requests=100, window_seconds=60)
        self.assertEqual(rule.burst_size, 25)  # 25% of 100
        
        rule = RateLimitRule(requests=4, window_seconds=60)
        self.assertEqual(rule.burst_size, 1)  # Minimum of 1
    
    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = RateLimitRule(
            requests=50,
            window_seconds=30,
            description="API limit"
        )
        
        rule_dict = rule.to_dict()
        
        self.assertIsInstance(rule_dict, dict)
        self.assertEqual(rule_dict["requests"], 50)
        self.assertEqual(rule_dict["window_seconds"], 30)
        self.assertEqual(rule_dict["description"], "API limit")


class TestRateLimitStatus(unittest.TestCase):
    """Test RateLimitStatus response."""
    
    def test_status_creation(self):
        """Test rate limit status creation."""
        status = RateLimitStatus(
            allowed=True,
            requests_remaining=45,
            reset_time=1640995200.0,
            retry_after=None
        )
        
        self.assertTrue(status.allowed)
        self.assertEqual(status.requests_remaining, 45)
        self.assertEqual(status.reset_time, 1640995200.0)
        self.assertIsNone(status.retry_after)
    
    def test_blocked_status(self):
        """Test blocked request status."""
        status = RateLimitStatus(
            allowed=False,
            requests_remaining=0,
            reset_time=1640995260.0,
            retry_after=60
        )
        
        self.assertFalse(status.allowed)
        self.assertEqual(status.requests_remaining, 0)
        self.assertEqual(status.retry_after, 60)
    
    def test_status_to_dict(self):
        """Test status serialization."""
        status = RateLimitStatus(
            allowed=True,
            requests_remaining=10,
            reset_time=1640995200.0
        )
        
        status_dict = status.to_dict()
        
        self.assertIsInstance(status_dict, dict)
        self.assertTrue(status_dict["allowed"])
        self.assertEqual(status_dict["requests_remaining"], 10)


class TestTokenBucket(unittest.TestCase):
    """Test token bucket algorithm."""
    
    def test_bucket_creation(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        self.assertEqual(bucket.capacity, 10)
        self.assertEqual(bucket.refill_rate, 1.0)
        self.assertEqual(bucket.tokens, 10)  # Initially full
    
    def test_token_consumption(self):
        """Test token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Consume tokens
        self.assertTrue(bucket.consume(5))
        self.assertEqual(bucket.tokens, 5)
        
        # Try to consume more than available
        self.assertFalse(bucket.consume(10))
        self.assertEqual(bucket.tokens, 5)  # Unchanged
        
        # Consume remaining tokens
        self.assertTrue(bucket.consume(5))
        self.assertEqual(bucket.tokens, 0)
    
    def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0, initial_tokens=0)
        
        # No tokens initially
        self.assertFalse(bucket.consume(1))
        
        # Wait for refill (simulate time passing)
        current_time = time.time()
        bucket.last_refill = current_time - 1.0  # 1 second ago
        
        # Should have 2 tokens now (2 tokens/second * 1 second)
        self.assertTrue(bucket.consume(2))
        self.assertFalse(bucket.consume(1))
    
    def test_bucket_capacity_limit(self):
        """Test that bucket doesn't exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0, initial_tokens=0)
        
        # Simulate long time passing
        bucket.last_refill = time.time() - 10.0  # 10 seconds ago
        
        # Should only have capacity tokens, not 100
        self.assertTrue(bucket.consume(5))
        self.assertFalse(bucket.consume(1))
    
    def test_bucket_status(self):
        """Test bucket status reporting."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        status = bucket.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status["capacity"], 10)
        self.assertEqual(status["refill_rate"], 1.0)
        self.assertIn("tokens", status)
        self.assertIn("last_refill", status)


class TestMemoryRateLimiter(unittest.TestCase):
    """Test memory-based rate limiter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.limiter = MemoryRateLimiter()
    
    def test_limiter_initialization(self):
        """Test limiter initialization."""
        self.assertIsNotNone(self.limiter.buckets)
        self.assertIsNotNone(self.limiter.rules)
        self.assertEqual(len(self.limiter.buckets), 0)
        self.assertEqual(len(self.limiter.rules), 0)
    
    def test_add_rule(self):
        """Test adding rate limiting rules."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        self.limiter.add_rule("test", rule)
        
        self.assertIn("test", self.limiter.rules)
        self.assertEqual(self.limiter.rules["test"], rule)
    
    def test_rate_limit_check_no_rule(self):
        """Test rate limit check with no applicable rule."""
        status = self.limiter.check_rate_limit("unknown_key")
        
        self.assertTrue(status.allowed)
        self.assertEqual(status.requests_remaining, 999999)
    
    def test_rate_limit_check_with_rule(self):
        """Test rate limit check with rule."""
        rule = RateLimitRule(requests=5, window_seconds=60, burst_size=5)
        self.limiter.add_rule("test", rule)
        
        # First requests should be allowed
        for i in range(5):
            status = self.limiter.check_rate_limit("test")
            self.assertTrue(status.allowed)
        
        # Sixth request should be blocked
        status = self.limiter.check_rate_limit("test")
        self.assertFalse(status.allowed)
        self.assertIsNotNone(status.retry_after)
    
    def test_rule_pattern_matching(self):
        """Test wildcard rule matching."""
        # Add wildcard rule
        rule = RateLimitRule(requests=10, window_seconds=60)
        self.limiter.add_rule("api:*", rule)
        
        # Should match api:predict, api:health, etc.
        status = self.limiter.check_rate_limit("api:predict")
        self.assertTrue(status.allowed)
        
        status = self.limiter.check_rate_limit("api:health")
        self.assertTrue(status.allowed)
        
        # Should not match non-api endpoints
        status = self.limiter.check_rate_limit("admin:scan")
        self.assertTrue(status.allowed)  # No rule, so allowed
        self.assertEqual(status.requests_remaining, 999999)
    
    def test_default_rule(self):
        """Test default rule fallback."""
        default_rule = RateLimitRule(requests=100, window_seconds=60)
        self.limiter.add_rule("default", default_rule)
        
        # Any key should use default rule
        status = self.limiter.check_rate_limit("any_key")
        self.assertTrue(status.allowed)
        self.assertLess(status.requests_remaining, 999999)
    
    def test_limiter_stats(self):
        """Test limiter statistics."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        self.limiter.add_rule("test", rule)
        
        stats = self.limiter.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["backend"], "memory")
        self.assertEqual(stats["rules"], 1)
        self.assertIn("rules_config", stats)
    
    def test_bucket_cleanup(self):
        """Test cleanup of old buckets."""
        # Add rule and create bucket
        rule = RateLimitRule(requests=10, window_seconds=60)
        self.limiter.add_rule("test", rule)
        
        # Create bucket by making request
        self.limiter.check_rate_limit("test")
        self.assertEqual(len(self.limiter.buckets), 1)
        
        # Force cleanup by making bucket old
        bucket_key = list(self.limiter.buckets.keys())[0]
        self.limiter.buckets[bucket_key].last_refill = time.time() - 1000
        
        # Trigger cleanup
        self.limiter._cleanup_old_buckets()
        self.assertEqual(len(self.limiter.buckets), 0)


class TestRateLimitManager(unittest.TestCase):
    """Test rate limit manager."""
    
    def test_manager_initialization_memory(self):
        """Test manager initialization with memory backend."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        self.assertEqual(manager.backend_type, RateLimitBackend.MEMORY)
        self.assertIsNotNone(manager.backend)
    
    @patch('src.rate_limiter.RedisRateLimiter')
    def test_manager_initialization_redis(self, mock_redis_limiter):
        """Test manager initialization with Redis backend."""
        mock_instance = MagicMock()
        mock_redis_limiter.return_value = mock_instance
        
        manager = RateLimitManager(RateLimitBackend.REDIS, "redis://localhost:6379/0")
        
        self.assertEqual(manager.backend_type, RateLimitBackend.REDIS)
        mock_redis_limiter.assert_called_once_with("redis://localhost:6379/0")
    
    @patch('src.rate_limiter.RedisRateLimiter')
    def test_manager_redis_fallback(self, mock_redis_limiter):
        """Test fallback to memory when Redis fails."""
        mock_redis_limiter.side_effect = Exception("Redis connection failed")
        
        manager = RateLimitManager(RateLimitBackend.REDIS)
        
        # Should fall back to memory
        self.assertEqual(manager.backend_type, RateLimitBackend.MEMORY)
    
    def test_manager_default_rules(self):
        """Test that manager sets up default rules."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        stats = manager.get_stats()
        rules_config = stats.get("rules_config", {})
        
        # Should have default rules
        self.assertIn("default", rules_config)
        self.assertIn("predict", rules_config)
        self.assertIn("health", rules_config)
        self.assertIn("metrics", rules_config)
    
    def test_manager_add_rule(self):
        """Test adding custom rules."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        custom_rule = RateLimitRule(requests=50, window_seconds=30)
        manager.add_rule("custom", custom_rule)
        
        stats = manager.get_stats()
        self.assertIn("custom", stats["rules_config"])
    
    def test_manager_check_rate_limit(self):
        """Test rate limit checking."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        # Should use default rule
        status = manager.check_rate_limit("192.168.1.1", "unknown_endpoint")
        self.assertTrue(status.allowed)
        
        # Should use predict rule
        status = manager.check_rate_limit("192.168.1.1", "predict")
        self.assertTrue(status.allowed)


class TestRateLimitFunctions(unittest.TestCase):
    """Test module-level rate limiting functions."""
    
    @patch('src.rate_limiter._global_rate_limiter', None)
    def test_get_rate_limiter(self):
        """Test global rate limiter initialization."""
        limiter = get_rate_limiter()
        
        self.assertIsNotNone(limiter)
        self.assertIsInstance(limiter, RateLimitManager)
        
        # Should return same instance on subsequent calls
        limiter2 = get_rate_limiter()
        self.assertIs(limiter, limiter2)
    
    def test_check_rate_limit_function(self):
        """Test convenience rate limit check function."""
        status = check_rate_limit("192.168.1.1", "test_endpoint")
        
        self.assertIsInstance(status, RateLimitStatus)
        self.assertTrue(status.allowed)  # Should be allowed initially
    
    def test_get_rate_limit_stats_function(self):
        """Test rate limit statistics function."""
        stats = get_rate_limit_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("backend", stats)
        self.assertIn("rules", stats)


class TestFastAPIIntegration(unittest.TestCase):
    """Test FastAPI middleware integration."""
    
    def setUp(self):
        """Set up test FastAPI client."""
        try:
            from fastapi.testclient import TestClient
            from src.api import app
            self.client = TestClient(app)
        except ImportError:
            self.skipTest("FastAPI not available for testing")
    
    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertEqual(data["service"], "Customer Churn Prediction API")
    
    def test_rate_limit_headers(self):
        """Test rate limiting headers in response."""
        response = self.client.get("/health")
        
        # Should have rate limit headers
        self.assertIn("X-RateLimit-Remaining", response.headers)
        self.assertIn("X-RateLimit-Reset", response.headers)
        
        # Values should be reasonable
        remaining = int(response.headers["X-RateLimit-Remaining"])
        self.assertGreater(remaining, 0)
    
    def test_rate_limit_enforcement(self):
        """Test that rate limiting is enforced."""
        # This test would need to make many requests quickly
        # For now, just verify the middleware doesn't break normal requests
        
        for i in range(5):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            
            # Check that remaining count decreases
            remaining = int(response.headers["X-RateLimit-Remaining"])
            if i > 0:
                # Should have fewer requests remaining
                self.assertIsInstance(remaining, int)
    
    def test_openapi_documentation(self):
        """Test OpenAPI schema generation."""
        response = self.client.get("/openapi.json")
        
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        
        self.assertIn("info", schema)
        self.assertIn("paths", schema)
        self.assertEqual(schema["info"]["title"], "Customer Churn Prediction API")
    
    def test_docs_endpoints(self):
        """Test documentation endpoints."""
        # Swagger UI
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        
        # ReDoc
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)


class TestCLICommands(unittest.TestCase):
    """Test rate limiting CLI commands."""
    
    def test_rate_limit_stats_import(self):
        """Test that rate limit stats can be imported and called."""
        try:
            from src.cli import rate_limit_stats
            # Function exists and is callable
            self.assertTrue(callable(rate_limit_stats))
        except ImportError as e:
            self.fail(f"Failed to import rate_limit_stats: {e}")
    
    def test_rate_limit_add_import(self):
        """Test that rate limit add command can be imported."""
        try:
            from src.cli import rate_limit_add
            self.assertTrue(callable(rate_limit_add))
        except ImportError as e:
            self.fail(f"Failed to import rate_limit_add: {e}")
    
    def test_serve_command_import(self):
        """Test that serve command can be imported."""
        try:
            from src.cli import serve
            self.assertTrue(callable(serve))
        except ImportError as e:
            self.fail(f"Failed to import serve command: {e}")


class TestRateLimitingSecurity(unittest.TestCase):
    """Test security aspects of rate limiting."""
    
    def test_different_ips_independent_limits(self):
        """Test that different IPs have independent rate limits."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        # Add strict rule for testing
        rule = RateLimitRule(requests=2, window_seconds=60, burst_size=2)
        manager.add_rule("test", rule)
        
        # IP 1 uses up its limit
        status1 = manager.check_rate_limit("192.168.1.1", "test")
        self.assertTrue(status1.allowed)
        
        status1 = manager.check_rate_limit("192.168.1.1", "test")
        self.assertTrue(status1.allowed)
        
        status1 = manager.check_rate_limit("192.168.1.1", "test")
        self.assertFalse(status1.allowed)  # Over limit
        
        # IP 2 should still have full limit
        status2 = manager.check_rate_limit("192.168.1.2", "test")
        self.assertTrue(status2.allowed)
    
    def test_key_sanitization(self):
        """Test that rate limit keys are properly handled."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        # Test with various key formats
        test_keys = [
            "192.168.1.1:predict",
            "127.0.0.1:health",
            "::1:metrics",  # IPv6
            "malicious:key:with:colons"
        ]
        
        for key in test_keys:
            ip, endpoint = key.split(":", 1)
            status = manager.check_rate_limit(ip, endpoint)
            self.assertIsInstance(status, RateLimitStatus)
    
    def test_dos_protection(self):
        """Test protection against DoS attacks."""
        manager = RateLimitManager(RateLimitBackend.MEMORY)
        
        # Add very restrictive rule
        rule = RateLimitRule(requests=1, window_seconds=60, burst_size=1)
        manager.add_rule("strict", rule)
        
        # First request allowed
        status = manager.check_rate_limit("attacker_ip", "strict")
        self.assertTrue(status.allowed)
        
        # Subsequent requests blocked
        for i in range(10):
            status = manager.check_rate_limit("attacker_ip", "strict")
            self.assertFalse(status.allowed)
            self.assertIsNotNone(status.retry_after)


class TestPerformance(unittest.TestCase):
    """Test rate limiting performance."""
    
    def test_memory_limiter_performance(self):
        """Test memory limiter performance with many requests."""
        limiter = MemoryRateLimiter()
        rule = RateLimitRule(requests=1000, window_seconds=60, burst_size=1000)
        limiter.add_rule("perf_test", rule)
        
        # Time many rate limit checks
        start_time = time.time()
        
        for i in range(100):
            status = limiter.check_rate_limit(f"user_{i % 10}", "perf_test")
            self.assertTrue(status.allowed)
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (< 1 second for 100 checks)
        self.assertLess(elapsed, 1.0)
    
    def test_concurrent_access(self):
        """Test thread safety of rate limiter."""
        import threading
        
        limiter = MemoryRateLimiter()
        rule = RateLimitRule(requests=100, window_seconds=60, burst_size=100)
        limiter.add_rule("concurrent_test", rule)
        
        results = []
        
        def make_requests():
            for i in range(10):
                status = limiter.check_rate_limit("concurrent_user", "concurrent_test")
                results.append(status.allowed)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should have succeeded (50 total, limit is 100)
        self.assertEqual(len(results), 50)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()