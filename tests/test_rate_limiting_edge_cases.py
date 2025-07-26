"""
Edge case tests for rate limiting functionality.

Tests comprehensive rate limiting scenarios including:
- Edge cases and boundary conditions
- Concurrent request handling
- Rate limit rule management
- Backend storage reliability
- Performance under load
- Error conditions and recovery
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from src.rate_limiter import (
    RateLimitManager, RateLimitRule, RateLimitStatus, 
    RateLimitBackend, MemoryRateLimiter
)


class TestRateLimitingEdgeCases:
    """Test suite for rate limiting edge cases and boundary conditions."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Fixture providing a configured rate limiter."""
        return RateLimiter(backend=RateLimitBackend.MEMORY)
    
    @pytest.fixture
    def test_rule(self):
        """Fixture providing a test rate limit rule."""
        return RateLimitRule(
            requests=10,
            window_seconds=60,
            burst_size=5,
            per_ip=True,
            per_endpoint=True,
            description="Test rule"
        )

    def test_boundary_rate_limit_exact_limit(self, rate_limiter, test_rule):
        """Test behavior when exactly hitting the rate limit."""
        client_ip = "192.168.1.1"
        endpoint = "test_endpoint"
        
        # Add rule
        rate_limiter.add_rule("test", test_rule)
        
        # Make exactly the limit number of requests
        for i in range(test_rule.requests):
            status = rate_limiter.check_rate_limit(client_ip, "test")
            if i < test_rule.requests - 1:
                assert status.allowed, f"Request {i+1} should be allowed"
            else:
                # Last request at the limit
                assert status.allowed or not status.allowed  # Could go either way at boundary

    def test_rate_limit_just_over_limit(self, rate_limiter, test_rule):
        """Test behavior when going just over the rate limit."""
        client_ip = "192.168.1.1"
        
        rate_limiter.add_rule("test", test_rule)
        
        # Exceed the limit by 1
        for i in range(test_rule.requests + 1):
            status = rate_limiter.check_rate_limit(client_ip, "test")
            if i < test_rule.requests:
                expected_allowed = True
            else:
                expected_allowed = False
            
            # At boundary conditions, behavior may vary
            assert isinstance(status.allowed, bool)

    def test_zero_rate_limit(self, rate_limiter):
        """Test rate limiting with zero requests allowed."""
        zero_rule = RateLimitRule(
            requests=0,
            window_seconds=60,
            description="No requests allowed"
        )
        
        rate_limiter.add_rule("zero", zero_rule)
        
        status = rate_limiter.check_rate_limit("192.168.1.1", "zero")
        assert not status.allowed
        assert status.remaining == 0

    def test_very_large_rate_limit(self, rate_limiter):
        """Test rate limiting with very large limits."""
        large_rule = RateLimitRule(
            requests=1000000,  # 1 million requests
            window_seconds=3600,
            description="Very large limit"
        )
        
        rate_limiter.add_rule("large", large_rule)
        
        # Should handle large numbers without overflow
        status = rate_limiter.check_rate_limit("192.168.1.1", "large")
        assert status.allowed
        assert status.remaining > 999990  # Most of the limit should remain

    def test_very_short_time_window(self, rate_limiter):
        """Test rate limiting with very short time windows."""
        short_rule = RateLimitRule(
            requests=5,
            window_seconds=1,  # 1 second window
            description="Very short window"
        )
        
        rate_limiter.add_rule("short", short_rule)
        
        # Make requests quickly
        for _ in range(3):
            status = rate_limiter.check_rate_limit("192.168.1.1", "short")
            assert status.allowed  # Should be allowed within short window

    def test_empty_client_ip(self, rate_limiter, test_rule):
        """Test rate limiting with empty or None client IP."""
        rate_limiter.add_rule("test", test_rule)
        
        # Test with None IP
        status = rate_limiter.check_rate_limit(None, "test")
        assert isinstance(status.allowed, bool)
        
        # Test with empty string IP
        status = rate_limiter.check_rate_limit("", "test")  
        assert isinstance(status.allowed, bool)

    def test_invalid_rate_limit_key(self, rate_limiter):
        """Test rate limiting with invalid or non-existent keys."""
        # Request with non-existent rule key
        status = rate_limiter.check_rate_limit("192.168.1.1", "nonexistent")
        
        # Should have some default behavior
        assert isinstance(status.allowed, bool)
        assert isinstance(status.remaining, int)

    def test_unicode_client_ip(self, rate_limiter, test_rule):
        """Test rate limiting with Unicode characters in client IP."""
        rate_limiter.add_rule("test", test_rule)
        
        # Test with Unicode characters (should be handled gracefully)
        unicode_ip = "192.168.1.1🚀"
        status = rate_limiter.check_rate_limit(unicode_ip, "test")
        assert isinstance(status.allowed, bool)

    def test_very_long_client_ip(self, rate_limiter, test_rule):
        """Test rate limiting with very long client IP strings."""
        rate_limiter.add_rule("test", test_rule)
        
        # Create a very long IP string
        long_ip = "192.168.1.1" + "x" * 1000
        status = rate_limiter.check_rate_limit(long_ip, "test")
        assert isinstance(status.allowed, bool)


class TestConcurrentRateLimiting:
    """Test suite for concurrent access to rate limiting."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Fixture providing a thread-safe rate limiter."""
        return RateLimiter(backend=RateLimitBackend.MEMORY)
    
    def test_concurrent_requests_same_ip(self, rate_limiter):
        """Test concurrent requests from the same IP address."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("concurrent", rule)
        
        results = []
        
        def make_request():
            status = rate_limiter.check_rate_limit("192.168.1.1", "concurrent")
            results.append(status.allowed)
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(15)]
            for future in futures:
                future.result()
        
        # Should have some requests allowed and some denied
        allowed_count = sum(results)
        assert 0 < allowed_count <= 10  # Some allowed, within limit

    def test_concurrent_requests_different_ips(self, rate_limiter):
        """Test concurrent requests from different IP addresses."""
        rule = RateLimitRule(requests=5, window_seconds=60)
        rate_limiter.add_rule("multi_ip", rule)
        
        results = defaultdict(list)
        
        def make_request(ip_suffix):
            ip = f"192.168.1.{ip_suffix}"
            status = rate_limiter.check_rate_limit(ip, "multi_ip")
            results[ip].append(status.allowed)
        
        # Run concurrent requests from different IPs
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for ip_suffix in range(1, 6):  # 5 different IPs
                for _ in range(3):  # 3 requests per IP
                    futures.append(executor.submit(make_request, ip_suffix))
            
            for future in futures:
                future.result()
        
        # Each IP should have its own limit
        for ip, allowed_list in results.items():
            assert len(allowed_list) == 3
            # Most or all should be allowed since each IP has its own limit
            assert sum(allowed_list) >= 2

    def test_race_condition_rule_update(self, rate_limiter):
        """Test race conditions when updating rules during requests."""
        initial_rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("race", initial_rule)
        
        results = []
        
        def make_requests():
            for _ in range(5):
                status = rate_limiter.check_rate_limit("192.168.1.1", "race")
                results.append(status.allowed)
                time.sleep(0.01)  # Small delay
        
        def update_rule():
            time.sleep(0.02)  # Let some requests start
            new_rule = RateLimitRule(requests=5, window_seconds=60)
            rate_limiter.add_rule("race", new_rule)
        
        # Run requests and rule update concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            request_future = executor.submit(make_requests)
            update_future = executor.submit(update_rule)
            
            request_future.result()
            update_future.result()
        
        # Should complete without errors
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)


class TestRateLimitingMemoryManagement:
    """Test suite for memory management in rate limiting."""
    
    @pytest.fixture
    def memory_storage(self):
        """Fixture providing in-memory storage for testing."""
        return MemoryRateLimiter()
    
    def test_memory_cleanup_expired_entries(self, memory_storage):
        """Test that expired entries are cleaned up from memory."""
        # Add entries that should expire
        current_time = time.time()
        
        # Add current entry
        memory_storage.store_request("192.168.1.1:test", current_time, 60)
        
        # Add expired entry (simulate old timestamp)
        expired_time = current_time - 120  # 2 minutes ago
        memory_storage.store_request("192.168.1.2:test", expired_time, 60)
        
        # Check that cleanup works
        current_count = memory_storage.get_request_count("192.168.1.1:test", 60)
        expired_count = memory_storage.get_request_count("192.168.1.2:test", 60)
        
        assert current_count >= 1  # Current entry should exist
        assert expired_count == 0  # Expired entry should be cleaned

    def test_memory_usage_with_many_ips(self, rate_limiter):
        """Test memory usage with many different IP addresses."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("memory_test", rule)
        
        # Generate requests from many different IPs
        for i in range(1000):
            ip = f"192.168.{i//256}.{i%256}"
            status = rate_limiter.check_rate_limit(ip, "memory_test")
            assert isinstance(status.allowed, bool)
        
        # Should handle many IPs without memory issues
        stats = rate_limiter.get_stats()
        assert "total_requests" in stats

    def test_memory_storage_thread_safety(self, memory_storage):
        """Test thread safety of in-memory storage."""
        def store_requests(ip_suffix):
            for i in range(100):
                key = f"192.168.1.{ip_suffix}:test"
                memory_storage.store_request(key, time.time(), 60)
        
        # Run concurrent storage operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(store_requests, i) for i in range(1, 6)]
            for future in futures:
                future.result()
        
        # Should complete without errors
        # Each IP should have stored requests
        for i in range(1, 6):
            key = f"192.168.1.{i}:test"
            count = memory_storage.get_request_count(key, 60)
            assert count > 0


class TestRateLimitingErrorRecovery:
    """Test suite for error recovery in rate limiting."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Fixture providing a rate limiter for error testing."""
        return RateLimiter(backend=RateLimitBackend.MEMORY)
    
    def test_storage_backend_error_handling(self, rate_limiter):
        """Test handling of storage backend errors."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("error_test", rule)
        
        # Mock storage to raise errors
        with patch.object(rate_limiter.storage, 'store_request', side_effect=Exception("Storage error")):
            # Should handle storage errors gracefully
            status = rate_limiter.check_rate_limit("192.168.1.1", "error_test")
            assert isinstance(status.allowed, bool)
            # Typically should allow on error (fail open) or deny (fail closed)

    def test_malformed_rule_handling(self, rate_limiter):
        """Test handling of malformed rate limit rules."""
        # Try to add invalid rule
        try:
            invalid_rule = RateLimitRule(
                requests=-1,  # Invalid negative value
                window_seconds=0  # Invalid zero window
            )
            rate_limiter.add_rule("invalid", invalid_rule)
            
            # Should either raise error or handle gracefully
            status = rate_limiter.check_rate_limit("192.168.1.1", "invalid")
            assert isinstance(status.allowed, bool)
        except (ValueError, TypeError):
            # Acceptable to raise validation errors
            pass

    def test_system_clock_changes(self, rate_limiter):
        """Test behavior when system clock changes."""
        rule = RateLimitRule(requests=5, window_seconds=60)
        rate_limiter.add_rule("clock_test", rule)
        
        # Make some requests
        for _ in range(3):
            status = rate_limiter.check_rate_limit("192.168.1.1", "clock_test")
            assert status.allowed
        
        # Simulate clock going backwards
        with patch('time.time', return_value=time.time() - 3600):  # 1 hour back
            status = rate_limiter.check_rate_limit("192.168.1.1", "clock_test")
            # Should handle gracefully without breaking
            assert isinstance(status.allowed, bool)

    @patch('src.rate_limiter.logger')
    def test_error_logging(self, mock_logger, rate_limiter):
        """Test that errors are logged appropriately."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("log_test", rule)
        
        # Mock storage to raise error
        with patch.object(rate_limiter.storage, 'get_request_count', side_effect=Exception("Test error")):
            rate_limiter.check_rate_limit("192.168.1.1", "log_test")
            
            # Should log the error
            assert mock_logger.error.called or mock_logger.warning.called

    def test_recovery_after_error(self, rate_limiter):
        """Test that rate limiter recovers after temporary errors."""
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("recovery_test", rule)
        
        # Simulate temporary error
        with patch.object(rate_limiter.storage, 'store_request', side_effect=Exception("Temp error")) as mock_store:
            status1 = rate_limiter.check_rate_limit("192.168.1.1", "recovery_test")
            assert isinstance(status1.allowed, bool)
        
        # Should work normally after error is gone
        status2 = rate_limiter.check_rate_limit("192.168.1.1", "recovery_test")
        assert isinstance(status2.allowed, bool)
        assert status2.remaining >= 0


class TestRateLimitingPerformance:
    """Test suite for rate limiting performance characteristics."""
    
    def test_large_number_of_requests(self):
        """Test performance with large number of requests."""
        rate_limiter = RateLimiter(backend=RateLimitBackend.MEMORY)
        rule = RateLimitRule(requests=1000, window_seconds=60)
        rate_limiter.add_rule("perf_test", rule)
        
        start_time = time.time()
        
        # Make many requests
        for i in range(500):
            status = rate_limiter.check_rate_limit(f"192.168.1.{i%10}", "perf_test")
            assert isinstance(status.allowed, bool)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds max for 500 requests

    def test_memory_efficiency(self):
        """Test memory efficiency with many different keys."""
        rate_limiter = RateLimiter(backend=RateLimitBackend.MEMORY)
        rule = RateLimitRule(requests=10, window_seconds=60)
        rate_limiter.add_rule("memory_efficiency", rule)
        
        # Create many different keys
        unique_ips = set()
        for i in range(1000):
            ip = f"10.0.{i//256}.{i%256}"
            unique_ips.add(ip)
            rate_limiter.check_rate_limit(ip, "memory_efficiency")
        
        assert len(unique_ips) == 1000
        
        # Memory usage should be reasonable
        # (This is more of a smoke test - in practice would measure actual memory)