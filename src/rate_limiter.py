"""
Rate limiting functionality for API endpoints.

This module provides comprehensive rate limiting capabilities including:
- Token bucket algorithm for smooth rate limiting
- Per-IP and per-endpoint rate limiting
- Redis and in-memory storage backends
- Configurable limits and time windows
- Integration with FastAPI middleware
"""

import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union, Tuple
from enum import Enum
from collections import defaultdict, deque
from threading import Lock
import asyncio
from datetime import datetime, timedelta

from .logging_config import get_logger

logger = get_logger(__name__)


class RateLimitBackend(Enum):
    """Available rate limiting storage backends."""
    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class RateLimitRule:
    """
    Rate limiting rule configuration.
    
    Attributes:
        requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        burst_size: Maximum burst size (for token bucket)
        per_ip: Apply rate limit per IP address
        per_endpoint: Apply rate limit per endpoint
        description: Human-readable description of the rule
    """
    requests: int
    window_seconds: int
    burst_size: Optional[int] = None
    per_ip: bool = True
    per_endpoint: bool = True
    description: str = ""
    
    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = max(1, self.requests // 4)  # Default burst to 25% of limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return asdict(self)


@dataclass
class RateLimitStatus:
    """
    Current rate limiting status for a client.
    
    Attributes:
        allowed: Whether the request is allowed
        requests_remaining: Number of requests remaining in window
        reset_time: When the rate limit resets (Unix timestamp)
        retry_after: Seconds to wait before retrying (if blocked)
    """
    allowed: bool
    requests_remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return asdict(self)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    Provides smooth rate limiting with burst capability using the token bucket algorithm.
    """
    
    def __init__(self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Try to consume requested tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bucket status."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            current_tokens = min(self.capacity, self.tokens + tokens_to_add)
            
            return {
                "tokens": current_tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "last_refill": self.last_refill
            }


class MemoryRateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.
    
    Stores rate limiting state in memory. Suitable for single-instance deployments
    or when Redis is not available.
    """
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.rules: Dict[str, RateLimitRule] = {}
        self._lock = Lock()
        
        # Cleanup old buckets periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def add_rule(self, key: str, rule: RateLimitRule) -> None:
        """Add or update a rate limiting rule."""
        with self._lock:
            self.rules[key] = rule
            logger.info(f"Added rate limit rule for {key}: {rule.requests} req/{rule.window_seconds}s")
    
    def check_rate_limit(self, key: str, tokens: int = 1) -> RateLimitStatus:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier for the client/endpoint combination
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        with self._lock:
            # Periodic cleanup
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_buckets()
                self._last_cleanup = now
            
            # Find applicable rule
            rule = self._find_rule(key)
            if not rule:
                # No rule found - allow request
                return RateLimitStatus(
                    allowed=True,
                    requests_remaining=999999,
                    reset_time=now + 3600
                )
            
            # Get or create token bucket
            bucket_key = f"{key}:{rule.window_seconds}"
            if bucket_key not in self.buckets:
                refill_rate = rule.requests / rule.window_seconds
                self.buckets[bucket_key] = TokenBucket(
                    capacity=rule.burst_size,
                    refill_rate=refill_rate
                )
            
            bucket = self.buckets[bucket_key]
            
            # Check if tokens can be consumed
            allowed = bucket.consume(tokens)
            bucket_status = bucket.get_status()
            
            # Calculate remaining requests and reset time
            requests_remaining = int(bucket_status["tokens"])
            reset_time = now + (rule.burst_size - bucket_status["tokens"]) / bucket.refill_rate
            
            retry_after = None
            if not allowed:
                # Calculate how long to wait for next token
                retry_after = int(tokens / bucket.refill_rate) + 1
            
            return RateLimitStatus(
                allowed=allowed,
                requests_remaining=max(0, requests_remaining),
                reset_time=reset_time,
                retry_after=retry_after
            )
    
    def _find_rule(self, key: str) -> Optional[RateLimitRule]:
        """Find the most specific rate limiting rule for a key."""
        # Try exact match first
        if key in self.rules:
            return self.rules[key]
        
        # Try endpoint-only match (remove IP part)
        if ":" in key:
            endpoint_key = key.split(":", 1)[1]
            if endpoint_key in self.rules:
                return self.rules[endpoint_key]
        
        # Try wildcard matches
        for rule_key, rule in self.rules.items():
            if "*" in rule_key and self._matches_pattern(key, rule_key):
                return rule
        
        # Try default rule
        if "default" in self.rules:
            return self.rules["default"]
        
        return None
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches a wildcard pattern."""
        # Simple wildcard matching
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return key.startswith(prefix)
        
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return key.endswith(suffix)
        
        return key == pattern
    
    def _cleanup_old_buckets(self) -> None:
        """Remove old, unused token buckets to prevent memory leaks."""
        now = time.time()
        keys_to_remove = []
        
        for key, bucket in self.buckets.items():
            # Remove buckets that haven't been accessed recently
            if now - bucket.last_refill > self._cleanup_interval * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.buckets[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old rate limit buckets")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "backend": "memory",
                "active_buckets": len(self.buckets),
                "rules": len(self.rules),
                "rules_config": {k: v.to_dict() for k, v in self.rules.items()}
            }


class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed deployments.
    
    Uses Redis for storing rate limiting state, allowing multiple application
    instances to share rate limiting information.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis rate limiter."""
        try:
            import redis
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()  # Test connection
            logger.info(f"Connected to Redis for rate limiting: {redis_url}")
        except ImportError:
            raise ImportError("Redis library not installed. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        self.rules: Dict[str, RateLimitRule] = {}
        self._lock = Lock()
    
    def add_rule(self, key: str, rule: RateLimitRule) -> None:
        """Add or update a rate limiting rule."""
        with self._lock:
            self.rules[key] = rule
            # Store rule in Redis for persistence
            self.redis.hset("rate_limit_rules", key, json.dumps(rule.to_dict()))
            logger.info(f"Added rate limit rule for {key}: {rule.requests} req/{rule.window_seconds}s")
    
    def check_rate_limit(self, key: str, tokens: int = 1) -> RateLimitStatus:
        """
        Check if request is within rate limit using Redis.
        
        Uses Redis Lua scripting for atomic rate limit checks.
        """
        rule = self._find_rule(key)
        if not rule:
            return RateLimitStatus(
                allowed=True,
                requests_remaining=999999,
                reset_time=time.time() + 3600
            )
        
        # Redis Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- Get current bucket state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens based on elapsed time
        local elapsed = now - last_refill
        local tokens_to_add = elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- Try to consume tokens
        local allowed = tokens >= tokens_requested
        if allowed then
            tokens = tokens - tokens_requested
        end
        
        -- Update bucket state
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity
        
        return {allowed and 1 or 0, tokens, last_refill}
        """
        
        now = time.time()
        bucket_key = f"rate_limit:{key}:{rule.window_seconds}"
        refill_rate = rule.requests / rule.window_seconds
        
        try:
            result = self.redis.eval(
                lua_script,
                1,
                bucket_key,
                rule.burst_size,
                refill_rate,
                tokens,
                now
            )
            
            allowed = bool(result[0])
            current_tokens = float(result[1])
            
            # Calculate reset time and retry after
            reset_time = now + (rule.burst_size - current_tokens) / refill_rate
            retry_after = None
            if not allowed:
                retry_after = int(tokens / refill_rate) + 1
            
            return RateLimitStatus(
                allowed=allowed,
                requests_remaining=max(0, int(current_tokens)),
                reset_time=reset_time,
                retry_after=retry_after
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is unavailable
            return RateLimitStatus(
                allowed=True,
                requests_remaining=0,
                reset_time=now + rule.window_seconds
            )
    
    def _find_rule(self, key: str) -> Optional[RateLimitRule]:
        """Find rate limiting rule, checking Redis first for persistence."""
        # Check memory cache first
        if key in self.rules:
            return self.rules[key]
        
        # Check Redis for persisted rules
        try:
            rule_data = self.redis.hget("rate_limit_rules", key)
            if rule_data:
                rule_dict = json.loads(rule_data)
                rule = RateLimitRule(**rule_dict)
                self.rules[key] = rule  # Cache in memory
                return rule
        except Exception as e:
            logger.warning(f"Failed to load rule from Redis: {e}")
        
        # Fallback to memory-based rule finding
        return self._find_rule_memory(key)
    
    def _find_rule_memory(self, key: str) -> Optional[RateLimitRule]:
        """Memory-based rule finding (same logic as MemoryRateLimiter)."""
        # Try endpoint-only match
        if ":" in key:
            endpoint_key = key.split(":", 1)[1]
            if endpoint_key in self.rules:
                return self.rules[endpoint_key]
        
        # Try wildcard matches
        for rule_key, rule in self.rules.items():
            if "*" in rule_key and self._matches_pattern(key, rule_key):
                return rule
        
        # Try default rule
        if "default" in self.rules:
            return self.rules["default"]
        
        return None
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches a wildcard pattern."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return key.endswith(pattern[1:])
        return key == pattern
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        try:
            info = self.redis.info()
            rule_count = self.redis.hlen("rate_limit_rules")
            
            return {
                "backend": "redis",
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "rules": rule_count,
                "rules_config": {k: v.to_dict() for k, v in self.rules.items()}
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {
                "backend": "redis",
                "error": str(e),
                "rules": len(self.rules)
            }


class RateLimitManager:
    """
    Main rate limiting manager that provides a unified interface.
    
    Automatically selects appropriate backend (Redis or memory) and provides
    high-level rate limiting functionality.
    """
    
    def __init__(self, backend: RateLimitBackend = RateLimitBackend.MEMORY, redis_url: Optional[str] = None):
        """
        Initialize rate limit manager.
        
        Args:
            backend: Rate limiting backend to use
            redis_url: Redis connection URL (required for Redis backend)
        """
        self.backend_type = backend
        
        if backend == RateLimitBackend.REDIS:
            if not redis_url:
                redis_url = "redis://localhost:6379/0"
            try:
                self.backend = RedisRateLimiter(redis_url)
                logger.info("Initialized Redis rate limiter")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis rate limiter, falling back to memory: {e}")
                self.backend = MemoryRateLimiter()
                self.backend_type = RateLimitBackend.MEMORY
        else:
            self.backend = MemoryRateLimiter()
            logger.info("Initialized memory rate limiter")
        
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default rate limiting rules."""
        # Default rules for different endpoint types
        default_rules = {
            "default": RateLimitRule(
                requests=100,
                window_seconds=60,
                description="Default rate limit: 100 requests per minute"
            ),
            "predict": RateLimitRule(
                requests=30,
                window_seconds=60,
                burst_size=10,
                description="Prediction endpoint: 30 requests per minute with burst of 10"
            ),
            "health": RateLimitRule(
                requests=200,
                window_seconds=60,
                description="Health check endpoint: 200 requests per minute"
            ),
            "metrics": RateLimitRule(
                requests=60,
                window_seconds=60,
                description="Metrics endpoint: 60 requests per minute"
            ),
            "train": RateLimitRule(
                requests=5,
                window_seconds=300,
                burst_size=2,
                description="Training endpoint: 5 requests per 5 minutes"
            ),
            "admin": RateLimitRule(
                requests=10,
                window_seconds=60,
                description="Admin endpoints: 10 requests per minute"
            )
        }
        
        for key, rule in default_rules.items():
            self.add_rule(key, rule)
    
    def add_rule(self, key: str, rule: RateLimitRule) -> None:
        """Add or update a rate limiting rule."""
        self.backend.add_rule(key, rule)
    
    def check_rate_limit(self, client_id: str, endpoint: str, tokens: int = 1) -> RateLimitStatus:
        """
        Check rate limit for a client and endpoint.
        
        Args:
            client_id: Client identifier (typically IP address)
            endpoint: Endpoint being accessed
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        # Create composite key for per-IP + per-endpoint limiting
        key = f"{client_id}:{endpoint}"
        
        return self.backend.check_rate_limit(key, tokens)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        stats = self.backend.get_stats()
        stats["manager_backend"] = self.backend_type.value
        return stats


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimitManager] = None


def get_rate_limiter() -> RateLimitManager:
    """
    Get global rate limiter instance.
    
    Returns:
        Global RateLimitManager instance
    """
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        # Try Redis first, fall back to memory
        try:
            _global_rate_limiter = RateLimitManager(RateLimitBackend.REDIS)
        except Exception:
            _global_rate_limiter = RateLimitManager(RateLimitBackend.MEMORY)
    
    return _global_rate_limiter


def check_rate_limit(client_id: str, endpoint: str, tokens: int = 1) -> RateLimitStatus:
    """
    Convenience function to check rate limit.
    
    Args:
        client_id: Client identifier
        endpoint: Endpoint being accessed
        tokens: Number of tokens to consume
        
    Returns:
        RateLimitStatus
    """
    limiter = get_rate_limiter()
    return limiter.check_rate_limit(client_id, endpoint, tokens)


def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiting statistics."""
    limiter = get_rate_limiter()
    return limiter.get_stats()