"""
Model and preprocessor caching system for the churn prediction application.

This module provides efficient caching mechanisms for machine learning models,
preprocessors, and feature metadata to improve prediction performance and
reduce resource usage.
"""

import os
import time
import threading
import hashlib
import weakref
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """
    Cache entry containing model/preprocessor data and metadata.
    
    Attributes:
        data: The cached object (model, preprocessor, etc.)
        timestamp: When the entry was cached
        access_count: Number of times accessed
        last_access: Last access timestamp
        file_path: Source file path for invalidation checking
        file_mtime: Source file modification time
        size_estimate: Estimated memory size in bytes
    """
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    file_path: Optional[Path] = None
    file_mtime: Optional[float] = None
    size_estimate: int = 0
    
    def __post_init__(self):
        """Initialize timestamps on creation."""
        current_time = time.time()
        if self.last_access == 0.0:
            self.last_access = current_time
        if self.timestamp == 0.0:
            self.timestamp = current_time


class ModelCache:
    """
    Thread-safe cache for ML models, preprocessors, and related artifacts.
    
    Features:
    - LRU eviction policy
    - File modification time invalidation
    - Memory usage tracking
    - Thread-safe operations
    - Configurable size limits
    - Access statistics
    """
    
    def __init__(
        self,
        max_entries: int = 10,
        max_memory_mb: int = 500,
        default_ttl_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize model cache.
        
        Args:
            max_entries: Maximum number of cached entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default time-to-live for entries
            cleanup_interval: Cleanup thread interval in seconds
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._memory_used = 0
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "total_requests": 0
        }
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"ModelCache initialized: max_entries={max_entries}, "
                   f"max_memory_mb={max_memory_mb}, ttl={default_ttl_seconds}s")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="ModelCache-Cleanup"
            )
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while not self._shutdown_event.wait(self.cleanup_interval):
            try:
                self._cleanup_expired()
                self._check_file_invalidation()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _generate_key(self, identifier: str, **kwargs) -> str:
        """
        Generate cache key from identifier and optional parameters.
        
        Args:
            identifier: Base identifier (file path, run_id, etc.)
            **kwargs: Additional parameters to include in key
            
        Returns:
            Generated cache key
        """
        # Create deterministic key from identifier and parameters
        key_parts = [str(identifier)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate memory size of an object.
        
        Args:
            obj: Object to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            import sys
            
            # Basic size estimation
            size = sys.getsizeof(obj)
            
            # For common ML objects, add more sophisticated estimation
            if hasattr(obj, 'coef_'):  # sklearn models
                size += sum(sys.getsizeof(attr) for attr in [obj.coef_] if hasattr(obj, attr))
            elif hasattr(obj, 'feature_importances_'):  # tree-based models
                size += sys.getsizeof(obj.feature_importances_)
            elif hasattr(obj, 'components_'):  # PCA, etc.
                size += sys.getsizeof(obj.components_)
            
            return max(size, 1024)  # Minimum 1KB
        except Exception:
            return 1024  # Default to 1KB if estimation fails
    
    def _evict_lru(self):
        """Evict least recently used entries to make space."""
        with self._lock:
            if not self._cache:
                return
            
            # Sort by last access time (oldest first)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda item: item[1].last_access
            )
            
            # Remove oldest entries until under limits
            for key, entry in sorted_entries:
                if (len(self._cache) <= self.max_entries and 
                    self._memory_used <= self.max_memory_bytes):
                    break
                
                self._remove_entry(key, entry)
                self._stats["evictions"] += 1
                logger.debug(f"Evicted cache entry: {key}")
    
    def _remove_entry(self, key: str, entry: CacheEntry):
        """Remove entry from cache and update memory tracking."""
        if key in self._cache:
            del self._cache[key]
            self._memory_used -= entry.size_estimate
            logger.debug(f"Removed cache entry {key}, memory freed: {entry.size_estimate} bytes")
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.default_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._remove_entry(key, entry)
                logger.debug(f"Expired cache entry: {key}")
    
    def _check_file_invalidation(self):
        """Check for file modifications and invalidate stale entries."""
        invalidated_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if entry.file_path and entry.file_path.exists():
                    current_mtime = entry.file_path.stat().st_mtime
                    if entry.file_mtime and current_mtime > entry.file_mtime:
                        invalidated_keys.append(key)
            
            for key in invalidated_keys:
                entry = self._cache[key]
                self._remove_entry(key, entry)
                self._stats["invalidations"] += 1
                logger.info(f"Invalidated cache entry due to file change: {key}")
    
    def get(self, identifier: str, **kwargs) -> Optional[Any]:
        """
        Get cached object by identifier.
        
        Args:
            identifier: Object identifier
            **kwargs: Additional key parameters
            
        Returns:
            Cached object or None if not found
        """
        key = self._generate_key(identifier, **kwargs)
        
        with self._lock:
            self._stats["total_requests"] += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if time.time() - entry.timestamp > self.default_ttl:
                    self._remove_entry(key, entry)
                    self._stats["misses"] += 1
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_access = time.time()
                
                self._stats["hits"] += 1
                logger.debug(f"Cache hit: {key} (access count: {entry.access_count})")
                return entry.data
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
                return None
    
    def put(
        self,
        identifier: str,
        data: Any,
        file_path: Optional[Path] = None,
        **kwargs
    ):
        """
        Store object in cache.
        
        Args:
            identifier: Object identifier
            data: Object to cache
            file_path: Source file path for invalidation
            **kwargs: Additional key parameters
        """
        key = self._generate_key(identifier, **kwargs)
        size_estimate = self._estimate_size(data)
        
        # Check if object is too large
        if size_estimate > self.max_memory_bytes:
            logger.warning(f"Object too large to cache: {size_estimate} bytes > {self.max_memory_bytes}")
            return
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._memory_used -= old_entry.size_estimate
            
            # Create new entry
            file_mtime = file_path.stat().st_mtime if file_path and file_path.exists() else None
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                file_path=file_path,
                file_mtime=file_mtime,
                size_estimate=size_estimate
            )
            
            self._cache[key] = entry
            self._memory_used += size_estimate
            
            logger.debug(f"Cached object: {key} (size: {size_estimate} bytes)")
            
            # Evict if over limits
            self._evict_lru()
    
    def invalidate(self, identifier: str, **kwargs):
        """
        Invalidate cached entry.
        
        Args:
            identifier: Object identifier
            **kwargs: Additional key parameters
        """
        key = self._generate_key(identifier, **kwargs)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._remove_entry(key, entry)
                logger.info(f"Manually invalidated cache entry: {key}")
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._memory_used = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            hit_rate = (self._stats["hits"] / max(self._stats["total_requests"], 1)) * 100
            
            return {
                "entries": len(self._cache),
                "memory_used_mb": self._memory_used / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": (self._memory_used / self.max_memory_bytes) * 100,
                "hit_rate": hit_rate,
                **self._stats
            }
    
    def get_entry_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cached entries.
        
        Returns:
            List of entry information dictionaries
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    "key": key,
                    "age_seconds": time.time() - entry.timestamp,
                    "access_count": entry.access_count,
                    "last_access_ago": time.time() - entry.last_access,
                    "size_mb": entry.size_estimate / (1024 * 1024),
                    "file_path": str(entry.file_path) if entry.file_path else None
                })
            
            return sorted(entries, key=lambda x: x["last_access_ago"])
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        logger.info("Shutting down ModelCache")
        self._shutdown_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self.clear()


# Global cache instance
_global_cache: Optional[ModelCache] = None
_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    """
    Get global model cache instance.
    
    Returns:
        Global ModelCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                # Configure cache from environment or defaults
                max_entries = int(os.environ.get('MODEL_CACHE_MAX_ENTRIES', '10'))
                max_memory_mb = int(os.environ.get('MODEL_CACHE_MAX_MEMORY_MB', '500'))
                ttl_seconds = int(os.environ.get('MODEL_CACHE_TTL_SECONDS', '3600'))
                
                _global_cache = ModelCache(
                    max_entries=max_entries,
                    max_memory_mb=max_memory_mb,
                    default_ttl_seconds=ttl_seconds
                )
                
                logger.info("Global ModelCache instance created")
    
    return _global_cache


def cached_load_model(model_path: Path, loader_func, **kwargs) -> Any:
    """
    Load model with caching.
    
    Args:
        model_path: Path to model file
        loader_func: Function to load model (e.g., joblib.load)
        **kwargs: Additional parameters for cache key
        
    Returns:
        Loaded model (from cache or fresh load)
    """
    cache = get_model_cache()
    identifier = f"model:{model_path}"
    
    # Try cache first
    cached_model = cache.get(identifier, **kwargs)
    if cached_model is not None:
        logger.debug(f"Loaded model from cache: {model_path}")
        return cached_model
    
    # Load fresh and cache
    logger.info(f"Loading model from file: {model_path}")
    model = loader_func(model_path)
    cache.put(identifier, model, file_path=model_path, **kwargs)
    
    return model


def cached_load_preprocessor(preprocessor_path: Path, loader_func, **kwargs) -> Any:
    """
    Load preprocessor with caching.
    
    Args:
        preprocessor_path: Path to preprocessor file
        loader_func: Function to load preprocessor
        **kwargs: Additional parameters for cache key
        
    Returns:
        Loaded preprocessor (from cache or fresh load)
    """
    cache = get_model_cache()
    identifier = f"preprocessor:{preprocessor_path}"
    
    # Try cache first
    cached_preprocessor = cache.get(identifier, **kwargs)
    if cached_preprocessor is not None:
        logger.debug(f"Loaded preprocessor from cache: {preprocessor_path}")
        return cached_preprocessor
    
    # Load fresh and cache
    logger.info(f"Loading preprocessor from file: {preprocessor_path}")
    preprocessor = loader_func(preprocessor_path)
    cache.put(identifier, preprocessor, file_path=preprocessor_path, **kwargs)
    
    return preprocessor


def cached_load_metadata(metadata_path: Path, loader_func, **kwargs) -> Any:
    """
    Load metadata (feature columns, config, etc.) with caching.
    
    Args:
        metadata_path: Path to metadata file
        loader_func: Function to load metadata
        **kwargs: Additional parameters for cache key
        
    Returns:
        Loaded metadata (from cache or fresh load)
    """
    cache = get_model_cache()
    identifier = f"metadata:{metadata_path}"
    
    # Try cache first
    cached_metadata = cache.get(identifier, **kwargs)
    if cached_metadata is not None:
        logger.debug(f"Loaded metadata from cache: {metadata_path}")
        return cached_metadata
    
    # Load fresh and cache
    logger.info(f"Loading metadata from file: {metadata_path}")
    metadata = loader_func(metadata_path)
    cache.put(identifier, metadata, file_path=metadata_path, **kwargs)
    
    return metadata


def invalidate_model_cache(model_path: Optional[Path] = None):
    """
    Invalidate model cache entries.
    
    Args:
        model_path: Specific model path to invalidate, or None to clear all
    """
    cache = get_model_cache()
    
    if model_path:
        cache.invalidate(f"model:{model_path}")
        logger.info(f"Invalidated model cache for: {model_path}")
    else:
        cache.clear()
        logger.info("Cleared entire model cache")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary containing cache statistics and entry information
    """
    cache = get_model_cache()
    return {
        "stats": cache.get_stats(),
        "entries": cache.get_entry_info()
    }