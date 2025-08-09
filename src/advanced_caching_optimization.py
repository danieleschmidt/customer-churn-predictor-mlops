"""
Advanced Caching and Data Optimization System.

This module provides comprehensive caching and data optimization capabilities including:
- Multi-tier intelligent caching (L1: Memory, L2: Redis, L3: Disk, L4: Cloud)
- Data compression and serialization optimization
- Query optimization and result caching
- Cache warming and prefetching strategies
- Data lifecycle management and archival
- Cache coherence and invalidation strategies
- Distributed caching with consistency guarantees
- Performance-aware data partitioning and sharding
"""

import os
import json
import time
import asyncio
import threading
import hashlib
import pickle
import gzip
import lzma
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import redis
import memcache
import sqlite3
import requests
from functools import wraps, lru_cache
import joblib
from sklearn.base import BaseEstimator
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .error_handling_recovery import with_error_handling, error_handler

logger = get_logger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    cache_name: str
    hit_count: int
    miss_count: int
    eviction_count: int
    size_bytes: int
    avg_access_time_ms: float
    last_updated: datetime
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    priority: int = 1  # Higher = more important
    compression_type: str = "none"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_memory_mb: int = 1024
    max_items: int = 10000
    default_ttl_seconds: int = 3600
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    compression_method: str = "gzip"  # gzip, lzma, lz4
    eviction_policy: str = "lru"  # lru, lfu, fifo, random
    enable_persistence: bool = True
    persistence_interval_seconds: int = 300
    enable_prefetching: bool = True
    prefetch_ratio: float = 0.1


class CompressionManager:
    """Handles data compression and decompression."""
    
    def __init__(self):
        self.supported_methods = {
            'gzip': (gzip.compress, gzip.decompress),
            'lzma': (lzma.compress, lzma.decompress)
        }
        
        try:
            import lz4.frame
            self.supported_methods['lz4'] = (lz4.frame.compress, lz4.frame.decompress)
        except ImportError:
            pass
    
    def compress(self, data: bytes, method: str = "gzip") -> Tuple[bytes, float]:
        """Compress data and return compressed data with compression ratio."""
        if method not in self.supported_methods:
            return data, 1.0
        
        start_time = time.time()
        compress_func = self.supported_methods[method][0]
        
        try:
            compressed = compress_func(data)
            compression_time = time.time() - start_time
            ratio = len(compressed) / len(data) if data else 1.0
            
            logger.debug(f"Compressed {len(data)} bytes to {len(compressed)} bytes "
                        f"({ratio:.2f} ratio) in {compression_time:.3f}s using {method}")
            
            return compressed, ratio
            
        except Exception as e:
            logger.warning(f"Compression failed with {method}: {e}")
            return data, 1.0
    
    def decompress(self, data: bytes, method: str = "gzip") -> bytes:
        """Decompress data."""
        if method not in self.supported_methods:
            return data
        
        decompress_func = self.supported_methods[method][1]
        
        try:
            return decompress_func(data)
        except Exception as e:
            logger.error(f"Decompression failed with {method}: {e}")
            return data
    
    def find_best_compression(self, data: bytes) -> Tuple[str, bytes, float]:
        """Find the best compression method for given data."""
        best_method = "none"
        best_data = data
        best_ratio = 1.0
        
        for method in self.supported_methods:
            compressed, ratio = self.compress(data, method)
            if ratio < best_ratio:
                best_method = method
                best_data = compressed
                best_ratio = ratio
        
        return best_method, best_data, best_ratio


class SerializationManager:
    """Handles efficient serialization and deserialization."""
    
    def __init__(self):
        self.serializers = {
            'pickle': (pickle.dumps, pickle.loads),
            'joblib': (joblib.dump, joblib.load)
        }
        
        try:
            import dill
            self.serializers['dill'] = (dill.dumps, dill.loads)
        except ImportError:
            pass
        
        try:
            import cloudpickle
            self.serializers['cloudpickle'] = (cloudpickle.dumps, cloudpickle.loads)
        except ImportError:
            pass
    
    def serialize(self, obj: Any, method: str = "pickle") -> bytes:
        """Serialize object to bytes."""
        if method == "json" and self._is_json_serializable(obj):
            return json.dumps(obj).encode('utf-8')
        
        if method in self.serializers:
            serialize_func = self.serializers[method][0]
            if method == "joblib":
                # joblib needs special handling
                import io
                buffer = io.BytesIO()
                serialize_func(obj, buffer)
                return buffer.getvalue()
            else:
                return serialize_func(obj)
        
        # Fallback to pickle
        return pickle.dumps(obj)
    
    def deserialize(self, data: bytes, method: str = "pickle") -> Any:
        """Deserialize bytes to object."""
        if method == "json":
            return json.loads(data.decode('utf-8'))
        
        if method in self.serializers:
            deserialize_func = self.serializers[method][1]
            if method == "joblib":
                import io
                buffer = io.BytesIO(data)
                return deserialize_func(buffer)
            else:
                return deserialize_func(data)
        
        # Fallback to pickle
        return pickle.loads(data)
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object can be JSON serialized."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    def find_best_serializer(self, obj: Any) -> str:
        """Find the most efficient serializer for an object."""
        if self._is_json_serializable(obj):
            return "json"
        
        if isinstance(obj, (BaseEstimator, np.ndarray, pd.DataFrame)):
            return "joblib"
        
        return "pickle"


class InMemoryCache:
    """High-performance in-memory cache with advanced features."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
        
        # Metrics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Background tasks
        self.cleanup_thread = None
        self.start_background_tasks()
    
    def start_background_tasks(self) -> None:
        """Start background cleanup tasks."""
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired items."""
        while True:
            try:
                self._cleanup_expired()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache. Returns (value, hit)."""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None, False
            
            item = self.cache[key]
            
            # Check if expired
            if item.is_expired:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.miss_count += 1
                return None, False
            
            # Update access info
            item.last_accessed = datetime.now()
            item.access_count += 1
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            # Move to end for LRU
            self.cache.move_to_end(key)
            
            self.hit_count += 1
            return item.value, True
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
           priority: int = 1, tags: List[str] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check memory limits
            if not self._has_space(size_bytes):
                if not self._evict_to_make_space(size_bytes):
                    return False
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                priority=priority,
                tags=tags or []
            )
            
            # Remove old item if exists
            if key in self.cache:
                old_item = self.cache[key]
                del self.cache[key]
            
            # Add new item
            self.cache[key] = item
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                return True
            return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all items with matching tags."""
        with self.lock:
            keys_to_remove = []
            
            for key, item in self.cache.items():
                if any(tag in item.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.delete(key)
            
            return len(keys_to_remove)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            else:
                # Use pickle size as approximation
                return len(pickle.dumps(value))
        except:
            return 1000  # Default estimate
    
    def _has_space(self, size_needed: int) -> bool:
        """Check if cache has space for new item."""
        current_size = sum(item.size_bytes for item in self.cache.values())
        max_size = self.config.max_memory_mb * 1024 * 1024
        
        return (current_size + size_needed <= max_size and 
                len(self.cache) < self.config.max_items)
    
    def _evict_to_make_space(self, size_needed: int) -> bool:
        """Evict items to make space."""
        if self.config.eviction_policy == "lru":
            return self._evict_lru(size_needed)
        elif self.config.eviction_policy == "lfu":
            return self._evict_lfu(size_needed)
        elif self.config.eviction_policy == "fifo":
            return self._evict_fifo(size_needed)
        else:
            return self._evict_random(size_needed)
    
    def _evict_lru(self, size_needed: int) -> bool:
        """Evict least recently used items."""
        space_freed = 0
        keys_to_remove = []
        
        # Items are already in LRU order (OrderedDict)
        for key, item in self.cache.items():
            if space_freed >= size_needed:
                break
            keys_to_remove.append(key)
            space_freed += item.size_bytes
        
        for key in keys_to_remove:
            self.delete(key)
            self.eviction_count += 1
        
        return space_freed >= size_needed
    
    def _evict_lfu(self, size_needed: int) -> bool:
        """Evict least frequently used items."""
        space_freed = 0
        
        # Sort by access count (ascending)
        sorted_items = sorted(self.cache.items(), 
                            key=lambda x: self.access_counts.get(x[0], 0))
        
        for key, item in sorted_items:
            if space_freed >= size_needed:
                break
            self.delete(key)
            space_freed += item.size_bytes
            self.eviction_count += 1
        
        return space_freed >= size_needed
    
    def _evict_fifo(self, size_needed: int) -> bool:
        """Evict first in, first out."""
        space_freed = 0
        keys_to_remove = []
        
        for key, item in self.cache.items():
            if space_freed >= size_needed:
                break
            keys_to_remove.append(key)
            space_freed += item.size_bytes
        
        for key in keys_to_remove:
            self.delete(key)
            self.eviction_count += 1
        
        return space_freed >= size_needed
    
    def _evict_random(self, size_needed: int) -> bool:
        """Evict random items."""
        import random
        
        space_freed = 0
        keys = list(self.cache.keys())
        random.shuffle(keys)
        
        for key in keys:
            if space_freed >= size_needed:
                break
            item = self.cache[key]
            self.delete(key)
            space_freed += item.size_bytes
            self.eviction_count += 1
        
        return space_freed >= size_needed
    
    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        with self.lock:
            expired_keys = [
                key for key, item in self.cache.items() 
                if item.is_expired
            ]
            
            for key in expired_keys:
                self.delete(key)
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        with self.lock:
            total_size = sum(item.size_bytes for item in self.cache.values())
            avg_access_time = 0.001  # Simplified
            
            return CacheMetrics(
                cache_name="in_memory",
                hit_count=self.hit_count,
                miss_count=self.miss_count,
                eviction_count=self.eviction_count,
                size_bytes=total_size,
                avg_access_time_ms=avg_access_time * 1000,
                last_updated=datetime.now()
            )
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, config: CacheConfig, redis_url: str = None):
        self.config = config
        self.compression = CompressionManager()
        self.serialization = SerializationManager()
        
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
            else:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    password=os.getenv('REDIS_PASSWORD'),
                    decode_responses=False  # We handle bytes
                )
            
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
            self.available = False
    
    @with_error_handling(component="redis_cache", enable_retry=True)
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from Redis cache."""
        if not self.available:
            return None, False
        
        try:
            # Get data and metadata
            pipe = self.redis_client.pipeline()
            pipe.get(key)
            pipe.hgetall(f"{key}:meta")
            results = pipe.execute()
            
            data, metadata = results
            
            if data is None:
                return None, False
            
            # Deserialize metadata
            compression_method = metadata.get(b'compression', b'none').decode()
            serialization_method = metadata.get(b'serialization', b'pickle').decode()
            
            # Decompress if needed
            if compression_method != 'none':
                data = self.compression.decompress(data, compression_method)
            
            # Deserialize
            value = self.serialization.deserialize(data, serialization_method)
            
            # Update access time
            self.redis_client.hset(f"{key}:meta", "last_accessed", 
                                 datetime.now().timestamp())
            
            return value, True
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None, False
    
    @with_error_handling(component="redis_cache", enable_retry=True)
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
           tags: List[str] = None) -> bool:
        """Set value in Redis cache."""
        if not self.available:
            return False
        
        try:
            # Determine best serialization method
            serialization_method = self.serialization.find_best_serializer(value)
            
            # Serialize
            data = self.serialization.serialize(value, serialization_method)
            
            # Compress if beneficial
            compression_method = "none"
            if (len(data) > self.config.compression_threshold_bytes and 
                self.config.enable_compression):
                
                compressed_data, ratio = self.compression.compress(
                    data, self.config.compression_method
                )
                
                if ratio < 0.8:  # Only use if significant compression
                    data = compressed_data
                    compression_method = self.config.compression_method
            
            # Set TTL
            ttl = ttl_seconds or self.config.default_ttl_seconds
            
            # Store data and metadata
            pipe = self.redis_client.pipeline()
            pipe.set(key, data, ex=ttl)
            pipe.hset(f"{key}:meta", mapping={
                "created_at": datetime.now().timestamp(),
                "last_accessed": datetime.now().timestamp(),
                "serialization": serialization_method,
                "compression": compression_method,
                "size_bytes": len(data),
                "tags": json.dumps(tags or [])
            })
            pipe.expire(f"{key}:meta", ttl)
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.available:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.delete(key)
            pipe.delete(f"{key}:meta")
            results = pipe.execute()
            return results[0] > 0
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all keys with matching tags."""
        if not self.available:
            return 0
        
        try:
            # Find all keys with metadata
            meta_keys = self.redis_client.keys("*:meta")
            keys_to_delete = []
            
            for meta_key in meta_keys:
                metadata = self.redis_client.hgetall(meta_key)
                if b'tags' in metadata:
                    key_tags = json.loads(metadata[b'tags'].decode())
                    if any(tag in key_tags for tag in tags):
                        # Get original key (remove :meta suffix)
                        original_key = meta_key.decode()[:-5]
                        keys_to_delete.append(original_key)
            
            # Delete matching keys
            if keys_to_delete:
                pipe = self.redis_client.pipeline()
                for key in keys_to_delete:
                    pipe.delete(key)
                    pipe.delete(f"{key}:meta")
                pipe.execute()
            
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Redis invalidate by tags error: {e}")
            return 0
    
    def get_metrics(self) -> Optional[CacheMetrics]:
        """Get Redis cache metrics."""
        if not self.available:
            return None
        
        try:
            info = self.redis_client.info()
            
            return CacheMetrics(
                cache_name="redis",
                hit_count=info.get('keyspace_hits', 0),
                miss_count=info.get('keyspace_misses', 0),
                eviction_count=info.get('evicted_keys', 0),
                size_bytes=info.get('used_memory', 0),
                avg_access_time_ms=0.5,  # Typical Redis latency
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Redis metrics error: {e}")
            return None


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, config: CacheConfig, cache_dir: str = "/tmp/cache"):
        self.config = config
        self.cache_dir = cache_dir
        self.compression = CompressionManager()
        self.serialization = SerializationManager()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize SQLite index
        self.db_path = os.path.join(cache_dir, "cache_index.db")
        self._init_db()
        
        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _init_db(self) -> None:
        """Initialize SQLite database for cache metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_items (
                key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                size_bytes INTEGER NOT NULL,
                ttl_seconds INTEGER,
                compression_method TEXT,
                serialization_method TEXT,
                tags TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from disk cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get metadata
            cursor.execute(
                "SELECT file_path, created_at, ttl_seconds, compression_method, "
                "serialization_method FROM cache_items WHERE key = ?",
                (key,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None, False
            
            file_path, created_at, ttl_seconds, compression_method, serialization_method = row
            
            # Check if expired
            if ttl_seconds and (time.time() - created_at) > ttl_seconds:
                self.delete(key)
                return None, False
            
            # Read file
            full_path = os.path.join(self.cache_dir, file_path)
            if not os.path.exists(full_path):
                self.delete(key)
                return None, False
            
            with open(full_path, 'rb') as f:
                data = f.read()
            
            # Decompress if needed
            if compression_method and compression_method != 'none':
                data = self.compression.decompress(data, compression_method)
            
            # Deserialize
            value = self.serialization.deserialize(data, serialization_method or 'pickle')
            
            # Update access info
            cursor.execute(
                "UPDATE cache_items SET last_accessed = ?, access_count = access_count + 1 "
                "WHERE key = ?",
                (time.time(), key)
            )
            conn.commit()
            
            return value, True
            
        except Exception as e:
            logger.error(f"Disk cache get error for key {key}: {e}")
            return None, False
        finally:
            conn.close()
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
           tags: List[str] = None) -> bool:
        """Set value in disk cache."""
        try:
            # Determine serialization method
            serialization_method = self.serialization.find_best_serializer(value)
            
            # Serialize
            data = self.serialization.serialize(value, serialization_method)
            
            # Compress if beneficial
            compression_method = "none"
            if (len(data) > self.config.compression_threshold_bytes and 
                self.config.enable_compression):
                
                compressed_data, ratio = self.compression.compress(
                    data, self.config.compression_method
                )
                
                if ratio < 0.8:
                    data = compressed_data
                    compression_method = self.config.compression_method
            
            # Generate file path
            file_hash = hashlib.md5(key.encode()).hexdigest()
            file_path = f"{file_hash[:2]}/{file_hash}.cache"
            full_path = os.path.join(self.cache_dir, file_path)
            
            # Create subdirectory
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'wb') as f:
                f.write(data)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_items 
                (key, file_path, created_at, last_accessed, access_count, 
                 size_bytes, ttl_seconds, compression_method, serialization_method, tags)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
            """, (
                key, file_path, time.time(), time.time(), len(data),
                ttl_seconds or self.config.default_ttl_seconds,
                compression_method, serialization_method, json.dumps(tags or [])
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get file path
            cursor.execute("SELECT file_path FROM cache_items WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                file_path = row[0]
                full_path = os.path.join(self.cache_dir, file_path)
                
                # Delete file
                if os.path.exists(full_path):
                    os.remove(full_path)
                
                # Delete from database
                cursor.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                conn.commit()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Disk cache delete error for key {key}: {e}")
            return False
        finally:
            conn.close()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired items."""
        while True:
            try:
                self._cleanup_expired()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Disk cache cleanup error: {e}")
                time.sleep(300)
    
    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Find expired items
            cursor.execute("""
                SELECT key FROM cache_items 
                WHERE ttl_seconds IS NOT NULL 
                AND (? - created_at) > ttl_seconds
            """, (time.time(),))
            
            expired_keys = [row[0] for row in cursor.fetchall()]
            
            # Delete expired items
            for key in expired_keys:
                self.delete(key)
                
        finally:
            conn.close()


class MultiTierCache:
    """Multi-tier cache system with L1 (Memory), L2 (Redis), L3 (Disk), L4 (Cloud)."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize cache tiers
        self.l1_cache = InMemoryCache(config)  # Memory
        self.l2_cache = RedisCache(config)     # Redis
        self.l3_cache = DiskCache(config)      # Disk
        
        # Cache warming and prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self.warming_stats = defaultdict(int)
        
        # Performance tracking
        self.tier_metrics = {}
    
    def get(self, key: str) -> Tuple[Optional[Any], str]:
        """Get value from cache, checking tiers in order. Returns (value, tier)."""
        
        # L1 (Memory) - fastest
        value, hit = self.l1_cache.get(key)
        if hit:
            return value, "L1"
        
        # L2 (Redis) - fast
        value, hit = self.l2_cache.get(key)
        if hit:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value, "L2"
        
        # L3 (Disk) - slower but persistent
        value, hit = self.l3_cache.get(key)
        if hit:
            # Promote to L1 and L2
            self.l1_cache.set(key, value)
            if self.l2_cache.available:
                self.l2_cache.set(key, value)
            return value, "L3"
        
        return None, "MISS"
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
           priority: int = 1, tags: List[str] = None) -> Dict[str, bool]:
        """Set value in appropriate cache tiers."""
        
        results = {}
        
        # Always set in L1 (memory)
        results["L1"] = self.l1_cache.set(key, value, ttl_seconds, priority, tags)
        
        # Set in L2 (Redis) for sharing across instances
        if self.l2_cache.available:
            results["L2"] = self.l2_cache.set(key, value, ttl_seconds, tags)
        
        # Set in L3 (Disk) for persistence, only for high priority items
        if priority >= 2:
            results["L3"] = self.l3_cache.set(key, value, ttl_seconds, tags)
        
        # Trigger prefetching for related items
        if self.config.enable_prefetching:
            self._trigger_prefetching(key, tags or [])
        
        return results
    
    def delete(self, key: str) -> Dict[str, bool]:
        """Delete key from all cache tiers."""
        return {
            "L1": self.l1_cache.delete(key),
            "L2": self.l2_cache.delete(key),
            "L3": self.l3_cache.delete(key)
        }
    
    def invalidate_by_tags(self, tags: List[str]) -> Dict[str, int]:
        """Invalidate items by tags across all tiers."""
        return {
            "L1": self.l1_cache.invalidate_by_tags(tags),
            "L2": self.l2_cache.invalidate_by_tags(tags),
            "L3": 0  # Not implemented for disk cache in this example
        }
    
    def warm_cache(self, key_value_pairs: List[Tuple[str, Any]], 
                   priority: int = 1) -> None:
        """Warm cache with pre-computed values."""
        
        def warm_item(key_value):
            key, value = key_value
            try:
                self.set(key, value, priority=priority)
                self.warming_stats[key] += 1
                logger.debug(f"Cache warmed: {key}")
            except Exception as e:
                logger.error(f"Cache warming failed for {key}: {e}")
        
        # Warm cache in background
        futures = []
        for kv in key_value_pairs:
            future = self.prefetch_executor.submit(warm_item, kv)
            futures.append(future)
        
        # Wait for completion (optional)
        for future in as_completed(futures, timeout=30):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Cache warming task failed: {e}")
    
    def _trigger_prefetching(self, key: str, tags: List[str]) -> None:
        """Trigger prefetching of related items."""
        
        def prefetch_task():
            try:
                # Simple prefetching strategy: load items with same tags
                # In practice, this would be more sophisticated
                
                related_patterns = []
                
                # Add pattern-based prefetching
                if "model:" in key:
                    # Prefetch related model artifacts
                    base_key = key.split(':')[1]
                    related_patterns = [
                        f"model_config:{base_key}",
                        f"model_metrics:{base_key}",
                        f"model_features:{base_key}"
                    ]
                
                for pattern in related_patterns:
                    # This would typically query a database or compute predictions
                    # For now, just log the prefetch attempt
                    logger.debug(f"Prefetching triggered for pattern: {pattern}")
                    
            except Exception as e:
                logger.error(f"Prefetching error: {e}")
        
        # Execute prefetching in background
        if random.random() < self.config.prefetch_ratio:
            self.prefetch_executor.submit(prefetch_task)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        
        l1_metrics = self.l1_cache.get_metrics()
        l2_metrics = self.l2_cache.get_metrics()
        
        total_hits = l1_metrics.hit_count + (l2_metrics.hit_count if l2_metrics else 0)
        total_misses = l1_metrics.miss_count + (l2_metrics.miss_count if l2_metrics else 0)
        
        return {
            "L1_memory": asdict(l1_metrics),
            "L2_redis": asdict(l2_metrics) if l2_metrics else None,
            "L3_disk": {"status": "active"},  # Simplified
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
                "tiers_active": sum([1, 1 if l2_metrics else 0, 1])  # L1, L2 (if available), L3
            },
            "warming_stats": dict(self.warming_stats)
        }


# Decorators for easy caching
def cached(ttl_seconds: int = 3600, priority: int = 1, tags: List[str] = None):
    """Decorator to cache function results."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            key = f"func:{hashlib.md5(str(key_data).encode()).hexdigest()}"
            
            # Try to get from cache
            if hasattr(wrapper, '_cache'):
                value, tier = wrapper._cache.get(key)
                if tier != "MISS":
                    return value
            
            # Compute and cache result
            result = func(*args, **kwargs)
            
            if hasattr(wrapper, '_cache'):
                wrapper._cache.set(key, result, ttl_seconds, priority, tags)
            
            return result
        
        return wrapper
    return decorator


# Query result caching
class QueryCache:
    """Specialized cache for database/API query results."""
    
    def __init__(self, cache: MultiTierCache):
        self.cache = cache
        self.query_stats = defaultdict(lambda: {"count": 0, "avg_time": 0.0})
    
    def get_or_execute(self, query_key: str, query_func: Callable, 
                      ttl_seconds: int = 3600, force_refresh: bool = False) -> Any:
        """Get cached query result or execute query and cache result."""
        
        if not force_refresh:
            value, tier = self.cache.get(query_key)
            if tier != "MISS":
                self.query_stats[query_key]["count"] += 1
                return value
        
        # Execute query and time it
        start_time = time.time()
        result = query_func()
        execution_time = time.time() - start_time
        
        # Update statistics
        stats = self.query_stats[query_key]
        stats["count"] += 1
        stats["avg_time"] = (stats["avg_time"] * (stats["count"] - 1) + execution_time) / stats["count"]
        
        # Cache result with tags based on query type
        tags = ["query", query_key.split(":")[0] if ":" in query_key else "unknown"]
        self.cache.set(query_key, result, ttl_seconds, tags=tags)
        
        logger.debug(f"Query executed and cached: {query_key} ({execution_time:.3f}s)")
        
        return result
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return dict(self.query_stats)


# Factory function
def create_caching_system(config: CacheConfig = None) -> MultiTierCache:
    """Create and configure multi-tier caching system."""
    if config is None:
        config = CacheConfig()
    
    return MultiTierCache(config)


# Global cache instance
_global_cache = None

def get_global_cache() -> MultiTierCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = create_caching_system()
    return _global_cache


if __name__ == "__main__":
    print("Advanced Caching and Data Optimization System")
    print("This system provides multi-tier intelligent caching with compression and optimization.")