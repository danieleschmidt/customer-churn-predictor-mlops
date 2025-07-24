"""
Tests for model caching functionality.

This module tests the model cache system used for improving prediction
performance by caching frequently accessed models and preprocessors.
"""

import unittest
import tempfile
import time
import threading
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any

from src.model_cache import (
    ModelCache,
    CacheEntry,
    cached_load_model,
    cached_load_preprocessor,
    cached_load_metadata,
    get_model_cache,
    invalidate_model_cache,
    get_cache_stats
)


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry dataclass functionality."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation with default values."""
        data = {"test": "data"}
        entry = CacheEntry(data=data, timestamp=time.time())
        
        self.assertEqual(entry.data, data)
        self.assertGreater(entry.timestamp, 0)
        self.assertEqual(entry.access_count, 0)
        self.assertGreater(entry.last_access, 0)
        self.assertIsNone(entry.file_path)
        self.assertIsNone(entry.file_mtime)
        self.assertEqual(entry.size_estimate, 0)
    
    def test_cache_entry_post_init(self):
        """Test CacheEntry post-initialization timestamp setting."""
        data = {"test": "data"}
        current_time = time.time()
        
        # Test with zero timestamps
        entry = CacheEntry(data=data, timestamp=0.0, last_access=0.0)
        
        self.assertGreaterEqual(entry.timestamp, current_time)
        self.assertGreaterEqual(entry.last_access, current_time)


class TestModelCache(unittest.TestCase):
    """Test ModelCache class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ModelCache(
            max_entries=3,
            max_memory_mb=1,  # Small for testing
            default_ttl_seconds=1,
            cleanup_interval=0.1
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.cache.shutdown()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization with parameters."""
        self.assertEqual(self.cache.max_entries, 3)
        self.assertEqual(self.cache.max_memory_bytes, 1024 * 1024)  # 1MB
        self.assertEqual(self.cache.default_ttl, 1)
        self.assertIsNotNone(self.cache._cleanup_thread)
    
    def test_generate_key(self):
        """Test cache key generation."""
        key1 = self.cache._generate_key("test_id")
        key2 = self.cache._generate_key("test_id")
        key3 = self.cache._generate_key("test_id", param1="value1")
        key4 = self.cache._generate_key("different_id")
        
        # Same identifier should generate same key
        self.assertEqual(key1, key2)
        
        # Different parameters should generate different key
        self.assertNotEqual(key1, key3)
        
        # Different identifier should generate different key
        self.assertNotEqual(key1, key4)
        
        # Keys should be hex strings of reasonable length
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 16)
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        test_data = {"model": "test_model"}
        identifier = "test_model_1"
        
        # Put data in cache
        self.cache.put(identifier, test_data)
        
        # Get data from cache
        cached_data = self.cache.get(identifier)
        
        self.assertEqual(cached_data, test_data)
        
        # Check statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["total_requests"], 1)
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        cached_data = self.cache.get("nonexistent")
        
        self.assertIsNone(cached_data)
        
        # Check statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total_requests"], 1)
    
    def test_cache_replacement(self):
        """Test cache replacement of existing entries."""
        identifier = "test_model"
        data1 = {"version": 1}
        data2 = {"version": 2}
        
        # Put first version
        self.cache.put(identifier, data1)
        
        # Put second version (should replace)
        self.cache.put(identifier, data2)
        
        # Should get second version
        cached_data = self.cache.get(identifier)
        self.assertEqual(cached_data, data2)
        
        # Should still have only one entry
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 1)
    
    def test_lru_eviction(self):
        """Test LRU eviction when max entries exceeded."""
        # Fill cache to max capacity
        for i in range(3):
            self.cache.put(f"model_{i}", {"data": i})
        
        # Access model_0 and model_1 to make them more recently used
        self.cache.get("model_0")
        self.cache.get("model_1")
        
        # Add one more item (should evict model_2 as least recently used)
        self.cache.put("model_3", {"data": 3})
        
        # Check that model_2 was evicted
        self.assertIsNone(self.cache.get("model_2"))
        self.assertIsNotNone(self.cache.get("model_0"))
        self.assertIsNotNone(self.cache.get("model_1"))
        self.assertIsNotNone(self.cache.get("model_3"))
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 3)
        self.assertEqual(stats["evictions"], 1)
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        identifier = "test_model"
        test_data = {"model": "test"}
        
        # Put data in cache
        self.cache.put(identifier, test_data)
        
        # Should be available immediately
        self.assertIsNotNone(self.cache.get(identifier))
        
        # Wait for TTL expiration (1 second + buffer)
        time.sleep(1.2)
        
        # Should be expired now
        self.assertIsNone(self.cache.get(identifier))
    
    def test_file_invalidation(self):
        """Test file modification time invalidation."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_model.pkl"
        test_file.write_text("initial content")
        initial_mtime = test_file.stat().st_mtime
        
        # Cache with file path
        test_data = {"model": "test"}
        self.cache.put("test_model", test_data, file_path=test_file)
        
        # Should be available
        self.assertIsNotNone(self.cache.get("test_model"))
        
        # Modify file (simulate file update)
        time.sleep(0.1)  # Ensure different mtime
        test_file.write_text("updated content")
        
        # Force file invalidation check
        self.cache._check_file_invalidation()
        
        # Should be invalidated now
        self.assertIsNone(self.cache.get("test_model"))
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        # Put some data in cache
        large_data = {"data": "x" * 1000}  # ~1KB
        self.cache.put("large_model", large_data)
        
        stats = self.cache.get_stats()
        self.assertGreater(stats["memory_used_mb"], 0)
        self.assertLessEqual(stats["memory_used_mb"], stats["max_memory_mb"])
    
    def test_access_count_tracking(self):
        """Test access count tracking."""
        test_data = {"model": "test"}
        identifier = "test_model"
        
        # Put data
        self.cache.put(identifier, test_data)
        
        # Access multiple times
        for i in range(5):
            self.cache.get(identifier)
        
        # Check entry info
        entries = self.cache.get_entry_info()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["access_count"], 5)
    
    def test_invalidate(self):
        """Test manual invalidation."""
        test_data = {"model": "test"}
        identifier = "test_model"
        
        # Put and verify
        self.cache.put(identifier, test_data)
        self.assertIsNotNone(self.cache.get(identifier))
        
        # Invalidate
        self.cache.invalidate(identifier)
        
        # Should be gone
        self.assertIsNone(self.cache.get(identifier))
    
    def test_clear(self):
        """Test cache clearing."""
        # Add multiple entries
        for i in range(3):
            self.cache.put(f"model_{i}", {"data": i})
        
        # Verify entries exist
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 3)
        
        # Clear cache
        self.cache.clear()
        
        # Should be empty
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["memory_used_mb"], 0)
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        def worker(thread_id):
            for i in range(10):
                identifier = f"thread_{thread_id}_model_{i}"
                data = {"thread": thread_id, "model": i}
                self.cache.put(identifier, data)
                retrieved = self.cache.get(identifier)
                self.assertEqual(retrieved, data)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Cache should be functional
        stats = self.cache.get_stats()
        self.assertGreaterEqual(stats["entries"], 0)


class TestCachingFunctions(unittest.TestCase):
    """Test module-level caching functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Clear global cache
        invalidate_model_cache()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        invalidate_model_cache()
    
    def test_cached_load_model(self):
        """Test cached model loading."""
        # Create a test model file
        model_file = Path(self.temp_dir) / "test_model.pkl"
        test_model = {"weights": [1, 2, 3]}
        
        # Mock loader function
        loader_mock = MagicMock(return_value=test_model)
        
        # First load should call loader
        result1 = cached_load_model(model_file, loader_mock)
        self.assertEqual(result1, test_model)
        self.assertEqual(loader_mock.call_count, 1)
        
        # Second load should use cache
        result2 = cached_load_model(model_file, loader_mock)
        self.assertEqual(result2, test_model)
        self.assertEqual(loader_mock.call_count, 1)  # Should not increase
    
    def test_cached_load_preprocessor(self):
        """Test cached preprocessor loading."""
        # Create a test preprocessor file
        preprocessor_file = Path(self.temp_dir) / "test_preprocessor.pkl"
        test_preprocessor = {"scaler": "StandardScaler"}
        
        # Mock loader function
        loader_mock = MagicMock(return_value=test_preprocessor)
        
        # First load should call loader
        result1 = cached_load_preprocessor(preprocessor_file, loader_mock)
        self.assertEqual(result1, test_preprocessor)
        self.assertEqual(loader_mock.call_count, 1)
        
        # Second load should use cache
        result2 = cached_load_preprocessor(preprocessor_file, loader_mock)
        self.assertEqual(result2, test_preprocessor)
        self.assertEqual(loader_mock.call_count, 1)
    
    def test_cached_load_metadata(self):
        """Test cached metadata loading."""
        # Create a test metadata file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        test_metadata = ["feature1", "feature2", "feature3"]
        
        # Mock loader function
        loader_mock = MagicMock(return_value=test_metadata)
        
        # First load should call loader
        result1 = cached_load_metadata(metadata_file, loader_mock)
        self.assertEqual(result1, test_metadata)
        self.assertEqual(loader_mock.call_count, 1)
        
        # Second load should use cache
        result2 = cached_load_metadata(metadata_file, loader_mock)
        self.assertEqual(result2, test_metadata)
        self.assertEqual(loader_mock.call_count, 1)
    
    def test_cache_parameters(self):
        """Test that cache keys include parameters."""
        model_file = Path(self.temp_dir) / "test_model.pkl"
        test_model1 = {"version": 1}
        test_model2 = {"version": 2}
        
        loader_mock1 = MagicMock(return_value=test_model1)
        loader_mock2 = MagicMock(return_value=test_model2)
        
        # Load with different parameters should create different cache entries
        result1 = cached_load_model(model_file, loader_mock1, version=1)
        result2 = cached_load_model(model_file, loader_mock2, version=2)
        
        self.assertEqual(result1, test_model1)
        self.assertEqual(result2, test_model2)
        self.assertEqual(loader_mock1.call_count, 1)
        self.assertEqual(loader_mock2.call_count, 1)
        
        # Loading again with same parameters should use cache
        result1_cached = cached_load_model(model_file, loader_mock1, version=1)
        result2_cached = cached_load_model(model_file, loader_mock2, version=2)
        
        self.assertEqual(result1_cached, test_model1)
        self.assertEqual(result2_cached, test_model2)
        self.assertEqual(loader_mock1.call_count, 1)  # No additional calls
        self.assertEqual(loader_mock2.call_count, 1)
    
    def test_get_global_cache(self):
        """Test global cache instance retrieval."""
        cache1 = get_model_cache()
        cache2 = get_model_cache()
        
        # Should return same instance
        self.assertIs(cache1, cache2)
        self.assertIsInstance(cache1, ModelCache)
    
    def test_get_cache_stats_function(self):
        """Test get_cache_stats function."""
        # Load something to create cache entry
        model_file = Path(self.temp_dir) / "test_model.pkl"
        test_model = {"test": "data"}
        loader_mock = MagicMock(return_value=test_model)
        
        cached_load_model(model_file, loader_mock)
        
        # Get stats
        stats = get_cache_stats()
        
        self.assertIn("stats", stats)
        self.assertIn("entries", stats)
        self.assertGreater(stats["stats"]["entries"], 0)
    
    def test_invalidate_model_cache_function(self):
        """Test cache invalidation function."""
        # Load something to create cache entry
        model_file = Path(self.temp_dir) / "test_model.pkl"
        test_model = {"test": "data"}
        loader_mock = MagicMock(return_value=test_model)
        
        cached_load_model(model_file, loader_mock)
        
        # Verify it's cached
        stats = get_cache_stats()
        self.assertGreater(stats["stats"]["entries"], 0)
        
        # Invalidate
        invalidate_model_cache()
        
        # Should be empty
        stats = get_cache_stats()
        self.assertEqual(stats["stats"]["entries"], 0)


class TestCacheConfiguration(unittest.TestCase):
    """Test cache configuration from environment."""
    
    def test_environment_configuration(self):
        """Test cache configuration from environment variables."""
        with patch.dict(os.environ, {
            'MODEL_CACHE_MAX_ENTRIES': '20',
            'MODEL_CACHE_MAX_MEMORY_MB': '1000',
            'MODEL_CACHE_TTL_SECONDS': '7200'
        }):
            # Clear global cache to force recreation
            global _global_cache
            from src.model_cache import _global_cache
            _global_cache = None
            
            cache = get_model_cache()
            
            self.assertEqual(cache.max_entries, 20)
            self.assertEqual(cache.max_memory_bytes, 1000 * 1024 * 1024)
            self.assertEqual(cache.default_ttl, 7200)


class TestCacheIntegration(unittest.TestCase):
    """Integration tests for cache system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        invalidate_model_cache()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        invalidate_model_cache()
    
    def test_cache_with_real_files(self):
        """Test cache with actual file operations."""
        import json
        
        # Create test files
        model_file = Path(self.temp_dir) / "model.pkl"
        metadata_file = Path(self.temp_dir) / "metadata.json"
        
        # Write test data
        model_data = {"test": "model"}
        metadata_data = ["feature1", "feature2"]
        
        # Mock joblib.load for model
        def mock_joblib_load(path):
            return model_data
        
        # Mock json loading for metadata
        def mock_json_load(path):
            return metadata_data
        
        # Create actual files (for file modification tracking)
        model_file.write_text("dummy model")
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f)
        
        # Load using cache
        cached_model = cached_load_model(model_file, mock_joblib_load)
        cached_metadata = cached_load_metadata(metadata_file, mock_json_load)
        
        self.assertEqual(cached_model, model_data)
        self.assertEqual(cached_metadata, metadata_data)
        
        # Verify caching works
        stats = get_cache_stats()
        self.assertEqual(stats["stats"]["entries"], 2)
        
        # Load again - should use cache
        cached_model2 = cached_load_model(model_file, mock_joblib_load)
        cached_metadata2 = cached_load_metadata(metadata_file, mock_json_load)
        
        self.assertEqual(cached_model2, model_data)
        self.assertEqual(cached_metadata2, metadata_data)
        
        # Should have cache hits
        stats = get_cache_stats()
        self.assertGreater(stats["stats"]["hits"], 0)


if __name__ == "__main__":
    unittest.main()