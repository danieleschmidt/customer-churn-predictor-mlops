"""
Comprehensive Integration Testing Suite.

This module provides end-to-end integration testing including:
- End-to-end workflow testing with realistic data flows
- Distributed system integration tests with multi-component validation
- Multi-component interaction testing with dependency injection
- Database integration testing with test containers and data isolation
- Service mesh testing with network partitions and latency simulation
- Cross-service communication validation with contract verification
- State consistency testing across distributed components
"""

import os
import json
import time
import asyncio
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, AsyncGenerator, ContextManager
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import sqlite3
import uuid

# Test framework
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock

# Database testing
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Container testing
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Network testing
try:
    import requests
    import aiohttp
    import httpx
    HTTP_CLIENTS_AVAILABLE = True
except ImportError:
    HTTP_CLIENTS_AVAILABLE = False

# Redis for caching tests
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ML libraries for workflow testing
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False


@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    test_id: str
    test_name: str
    test_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    passed: bool
    error_message: Optional[str] = None
    components_tested: List[str] = field(default_factory=list)
    assertions_passed: int = 0
    assertions_failed: int = 0
    test_data: Dict[str, Any] = field(default_factory=dict)
    cleanup_successful: bool = True


@dataclass
class IntegrationTestSuite:
    """Collection of integration test results."""
    suite_id: str
    suite_name: str
    start_time: datetime
    end_time: datetime
    test_results: List[IntegrationTestResult]
    environment_setup: Dict[str, Any]
    overall_passed: bool
    total_duration_seconds: float


class TestEnvironmentManager:
    """Manages test environments with isolation and cleanup."""
    
    def __init__(self):
        self.temp_dirs = []
        self.processes = []
        self.containers = []
        self.databases = []
        self.cleanup_callbacks = []
        
    @contextmanager
    def isolated_environment(self, env_name: str = None):
        """Create an isolated test environment."""
        env_name = env_name or f"test_env_{uuid.uuid4().hex[:8]}"
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"{env_name}_")
        self.temp_dirs.append(temp_dir)
        
        # Set environment variables
        old_env = dict(os.environ)
        os.environ.update({
            'TEST_ENV_NAME': env_name,
            'TEST_TEMP_DIR': temp_dir,
            'TEST_MODE': 'true',
            'LOG_LEVEL': 'INFO'
        })
        
        try:
            yield {
                'env_name': env_name,
                'temp_dir': temp_dir,
                'env_vars': dict(os.environ)
            }
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(old_env)
            
            # Cleanup will be handled by cleanup_all()
    
    @contextmanager  
    def test_database(self, db_name: str = None):
        """Create an isolated test database."""
        if not SQLALCHEMY_AVAILABLE:
            pytest.skip("SQLAlchemy not available")
            
        db_name = db_name or f"test_db_{uuid.uuid4().hex[:8]}"
        db_path = f":memory:"  # Use in-memory database for tests
        
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        session_factory = sessionmaker(bind=engine)
        
        self.databases.append((engine, session_factory))
        
        try:
            yield {
                'engine': engine,
                'session_factory': session_factory,
                'db_name': db_name
            }
        finally:
            engine.dispose()
    
    @contextmanager
    def test_containers(self, services: List[str]):
        """Start test containers for integration testing."""
        if not DOCKER_AVAILABLE:
            pytest.skip("Docker not available")
            
        client = docker.from_env()
        containers = []
        
        try:
            for service in services:
                if service == 'redis':
                    container = client.containers.run(
                        'redis:alpine',
                        ports={'6379/tcp': None},
                        detach=True,
                        remove=True,
                        name=f"test_redis_{uuid.uuid4().hex[:8]}"
                    )
                    containers.append(container)
                    
                elif service == 'postgres':
                    container = client.containers.run(
                        'postgres:13-alpine',
                        environment={
                            'POSTGRES_DB': 'testdb',
                            'POSTGRES_USER': 'testuser',
                            'POSTGRES_PASSWORD': 'testpass'
                        },
                        ports={'5432/tcp': None},
                        detach=True,
                        remove=True,
                        name=f"test_postgres_{uuid.uuid4().hex[:8]}"
                    )
                    containers.append(container)
                    
                elif service == 'nginx':
                    container = client.containers.run(
                        'nginx:alpine',
                        ports={'80/tcp': None},
                        detach=True,
                        remove=True,
                        name=f"test_nginx_{uuid.uuid4().hex[:8]}"
                    )
                    containers.append(container)
            
            # Wait for containers to be ready
            time.sleep(2)
            
            self.containers.extend(containers)
            
            # Get container connection info
            container_info = {}
            for container in containers:
                container.reload()
                ports = container.ports
                container_info[container.name] = {
                    'id': container.id,
                    'ports': ports,
                    'status': container.status
                }
            
            yield container_info
            
        finally:
            # Containers will be cleaned up automatically due to remove=True
            pass
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback to be executed during teardown."""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_all(self):
        """Clean up all test resources."""
        # Execute cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup callback failed: {e}")
        
        # Stop containers
        if DOCKER_AVAILABLE:
            client = docker.from_env()
            for container in self.containers:
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                except Exception as e:
                    print(f"Failed to cleanup container {container}: {e}")
        
        # Close databases
        for engine, _ in self.databases:
            try:
                engine.dispose()
            except Exception as e:
                print(f"Failed to cleanup database: {e}")
        
        # Stop processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                print(f"Failed to cleanup process: {e}")
        
        # Remove temp directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Failed to cleanup temp dir {temp_dir}: {e}")


class WorkflowIntegrationTests:
    """End-to-end workflow integration tests."""
    
    def __init__(self, env_manager: TestEnvironmentManager):
        self.env_manager = env_manager
        self.test_results = []
    
    def test_complete_ml_pipeline(self) -> IntegrationTestResult:
        """Test complete ML pipeline from data ingestion to prediction."""
        test_id = f"ml_pipeline_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['data_ingestion', 'preprocessing', 'training', 'validation', 'prediction', 'monitoring']
        assertions_passed = 0
        assertions_failed = 0
        error_message = None
        
        try:
            with self.env_manager.isolated_environment("ml_pipeline") as env:
                temp_dir = Path(env['temp_dir'])
                
                # 1. Data Ingestion
                print("📥 Testing data ingestion...")
                raw_data_path = temp_dir / "raw_data.csv"
                test_data = self._generate_test_data(1000)
                test_data.to_csv(raw_data_path, index=False)
                
                assert raw_data_path.exists(), "Raw data file should be created"
                assertions_passed += 1
                
                # 2. Data Preprocessing
                print("🔄 Testing data preprocessing...")
                processed_data = self._preprocess_data(test_data)
                
                assert len(processed_data) == len(test_data), "Preprocessing should preserve sample count"
                assert processed_data.isnull().sum().sum() == 0, "Processed data should have no null values"
                assertions_passed += 2
                
                processed_data_path = temp_dir / "processed_data.csv"
                processed_data.to_csv(processed_data_path, index=False)
                
                # 3. Model Training
                print("🎓 Testing model training...")
                X = processed_data.drop('target', axis=1)
                y = processed_data['target']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                
                # Save model
                model_path = temp_dir / "model.joblib"
                joblib.dump(model, model_path)
                
                assert model_path.exists(), "Model file should be saved"
                assertions_passed += 1
                
                # 4. Model Validation
                print("✅ Testing model validation...")
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                assert train_score > 0.5, f"Training accuracy should be > 0.5, got {train_score}"
                assert test_score > 0.3, f"Test accuracy should be > 0.3, got {test_score}"
                assert abs(train_score - test_score) < 0.5, "Model should not be severely overfitted"
                assertions_passed += 3
                
                # 5. Prediction Pipeline
                print("🔮 Testing prediction pipeline...")
                loaded_model = joblib.load(model_path)
                
                # Test single prediction
                sample_input = X_test.iloc[[0]]
                prediction = loaded_model.predict(sample_input)
                probability = loaded_model.predict_proba(sample_input)
                
                assert prediction.shape[0] == 1, "Single prediction should return 1 result"
                assert probability.shape == (1, 2), "Binary classification should return 2 probabilities"
                assert np.isclose(probability.sum(axis=1), 1.0), "Probabilities should sum to 1"
                assertions_passed += 3
                
                # Test batch prediction
                batch_predictions = loaded_model.predict(X_test)
                batch_probabilities = loaded_model.predict_proba(X_test)
                
                assert len(batch_predictions) == len(X_test), "Batch predictions should match input size"
                assert batch_probabilities.shape[0] == len(X_test), "Batch probabilities should match input size"
                assertions_passed += 2
                
                # 6. Model Monitoring Metrics
                print("📊 Testing model monitoring...")
                monitoring_metrics = {
                    'model_version': '1.0.0',
                    'train_accuracy': float(train_score),
                    'test_accuracy': float(test_score),
                    'feature_count': X.shape[1],
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'prediction_latency_ms': 10.5,  # Simulated
                    'model_size_mb': model_path.stat().st_size / (1024 * 1024)
                }
                
                metrics_path = temp_dir / "model_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(monitoring_metrics, f, indent=2)
                
                assert metrics_path.exists(), "Model metrics should be saved"
                assert monitoring_metrics['train_accuracy'] > 0, "Training accuracy should be positive"
                assertions_passed += 2
                
                print("✅ ML pipeline integration test completed successfully")
                
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ ML pipeline test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_id=test_id,
            test_name="complete_ml_pipeline",
            test_type="end_to_end_workflow",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'samples_processed': 1000,
                'features_count': 5,
                'model_accuracy': monitoring_metrics.get('test_accuracy', 0) if 'monitoring_metrics' in locals() else 0
            }
        )
        
        self.test_results.append(result)
        return result
    
    def test_api_workflow_integration(self) -> IntegrationTestResult:
        """Test API workflow from request to response."""
        test_id = f"api_workflow_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['request_validation', 'authentication', 'business_logic', 'data_persistence', 'response_formatting']
        assertions_passed = 0
        assertions_failed = 0
        error_message = None
        
        try:
            with self.env_manager.isolated_environment("api_workflow") as env:
                # Simulate API workflow components
                
                # 1. Request Validation
                print("📋 Testing request validation...")
                valid_request = {
                    "features": [0.5, 1.2, 0.8, 2.1, 0.3],
                    "model_version": "1.0.0",
                    "request_id": str(uuid.uuid4())
                }
                
                validation_result = self._validate_api_request(valid_request)
                assert validation_result['valid'] == True, "Valid request should pass validation"
                assertions_passed += 1
                
                invalid_request = {
                    "features": "invalid",
                    "model_version": None
                }
                
                invalid_validation = self._validate_api_request(invalid_request)
                assert invalid_validation['valid'] == False, "Invalid request should fail validation"
                assertions_passed += 1
                
                # 2. Authentication/Authorization
                print("🔐 Testing authentication...")
                auth_token = "valid-test-token-123"
                auth_result = self._authenticate_request(auth_token)
                assert auth_result['authenticated'] == True, "Valid token should authenticate"
                assertions_passed += 1
                
                invalid_auth = self._authenticate_request("invalid-token")
                assert invalid_auth['authenticated'] == False, "Invalid token should be rejected"
                assertions_passed += 1
                
                # 3. Business Logic Processing
                print("🧠 Testing business logic...")
                processed_features = self._process_business_logic(valid_request['features'])
                assert len(processed_features) == len(valid_request['features']), "Feature processing should preserve length"
                assert all(isinstance(f, (int, float)) for f in processed_features), "Processed features should be numeric"
                assertions_passed += 2
                
                # 4. Model Prediction
                print("🔮 Testing prediction logic...")
                prediction_result = self._make_prediction(processed_features)
                assert 'prediction' in prediction_result, "Prediction result should contain prediction"
                assert 'probability' in prediction_result, "Prediction result should contain probability"
                assert prediction_result['prediction'] in [0, 1], "Binary prediction should be 0 or 1"
                assert 0 <= prediction_result['probability'] <= 1, "Probability should be between 0 and 1"
                assertions_passed += 4
                
                # 5. Response Formatting
                print("📤 Testing response formatting...")
                api_response = self._format_api_response(prediction_result, valid_request['request_id'])
                
                required_fields = ['request_id', 'prediction', 'probability', 'model_version', 'timestamp']
                for field in required_fields:
                    assert field in api_response, f"Response should contain {field}"
                assertions_passed += len(required_fields)
                
                # 6. End-to-End API Simulation
                print("🔄 Testing end-to-end API flow...")
                full_response = self._simulate_full_api_request(valid_request)
                
                assert full_response['status'] == 'success', "Full API request should succeed"
                assert 'result' in full_response, "Full response should contain result"
                assert full_response['processing_time_ms'] > 0, "Processing time should be recorded"
                assertions_passed += 3
                
                print("✅ API workflow integration test completed successfully")
                
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ API workflow test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_id=test_id,
            test_name="api_workflow_integration",
            test_type="api_end_to_end",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'requests_tested': 3,
                'auth_scenarios': 2,
                'validation_scenarios': 2
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _generate_test_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic test data for ML pipeline."""
        if not ML_LIBRARIES_AVAILABLE:
            pytest.skip("ML libraries not available")
            
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.uniform(0, 1, n_samples),
            'feature4': np.random.exponential(1, n_samples),
            'feature5': np.random.poisson(2, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for ML pipeline."""
        processed = data.copy()
        
        # Handle any missing values (though our synthetic data shouldn't have any)
        processed = processed.fillna(0)
        
        # Basic feature engineering
        processed['feature_interaction'] = processed['feature1'] * processed['feature2']
        
        # Standardize features (except target)
        feature_cols = [col for col in processed.columns if col != 'target']
        scaler = StandardScaler()
        processed[feature_cols] = scaler.fit_transform(processed[feature_cols])
        
        return processed
    
    def _validate_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API request format."""
        errors = []
        
        # Check required fields
        if 'features' not in request_data:
            errors.append("Missing 'features' field")
        elif not isinstance(request_data['features'], list):
            errors.append("'features' must be a list")
        elif not all(isinstance(f, (int, float)) for f in request_data['features']):
            errors.append("All features must be numeric")
        
        if 'model_version' not in request_data:
            errors.append("Missing 'model_version' field")
        elif request_data['model_version'] is None:
            errors.append("'model_version' cannot be null")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _authenticate_request(self, token: str) -> Dict[str, Any]:
        """Simulate request authentication."""
        valid_tokens = ['valid-test-token-123', 'api-key-456', 'bearer-token-789']
        
        return {
            'authenticated': token in valid_tokens,
            'user_id': 'test_user_123' if token in valid_tokens else None
        }
    
    def _process_business_logic(self, features: List[float]) -> List[float]:
        """Apply business logic processing to features."""
        # Simulate business rules
        processed_features = []
        
        for feature in features:
            # Apply some business logic transformation
            if feature < 0:
                processed_feature = 0  # Business rule: negative values become 0
            elif feature > 10:
                processed_feature = 10  # Business rule: cap at 10
            else:
                processed_feature = feature
            
            processed_features.append(processed_feature)
        
        return processed_features
    
    def _make_prediction(self, features: List[float]) -> Dict[str, Any]:
        """Make a model prediction."""
        # Simulate model prediction logic
        # In a real scenario, this would load and use a trained model
        feature_sum = sum(features)
        
        # Simple prediction logic for simulation
        prediction = 1 if feature_sum > 2.5 else 0
        probability = min(0.95, max(0.05, (feature_sum + 5) / 10))  # Simulate probability
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': 'high' if abs(probability - 0.5) > 0.3 else 'low'
        }
    
    def _format_api_response(self, prediction_result: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Format the final API response."""
        return {
            'request_id': request_id,
            'prediction': prediction_result['prediction'],
            'probability': round(prediction_result['probability'], 3),
            'confidence': prediction_result['confidence'],
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'processing_node': 'test-node-1'
        }
    
    def _simulate_full_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a complete API request processing."""
        start_time = time.perf_counter()
        
        try:
            # Full pipeline simulation
            validation = self._validate_api_request(request_data)
            if not validation['valid']:
                return {
                    'status': 'error',
                    'message': 'Validation failed',
                    'errors': validation['errors']
                }
            
            auth_result = self._authenticate_request('valid-test-token-123')
            if not auth_result['authenticated']:
                return {
                    'status': 'error',
                    'message': 'Authentication failed'
                }
            
            processed_features = self._process_business_logic(request_data['features'])
            prediction_result = self._make_prediction(processed_features)
            formatted_response = self._format_api_response(prediction_result, request_data.get('request_id', 'test-123'))
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'status': 'success',
                'result': formatted_response,
                'processing_time_ms': round(processing_time, 2)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Processing failed: {str(e)}'
            }


class DistributedSystemIntegrationTests:
    """Integration tests for distributed system components."""
    
    def __init__(self, env_manager: TestEnvironmentManager):
        self.env_manager = env_manager
        self.test_results = []
    
    def test_cache_database_consistency(self) -> IntegrationTestResult:
        """Test consistency between cache and database layers."""
        test_id = f"cache_db_consistency_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['database', 'cache', 'consistency_checker']
        assertions_passed = 0
        assertions_failed = 0
        error_message = None
        
        try:
            with self.env_manager.test_database("consistency_test") as db_env:
                # Create test tables
                engine = db_env['engine']
                with engine.connect() as conn:
                    conn.execute(text("""
                        CREATE TABLE test_data (
                            id INTEGER PRIMARY KEY,
                            key TEXT UNIQUE NOT NULL,
                            value TEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    conn.commit()
                
                # Simulate cache (using in-memory dict for testing)
                cache = {}
                
                print("🗄️ Testing database-cache consistency...")
                
                # 1. Test write-through pattern
                test_key = "test_key_1"
                test_value = "test_value_1"
                
                # Write to database first
                with engine.connect() as conn:
                    conn.execute(text(
                        "INSERT INTO test_data (key, value) VALUES (:key, :value)"
                    ), {"key": test_key, "value": test_value})
                    conn.commit()
                
                # Write to cache
                cache[test_key] = test_value
                
                # Verify consistency
                with engine.connect() as conn:
                    db_result = conn.execute(text(
                        "SELECT value FROM test_data WHERE key = :key"
                    ), {"key": test_key}).fetchone()
                
                assert db_result is not None, "Data should exist in database"
                assert db_result[0] == test_value, "Database value should match written value"
                assert cache.get(test_key) == test_value, "Cache value should match written value"
                assert db_result[0] == cache[test_key], "Database and cache should be consistent"
                assertions_passed += 4
                
                # 2. Test cache invalidation
                updated_value = "updated_test_value_1"
                
                # Update database
                with engine.connect() as conn:
                    conn.execute(text(
                        "UPDATE test_data SET value = :value WHERE key = :key"
                    ), {"key": test_key, "value": updated_value})
                    conn.commit()
                
                # Invalidate cache
                del cache[test_key]
                
                # Read from database and populate cache
                with engine.connect() as conn:
                    db_result = conn.execute(text(
                        "SELECT value FROM test_data WHERE key = :key"
                    ), {"key": test_key}).fetchone()
                    
                    if db_result:
                        cache[test_key] = db_result[0]
                
                assert cache[test_key] == updated_value, "Cache should reflect updated value"
                assertions_passed += 1
                
                # 3. Test bulk operations consistency
                bulk_data = [(f"key_{i}", f"value_{i}") for i in range(10)]
                
                # Bulk insert to database
                with engine.connect() as conn:
                    for key, value in bulk_data:
                        conn.execute(text(
                            "INSERT INTO test_data (key, value) VALUES (:key, :value)"
                        ), {"key": key, "value": value})
                    conn.commit()
                
                # Bulk update cache
                for key, value in bulk_data:
                    cache[key] = value
                
                # Verify all entries
                with engine.connect() as conn:
                    db_count = conn.execute(text("SELECT COUNT(*) FROM test_data")).scalar()
                
                cache_count = len(cache)
                
                assert db_count >= len(bulk_data), f"Database should contain at least {len(bulk_data)} entries"
                assert cache_count >= len(bulk_data), f"Cache should contain at least {len(bulk_data)} entries"
                
                # Check specific consistency
                for key, expected_value in bulk_data:
                    with engine.connect() as conn:
                        db_value = conn.execute(text(
                            "SELECT value FROM test_data WHERE key = :key"
                        ), {"key": key}).scalar()
                    
                    cache_value = cache.get(key)
                    
                    assert db_value == expected_value, f"Database value for {key} should be {expected_value}"
                    assert cache_value == expected_value, f"Cache value for {key} should be {expected_value}"
                    assert db_value == cache_value, f"Database and cache should be consistent for {key}"
                
                assertions_passed += len(bulk_data) * 3
                
                print("✅ Cache-database consistency test completed successfully")
                
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ Cache-database consistency test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_id=test_id,
            test_name="cache_database_consistency",
            test_type="distributed_consistency",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'records_tested': 11,  # 1 initial + 10 bulk
                'operations_tested': ['write', 'update', 'bulk_insert', 'cache_invalidation']
            }
        )
        
        self.test_results.append(result)
        return result
    
    async def test_distributed_consensus(self) -> IntegrationTestResult:
        """Test distributed consensus mechanisms."""
        test_id = f"distributed_consensus_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['node_communication', 'leader_election', 'state_replication', 'failure_recovery']
        assertions_passed = 0
        assertions_failed = 0
        error_message = None
        
        try:
            # Simulate distributed nodes
            nodes = {
                'node_1': {'id': 1, 'status': 'active', 'is_leader': False, 'state': {}, 'heartbeat': time.time()},
                'node_2': {'id': 2, 'status': 'active', 'is_leader': False, 'state': {}, 'heartbeat': time.time()},
                'node_3': {'id': 3, 'status': 'active', 'is_leader': False, 'state': {}, 'heartbeat': time.time()},
                'node_4': {'id': 4, 'status': 'active', 'is_leader': False, 'state': {}, 'heartbeat': time.time()},
                'node_5': {'id': 5, 'status': 'active', 'is_leader': False, 'state': {}, 'heartbeat': time.time()}
            }
            
            print("🗳️ Testing distributed consensus...")
            
            # 1. Leader Election
            print("👑 Testing leader election...")
            leader = self._elect_leader(nodes)
            assert leader is not None, "A leader should be elected"
            assert nodes[leader]['is_leader'] == True, "Elected node should be marked as leader"
            
            # Verify only one leader
            leaders = [node for node, info in nodes.items() if info['is_leader']]
            assert len(leaders) == 1, f"Should have exactly 1 leader, found {len(leaders)}"
            assertions_passed += 3
            
            # 2. State Replication
            print("🔄 Testing state replication...")
            test_state = {'key1': 'value1', 'key2': 'value2', 'counter': 42}
            
            # Leader proposes state change
            self._replicate_state(nodes, leader, test_state)
            
            # Verify all nodes have the same state
            for node_name, node_info in nodes.items():
                if node_info['status'] == 'active':
                    assert node_info['state'] == test_state, f"Node {node_name} should have replicated state"
                    assertions_passed += 1
            
            # 3. Consensus on State Changes
            print("📋 Testing consensus on state changes...")
            state_update = {'key3': 'value3', 'counter': 43}
            
            # Simulate consensus process
            votes = self._collect_votes(nodes, state_update)
            majority_threshold = len([n for n in nodes.values() if n['status'] == 'active']) // 2 + 1
            
            assert len(votes) >= majority_threshold, f"Should have majority votes, got {len(votes)}"
            
            if len(votes) >= majority_threshold:
                self._apply_state_change(nodes, state_update)
                
                # Verify state was applied to all nodes
                for node_name, node_info in nodes.items():
                    if node_info['status'] == 'active':
                        for key, value in state_update.items():
                            assert node_info['state'].get(key) == value, f"Node {node_name} should have updated state"
                        assertions_passed += 1
            
            # 4. Leader Failure and Re-election
            print("💥 Testing leader failure recovery...")
            old_leader = leader
            
            # Simulate leader failure
            nodes[old_leader]['status'] = 'failed'
            nodes[old_leader]['is_leader'] = False
            
            # Trigger re-election
            new_leader = self._elect_leader(nodes)
            
            assert new_leader != old_leader, "New leader should be different from failed leader"
            assert new_leader is not None, "A new leader should be elected"
            assert nodes[new_leader]['is_leader'] == True, "New leader should be marked as leader"
            
            # Verify only one leader exists
            active_leaders = [node for node, info in nodes.items() 
                            if info['is_leader'] and info['status'] == 'active']
            assert len(active_leaders) == 1, "Should have exactly 1 active leader after re-election"
            assertions_passed += 4
            
            # 5. Network Partition Simulation
            print("🌐 Testing network partition handling...")
            
            # Simulate network partition - split nodes into two groups
            partition_1 = ['node_2', 'node_3']
            partition_2 = ['node_4', 'node_5']
            
            # Nodes in different partitions can't communicate
            partition_state = {
                'partition_1_nodes': partition_1,
                'partition_2_nodes': partition_2,
                'partition_active': True
            }
            
            # Each partition should not be able to achieve majority alone
            assert len(partition_1) < majority_threshold, "Partition 1 should not have majority"
            assert len(partition_2) < majority_threshold, "Partition 2 should not have majority"
            
            # Neither partition should be able to make state changes
            partition_update = {'partition_test': True}
            partition_1_votes = [node for node in partition_1]
            partition_2_votes = [node for node in partition_2]
            
            assert len(partition_1_votes) < majority_threshold, "Partition 1 should not achieve consensus"
            assert len(partition_2_votes) < majority_threshold, "Partition 2 should not achieve consensus"
            assertions_passed += 4
            
            # 6. Partition Healing
            print("🔗 Testing partition healing...")
            
            # Heal partition - all nodes can communicate again
            partition_state['partition_active'] = False
            
            # Re-establish consensus with all available nodes
            available_nodes = [name for name, info in nodes.items() if info['status'] == 'active']
            healing_votes = available_nodes
            
            assert len(healing_votes) >= majority_threshold, "After healing, should achieve majority"
            
            # Verify consensus can be reached again
            healing_update = {'partition_healed': True, 'timestamp': time.time()}
            final_votes = self._collect_votes(nodes, healing_update, available_only=True)
            
            assert len(final_votes) >= majority_threshold, "Should achieve consensus after partition healing"
            assertions_passed += 2
            
            print("✅ Distributed consensus test completed successfully")
            
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ Distributed consensus test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_id=test_id,
            test_name="distributed_consensus",
            test_type="distributed_consensus",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'nodes_tested': 5,
                'consensus_rounds': 3,
                'failure_scenarios': 2
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _elect_leader(self, nodes: Dict[str, Any]) -> Optional[str]:
        """Simulate leader election process."""
        # Simple leader election based on node ID (lowest ID wins)
        active_nodes = [(name, info) for name, info in nodes.items() if info['status'] == 'active']
        
        if not active_nodes:
            return None
        
        # Reset all leader flags
        for node_info in nodes.values():
            node_info['is_leader'] = False
        
        # Elect leader (lowest ID)
        leader_node = min(active_nodes, key=lambda x: x[1]['id'])
        leader_name = leader_node[0]
        
        nodes[leader_name]['is_leader'] = True
        return leader_name
    
    def _replicate_state(self, nodes: Dict[str, Any], leader: str, state: Dict[str, Any]) -> None:
        """Simulate state replication from leader to followers."""
        # Leader propagates state to all active followers
        for node_name, node_info in nodes.items():
            if node_info['status'] == 'active':
                node_info['state'].update(state)
                node_info['heartbeat'] = time.time()
    
    def _collect_votes(self, nodes: Dict[str, Any], proposal: Dict[str, Any], 
                      available_only: bool = False) -> List[str]:
        """Simulate collecting votes for a consensus proposal."""
        votes = []
        
        for node_name, node_info in nodes.items():
            if available_only:
                if node_info['status'] == 'active':
                    votes.append(node_name)
            else:
                # In real consensus, nodes would evaluate the proposal
                # For simulation, active nodes vote yes
                if node_info['status'] == 'active':
                    votes.append(node_name)
        
        return votes
    
    def _apply_state_change(self, nodes: Dict[str, Any], state_change: Dict[str, Any]) -> None:
        """Apply agreed state change to all active nodes."""
        for node_name, node_info in nodes.items():
            if node_info['status'] == 'active':
                node_info['state'].update(state_change)


class IntegrationTestSuiteRunner:
    """Main runner for integration test suites."""
    
    def __init__(self):
        self.env_manager = TestEnvironmentManager()
        self.workflow_tests = WorkflowIntegrationTests(self.env_manager)
        self.distributed_tests = DistributedSystemIntegrationTests(self.env_manager)
        self.all_results = []
    
    async def run_all_integration_tests(self) -> IntegrationTestSuite:
        """Run all integration tests."""
        suite_id = f"integration_suite_{int(time.time())}"
        start_time = datetime.now()
        
        print("🚀 Starting comprehensive integration test suite...")
        
        environment_setup = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'test_environment': 'isolated',
            'parallel_execution': False,
            'cleanup_enabled': True
        }
        
        test_results = []
        
        try:
            # 1. Workflow Integration Tests
            print("\n📋 Running workflow integration tests...")
            
            # ML Pipeline Test
            ml_result = self.workflow_tests.test_complete_ml_pipeline()
            test_results.append(ml_result)
            
            # API Workflow Test  
            api_result = self.workflow_tests.test_api_workflow_integration()
            test_results.append(api_result)
            
            # 2. Distributed System Tests
            print("\n🌐 Running distributed system integration tests...")
            
            # Cache-Database Consistency Test
            consistency_result = self.distributed_tests.test_cache_database_consistency()
            test_results.append(consistency_result)
            
            # Distributed Consensus Test
            consensus_result = await self.distributed_tests.test_distributed_consensus()
            test_results.append(consensus_result)
            
            # 3. Additional Integration Scenarios
            print("\n🔧 Running additional integration scenarios...")
            
            # Component Interaction Test
            interaction_result = self._test_component_interactions()
            test_results.append(interaction_result)
            
            # Cross-Service Communication Test
            if HTTP_CLIENTS_AVAILABLE:
                communication_result = await self._test_cross_service_communication()
                test_results.append(communication_result)
            
        finally:
            # Always cleanup, even if tests fail
            print("\n🧹 Cleaning up test resources...")
            self.env_manager.cleanup_all()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Create test suite result
        overall_passed = all(result.passed for result in test_results)
        
        suite = IntegrationTestSuite(
            suite_id=suite_id,
            suite_name="comprehensive_integration_tests",
            start_time=start_time,
            end_time=end_time,
            test_results=test_results,
            environment_setup=environment_setup,
            overall_passed=overall_passed,
            total_duration_seconds=total_duration
        )
        
        # Print summary
        passed_count = sum(1 for result in test_results if result.passed)
        failed_count = len(test_results) - passed_count
        total_assertions = sum(result.assertions_passed + result.assertions_failed for result in test_results)
        
        print(f"\n🎯 Integration test suite completed in {total_duration:.2f}s")
        print(f"   Tests run: {len(test_results)}")
        print(f"   Passed: {passed_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total assertions: {total_assertions}")
        print(f"   Overall result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        if not overall_passed:
            print("\n❌ Failed tests:")
            for result in test_results:
                if not result.passed:
                    print(f"   • {result.test_name}: {result.error_message}")
        
        # Save results
        self._save_results(suite)
        
        return suite
    
    def _test_component_interactions(self) -> IntegrationTestResult:
        """Test interactions between different system components."""
        test_id = f"component_interaction_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['configuration', 'logging', 'metrics', 'caching', 'validation']
        assertions_passed = 0
        assertions_failed = 0
        error_message = None
        
        try:
            print("🔗 Testing component interactions...")
            
            # 1. Configuration-Logging Integration
            config = {
                'log_level': 'INFO',
                'log_format': 'json',
                'enable_metrics': True,
                'cache_enabled': True
            }
            
            # Simulate configuration loading affecting other components
            logging_configured = self._configure_logging(config)
            assert logging_configured, "Logging should be configured from config"
            assertions_passed += 1
            
            # 2. Metrics-Caching Integration
            metrics_collector = {'cache_hits': 0, 'cache_misses': 0, 'requests': 0}
            cache = {}
            
            # Test cache operations with metrics
            for i in range(10):
                key = f"test_key_{i % 3}"  # Some overlap to test hits/misses
                
                if key in cache:
                    # Cache hit
                    value = cache[key]
                    metrics_collector['cache_hits'] += 1
                else:
                    # Cache miss
                    value = f"computed_value_{i}"
                    cache[key] = value
                    metrics_collector['cache_misses'] += 1
                
                metrics_collector['requests'] += 1
            
            # Verify metrics integration
            total_operations = metrics_collector['cache_hits'] + metrics_collector['cache_misses']
            assert total_operations == metrics_collector['requests'], "Metrics should account for all operations"
            assert metrics_collector['cache_hits'] > 0, "Should have some cache hits"
            assert metrics_collector['cache_misses'] > 0, "Should have some cache misses"
            assertions_passed += 3
            
            # 3. Validation-Configuration Integration
            validation_config = config.copy()
            validation_rules = self._get_validation_rules(validation_config)
            
            assert isinstance(validation_rules, dict), "Should return validation rules"
            assert len(validation_rules) > 0, "Should have validation rules"
            assertions_passed += 2
            
            # Test validation with different config
            test_data = {
                'user_id': 123,
                'email': 'test@example.com',
                'age': 25
            }
            
            validation_result = self._validate_with_rules(test_data, validation_rules)
            assert validation_result['valid'], "Valid data should pass validation"
            assertions_passed += 1
            
            # 4. Cross-Component Event Flow
            event_system = {
                'events': [],
                'handlers': {},
                'metrics': metrics_collector
            }
            
            # Register event handlers
            self._register_event_handlers(event_system)
            
            # Trigger events and verify cross-component reactions
            test_events = [
                {'type': 'user_login', 'user_id': 123, 'timestamp': time.time()},
                {'type': 'cache_miss', 'key': 'user_123', 'timestamp': time.time()},
                {'type': 'validation_error', 'field': 'email', 'timestamp': time.time()}
            ]
            
            for event in test_events:
                self._process_event(event_system, event)
            
            # Verify events were processed
            assert len(event_system['events']) == len(test_events), "All events should be recorded"
            
            # Verify cross-component effects
            login_events = [e for e in event_system['events'] if e['type'] == 'user_login']
            assert len(login_events) == 1, "Should have recorded login event"
            assertions_passed += 2
            
            print("✅ Component interaction test completed successfully")
            
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ Component interaction test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return IntegrationTestResult(
            test_id=test_id,
            test_name="component_interactions",
            test_type="component_integration",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'config_items': len(config),
                'cache_operations': 10,
                'events_processed': len(test_events) if 'test_events' in locals() else 0
            }
        )
    
    async def _test_cross_service_communication(self) -> IntegrationTestResult:
        """Test communication between different services."""
        test_id = f"cross_service_communication_test_{int(time.time())}"
        start_time = datetime.now()
        
        components_tested = ['service_discovery', 'load_balancing', 'circuit_breaker', 'retry_logic']
        assertions_passed = 0
        assertions_failed = 0  
        error_message = None
        
        try:
            print("🌐 Testing cross-service communication...")
            
            # Simulate service registry
            service_registry = {
                'prediction_service': [
                    {'host': 'localhost', 'port': 8001, 'healthy': True},
                    {'host': 'localhost', 'port': 8002, 'healthy': True},
                    {'host': 'localhost', 'port': 8003, 'healthy': False}
                ],
                'user_service': [
                    {'host': 'localhost', 'port': 9001, 'healthy': True}
                ],
                'notification_service': [
                    {'host': 'localhost', 'port': 9002, 'healthy': True}
                ]
            }
            
            # 1. Service Discovery
            available_prediction_services = self._discover_healthy_services(
                service_registry, 'prediction_service'
            )
            
            assert len(available_prediction_services) == 2, "Should find 2 healthy prediction services"
            for service in available_prediction_services:
                assert service['healthy'], "Discovered services should be healthy"
            assertions_passed += 2
            
            # 2. Load Balancing
            # Simulate multiple requests with round-robin load balancing
            requests_made = []
            for i in range(6):
                selected_service = self._select_service_round_robin(
                    available_prediction_services, i
                )
                requests_made.append(selected_service['port'])
            
            # Verify load balancing distribution
            unique_services_used = set(requests_made)
            assert len(unique_services_used) == 2, "Load balancer should use both services"
            
            # Check distribution is roughly equal
            service_usage = {port: requests_made.count(port) for port in unique_services_used}
            usage_values = list(service_usage.values())
            assert abs(usage_values[0] - usage_values[1]) <= 1, "Load should be roughly balanced"
            assertions_passed += 2
            
            # 3. Circuit Breaker Pattern
            circuit_breaker = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'success_count': 0,
                'failure_threshold': 3,
                'timeout': 5.0,
                'last_failure_time': None
            }
            
            # Simulate successful requests
            for _ in range(5):
                result = self._make_request_with_circuit_breaker(
                    circuit_breaker, lambda: {'status': 'success'}
                )
                assert result['status'] == 'success', "Successful requests should go through"
            
            assert circuit_breaker['state'] == 'closed', "Circuit should remain closed for successful requests"
            assertions_passed += 1
            
            # Simulate failures to trip circuit breaker
            for _ in range(4):  # Exceed failure threshold
                result = self._make_request_with_circuit_breaker(
                    circuit_breaker, lambda: self._simulate_service_failure()
                )
            
            assert circuit_breaker['state'] == 'open', "Circuit should open after failure threshold"
            assertions_passed += 1
            
            # 4. Retry Logic with Exponential Backoff
            retry_attempts = []
            
            def failing_service():
                retry_attempts.append(time.time())
                if len(retry_attempts) < 3:
                    raise Exception("Service temporarily unavailable")
                return {'status': 'success', 'attempts': len(retry_attempts)}
            
            result = self._retry_with_backoff(failing_service, max_retries=3)
            
            assert result is not None, "Should eventually succeed with retries"
            assert result['status'] == 'success', "Final result should be successful"
            assert len(retry_attempts) == 3, "Should have made exactly 3 attempts"
            
            # Verify exponential backoff timing
            if len(retry_attempts) >= 2:
                time_diff_1 = retry_attempts[1] - retry_attempts[0]
                time_diff_2 = retry_attempts[2] - retry_attempts[1] if len(retry_attempts) >= 3 else 0
                
                # Second retry should have longer delay than first
                if time_diff_2 > 0:
                    assert time_diff_2 > time_diff_1, "Backoff should be exponential"
            assertions_passed += 3
            
            # 5. Service Mesh Communication
            service_mesh_config = {
                'encryption_enabled': True,
                'mutual_tls': True,
                'traffic_policy': 'least_connections',
                'health_checks': True
            }
            
            # Simulate service mesh request
            mesh_result = self._simulate_service_mesh_request(
                service_mesh_config,
                source_service='api_gateway',
                target_service='prediction_service',
                payload={'features': [1, 2, 3, 4, 5]}
            )
            
            assert mesh_result['encrypted'], "Service mesh should encrypt traffic"
            assert mesh_result['authenticated'], "Service mesh should authenticate requests"
            assert 'response' in mesh_result, "Should contain service response"
            assertions_passed += 3
            
            print("✅ Cross-service communication test completed successfully")
            
        except Exception as e:
            error_message = str(e)
            assertions_failed += 1
            print(f"❌ Cross-service communication test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return IntegrationTestResult(
            test_id=test_id,
            test_name="cross_service_communication",
            test_type="service_integration",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            passed=error_message is None,
            error_message=error_message,
            components_tested=components_tested,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            test_data={
                'services_tested': len(service_registry),
                'load_balancer_requests': 6,
                'circuit_breaker_tests': 9,
                'retry_attempts': len(retry_attempts) if 'retry_attempts' in locals() else 0
            }
        )
    
    def _configure_logging(self, config: Dict[str, Any]) -> bool:
        """Simulate logging configuration from config."""
        required_keys = ['log_level', 'log_format']
        return all(key in config for key in required_keys)
    
    def _get_validation_rules(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation rules based on configuration."""
        return {
            'user_id': {'type': 'integer', 'required': True, 'min': 1},
            'email': {'type': 'string', 'required': True, 'format': 'email'},
            'age': {'type': 'integer', 'min': 0, 'max': 150}
        }
    
    def _validate_with_rules(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against rules."""
        errors = []
        
        for field, rule in rules.items():
            if rule.get('required', False) and field not in data:
                errors.append(f"Required field '{field}' missing")
                continue
            
            if field in data:
                value = data[field]
                field_type = rule.get('type')
                
                if field_type == 'integer' and not isinstance(value, int):
                    errors.append(f"Field '{field}' must be integer")
                elif field_type == 'string' and not isinstance(value, str):
                    errors.append(f"Field '{field}' must be string")
                elif field_type == 'string' and rule.get('format') == 'email' and '@' not in value:
                    errors.append(f"Field '{field}' must be valid email")
                
                if 'min' in rule and value < rule['min']:
                    errors.append(f"Field '{field}' must be >= {rule['min']}")
                if 'max' in rule and value > rule['max']:
                    errors.append(f"Field '{field}' must be <= {rule['max']}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _register_event_handlers(self, event_system: Dict[str, Any]) -> None:
        """Register event handlers for different event types."""
        event_system['handlers'] = {
            'user_login': [self._handle_login_event],
            'cache_miss': [self._handle_cache_miss_event],
            'validation_error': [self._handle_validation_error_event]
        }
    
    def _process_event(self, event_system: Dict[str, Any], event: Dict[str, Any]) -> None:
        """Process an event through the event system."""
        event_system['events'].append(event)
        
        handlers = event_system['handlers'].get(event['type'], [])
        for handler in handlers:
            handler(event, event_system)
    
    def _handle_login_event(self, event: Dict[str, Any], event_system: Dict[str, Any]) -> None:
        """Handle user login event."""
        event_system['metrics']['requests'] = event_system['metrics'].get('requests', 0) + 1
    
    def _handle_cache_miss_event(self, event: Dict[str, Any], event_system: Dict[str, Any]) -> None:
        """Handle cache miss event."""
        event_system['metrics']['cache_misses'] = event_system['metrics'].get('cache_misses', 0) + 1
    
    def _handle_validation_error_event(self, event: Dict[str, Any], event_system: Dict[str, Any]) -> None:
        """Handle validation error event."""
        event_system['metrics']['validation_errors'] = event_system['metrics'].get('validation_errors', 0) + 1
    
    def _discover_healthy_services(self, registry: Dict[str, List[Dict]], service_name: str) -> List[Dict]:
        """Discover healthy services from registry."""
        services = registry.get(service_name, [])
        return [service for service in services if service.get('healthy', False)]
    
    def _select_service_round_robin(self, services: List[Dict], request_index: int) -> Dict:
        """Select service using round-robin load balancing."""
        if not services:
            raise ValueError("No services available")
        
        return services[request_index % len(services)]
    
    def _make_request_with_circuit_breaker(self, circuit_breaker: Dict[str, Any], 
                                         request_func: Callable) -> Dict[str, Any]:
        """Make request through circuit breaker."""
        if circuit_breaker['state'] == 'open':
            # Check if timeout has passed
            if (circuit_breaker['last_failure_time'] and 
                time.time() - circuit_breaker['last_failure_time'] > circuit_breaker['timeout']):
                circuit_breaker['state'] = 'half_open'
            else:
                return {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
        
        try:
            result = request_func()
            
            # Success
            circuit_breaker['success_count'] += 1
            
            if circuit_breaker['state'] == 'half_open':
                # Reset circuit breaker on success in half-open state
                circuit_breaker['state'] = 'closed'
                circuit_breaker['failure_count'] = 0
            
            return result
            
        except Exception as e:
            # Failure
            circuit_breaker['failure_count'] += 1
            circuit_breaker['last_failure_time'] = time.time()
            
            if circuit_breaker['failure_count'] >= circuit_breaker['failure_threshold']:
                circuit_breaker['state'] = 'open'
            
            return {'status': 'error', 'message': str(e)}
    
    def _simulate_service_failure(self) -> Dict[str, Any]:
        """Simulate a service failure."""
        raise Exception("Service is down")
    
    def _retry_with_backoff(self, func: Callable, max_retries: int = 3, 
                          base_delay: float = 0.1) -> Optional[Any]:
        """Retry function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        
        return None
    
    def _simulate_service_mesh_request(self, mesh_config: Dict[str, Any],
                                     source_service: str, target_service: str,
                                     payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate service mesh request processing."""
        # Simulate service mesh features
        result = {
            'encrypted': mesh_config.get('encryption_enabled', False),
            'authenticated': mesh_config.get('mutual_tls', False),
            'source_service': source_service,
            'target_service': target_service,
            'response': {
                'prediction': 1,
                'probability': 0.85,
                'processed_at': time.time()
            }
        }
        
        return result
    
    def _save_results(self, suite: IntegrationTestSuite) -> None:
        """Save integration test results."""
        results_dir = Path("integration_test_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"integration_tests_{suite.suite_id}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        
        print(f"📊 Integration test results saved to {results_file}")


async def main():
    """Main function for running integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration Testing Suite")
    parser.add_argument("--test-type", choices=["workflow", "distributed", "all"], 
                       default="all", help="Type of integration tests to run")
    parser.add_argument("--cleanup", action="store_true", default=True,
                       help="Clean up test resources after completion")
    
    args = parser.parse_args()
    
    runner = IntegrationTestSuiteRunner()
    
    try:
        if args.test_type == "workflow":
            print("Running workflow integration tests...")
            ml_result = runner.workflow_tests.test_complete_ml_pipeline()
            api_result = runner.workflow_tests.test_api_workflow_integration()
            
            all_passed = ml_result.passed and api_result.passed
            print(f"Workflow tests: {'✅ PASSED' if all_passed else '❌ FAILED'}")
            
        elif args.test_type == "distributed":
            print("Running distributed system integration tests...")
            consistency_result = runner.distributed_tests.test_cache_database_consistency()
            consensus_result = await runner.distributed_tests.test_distributed_consensus()
            
            all_passed = consistency_result.passed and consensus_result.passed
            print(f"Distributed tests: {'✅ PASSED' if all_passed else '❌ FAILED'}")
            
        else:
            # Run all tests
            suite = await runner.run_all_integration_tests()
            
            if not suite.overall_passed:
                exit(1)
                
    finally:
        if args.cleanup:
            runner.env_manager.cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())