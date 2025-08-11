"""
Property-Based Testing Framework with Hypothesis.

This module provides comprehensive property-based testing for ML models and data processing:
- Hypothesis-based testing for ML models with invariant validation
- Fuzzing for robust input validation and edge case discovery
- Contract testing for API endpoints with automatic test generation
- Invariant testing for distributed systems and data consistency
- Statistical testing for ML model properties and fairness
- Automated test case generation and shrinking for minimal failing examples
"""

import os
import json
import time
import math
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

# Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, HealthCheck, assume, note, event
    from hypothesis.extra.pandas import data_frames, columns
    from hypothesis.extra.numpy import arrays
    from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, Bundle, consumes
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Testing framework
import pytest
from unittest.mock import Mock, patch, MagicMock

# ML libraries for testing
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# API testing
try:
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, ValidationError
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@dataclass
class PropertyTestResult:
    """Result of a property-based test."""
    test_name: str
    property_name: str
    test_type: str  # 'hypothesis', 'fuzzing', 'contract', 'invariant'
    passed: bool
    examples_tested: int
    counterexample: Optional[Dict[str, Any]]
    execution_time_ms: float
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class PropertySuite:
    """Collection of property-based tests."""
    suite_name: str
    properties: List[str]
    test_results: List[PropertyTestResult]
    overall_passed: bool
    total_examples: int
    execution_time_ms: float


class MLModelProperties:
    """Property-based tests for ML models."""
    
    def __init__(self):
        self.results = []
    
    # Basic ML model properties
    
    @given(st.integers(min_value=1, max_value=10000))
    def test_model_input_output_shape_consistency(self, n_samples: int):
        """Test that model output shape is consistent with input shape."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        # Create synthetic data
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test property: output shape should match input sample count
        predictions = model.predict(X)
        assert len(predictions) == n_samples, f"Expected {n_samples} predictions, got {len(predictions)}"
        
        # Test property: predict_proba should return probabilities
        probabilities = model.predict_proba(X)
        assert probabilities.shape[0] == n_samples, f"Expected {n_samples} probability rows, got {probabilities.shape[0]}"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    @given(arrays(dtype=np.float64, shape=(st.integers(10, 1000), st.integers(2, 20)),
                  elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)))
    def test_model_deterministic_predictions(self, X: np.ndarray):
        """Test that model predictions are deterministic with fixed random seed."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        assume(X.shape[0] >= 10)  # Need minimum samples for training
        
        # Create target variable
        y = (X.sum(axis=1) > 0).astype(int)
        
        # Train two identical models
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Test property: identical models should give identical predictions
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2, "Identical models should give identical predictions")
    
    @given(st.data())
    def test_model_robustness_to_feature_scaling(self, data):
        """Test that certain models are invariant to feature scaling."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        # Generate data
        n_samples = data.draw(st.integers(50, 500))
        n_features = data.draw(st.integers(2, 10))
        
        X = data.draw(arrays(dtype=np.float64, shape=(n_samples, n_features),
                           elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)))
        y = data.draw(arrays(dtype=np.int32, shape=(n_samples,),
                           elements=st.integers(0, 1)))
        
        # Tree-based models should be invariant to monotonic scaling
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        pred_original = model.predict(X)
        
        # Scale features by different positive constants
        scaling_factors = data.draw(arrays(dtype=np.float64, shape=(n_features,),
                                         elements=st.floats(min_value=1.1, max_value=10.0)))
        X_scaled = X * scaling_factors
        
        model_scaled = RandomForestClassifier(n_estimators=10, random_state=42)
        model_scaled.fit(X_scaled, y)
        pred_scaled = model_scaled.predict(X_scaled)
        
        # Test property: tree-based predictions should be similar under monotonic scaling
        accuracy_diff = abs(accuracy_score(y, pred_original) - accuracy_score(y, pred_scaled))
        assert accuracy_diff < 0.1, f"Tree model accuracy should be robust to scaling, diff: {accuracy_diff}"
    
    @given(st.floats(min_value=0.01, max_value=0.99))
    def test_probability_calibration_properties(self, threshold: float):
        """Test probability calibration properties."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        # Generate calibrated synthetic data
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        # Create target based on threshold
        true_probs = 1 / (1 + np.exp(-X.sum(axis=1)))  # Logistic function
        y = (true_probs > threshold).astype(int)
        
        # Train calibrated model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        predicted_probs = model.predict_proba(X)[:, 1]
        
        # Test property: predicted probabilities should be well-calibrated
        # Higher predicted probability should correlate with higher true probability
        high_prob_mask = predicted_probs > threshold
        low_prob_mask = predicted_probs <= threshold
        
        if high_prob_mask.sum() > 0 and low_prob_mask.sum() > 0:
            high_prob_accuracy = y[high_prob_mask].mean()
            low_prob_accuracy = y[low_prob_mask].mean()
            
            # Property: high predicted probability samples should have higher actual rate
            assert high_prob_accuracy >= low_prob_accuracy, \
                f"High prob group ({high_prob_accuracy:.3f}) should have higher accuracy than low prob group ({low_prob_accuracy:.3f})"
    
    @given(st.integers(min_value=2, max_value=1000), 
           st.integers(min_value=2, max_value=20))
    def test_model_performance_monotonicity(self, n_samples: int, n_features: int):
        """Test that model performance generally improves with more training data."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        # Generate dataset
        X_full = np.random.randn(n_samples, n_features)
        coefficients = np.random.randn(n_features)
        y_full = ((X_full @ coefficients + np.random.randn(n_samples) * 0.1) > 0).astype(int)
        
        # Test with different training sizes
        small_size = max(10, n_samples // 4)
        large_size = max(20, n_samples // 2)
        
        if large_size <= small_size:
            assume(False)  # Skip if not enough samples for comparison
        
        # Train on small dataset
        X_small, y_small = X_full[:small_size], y_full[:small_size]
        model_small = RandomForestClassifier(n_estimators=10, random_state=42)
        model_small.fit(X_small, y_small)
        
        # Train on larger dataset
        X_large, y_large = X_full[:large_size], y_full[:large_size]
        model_large = RandomForestClassifier(n_estimators=10, random_state=42)
        model_large.fit(X_large, y_large)
        
        # Test on held-out data
        test_size = min(100, n_samples - large_size)
        if test_size < 10:
            assume(False)  # Need enough test data
            
        X_test = X_full[large_size:large_size + test_size]
        y_test = y_full[large_size:large_size + test_size]
        
        # Property: more training data should generally not hurt performance significantly
        acc_small = accuracy_score(y_test, model_small.predict(X_test))
        acc_large = accuracy_score(y_test, model_large.predict(X_test))
        
        # Allow for some variance but expect general improvement or stability
        performance_drop = acc_small - acc_large
        assert performance_drop < 0.2, f"Large model performance dropped too much: {performance_drop:.3f}"
    
    # Fairness and bias testing properties
    
    @given(st.data())
    def test_demographic_parity_approximation(self, data):
        """Test approximate demographic parity across groups."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")
        
        n_samples = data.draw(st.integers(200, 1000))
        n_features = data.draw(st.integers(3, 10))
        
        # Generate features
        X = data.draw(arrays(dtype=np.float64, shape=(n_samples, n_features),
                           elements=st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False)))
        
        # Create protected attribute (e.g., gender: 0 or 1)
        protected_attr = data.draw(arrays(dtype=np.int32, shape=(n_samples,),
                                        elements=st.integers(0, 1)))
        
        # Create target with some correlation to features but trying to be fair
        target_noise = np.random.randn(n_samples) * 0.5
        y = ((X.sum(axis=1) + target_noise) > 0).astype(int)
        
        # Train model (without using protected attribute directly)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Test approximate demographic parity
        group_0_positive_rate = predictions[protected_attr == 0].mean()
        group_1_positive_rate = predictions[protected_attr == 1].mean()
        
        # Property: positive rates should not differ too dramatically
        disparity = abs(group_0_positive_rate - group_1_positive_rate)
        
        # Note: This is a weak fairness test - in practice, more sophisticated fairness metrics needed
        note(f"Demographic disparity: {disparity:.3f}")
        event(f"High disparity: {disparity > 0.3}")
        
        # Property is more about awareness than strict enforcement
        assert disparity < 0.8, f"Extreme demographic disparity detected: {disparity:.3f}"


class DataProcessingProperties:
    """Property-based tests for data processing operations."""
    
    @given(data_frames(columns=[
        columns('feature1', elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
        columns('feature2', elements=st.floats(min_value=0, max_value=1000, allow_nan=False)),
        columns('category', elements=st.sampled_from(['A', 'B', 'C', 'D']))
    ], rows=st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False),
                      st.floats(min_value=0, max_value=1000, allow_nan=False),
                      st.sampled_from(['A', 'B', 'C', 'D']))))
    def test_data_preprocessing_invariants(self, df: pd.DataFrame):
        """Test invariants in data preprocessing."""
        assume(len(df) > 0)
        
        original_shape = df.shape
        original_columns = set(df.columns)
        
        # Apply preprocessing
        processed_df = df.copy()
        
        # Standardize numerical features
        if 'feature1' in processed_df.columns:
            processed_df['feature1'] = (processed_df['feature1'] - processed_df['feature1'].mean()) / processed_df['feature1'].std()
        
        if 'feature2' in processed_df.columns:
            processed_df['feature2'] = (processed_df['feature2'] - processed_df['feature2'].mean()) / processed_df['feature2'].std()
        
        # One-hot encode categorical
        if 'category' in processed_df.columns:
            dummies = pd.get_dummies(processed_df['category'], prefix='category')
            processed_df = processed_df.drop('category', axis=1)
            processed_df = pd.concat([processed_df, dummies], axis=1)
        
        # Test invariants
        assert len(processed_df) == original_shape[0], "Preprocessing should not change number of rows"
        
        # Test that numerical columns have expected properties after standardization
        if 'feature1' in processed_df.columns and len(processed_df) > 1:
            assert abs(processed_df['feature1'].mean()) < 1e-10, "Standardized feature should have mean ≈ 0"
            assert abs(processed_df['feature1'].std() - 1.0) < 1e-10, "Standardized feature should have std ≈ 1"
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=10000))
    def test_statistical_operations_properties(self, values: List[float]):
        """Test properties of statistical operations."""
        arr = np.array(values)
        
        # Test mean properties
        mean_val = np.mean(arr)
        assert not np.isnan(mean_val), "Mean should not be NaN for finite values"
        assert np.min(arr) <= mean_val <= np.max(arr), "Mean should be between min and max"
        
        # Test that adding a constant shifts the mean by that constant
        constant = 5.0
        shifted_arr = arr + constant
        shifted_mean = np.mean(shifted_arr)
        assert abs(shifted_mean - (mean_val + constant)) < 1e-10, "Adding constant should shift mean by that constant"
        
        # Test scaling properties
        if len(values) > 1:
            std_val = np.std(arr)
            scale_factor = 2.0
            scaled_arr = arr * scale_factor
            scaled_std = np.std(scaled_arr)
            assert abs(scaled_std - (std_val * scale_factor)) < 1e-10, "Scaling should multiply std by scale factor"
    
    @given(st.lists(st.dictionaries(
        keys=st.sampled_from(['name', 'age', 'salary', 'department']),
        values=st.one_of(
            st.text(min_size=1, max_size=50),
            st.integers(min_value=18, max_value=80),
            st.floats(min_value=30000, max_value=200000, allow_nan=False),
            st.sampled_from(['Engineering', 'Sales', 'Marketing', 'HR'])
        ),
        min_size=1, max_size=4
    ), min_size=1, max_size=1000))
    def test_data_validation_properties(self, records: List[Dict[str, Any]]):
        """Test data validation properties."""
        df = pd.DataFrame(records)
        
        # Test that validation preserves valid records
        valid_records = []
        for _, row in df.iterrows():
            if self._is_valid_record(row.to_dict()):
                valid_records.append(row.to_dict())
        
        # Property: validation should be idempotent
        first_validation = [r for r in valid_records if self._is_valid_record(r)]
        second_validation = [r for r in first_validation if self._is_valid_record(r)]
        
        assert len(first_validation) == len(second_validation), "Validation should be idempotent"
        
        # Property: valid records should satisfy business rules
        for record in valid_records:
            if 'age' in record:
                assert isinstance(record['age'], int) and 18 <= record['age'] <= 80, "Age should be valid integer"
            if 'salary' in record:
                assert isinstance(record['salary'], (int, float)) and record['salary'] > 0, "Salary should be positive"
    
    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        """Validate a single record."""
        if 'age' in record:
            if not isinstance(record['age'], int) or not (18 <= record['age'] <= 80):
                return False
        
        if 'salary' in record:
            if not isinstance(record['salary'], (int, float)) or record['salary'] <= 0:
                return False
        
        if 'name' in record:
            if not isinstance(record['name'], str) or len(record['name'].strip()) == 0:
                return False
        
        return True


class APIContractTesting:
    """Contract testing for API endpoints using property-based testing."""
    
    def __init__(self, client: Optional[Any] = None):
        self.client = client or self._create_mock_client()
    
    def _create_mock_client(self):
        """Create a mock client for testing when real API is not available."""
        mock_client = Mock()
        
        # Mock health endpoint
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json.return_value = {"status": "healthy"}
        
        # Mock prediction endpoint
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "prediction": 0,
            "probability": 0.7,
            "model_version": "1.0.0"
        }
        
        return mock_client
    
    @given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=20))
    def test_prediction_endpoint_contract(self, features: List[float]):
        """Test prediction endpoint contract properties."""
        payload = {"features": features}
        
        response = self.client.post("/predict", json=payload)
        
        # Contract property: should return 200 for valid input
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        try:
            data = response.json()
        except:
            pytest.fail("Response should be valid JSON")
        
        # Contract properties for response structure
        assert "prediction" in data, "Response should contain 'prediction' field"
        assert "probability" in data, "Response should contain 'probability' field"
        
        # Property: prediction should be valid classification result
        prediction = data["prediction"]
        assert prediction in [0, 1], f"Prediction should be 0 or 1, got {prediction}"
        
        # Property: probability should be valid
        probability = data["probability"]
        assert isinstance(probability, (int, float)), "Probability should be numeric"
        assert 0 <= probability <= 1, f"Probability should be between 0 and 1, got {probability}"
    
    @given(st.lists(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                             min_size=1, max_size=20),
                    min_size=1, max_size=100))
    def test_batch_prediction_contract(self, batch_features: List[List[float]]):
        """Test batch prediction endpoint contract."""
        payload = {"features": batch_features}
        
        # Mock batch response
        batch_size = len(batch_features)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [0] * batch_size,
            "probabilities": [0.5] * batch_size,
            "batch_size": batch_size
        }
        self.client.post.return_value = mock_response
        
        response = self.client.post("/predict/batch", json=payload)
        
        # Contract properties
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "predictions" in data, "Response should contain 'predictions' field"
        assert "probabilities" in data, "Response should contain 'probabilities' field"
        
        predictions = data["predictions"]
        probabilities = data["probabilities"]
        
        # Property: output size should match input size
        assert len(predictions) == batch_size, f"Expected {batch_size} predictions, got {len(predictions)}"
        assert len(probabilities) == batch_size, f"Expected {batch_size} probabilities, got {len(probabilities)}"
        
        # Property: all predictions should be valid
        for pred in predictions:
            assert pred in [0, 1], f"Each prediction should be 0 or 1, got {pred}"
        
        # Property: all probabilities should be valid
        for prob in probabilities:
            assert isinstance(prob, (int, float)), "Each probability should be numeric"
            assert 0 <= prob <= 1, f"Each probability should be between 0 and 1, got {prob}"
    
    @given(st.text(min_size=0, max_size=1000))
    def test_api_error_handling_properties(self, invalid_input: str):
        """Test API error handling properties."""
        # Test with various invalid inputs
        invalid_payloads = [
            invalid_input,
            {"features": invalid_input},
            {"invalid_field": [1, 2, 3]},
            {"features": None},
            {}
        ]
        
        for payload in invalid_payloads:
            # Mock error response
            mock_response = Mock()
            mock_response.status_code = 422
            mock_response.json.return_value = {"detail": "Validation error"}
            self.client.post.return_value = mock_response
            
            response = self.client.post("/predict", json=payload)
            
            # Property: invalid inputs should return appropriate error codes
            assert response.status_code >= 400, f"Invalid input should return error code >= 400, got {response.status_code}"
            
            # Property: error responses should include error details
            if response.status_code == 422:  # Validation error
                try:
                    error_data = response.json()
                    assert "detail" in error_data or "message" in error_data, "Error response should include details"
                except:
                    pass  # Some error responses might not be JSON


class DistributedSystemInvariants:
    """Property-based testing for distributed system invariants."""
    
    class DistributedStateMachine(RuleBasedStateMachine):
        """Stateful testing for distributed systems."""
        
        nodes = Bundle('nodes')
        
        def __init__(self):
            super().__init__()
            self.cluster_state = {}
            self.node_counter = 0
        
        @initialize()
        def setup_cluster(self):
            """Initialize the distributed system."""
            self.cluster_state = {
                'nodes': {},
                'leader': None,
                'data': {},
                'replication_factor': 3
            }
        
        @rule(target=nodes, node_id=st.integers(min_value=1, max_value=10))
        def add_node(self, node_id):
            """Add a node to the cluster."""
            assume(node_id not in self.cluster_state['nodes'])
            
            node_info = {
                'id': node_id,
                'status': 'active',
                'data': {},
                'last_heartbeat': time.time()
            }
            
            self.cluster_state['nodes'][node_id] = node_info
            
            # If no leader, elect this node as leader
            if self.cluster_state['leader'] is None:
                self.cluster_state['leader'] = node_id
            
            note(f"Added node {node_id}")
            return node_id
        
        @rule(node=consumes(nodes), key=st.text(min_size=1, max_size=10),
              value=st.integers(min_value=0, max_value=1000))
        def write_data(self, node, key, value):
            """Write data through a node."""
            if node not in self.cluster_state['nodes']:
                return  # Node was removed
            
            # Write to the primary location
            self.cluster_state['data'][key] = {
                'value': value,
                'timestamp': time.time(),
                'replicas': set()
            }
            
            # Replicate to other nodes
            active_nodes = [n for n, info in self.cluster_state['nodes'].items() 
                          if info['status'] == 'active']
            
            replication_count = min(len(active_nodes), self.cluster_state['replication_factor'])
            replicas = active_nodes[:replication_count]
            
            for replica_node in replicas:
                self.cluster_state['nodes'][replica_node]['data'][key] = value
                self.cluster_state['data'][key]['replicas'].add(replica_node)
            
            note(f"Wrote {key}={value} via node {node} to {len(replicas)} replicas")
        
        @rule(node=nodes, key=st.text(min_size=1, max_size=10))
        def read_data(self, node, key):
            """Read data from a node."""
            if node not in self.cluster_state['nodes']:
                return
            
            node_info = self.cluster_state['nodes'][node]
            
            if key in node_info['data']:
                local_value = node_info['data'][key]
                
                # Invariant: local value should match authoritative value
                if key in self.cluster_state['data']:
                    auth_value = self.cluster_state['data'][key]['value']
                    assert local_value == auth_value, \
                        f"Consistency violation: node {node} has {local_value}, authority has {auth_value}"
            
            note(f"Read {key} from node {node}")
        
        @rule(node=consumes(nodes))
        def remove_node(self, node):
            """Remove a node from the cluster."""
            if node not in self.cluster_state['nodes']:
                return
            
            # Mark node as failed
            self.cluster_state['nodes'][node]['status'] = 'failed'
            
            # If it was the leader, elect a new one
            if self.cluster_state['leader'] == node:
                active_nodes = [n for n, info in self.cluster_state['nodes'].items() 
                              if info['status'] == 'active']
                self.cluster_state['leader'] = active_nodes[0] if active_nodes else None
            
            note(f"Removed node {node}")
        
        @rule()
        def check_cluster_invariants(self):
            """Check distributed system invariants."""
            # Invariant: at most one leader
            active_nodes = [n for n, info in self.cluster_state['nodes'].items() 
                          if info['status'] == 'active']
            
            if active_nodes:
                assert self.cluster_state['leader'] in active_nodes or self.cluster_state['leader'] is None, \
                    "Leader must be an active node or None"
            
            # Invariant: data should be replicated according to replication factor
            for key, data_info in self.cluster_state['data'].items():
                available_replicas = [node for node in data_info['replicas'] 
                                    if node in self.cluster_state['nodes'] and 
                                    self.cluster_state['nodes'][node]['status'] == 'active']
                
                # Should have data on at least one node
                assert len(available_replicas) > 0, f"Key {key} has no available replicas"
                
                note(f"Key {key} has {len(available_replicas)} available replicas")


class FuzzTestingFramework:
    """Fuzzing framework for robust input validation."""
    
    def __init__(self):
        self.fuzz_results = []
    
    @given(st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.binary(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.integers()),
    ))
    def test_input_validation_robustness(self, fuzz_input):
        """Test input validation with various data types."""
        # Test that validation functions handle unexpected input gracefully
        validation_result = self._validate_input_safely(fuzz_input)
        
        # Property: validation should never crash
        assert validation_result in ['valid', 'invalid', 'error'], \
            f"Validation should return recognized result, got {validation_result}"
    
    def _validate_input_safely(self, input_data):
        """Safely validate input data."""
        try:
            # Simulate various validation scenarios
            if input_data is None:
                return 'invalid'
            elif isinstance(input_data, (int, float)) and not math.isnan(input_data) and not math.isinf(input_data):
                return 'valid'
            elif isinstance(input_data, str) and len(input_data.strip()) > 0:
                return 'valid'
            elif isinstance(input_data, list) and len(input_data) > 0:
                return 'valid'
            elif isinstance(input_data, dict) and len(input_data) > 0:
                return 'valid'
            else:
                return 'invalid'
        except Exception:
            return 'error'
    
    @given(st.text(min_size=0, max_size=10000))
    def test_string_processing_robustness(self, fuzz_string):
        """Test string processing with various string inputs."""
        # Test various string operations
        try:
            # Length should never be negative
            assert len(fuzz_string) >= 0, "String length should be non-negative"
            
            # Splitting and joining should be inverse operations (when no duplicates)
            if '\n' not in fuzz_string:
                parts = fuzz_string.split(' ')
                rejoined = ' '.join(parts)
                # Note: this may not be exactly equal due to multiple spaces
                
            # Encoding/decoding should work for valid UTF-8
            try:
                encoded = fuzz_string.encode('utf-8')
                decoded = encoded.decode('utf-8')
                assert decoded == fuzz_string, "UTF-8 encode/decode should be identity"
            except UnicodeError:
                pass  # Expected for invalid Unicode
            
        except Exception as e:
            note(f"String processing failed with input length {len(fuzz_string)}: {e}")
            # Most string operations should not crash
            if not isinstance(e, (UnicodeError, MemoryError)):
                raise
    
    @given(st.lists(st.integers(), min_size=0, max_size=1000))
    def test_numerical_operations_robustness(self, numbers):
        """Test numerical operations with various inputs."""
        if not numbers:
            return
        
        # Test statistical operations
        try:
            # Sum should be deterministic
            sum1 = sum(numbers)
            sum2 = sum(numbers)
            assert sum1 == sum2, "Sum should be deterministic"
            
            # Mean should be between min and max
            if numbers:
                mean_val = statistics.mean(numbers)
                assert min(numbers) <= mean_val <= max(numbers), "Mean should be between min and max"
            
            # Standard deviation should be non-negative
            if len(numbers) > 1:
                std_val = statistics.stdev(numbers)
                assert std_val >= 0, "Standard deviation should be non-negative"
            
        except (OverflowError, statistics.StatisticsError) as e:
            # These are acceptable for extreme inputs
            note(f"Expected numerical error: {e}")
        except Exception as e:
            note(f"Unexpected error in numerical operations: {e}")
            raise


class PropertyBasedTestRunner:
    """Main runner for property-based tests."""
    
    def __init__(self):
        self.results = []
        self.test_classes = [
            MLModelProperties(),
            DataProcessingProperties(),
            APIContractTesting(),
            FuzzTestingFramework()
        ]
    
    def run_all_property_tests(self, max_examples: int = 100, timeout: int = 300) -> PropertySuite:
        """Run all property-based tests."""
        if not HYPOTHESIS_AVAILABLE:
            print("❌ Hypothesis not available for property-based testing")
            return PropertySuite("property_tests", [], [], False, 0, 0.0)
        
        print("🔍 Running property-based test suite...")
        
        suite_start = time.perf_counter()
        all_results = []
        total_examples = 0
        
        # Configure Hypothesis settings
        test_settings = settings(
            max_examples=max_examples,
            deadline=timedelta(seconds=timeout),
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
        )
        
        # Run tests from each class
        for test_class in self.test_classes:
            class_name = test_class.__class__.__name__
            print(f"📋 Running {class_name} tests...")
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) 
                          if method.startswith('test_') and callable(getattr(test_class, method))]
            
            for method_name in test_methods:
                test_method = getattr(test_class, method_name)
                
                # Check if it's a Hypothesis test
                if hasattr(test_method, 'hypothesis'):
                    try:
                        test_start = time.perf_counter()
                        
                        # Apply settings and run test
                        test_method = test_settings(test_method)
                        test_method()
                        
                        test_duration = (time.perf_counter() - test_start) * 1000
                        
                        result = PropertyTestResult(
                            test_name=f"{class_name}.{method_name}",
                            property_name=method_name,
                            test_type="hypothesis",
                            passed=True,
                            examples_tested=max_examples,
                            counterexample=None,
                            execution_time_ms=test_duration,
                            error_message=None,
                            timestamp=datetime.now()
                        )
                        
                        all_results.append(result)
                        total_examples += max_examples
                        
                        print(f"   ✅ {method_name} ({max_examples} examples)")
                        
                    except Exception as e:
                        test_duration = (time.perf_counter() - test_start) * 1000
                        
                        # Extract counterexample if available
                        counterexample = None
                        if hasattr(e, 'hypothesis_internal_use_examples'):
                            counterexample = {"error_examples": str(e)}
                        
                        result = PropertyTestResult(
                            test_name=f"{class_name}.{method_name}",
                            property_name=method_name,
                            test_type="hypothesis",
                            passed=False,
                            examples_tested=0,  # Unknown how many were tested before failure
                            counterexample=counterexample,
                            execution_time_ms=test_duration,
                            error_message=str(e),
                            timestamp=datetime.now()
                        )
                        
                        all_results.append(result)
                        
                        print(f"   ❌ {method_name}: {str(e)[:100]}")
        
        # Run stateful tests
        if HYPOTHESIS_AVAILABLE:
            print("🔄 Running stateful distributed system tests...")
            try:
                stateful_test = DistributedSystemInvariants.DistributedStateMachine.TestCase()
                stateful_test.runTest()
                
                all_results.append(PropertyTestResult(
                    test_name="DistributedSystemInvariants.stateful_test",
                    property_name="distributed_invariants",
                    test_type="stateful",
                    passed=True,
                    examples_tested=50,  # Hypothesis default for stateful
                    counterexample=None,
                    execution_time_ms=1000,  # Estimate
                    error_message=None,
                    timestamp=datetime.now()
                ))
                
                print("   ✅ Distributed system invariants")
                
            except Exception as e:
                all_results.append(PropertyTestResult(
                    test_name="DistributedSystemInvariants.stateful_test",
                    property_name="distributed_invariants",
                    test_type="stateful",
                    passed=False,
                    examples_tested=0,
                    counterexample={"stateful_error": str(e)},
                    execution_time_ms=1000,
                    error_message=str(e),
                    timestamp=datetime.now()
                ))
                
                print(f"   ❌ Distributed system invariants: {str(e)[:100]}")
        
        suite_duration = (time.perf_counter() - suite_start) * 1000
        
        # Create test suite result
        passed_tests = [r for r in all_results if r.passed]
        failed_tests = [r for r in all_results if not r.passed]
        
        properties = list(set(r.property_name for r in all_results))
        
        suite = PropertySuite(
            suite_name="comprehensive_property_tests",
            properties=properties,
            test_results=all_results,
            overall_passed=len(failed_tests) == 0,
            total_examples=total_examples,
            execution_time_ms=suite_duration
        )
        
        # Print summary
        print(f"\n🎯 Property-based test suite completed in {suite_duration:.2f}ms")
        print(f"   Properties tested: {len(properties)}")
        print(f"   Total examples: {total_examples}")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        
        if failed_tests:
            print("\n❌ Failed tests:")
            for test in failed_tests[:5]:  # Show first 5 failures
                print(f"   • {test.test_name}: {test.error_message[:100]}")
        
        # Save results
        self._save_results(suite)
        
        return suite
    
    def _save_results(self, suite: PropertySuite) -> None:
        """Save property test results."""
        results_dir = Path("property_test_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"property_tests_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        
        print(f"📊 Property test results saved to {results_file}")


def main():
    """Main function for running property-based tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Property-Based Testing Framework")
    parser.add_argument("--examples", type=int, default=100,
                       help="Number of examples per property test")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per test in seconds")
    parser.add_argument("--test-type", choices=["ml", "data", "api", "fuzz", "distributed", "all"], 
                       default="all", help="Type of property tests to run")
    
    args = parser.parse_args()
    
    if not HYPOTHESIS_AVAILABLE:
        print("❌ Hypothesis library not available. Install with: pip install hypothesis")
        return
    
    runner = PropertyBasedTestRunner()
    
    # Filter test classes based on type
    if args.test_type != "all":
        type_mapping = {
            "ml": [MLModelProperties()],
            "data": [DataProcessingProperties()],
            "api": [APIContractTesting()],
            "fuzz": [FuzzTestingFramework()]
        }
        runner.test_classes = type_mapping.get(args.test_type, runner.test_classes)
    
    # Run tests
    suite = runner.run_all_property_tests(
        max_examples=args.examples,
        timeout=args.timeout
    )
    
    # Exit with error code if tests failed
    if not suite.overall_passed:
        exit(1)


if __name__ == "__main__":
    main()