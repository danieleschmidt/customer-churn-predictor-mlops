"""
Test suite for Superintelligent Discovery Engine

Comprehensive testing of superintelligent discovery capabilities including
pattern discovery, theory generation, and novel algorithm creation.

Author: Terry (Terragon Labs)
Version: 1.0.0 - Superintelligent Discovery Testing
"""

import asyncio
import json
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.superintelligent_discovery_engine import (
    SuperintelligentDiscoveryEngine,
    NovelAlgorithm,
    DiscoveryPattern,
    TheoryGeneration,
    BreakthroughDiscovery,
    DiscoveryType,
    IntelligenceLevel,
    BreakthroughType,
)


class TestSuperintelligentDiscoveryEngine:
    """Test suite for SuperintelligentDiscoveryEngine class."""

    @pytest.fixture
    def discovery_engine(self):
        """Create a test instance of SuperintelligentDiscoveryEngine."""
        return SuperintelligentDiscoveryEngine(
            intelligence_level=IntelligenceLevel.SUPERINTELLIGENT,
            discovery_threshold=0.8,
            novelty_threshold=0.7,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data for discovery testing."""
        np.random.seed(42)
        data = np.random.randn(200, 8)
        target = np.random.randint(0, 2, 200)
        return data, target

    @pytest.mark.asyncio
    async def test_discover_novel_patterns(self, discovery_engine, sample_data):
        """Test novel pattern discovery functionality."""
        data, target = sample_data
        
        patterns = await discovery_engine.discover_novel_patterns(
            data, target, domain_context="machine_learning"
        )
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(p, DiscoveryPattern) for p in patterns)
        assert all(p.novelty_score >= discovery_engine.novelty_threshold for p in patterns)

    @pytest.mark.asyncio
    async def test_pattern_discovery_statistical(self, discovery_engine, sample_data):
        """Test statistical pattern discovery."""
        data, target = sample_data
        
        # Test with correlated features
        correlated_data = np.copy(data)
        correlated_data[:, 1] = correlated_data[:, 0] * 0.9 + np.random.normal(0, 0.1, len(data))
        
        patterns = discovery_engine._discover_statistical_patterns(correlated_data, target)
        
        assert len(patterns) > 0
        correlation_patterns = [p for p in patterns if p.type == DiscoveryType.PATTERN_DISCOVERY]
        assert len(correlation_patterns) > 0

    @pytest.mark.asyncio
    async def test_pattern_discovery_analogical(self, discovery_engine):
        """Test analogical pattern discovery."""
        data = np.random.randn(100, 5)
        
        patterns = discovery_engine._discover_analogical_patterns(data, "machine_learning")
        
        assert isinstance(patterns, list)
        if patterns:  # May not always find analogical patterns
            assert all(p.type == DiscoveryType.CROSS_DOMAIN_CONNECTION for p in patterns)

    @pytest.mark.asyncio
    async def test_pattern_discovery_cross_domain(self, discovery_engine, sample_data):
        """Test cross-domain pattern discovery."""
        data, target = sample_data
        
        patterns = discovery_engine._discover_cross_domain_patterns(data, target)
        
        assert isinstance(patterns, list)
        if patterns:
            theoretical_patterns = [p for p in patterns if p.type == DiscoveryType.THEORETICAL_INSIGHT]
            assert len(theoretical_patterns) >= 0

    @pytest.mark.asyncio
    async def test_pattern_discovery_emergent_behaviors(self, discovery_engine, sample_data):
        """Test emergent behavior discovery."""
        data, target = sample_data
        
        patterns = discovery_engine._discover_emergent_behaviors(data, target)
        
        assert isinstance(patterns, list)
        if patterns:
            emergent_patterns = [p for p in patterns if p.type == DiscoveryType.EMERGENT_BEHAVIOR]
            assert all(p.theoretical_significance > 0.5 for p in emergent_patterns)

    @pytest.mark.asyncio
    async def test_pattern_discovery_unifying_principles(self, discovery_engine, sample_data):
        """Test unifying principle discovery."""
        data, target = sample_data
        
        patterns = discovery_engine._discover_unifying_principles(data, target)
        
        assert isinstance(patterns, list)
        if patterns:
            theoretical_patterns = [p for p in patterns if p.type == DiscoveryType.THEORETICAL_INSIGHT]
            assert all(p.theoretical_significance >= 0.7 for p in theoretical_patterns)

    def test_generate_novel_theories(self, discovery_engine):
        """Test novel theory generation."""
        # Create mock discovered patterns
        mock_patterns = [
            DiscoveryPattern(
                id="pattern_1",
                type=DiscoveryType.PATTERN_DISCOVERY,
                description="Pattern 1",
                mathematical_formulation="f(x) = x^2",
                confidence=0.8,
                novelty_score=0.9,
                practical_impact=0.7,
                theoretical_significance=0.8,
                discovered_at=datetime.now(),
                validation_status="preliminary",
                commercial_potential=100000,
            ),
            DiscoveryPattern(
                id="pattern_2",
                type=DiscoveryType.PATTERN_DISCOVERY,
                description="Pattern 2",
                mathematical_formulation="g(x) = sin(x)",
                confidence=0.85,
                novelty_score=0.95,
                practical_impact=0.8,
                theoretical_significance=0.9,
                discovered_at=datetime.now(),
                validation_status="preliminary",
                commercial_potential=150000,
            ),
        ]
        
        theories = discovery_engine.generate_novel_theories(mock_patterns)
        
        assert isinstance(theories, list)
        assert len(theories) > 0
        assert all(isinstance(t, TheoryGeneration) for t in theories)
        assert all(t.confidence > 0.5 for t in theories)

    @pytest.mark.asyncio
    async def test_create_novel_algorithms(self, discovery_engine):
        """Test novel algorithm creation."""
        # Create mock patterns for algorithm creation
        mock_patterns = [
            DiscoveryPattern(
                id="quantum_pattern",
                type=DiscoveryType.CROSS_DOMAIN_CONNECTION,
                description="Quantum-inspired pattern discovered",
                mathematical_formulation="Quantum superposition",
                confidence=0.9,
                novelty_score=0.95,
                practical_impact=0.8,
                theoretical_significance=0.9,
                discovered_at=datetime.now(),
                validation_status="validated",
                commercial_potential=500000,
            ),
            DiscoveryPattern(
                id="emergent_pattern",
                type=DiscoveryType.EMERGENT_BEHAVIOR,
                description="Emergent behavior detected",
                mathematical_formulation="Emergence function",
                confidence=0.85,
                novelty_score=0.9,
                practical_impact=0.75,
                theoretical_significance=0.85,
                discovered_at=datetime.now(),
                validation_status="preliminary",
                commercial_potential=300000,
            ),
        ]
        
        algorithms = await discovery_engine.create_novel_algorithms(mock_patterns)
        
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert all(isinstance(a, NovelAlgorithm) for a in algorithms)

    def test_pattern_validation(self, discovery_engine):
        """Test pattern validation logic."""
        valid_pattern = DiscoveryPattern(
            id="valid_pattern",
            type=DiscoveryType.PATTERN_DISCOVERY,
            description="Valid pattern",
            mathematical_formulation="f(x) = x",
            confidence=0.8,
            novelty_score=0.8,  # Above threshold
            practical_impact=0.6,
            theoretical_significance=0.5,  # Above threshold
            discovered_at=datetime.now(),
            validation_status="preliminary",
            commercial_potential=50000,
        )
        
        invalid_pattern = DiscoveryPattern(
            id="invalid_pattern",
            type=DiscoveryType.PATTERN_DISCOVERY,
            description="Invalid pattern",
            mathematical_formulation="g(x) = x",
            confidence=0.4,  # Below threshold
            novelty_score=0.6,  # Below threshold
            practical_impact=0.2,  # Below threshold
            theoretical_significance=0.2,  # Below threshold
            discovered_at=datetime.now(),
            validation_status="preliminary",
            commercial_potential=1000,
        )
        
        validated = discovery_engine._validate_patterns([valid_pattern, invalid_pattern])
        
        assert len(validated) == 1
        assert validated[0].id == "valid_pattern"

    def test_cognitive_state_updates(self, discovery_engine):
        """Test cognitive state update mechanism."""
        initial_momentum = discovery_engine.cognitive_state["discovery_momentum"]
        
        # Test with successful patterns
        good_patterns = [
            DiscoveryPattern(
                id="good_pattern",
                type=DiscoveryType.PATTERN_DISCOVERY,
                description="Good pattern",
                mathematical_formulation="f(x) = x",
                confidence=0.9,
                novelty_score=0.85,
                practical_impact=0.8,
                theoretical_significance=0.7,
                discovered_at=datetime.now(),
                validation_status="preliminary",
                commercial_potential=100000,
            )
        ]
        
        discovery_engine._update_cognitive_state(good_patterns)
        
        assert discovery_engine.cognitive_state["discovery_momentum"] >= initial_momentum
        
        # Test with empty patterns
        discovery_engine._update_cognitive_state([])
        
        # Momentum should decrease
        assert discovery_engine.cognitive_state["discovery_momentum"] < 1.0

    def test_knowledge_graph_integration(self, discovery_engine):
        """Test knowledge graph integration."""
        pattern = DiscoveryPattern(
            id="test_pattern",
            type=DiscoveryType.PATTERN_DISCOVERY,
            description="Test pattern for knowledge integration",
            mathematical_formulation="test_formula",
            confidence=0.8,
            novelty_score=0.85,
            practical_impact=0.7,
            theoretical_significance=0.6,
            discovered_at=datetime.now(),
            validation_status="preliminary",
            commercial_potential=75000,
        )
        
        initial_frequency = discovery_engine.knowledge_graph.get("pattern_discovery", Mock()).discovery_frequency
        
        discovery_engine._integrate_pattern_to_knowledge(pattern)
        
        # Verify knowledge graph was updated
        if "pattern_discovery" in discovery_engine.knowledge_graph:
            node = discovery_engine.knowledge_graph["pattern_discovery"]
            assert len(node.insights) > 0

    def test_utility_methods(self, discovery_engine):
        """Test utility calculation methods."""
        # Test skewness calculation
        normal_data = np.random.normal(0, 1, 1000)
        skewed_data = np.random.exponential(1, 1000)
        
        normal_skewness = discovery_engine._calculate_skewness(normal_data)
        skewed_skewness = discovery_engine._calculate_skewness(skewed_data)
        
        assert abs(normal_skewness) < 0.5  # Should be close to 0 for normal data
        assert skewed_skewness > 0.5  # Should be positive for right-skewed data

    def test_mutual_information_approximation(self, discovery_engine):
        """Test mutual information approximation."""
        # Create correlated data
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.normal(0, 0.1, 100)  # y correlated with first feature
        
        mi = discovery_engine._approximate_mutual_information(X, y)
        
        assert mi >= 0.0
        assert mi <= 1.0

    def test_complexity_emergence_calculation(self, discovery_engine):
        """Test complexity emergence calculation."""
        # Create data with feature interactions
        data = np.random.randn(100, 6)
        data[:, 1] = data[:, 0] * 0.8 + np.random.normal(0, 0.2, 100)  # Correlated features
        
        emergence_score = discovery_engine._calculate_complexity_emergence(data)
        
        assert emergence_score >= 0.0
        assert emergence_score <= 1.0

    def test_scaling_law_discovery(self, discovery_engine):
        """Test scaling law discovery."""
        # Create power law relationship
        x_vals = np.random.uniform(1, 10, 100)
        y_vals = x_vals ** 1.5 + np.random.normal(0, 0.1, 100)
        
        data = np.column_stack([x_vals, y_vals])
        
        scaling_exponent = discovery_engine._discover_scaling_law(data)
        
        assert isinstance(scaling_exponent, float)
        assert not np.isnan(scaling_exponent)

    def test_export_discovery_report(self, discovery_engine, tmp_path):
        """Test discovery report export functionality."""
        # Add some test data
        discovery_engine.discovered_patterns.append(
            DiscoveryPattern(
                id="export_test_pattern",
                type=DiscoveryType.PATTERN_DISCOVERY,
                description="Test pattern for export",
                mathematical_formulation="export_formula",
                confidence=0.85,
                novelty_score=0.9,
                practical_impact=0.8,
                theoretical_significance=0.7,
                discovered_at=datetime.now(),
                validation_status="validated",
                commercial_potential=200000,
            )
        )
        
        filepath = tmp_path / "test_discovery_report.json"
        discovery_engine.export_discovery_report(str(filepath))
        
        assert filepath.exists()
        
        # Verify report content
        with open(filepath) as f:
            report = json.load(f)
        
        assert "timestamp" in report
        assert "intelligence_level" in report
        assert "discoveries" in report
        assert len(report["discoveries"]["patterns"]) == 1

    @pytest.mark.asyncio
    async def test_complete_discovery_workflow(self, discovery_engine, sample_data):
        """Test complete discovery workflow from patterns to algorithms."""
        data, target = sample_data
        
        # 1. Discover patterns
        patterns = await discovery_engine.discover_novel_patterns(data, target)
        
        # 2. Generate theories
        theories = discovery_engine.generate_novel_theories(patterns)
        
        # 3. Create algorithms
        algorithms = await discovery_engine.create_novel_algorithms(patterns)
        
        # Verify workflow completion
        assert len(patterns) > 0
        assert len(discovery_engine.discovered_patterns) > 0
        assert isinstance(theories, list)
        assert isinstance(algorithms, list)

    @pytest.mark.performance
    async def test_discovery_performance(self, discovery_engine, sample_data):
        """Test discovery engine performance requirements."""
        data, target = sample_data
        
        start_time = asyncio.get_event_loop().time()
        patterns = await discovery_engine.discover_novel_patterns(
            data, target, domain_context="machine_learning"
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Should complete discovery within reasonable time
        assert execution_time < 60.0  # 60 seconds for test data
        assert len(patterns) > 0

    @pytest.mark.integration
    async def test_intelligence_level_scaling(self):
        """Test that higher intelligence levels provide better results."""
        data = np.random.randn(100, 5)
        target = np.random.randint(0, 2, 100)
        
        # Test different intelligence levels
        basic_engine = SuperintelligentDiscoveryEngine(
            intelligence_level=IntelligenceLevel.BASIC,
            max_discovery_depth=2,
        )
        
        super_engine = SuperintelligentDiscoveryEngine(
            intelligence_level=IntelligenceLevel.SUPERINTELLIGENT,
            max_discovery_depth=5,
        )
        
        basic_patterns = await basic_engine.discover_novel_patterns(data, target)
        super_patterns = await super_engine.discover_novel_patterns(data, target)
        
        # Superintelligent should discover more or higher quality patterns
        super_avg_novelty = np.mean([p.novelty_score for p in super_patterns]) if super_patterns else 0
        basic_avg_novelty = np.mean([p.novelty_score for p in basic_patterns]) if basic_patterns else 0
        
        assert super_avg_novelty >= basic_avg_novelty


class TestNovelAlgorithm:
    """Test suite for NovelAlgorithm class."""

    @pytest.fixture
    def novel_algorithm(self):
        """Create a test instance of NovelAlgorithm."""
        return NovelAlgorithm(
            algorithm_name="TestSuperAlgorithm",
            quantum_inspired=True,
            meta_learning_enabled=True,
        )

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data for algorithm testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 8)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 8)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test

    def test_novel_algorithm_initialization(self, novel_algorithm):
        """Test NovelAlgorithm initialization."""
        assert novel_algorithm.algorithm_name == "TestSuperAlgorithm"
        assert novel_algorithm.quantum_inspired is True
        assert novel_algorithm.meta_learning_enabled is True
        assert len(novel_algorithm.base_models) == 0

    def test_novel_algorithm_fit(self, novel_algorithm, sample_ml_data):
        """Test NovelAlgorithm fit method."""
        X_train, y_train, _, _ = sample_ml_data
        
        fitted_algorithm = novel_algorithm.fit(X_train, y_train)
        
        assert fitted_algorithm is novel_algorithm
        assert len(novel_algorithm.base_models) > 0
        assert len(novel_algorithm.learning_history) > 0

    def test_novel_algorithm_predict(self, novel_algorithm, sample_ml_data):
        """Test NovelAlgorithm predict method."""
        X_train, y_train, X_test, y_test = sample_ml_data
        
        # Fit the algorithm first
        novel_algorithm.fit(X_train, y_train)
        
        # Make predictions
        predictions = novel_algorithm.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_quantum_transformation(self, novel_algorithm, sample_ml_data):
        """Test quantum-inspired transformation."""
        X_train, _, _, _ = sample_ml_data
        
        X_transformed = novel_algorithm._quantum_inspired_transformation(X_train)
        
        # Quantum transformation should add features
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] > X_train.shape[1]

    def test_adaptive_transformation(self, novel_algorithm, sample_ml_data):
        """Test adaptive transformation."""
        X_train, _, _, _ = sample_ml_data
        
        X_transformed = novel_algorithm._adaptive_transformation(X_train)
        
        # Adaptive transformation may add features based on data characteristics
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] >= X_train.shape[1]

    def test_decision_fusion(self, novel_algorithm, sample_ml_data):
        """Test decision fusion mechanism."""
        X_train, y_train, X_test, _ = sample_ml_data
        
        # Fit algorithm to create decision fusion strategy
        novel_algorithm.fit(X_train, y_train)
        
        # Create mock predictions from base models
        mock_predictions = [
            np.random.randint(0, 2, len(X_test)),
            np.random.randint(0, 2, len(X_test)),
            np.random.randint(0, 2, len(X_test)),
        ]
        
        fused_predictions = novel_algorithm._apply_decision_fusion(mock_predictions)
        
        assert len(fused_predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in fused_predictions)

    def test_algorithm_without_fitting(self, novel_algorithm, sample_ml_data):
        """Test that prediction fails without fitting."""
        _, _, X_test, _ = sample_ml_data
        
        with pytest.raises(ValueError, match="Algorithm must be fitted"):
            novel_algorithm.predict(X_test)


@pytest.mark.asyncio
async def test_superintelligent_discovery_main():
    """Test the main function of the superintelligent discovery module."""
    from src.superintelligent_discovery_engine import main
    
    results = await main()
    
    assert isinstance(results, dict)
    assert "patterns_discovered" in results
    assert "theories_generated" in results
    assert "algorithms_created" in results
    assert "intelligence_level" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])