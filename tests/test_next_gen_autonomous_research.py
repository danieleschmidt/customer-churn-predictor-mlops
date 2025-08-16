"""
Test suite for Next-Generation Autonomous Research Intelligence System

Comprehensive testing of the autonomous research capabilities including
hypothesis generation, experiment execution, and breakthrough detection.

Author: Terry (Terragon Labs)
Version: 1.0.0 - Autonomous Research Testing
"""

import asyncio
import json
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.next_gen_autonomous_research import (
    AutonomousResearchIntelligence,
    ResearchHypothesis,
    ExperimentResult,
    BreakthroughDiscovery,
    ResearchPhase,
    BreakthroughType,
)


class TestAutonomousResearchIntelligence:
    """Test suite for AutonomousResearchIntelligence class."""

    @pytest.fixture
    def research_system(self):
        """Create a test instance of AutonomousResearchIntelligence."""
        return AutonomousResearchIntelligence(
            confidence_threshold=0.7,
            breakthrough_threshold=0.85,
            max_concurrent_experiments=3,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training and testing data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    @pytest.mark.asyncio
    async def test_generate_research_hypotheses(self, research_system):
        """Test autonomous hypothesis generation."""
        hypotheses = await research_system.generate_research_hypotheses(n_hypotheses=5)
        
        assert len(hypotheses) == 5
        assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
        assert all(h.confidence >= 0.6 for h in hypotheses)
        assert all(h.confidence <= 1.0 for h in hypotheses)
        assert len(set(h.id for h in hypotheses)) == 5  # All IDs unique

    @pytest.mark.asyncio
    async def test_execute_autonomous_experiment_ensemble(self, research_system, sample_data):
        """Test ensemble method experiment execution."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_ensemble_001",
            description="Test ensemble methods",
            research_question="Can ensemble methods improve accuracy?",
            predicted_outcome="Improved accuracy",
            confidence=0.8,
            priority=1,
            created_at=datetime.now(),
            domain="ensemble_methods",
            approach="Adaptive ensemble weighting",
            expected_improvement=0.15,
            risk_level="low",
        )
        
        result = await research_system.execute_autonomous_experiment(
            hypothesis, X_train, y_train, X_test, y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert result.hypothesis_id == hypothesis.id
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] >= 0.0
        assert result.metrics["accuracy"] <= 1.0

    @pytest.mark.asyncio
    async def test_execute_autonomous_experiment_quantum_features(self, research_system, sample_data):
        """Test quantum feature engineering experiment."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_quantum_001",
            description="Test quantum feature engineering",
            research_question="Do quantum features improve performance?",
            predicted_outcome="Enhanced feature representation",
            confidence=0.75,
            priority=1,
            created_at=datetime.now(),
            domain="feature_engineering",
            approach="Quantum superposition-based features",
            expected_improvement=0.20,
            risk_level="medium",
        )
        
        result = await research_system.execute_autonomous_experiment(
            hypothesis, X_train, y_train, X_test, y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert "accuracy" in result.metrics
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_execute_autonomous_experiment_meta_learning(self, research_system, sample_data):
        """Test meta-learning experiment execution."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_meta_001",
            description="Test meta-learning approach",
            research_question="Can meta-learning optimize algorithm selection?",
            predicted_outcome="Improved algorithm selection",
            confidence=0.85,
            priority=1,
            created_at=datetime.now(),
            domain="meta_learning",
            approach="Automated algorithm selection",
            expected_improvement=0.25,
            risk_level="low",
        )
        
        result = await research_system.execute_autonomous_experiment(
            hypothesis, X_train, y_train, X_test, y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert "meta_learning_improvement" in result.metrics

    @pytest.mark.asyncio
    async def test_execute_autonomous_experiment_neural_search(self, research_system, sample_data):
        """Test neural architecture search experiment."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_nas_001",
            description="Test neural architecture search",
            research_question="What is the optimal neural architecture?",
            predicted_outcome="Discovered optimal architecture",
            confidence=0.7,
            priority=1,
            created_at=datetime.now(),
            domain="neural_architecture",
            approach="Evolutionary architecture search",
            expected_improvement=0.18,
            risk_level="medium",
        )
        
        result = await research_system.execute_autonomous_experiment(
            hypothesis, X_train, y_train, X_test, y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert "optimal_architecture" in result.metrics

    @pytest.mark.asyncio
    async def test_experiment_failure_handling(self, research_system, sample_data):
        """Test experiment failure handling."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_failure_001",
            description="Test failure handling",
            research_question="How does system handle failures?",
            predicted_outcome="Graceful failure handling",
            confidence=0.6,
            priority=1,
            created_at=datetime.now(),
            domain="unknown_domain",  # This should trigger baseline execution
            approach="Unknown approach",
            expected_improvement=0.1,
            risk_level="high",
        )
        
        # Mock a failure scenario
        with patch.object(research_system, '_experiment_baseline_comparison', side_effect=Exception("Test failure")):
            result = await research_system.execute_autonomous_experiment(
                hypothesis, X_train, y_train, X_test, y_test
            )
            
            assert isinstance(result, ExperimentResult)
            assert result.success is False
            assert "error" in result.metrics
            assert len(result.insights) > 0
            assert "failed" in result.insights[0].lower()

    @pytest.mark.asyncio
    async def test_breakthrough_detection(self, research_system, sample_data):
        """Test breakthrough detection functionality."""
        X_train, y_train, X_test, y_test = sample_data
        
        hypothesis = ResearchHypothesis(
            id="test_breakthrough_001",
            description="Test breakthrough detection",
            research_question="Can we detect breakthroughs?",
            predicted_outcome="Breakthrough detection",
            confidence=0.9,
            priority=1,
            created_at=datetime.now(),
            domain="ensemble_methods",
            approach="Revolutionary ensemble method",
            expected_improvement=0.5,
            risk_level="low",
        )
        
        # Mock high accuracy to trigger breakthrough detection
        with patch.object(research_system, '_experiment_adaptive_ensemble') as mock_experiment:
            mock_experiment.return_value = {
                "accuracy": 0.95,  # High accuracy to trigger breakthrough
                "f1_score": 0.94,
                "precision": 0.93,
                "recall": 0.96,
            }
            
            result = await research_system.execute_autonomous_experiment(
                hypothesis, X_train, y_train, X_test, y_test
            )
            
            assert result.breakthrough_detected is True
            assert result.breakthrough_type is not None
            assert len(research_system.breakthroughs) > 0

    @pytest.mark.asyncio
    async def test_run_autonomous_research_campaign(self, research_system, sample_data):
        """Test complete autonomous research campaign."""
        X_train, y_train, X_test, y_test = sample_data
        
        campaign_results = await research_system.run_autonomous_research_campaign(
            X_train, y_train, X_test, y_test,
            n_hypotheses=3,
            max_experiments=3,
        )
        
        assert isinstance(campaign_results, dict)
        assert "campaign_summary" in campaign_results
        assert "best_results" in campaign_results
        assert "breakthrough_summary" in campaign_results
        
        # Check campaign summary
        summary = campaign_results["campaign_summary"]
        assert summary["hypotheses_generated"] == 3
        assert summary["experiments_completed"] >= 0
        assert summary["success_rate"] >= 0.0
        assert summary["success_rate"] <= 1.0

    def test_generate_experimental_insights(self, research_system):
        """Test insight generation from experimental results."""
        result = {"accuracy": 0.92, "f1_score": 0.90}
        hypothesis = ResearchHypothesis(
            id="test_insights_001",
            description="Test insights",
            research_question="Test question",
            predicted_outcome="Test outcome",
            confidence=0.8,
            priority=1,
            created_at=datetime.now(),
            domain="test_domain",
            approach="Test approach",
            expected_improvement=0.1,
            risk_level="low",
        )
        
        insights = research_system._generate_experimental_insights(result, hypothesis)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert any("exceptional performance" in insight.lower() for insight in insights)

    def test_generate_recommendations(self, research_system):
        """Test recommendation generation from experimental results."""
        result = {"accuracy": 0.88, "f1_score": 0.85}
        hypothesis = ResearchHypothesis(
            id="test_recommendations_001",
            description="Test recommendations",
            research_question="Test question",
            predicted_outcome="Test outcome",
            confidence=0.75,
            priority=1,
            created_at=datetime.now(),
            domain="test_domain",
            approach="Test approach",
            expected_improvement=0.15,
            risk_level="medium",
        )
        
        recommendations = research_system._generate_recommendations(result, hypothesis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("optimize" in rec.lower() for rec in recommendations)

    def test_campaign_analysis(self, research_system):
        """Test campaign results analysis."""
        # Create mock experiment results
        mock_results = [
            ExperimentResult(
                hypothesis_id="test_001",
                approach="test_approach_1",
                metrics={"accuracy": 0.85, "f1_score": 0.83},
                execution_time=1.5,
                success=True,
                breakthrough_detected=False,
                breakthrough_type=None,
                insights=["Good performance"],
                recommendations=["Continue research"],
                reproducibility_score=0.9,
            ),
            ExperimentResult(
                hypothesis_id="test_002",
                approach="test_approach_2",
                metrics={"accuracy": 0.92, "f1_score": 0.90},
                execution_time=2.1,
                success=True,
                breakthrough_detected=True,
                breakthrough_type=BreakthroughType.ALGORITHMIC,
                insights=["Breakthrough achieved"],
                recommendations=["Scale experiment"],
                reproducibility_score=0.95,
            ),
        ]
        
        # Add mock hypotheses
        research_system.hypotheses = [
            ResearchHypothesis(
                id="test_001",
                description="Test 1",
                research_question="Question 1",
                predicted_outcome="Outcome 1",
                confidence=0.8,
                priority=1,
                created_at=datetime.now(),
                domain="domain_1",
                approach="approach_1",
                expected_improvement=0.1,
                risk_level="low",
            ),
            ResearchHypothesis(
                id="test_002",
                description="Test 2",
                research_question="Question 2",
                predicted_outcome="Outcome 2",
                confidence=0.9,
                priority=1,
                created_at=datetime.now(),
                domain="domain_2",
                approach="approach_2",
                expected_improvement=0.2,
                risk_level="medium",
            ),
        ]
        
        analysis = research_system._analyze_campaign_results(mock_results)
        
        assert isinstance(analysis, dict)
        assert "best_results" in analysis
        assert "breakthrough_summary" in analysis
        assert "knowledge_insights" in analysis
        assert analysis["best_results"]["accuracy"] == 0.92

    def test_export_research_report(self, research_system, tmp_path):
        """Test research report export functionality."""
        # Add some test data
        research_system.hypotheses.append(
            ResearchHypothesis(
                id="test_export_001",
                description="Test export",
                research_question="Test question",
                predicted_outcome="Test outcome",
                confidence=0.8,
                priority=1,
                created_at=datetime.now(),
                domain="test_domain",
                approach="Test approach",
                expected_improvement=0.1,
                risk_level="low",
            )
        )
        
        filepath = tmp_path / "test_research_report.json"
        research_system.export_research_report(str(filepath))
        
        assert filepath.exists()
        
        # Verify report content
        with open(filepath) as f:
            report = json.load(f)
        
        assert "timestamp" in report
        assert "system_info" in report
        assert "hypotheses" in report
        assert len(report["hypotheses"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_experiment_execution(self, research_system, sample_data):
        """Test concurrent execution of multiple experiments."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Generate multiple hypotheses
        hypotheses = await research_system.generate_research_hypotheses(n_hypotheses=4)
        
        # Execute experiments concurrently
        tasks = []
        for hypothesis in hypotheses:
            task = research_system.execute_autonomous_experiment(
                hypothesis, X_train, y_train, X_test, y_test
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all experiments completed
        successful_results = [r for r in results if isinstance(r, ExperimentResult)]
        assert len(successful_results) >= 3  # At least 3 should succeed

    @pytest.mark.performance
    async def test_performance_requirements(self, research_system, sample_data):
        """Test that the system meets performance requirements."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Test single experiment performance
        hypothesis = ResearchHypothesis(
            id="perf_test_001",
            description="Performance test",
            research_question="Is the system fast enough?",
            predicted_outcome="Fast execution",
            confidence=0.8,
            priority=1,
            created_at=datetime.now(),
            domain="ensemble_methods",
            approach="Fast ensemble",
            expected_improvement=0.1,
            risk_level="low",
        )
        
        start_time = asyncio.get_event_loop().time()
        result = await research_system.execute_autonomous_experiment(
            hypothesis, X_train, y_train, X_test, y_test
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Should complete within reasonable time (30 seconds for test data)
        assert execution_time < 30.0
        assert result.success is True

    @pytest.mark.integration
    async def test_integration_with_existing_system(self, research_system, sample_data):
        """Test integration with existing ML pipeline components."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Test that the research system can work with existing data formats
        assert X_train.shape[0] > 0
        assert y_train.shape[0] == X_train.shape[0]
        
        # Run a small campaign to verify integration
        campaign_results = await research_system.run_autonomous_research_campaign(
            X_train, y_train, X_test, y_test,
            n_hypotheses=2,
            max_experiments=2,
        )
        
        # Verify integration success
        assert campaign_results is not None
        assert isinstance(campaign_results, dict)


class TestResearchComponents:
    """Test individual research components."""

    def test_research_hypothesis_creation(self):
        """Test ResearchHypothesis data class."""
        hypothesis = ResearchHypothesis(
            id="test_hyp_001",
            description="Test hypothesis",
            research_question="Is this a test?",
            predicted_outcome="Yes, it is",
            confidence=0.9,
            priority=1,
            created_at=datetime.now(),
            domain="testing",
            approach="Unit testing",
            expected_improvement=0.5,
            risk_level="low",
        )
        
        assert hypothesis.id == "test_hyp_001"
        assert hypothesis.confidence == 0.9
        assert hypothesis.domain == "testing"

    def test_experiment_result_creation(self):
        """Test ExperimentResult data class."""
        result = ExperimentResult(
            hypothesis_id="test_hyp_001",
            approach="test_approach",
            metrics={"accuracy": 0.85, "f1_score": 0.83},
            execution_time=1.5,
            success=True,
            breakthrough_detected=False,
            breakthrough_type=None,
            insights=["Test insight"],
            recommendations=["Test recommendation"],
            reproducibility_score=0.9,
        )
        
        assert result.hypothesis_id == "test_hyp_001"
        assert result.success is True
        assert result.metrics["accuracy"] == 0.85

    def test_breakthrough_discovery_creation(self):
        """Test BreakthroughDiscovery data class."""
        breakthrough = BreakthroughDiscovery(
            id="breakthrough_001",
            type=BreakthroughType.ALGORITHMIC,
            description="Test breakthrough",
            significance_score=0.95,
            validation_status="preliminary",
            practical_applications=["Application 1"],
            theoretical_implications=["Implication 1"],
            experimental_evidence={"accuracy": 0.98},
            publication_potential="high",
            commercial_value=1000000.0,
        )
        
        assert breakthrough.type == BreakthroughType.ALGORITHMIC
        assert breakthrough.significance_score == 0.95
        assert breakthrough.commercial_value == 1000000.0


@pytest.mark.asyncio
async def test_autonomous_research_main():
    """Test the main function of the autonomous research module."""
    from src.next_gen_autonomous_research import main
    
    results = await main()
    
    assert isinstance(results, dict)
    assert "patterns_discovered" in results
    assert "theories_generated" in results
    assert "algorithms_created" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])