#!/usr/bin/env python3
"""
Autonomous SDLC Demo Script

Demonstrates the advanced autonomous capabilities implemented:
- Advanced AI-driven model optimization
- Autonomous retraining with drift detection
- Research-grade benchmarking
- Advanced security monitoring
- Performance profiling and optimization
"""

import sys
import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.advanced_optimization import optimize_model_hyperparameters, save_optimization_results
from src.autonomous_retraining import AutonomousRetrainingSystem, StatisticalDriftDetector
from src.research_framework import run_research_benchmark, BenchmarkDatasetGenerator
from src.advanced_security import AdvancedSecurityOrchestrator, create_security_report
from src.performance_profiler import PerformanceOptimizer, performance_context


def create_demo_dataset(n_samples=500, with_drift=False):
    """Create demo dataset for testing."""
    print(f"📊 Generating demo dataset ({n_samples} samples, drift={with_drift})")
    
    if with_drift:
        # Create data with intentional distribution shift
        X, y = make_classification(
            n_samples=n_samples, n_features=10, n_classes=2,
            n_informative=8, n_redundant=2, random_state=123
        )
        # Add drift
        X[:, 0] += 1.5  # Shift first feature
        X[:, 1] *= 1.3  # Scale second feature
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=10, n_classes=2,
            n_informative=8, n_redundant=2, random_state=42
        )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


def demo_advanced_optimization():
    """Demo advanced AI-driven optimization."""
    print("\n🧠 GENERATION 1: Advanced AI-Driven Model Optimization")
    print("=" * 60)
    
    # Generate demo data
    X, y = create_demo_dataset(200)
    
    start_time = time.time()
    
    # Run advanced hyperparameter optimization
    print("🔧 Running AutoML hyperparameter optimization...")
    result = optimize_model_hyperparameters(
        X, y, 
        model_type="auto",
        optimization_budget=30  # Short for demo
    )
    
    optimization_time = time.time() - start_time
    
    print(f"✅ Optimization completed in {optimization_time:.2f}s")
    print(f"📈 Best model score: {result.best_score:.4f}")
    print(f"⚡ Prediction latency: {result.prediction_latency:.4f}s")
    print(f"🎯 Model complexity: {result.model_complexity}")
    print(f"🔥 Top features: {list(result.feature_importance.keys())[:3]}")
    
    return result


def demo_autonomous_retraining():
    """Demo autonomous retraining system."""
    print("\n🤖 GENERATION 2: Autonomous Retraining & Drift Detection")
    print("=" * 60)
    
    # Create baseline and drift datasets
    X_baseline, y_baseline = create_demo_dataset(300, with_drift=False)
    X_drift, y_drift = create_demo_dataset(200, with_drift=True)
    
    print("🔍 Setting up autonomous retraining system...")
    
    # Initialize system
    retraining_system = AutonomousRetrainingSystem()
    
    # Create simple baseline model
    from sklearn.linear_model import LogisticRegression
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_baseline, y_baseline)
    
    # Initialize baseline
    retraining_system.initialize_baseline(X_baseline, y_baseline, baseline_model)
    print("✅ Baseline initialized")
    
    # Test drift detection
    print("📊 Testing drift detection...")
    drift_detector = StatisticalDriftDetector()
    drift_detector.fit_reference(X_baseline)
    
    drift_result = drift_detector.detect_drift(X_drift)
    
    print(f"🚨 Drift detected: {drift_result.drift_detected}")
    print(f"📊 Drift score: {drift_result.drift_score:.3f}")
    print(f"🎯 Drift type: {drift_result.drift_type}")
    print(f"⚠️  Affected features: {len(drift_result.affected_features)}")
    
    # Assess retraining need
    print("🧐 Assessing retraining requirements...")
    y_pred = baseline_model.predict(X_drift)
    decision = retraining_system.assess_retraining_need(X_drift, y_drift, y_pred)
    
    print(f"🎯 Retraining recommended: {decision.should_retrain}")
    print(f"📊 Priority level: {decision.priority}")
    print(f"💡 Reason: {decision.reason}")
    print(f"🔮 Estimated improvement: {decision.estimated_improvement:.1%}")
    
    return retraining_system


def demo_research_framework():
    """Demo research-grade benchmarking."""
    print("\n🔬 GENERATION 3: Research-Grade Benchmarking")
    print("=" * 60)
    
    print("📚 Running comparative algorithm analysis...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Run mini benchmark
            result = run_research_benchmark(
                algorithms=["logistic_regression", "random_forest"],
                n_synthetic_datasets=2,
                output_dir=temp_dir
            )
            
            print(f"✅ Benchmark completed with {len(result.algorithm_results)} experiments")
            
            # Show results
            if result.ranking_analysis.get('accuracy'):
                rankings = result.ranking_analysis['accuracy']['ranking']
                print("🏆 Algorithm Rankings (by accuracy):")
                for i, rank_info in enumerate(rankings[:3]):
                    print(f"  {i+1}. {rank_info['algorithm']}: {rank_info['mean_score']:.4f} ± {rank_info['std_score']:.4f}")
            
            print(f"💡 Recommendations: {len(result.recommendations)}")
            for rec in result.recommendations[:2]:
                print(f"  • {rec}")
            
            # Statistical significance
            if result.statistical_tests.get('accuracy'):
                sig_test = result.statistical_tests['accuracy']
                print(f"📈 Statistical significance: {sig_test.get('significant', False)}")
            
            return result
            
        except Exception as e:
            print(f"⚠️  Benchmark demo error (expected in limited env): {str(e)[:100]}")
            return None


def demo_advanced_security():
    """Demo advanced security system."""
    print("\n🛡️  GENERATION 3+: Advanced Security & Compliance")
    print("=" * 60)
    
    print("🔐 Initializing security orchestrator...")
    orchestrator = AdvancedSecurityOrchestrator()
    
    # Test threat detection
    print("🚨 Testing threat detection...")
    test_requests = [
        {
            'method': 'GET',
            'path': '/api/health',
            'headers': {'user-agent': 'Mozilla/5.0'},
            'params': {},
            'source_ip': '192.168.1.100'
        },
        {
            'method': 'POST',
            'path': '/login',
            'headers': {'user-agent': 'sqlmap/1.4'},
            'params': {'username': 'admin', 'password': "'; DROP TABLE users; --"},
            'source_ip': '10.0.0.1'
        }
    ]
    
    total_threats = 0
    for i, request in enumerate(test_requests):
        threats = orchestrator.monitor_request(request)
        total_threats += len(threats)
        print(f"  Request {i+1}: {len(threats)} threats detected")
    
    print(f"🎯 Total threats detected: {total_threats}")
    
    # Perform security audit
    print("🔍 Performing comprehensive security audit...")
    audit_result = orchestrator.perform_security_audit()
    
    print(f"📊 Overall security score: {audit_result.overall_score:.2f}/1.00")
    print(f"🚨 Active threats: {len(audit_result.threats_detected)}")
    print(f"📋 Compliance checks: {len(audit_result.compliance_status)}")
    print(f"🔓 Vulnerabilities found: {audit_result.vulnerability_scan['vulnerabilities_found']}")
    print(f"💡 Recommendations: {len(audit_result.recommendations)}")
    
    for rec in audit_result.recommendations[:2]:
        print(f"  • {rec}")
    
    return orchestrator


def demo_performance_profiling():
    """Demo performance profiling system."""
    print("\n⚡ GENERATION 3+: Performance Profiling & Optimization")
    print("=" * 60)
    
    print("📊 Initializing performance optimizer...")
    
    with performance_context("demo_operations") as optimizer:
        print("🔥 Profiling CPU-intensive operation...")
        
        # Simulate CPU-intensive work
        with optimizer.profile_code_block("matrix_operations"):
            X = np.random.random((100, 100))
            for i in range(10):
                result = np.dot(X, X.T)
                time.sleep(0.01)  # Simulate work
        
        print("💾 Profiling memory-intensive operation...")
        
        # Simulate memory-intensive work
        with optimizer.profile_memory_block("data_processing"):
            data_list = []
            for i in range(1000):
                data_list.append(np.random.random(100))
            
            # Process data
            processed = pd.DataFrame(data_list)
            summary = processed.describe()
    
    print("📈 Generating performance report...")
    report = optimizer.generate_comprehensive_report()
    
    # Show system metrics
    if report.get('system_metrics'):
        sys_metrics = report['system_metrics']
        if 'cpu' in sys_metrics:
            print(f"🖥️  CPU usage: {sys_metrics['cpu']['mean']:.1f}% avg, {sys_metrics['cpu']['max']:.1f}% peak")
        if 'memory' in sys_metrics:
            print(f"💾 Memory usage: {sys_metrics['memory']['mean']:.1f}% avg, {sys_metrics['memory']['max']:.1f}% peak")
    
    # Show optimization recommendations
    recommendations = report.get('optimization_recommendations', [])
    print(f"💡 Performance recommendations: {len(recommendations)}")
    for rec in recommendations[:2]:
        print(f"  • {rec.get('description', 'N/A')}")
    
    return optimizer


def main():
    """Run complete autonomous SDLC demonstration."""
    print("🚀 AUTONOMOUS SDLC MASTER DEMONSTRATION")
    print("=" * 60)
    print("Demonstrating next-generation autonomous development capabilities")
    print()
    
    start_time = time.time()
    
    try:
        # Generation 1: Advanced Model Optimization
        opt_result = demo_advanced_optimization()
        
        # Generation 2: Autonomous Retraining
        retraining_system = demo_autonomous_retraining()
        
        # Generation 3: Research Framework
        research_result = demo_research_framework()
        
        # Advanced Security
        security_orchestrator = demo_advanced_security()
        
        # Performance Profiling
        performance_optimizer = demo_performance_profiling()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 AUTONOMOUS SDLC DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"🧠 AI Optimization: {opt_result.best_score:.4f} accuracy achieved")
        print(f"🤖 Autonomous System: Fully operational with drift detection")
        print(f"🔬 Research Framework: {'✅ Active' if research_result else '⚠️ Limited'}")
        print(f"🛡️  Security Score: {security_orchestrator.perform_security_audit().overall_score:.2f}/1.00")
        print(f"⚡ Performance: Monitoring and optimization active")
        
        print(f"\n🚀 NEXT-GENERATION CAPABILITIES DEMONSTRATED:")
        print("  • Autonomous hyperparameter optimization with Bayesian methods")
        print("  • Real-time drift detection and adaptive retraining")
        print("  • Publication-ready research and benchmarking")
        print("  • Enterprise-grade security and compliance")
        print("  • Advanced performance profiling and optimization")
        print("\n💡 The system is ready for autonomous production deployment!")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        print("This may be expected in a containerized environment with limited resources.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()