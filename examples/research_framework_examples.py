"""
Research Framework Examples and Real-time Benchmarking.

This module provides comprehensive examples demonstrating all novel research
frameworks with real-time benchmarking and comparative studies for 
academic publication and industrial deployment.

Key Features:
- End-to-end examples for all 4 research frameworks
- Real-time performance benchmarking
- Statistical significance testing
- Comparative analysis with baselines
- Visualization and reporting
- Publication-ready experimental results
"""

import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import joblib
import mlflow
import mlflow.sklearn

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal_discovery_framework import (
    CausalGraphNeuralNetwork, 
    CausalDiscoveryConfig,
    run_causal_discovery_experiment
)
from src.temporal_graph_networks import (
    TemporalGraphNeuralNetwork,
    TemporalGraphConfig, 
    run_temporal_graph_experiment
)
from src.multimodal_fusion_framework import (
    MultiModalFusionNetwork,
    MultiModalConfig,
    run_multimodal_fusion_experiment,
    create_synthetic_multimodal_data
)
from src.uncertainty_aware_ensembles import (
    UncertaintyAwareEnsemble,
    UncertaintyConfig,
    run_uncertainty_experiment
)
from src.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_comprehensive_dataset(n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate comprehensive synthetic dataset for all research frameworks."""
    np.random.seed(42)
    logger.info(f"Generating comprehensive dataset with {n_samples} samples")
    
    # Generate core features
    data = {
        'customer_id': [f'C{i:06d}' for i in range(n_samples)],
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.4, 0.3]),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'DeviceProtection': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'StreamingTV': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
        'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples)
    }
    
    # Add temporal features
    start_date = pd.Timestamp('2022-01-01')
    data['timestamp'] = [start_date + pd.Timedelta(days=np.random.randint(0, 500)) 
                        for _ in range(n_samples)]
    
    # Add derived features for causal relationships
    data['ServiceComplexity'] = (
        (data['InternetService'] != 'No').astype(int) +
        (data['PhoneService'] == 'Yes').astype(int) +
        (data['StreamingTV'] == 'Yes').astype(int) +
        (data['StreamingMovies'] == 'Yes').astype(int)
    )
    
    data['SecurityServices'] = (
        (data['OnlineSecurity'] == 'Yes').astype(int) +
        (data['OnlineBackup'] == 'Yes').astype(int) +
        (data['DeviceProtection'] == 'Yes').astype(int) +
        (data['TechSupport'] == 'Yes').astype(int)
    )
    
    X = pd.DataFrame(data)
    
    # Generate realistic target variable
    # Create complex relationships for churn prediction
    churn_logits = (
        -2.0 +  # Base intercept
        0.8 * (X['Contract'] == 'Month-to-month').astype(int) +
        0.6 * (X['MonthlyCharges'] > 80).astype(int) +
        -0.4 * (X['tenure'] > 24).astype(int) +
        0.5 * (X['SeniorCitizen'] == 1).astype(int) +
        0.3 * (X['InternetService'] == 'Fiber optic').astype(int) +
        -0.3 * (X['SecurityServices'] > 2).astype(int) +
        0.4 * (X['PaymentMethod'] == 'Electronic check').astype(int) +
        -0.2 * (X['Partner'] == 'Yes').astype(int) +
        np.random.normal(0, 0.3, n_samples)  # Random noise
    )
    
    churn_prob = 1 / (1 + np.exp(-churn_logits))
    y = pd.Series((np.random.random(n_samples) < churn_prob).astype(int), name='churn')
    
    logger.info(f"Generated dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Churn rate: {y.mean():.3f} ({y.sum()} churned customers)")
    
    return X, y


class ResearchFrameworkBenchmark:
    """
    Comprehensive benchmarking system for all research frameworks.
    
    This class provides real-time performance comparison and statistical
    analysis of novel ML approaches against established baselines.
    """
    
    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.X, self.y = generate_comprehensive_dataset(n_samples)
        self.results = {}
        self.timing_results = {}
        
        # Initialize baseline models
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        self.baseline_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        logger.info("ResearchFrameworkBenchmark initialized")
    
    def benchmark_causal_discovery(self) -> Dict[str, Any]:
        """Benchmark causal discovery framework."""
        logger.info("🧬 Benchmarking Causal Discovery Framework")
        start_time = time.time()
        
        # Run causal experiment
        config = CausalDiscoveryConfig(
            significance_level=0.01,
            bootstrap_samples=50,
            max_iterations=500
        )
        
        causal_results = run_causal_discovery_experiment(self.X, self.y, config)
        
        # Extract key metrics
        causal_performance = causal_results['performance']
        
        benchmark_time = time.time() - start_time
        self.timing_results['causal'] = benchmark_time
        
        logger.info(f"Causal Discovery Benchmark completed in {benchmark_time:.2f} seconds")
        
        return {
            'framework': 'Causal Discovery',
            'performance': causal_performance,
            'improvement_over_baseline': causal_performance['improvement']['accuracy_improvement'],
            'statistical_significance': causal_performance['improvement']['statistical_significance'],
            'is_significant': causal_performance['improvement']['is_significant'],
            'benchmark_time': benchmark_time,
            'causal_edges_discovered': causal_results.get('causal_insights', {}).get('num_causal_edges', 0)
        }
    
    def benchmark_temporal_graphs(self) -> Dict[str, Any]:
        """Benchmark temporal graph neural networks."""
        logger.info("⏰ Benchmarking Temporal Graph Neural Networks")
        start_time = time.time()
        
        # Run temporal experiment
        config = TemporalGraphConfig(
            time_window_days=30,
            sequence_length=10,
            embedding_dim=32,
            max_epochs=50
        )
        
        temporal_results = run_temporal_graph_experiment(self.X, self.y, config)
        
        # Extract key metrics
        temporal_performance = temporal_results['performance']
        
        benchmark_time = time.time() - start_time
        self.timing_results['temporal'] = benchmark_time
        
        logger.info(f"Temporal Graph Benchmark completed in {benchmark_time:.2f} seconds")
        
        return {
            'framework': 'Temporal Graph Networks',
            'performance': temporal_performance,
            'improvement_vs_lr': temporal_performance['improvements']['vs_logistic_regression']['accuracy_improvement'],
            'improvement_vs_rf': temporal_performance['improvements']['vs_random_forest']['accuracy_improvement'],
            'significance_lr': temporal_performance['improvements']['vs_logistic_regression']['statistical_significance'],
            'significance_rf': temporal_performance['improvements']['vs_random_forest']['statistical_significance'],
            'is_significant': (temporal_performance['improvements']['vs_logistic_regression']['is_significant'] or
                             temporal_performance['improvements']['vs_random_forest']['is_significant']),
            'benchmark_time': benchmark_time
        }
    
    def benchmark_multimodal_fusion(self) -> Dict[str, Any]:
        """Benchmark multi-modal fusion framework."""
        logger.info("🎭 Benchmarking Multi-Modal Fusion Framework")
        start_time = time.time()
        
        # Generate multi-modal data
        text_data, behavioral_data = create_synthetic_multimodal_data(self.X, self.y)
        
        # Run multi-modal experiment
        config = MultiModalConfig(
            text_max_features=5000,
            sequence_length=15,
            fusion_strategy="attention",
            hidden_dims=[256, 128]
        )
        
        multimodal_results = run_multimodal_fusion_experiment(
            self.X, self.y, text_data, behavioral_data, config
        )
        
        # Extract key metrics
        multimodal_performance = multimodal_results['performance']
        
        benchmark_time = time.time() - start_time
        self.timing_results['multimodal'] = benchmark_time
        
        logger.info(f"Multi-Modal Fusion Benchmark completed in {benchmark_time:.2f} seconds")
        
        return {
            'framework': 'Multi-Modal Fusion',
            'performance': multimodal_performance,
            'improvement_vs_gb': multimodal_performance['improvements']['vs_gradient_boosting']['accuracy_improvement'],
            'improvement_vs_rf': multimodal_performance['improvements']['vs_random_forest']['accuracy_improvement'],
            'improvement_vs_lr': multimodal_performance['improvements']['vs_logistic_regression']['accuracy_improvement'],
            'best_improvement': max([
                multimodal_performance['improvements']['vs_gradient_boosting']['accuracy_improvement'],
                multimodal_performance['improvements']['vs_random_forest']['accuracy_improvement'],
                multimodal_performance['improvements']['vs_logistic_regression']['accuracy_improvement']
            ]),
            'is_significant': any([
                multimodal_performance['improvements']['vs_gradient_boosting']['is_significant'],
                multimodal_performance['improvements']['vs_random_forest']['is_significant'],
                multimodal_performance['improvements']['vs_logistic_regression']['is_significant']
            ]),
            'benchmark_time': benchmark_time
        }
    
    def benchmark_uncertainty_quantification(self) -> Dict[str, Any]:
        """Benchmark uncertainty-aware ensembles."""
        logger.info("🎯 Benchmarking Uncertainty-Aware Ensembles")
        start_time = time.time()
        
        # Run uncertainty experiment
        config = UncertaintyConfig(
            n_estimators=8,
            ensemble_methods=['rf', 'gb', 'lr'],
            n_monte_carlo_samples=50,
            confidence_threshold=0.8
        )
        
        uncertainty_results = run_uncertainty_experiment(self.X, self.y, config)
        
        # Extract key metrics
        uncertainty_performance = uncertainty_results['performance']
        
        benchmark_time = time.time() - start_time
        self.timing_results['uncertainty'] = benchmark_time
        
        logger.info(f"Uncertainty Quantification Benchmark completed in {benchmark_time:.2f} seconds")
        
        return {
            'framework': 'Uncertainty-Aware Ensembles',
            'performance': uncertainty_performance,
            'improvement_over_baseline': uncertainty_performance['improvement']['accuracy_improvement'],
            'statistical_significance': uncertainty_performance['improvement']['statistical_significance'],
            'is_significant': uncertainty_performance['improvement']['is_significant'],
            'calibration_metrics': uncertainty_results.get('calibration', {}),
            'expected_calibration_error': uncertainty_results.get('calibration', {}).get('expected_calibration_error', 0),
            'brier_score': uncertainty_results.get('calibration', {}).get('brier_score', 0),
            'benchmark_time': benchmark_time
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all research frameworks."""
        logger.info("🚀 Starting Comprehensive Research Framework Benchmark")
        
        # Start MLflow experiment
        mlflow.set_experiment("Research_Framework_Benchmark")
        
        with mlflow.start_run(run_name=f"comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log dataset info
            mlflow.log_param("n_samples", self.n_samples)
            mlflow.log_param("n_features", len(self.X.columns))
            mlflow.log_param("churn_rate", float(self.y.mean()))
            
            benchmark_results = {}
            
            # Benchmark each framework
            try:
                benchmark_results['causal'] = self.benchmark_causal_discovery()
            except Exception as e:
                logger.error(f"Causal discovery benchmark failed: {e}")
                benchmark_results['causal'] = {'framework': 'Causal Discovery', 'error': str(e)}
            
            try:
                benchmark_results['temporal'] = self.benchmark_temporal_graphs()
            except Exception as e:
                logger.error(f"Temporal graph benchmark failed: {e}")
                benchmark_results['temporal'] = {'framework': 'Temporal Graph Networks', 'error': str(e)}
            
            try:
                benchmark_results['multimodal'] = self.benchmark_multimodal_fusion()
            except Exception as e:
                logger.error(f"Multi-modal benchmark failed: {e}")
                benchmark_results['multimodal'] = {'framework': 'Multi-Modal Fusion', 'error': str(e)}
            
            try:
                benchmark_results['uncertainty'] = self.benchmark_uncertainty_quantification()
            except Exception as e:
                logger.error(f"Uncertainty benchmark failed: {e}")
                benchmark_results['uncertainty'] = {'framework': 'Uncertainty-Aware Ensembles', 'error': str(e)}
            
            # Generate comprehensive report
            report = self.generate_benchmark_report(benchmark_results)
            
            # Log summary metrics to MLflow
            for framework, results in benchmark_results.items():
                if 'error' not in results:
                    if 'improvement_over_baseline' in results:
                        mlflow.log_metric(f"{framework}_improvement", results['improvement_over_baseline'])
                    if 'benchmark_time' in results:
                        mlflow.log_metric(f"{framework}_time", results['benchmark_time'])
                    if 'is_significant' in results:
                        mlflow.log_metric(f"{framework}_significant", int(results['is_significant']))
            
            # Save report
            report_path = f"/tmp/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact(report_path)
            
            return report
    
    def generate_benchmark_report(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        logger.info("📊 Generating Benchmark Report")
        
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': self.n_samples,
                'n_features': len(self.X.columns),
                'churn_rate': float(self.y.mean()),
                'feature_names': list(self.X.columns)
            },
            'framework_results': benchmark_results,
            'timing_analysis': self.timing_results,
            'summary': {}
        }
        
        # Analyze results
        successful_frameworks = [name for name, results in benchmark_results.items() 
                               if 'error' not in results]
        
        if successful_frameworks:
            # Find best performing framework
            improvements = {}
            for name in successful_frameworks:
                results = benchmark_results[name]
                if 'improvement_over_baseline' in results:
                    improvements[name] = results['improvement_over_baseline']
                elif 'best_improvement' in results:
                    improvements[name] = results['best_improvement']
            
            if improvements:
                best_framework = max(improvements.keys(), key=lambda x: improvements[x])
                best_improvement = improvements[best_framework]
                
                report['summary']['best_framework'] = best_framework
                report['summary']['best_improvement'] = best_improvement
                report['summary']['best_framework_name'] = benchmark_results[best_framework]['framework']
            
            # Count significant improvements
            significant_frameworks = [
                name for name in successful_frameworks 
                if benchmark_results[name].get('is_significant', False)
            ]
            
            report['summary']['significant_frameworks'] = len(significant_frameworks)
            report['summary']['significant_framework_names'] = [
                benchmark_results[name]['framework'] for name in significant_frameworks
            ]
            
            # Total benchmark time
            report['summary']['total_benchmark_time'] = sum(self.timing_results.values())
            
            # Success rate
            report['summary']['success_rate'] = len(successful_frameworks) / len(benchmark_results)
            
        return report
    
    def visualize_benchmark_results(self, benchmark_results: Dict[str, Any], 
                                  save_path: Optional[str] = None) -> None:
        """Create comprehensive visualizations of benchmark results."""
        try:
            logger.info("📈 Creating Benchmark Visualizations")
            
            # Extract data for plotting
            frameworks = []
            improvements = []
            times = []
            significances = []
            
            for name, results in benchmark_results.items():
                if 'error' not in results:
                    frameworks.append(results['framework'])
                    
                    if 'improvement_over_baseline' in results:
                        improvements.append(results['improvement_over_baseline'])
                    elif 'best_improvement' in results:
                        improvements.append(results['best_improvement'])
                    else:
                        improvements.append(0)
                    
                    times.append(results.get('benchmark_time', 0))
                    significances.append(results.get('is_significant', False))
            
            if not frameworks:
                logger.warning("No successful frameworks to visualize")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Research Framework Benchmark Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Performance Improvements
            colors = ['green' if sig else 'orange' for sig in significances]
            bars1 = axes[0, 0].bar(frameworks, improvements, color=colors, alpha=0.7)
            axes[0, 0].set_title('Accuracy Improvement Over Baselines', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy Improvement')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, improvement in zip(bars1, improvements):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{improvement:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Statistically Significant'),
                             Patch(facecolor='orange', alpha=0.7, label='Not Significant')]
            axes[0, 0].legend(handles=legend_elements)
            
            # Plot 2: Execution Times
            bars2 = axes[0, 1].bar(frameworks, times, color='skyblue', alpha=0.7)
            axes[0, 1].set_title('Benchmark Execution Time', fontweight='bold')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, time_val in zip(bars2, times):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Performance vs Time Trade-off
            scatter = axes[1, 0].scatter(times, improvements, c=significances, 
                                       cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
            axes[1, 0].set_title('Performance vs. Execution Time Trade-off', fontweight='bold')
            axes[1, 0].set_xlabel('Execution Time (seconds)')
            axes[1, 0].set_ylabel('Accuracy Improvement')
            axes[1, 0].grid(alpha=0.3)
            
            # Add framework labels
            for i, framework in enumerate(frameworks):
                axes[1, 0].annotate(framework.split()[0], (times[i], improvements[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot 4: Summary Statistics
            stats_data = {
                'Total Frameworks': len(frameworks),
                'Significant Improvements': sum(significances),
                'Average Improvement': np.mean(improvements),
                'Best Improvement': max(improvements) if improvements else 0,
                'Total Time': sum(times)
            }
            
            stats_names = list(stats_data.keys())
            stats_values = list(stats_data.values())
            
            bars4 = axes[1, 1].bar(range(len(stats_names)), stats_values, 
                                  color=plt.cm.viridis(np.linspace(0, 1, len(stats_names))), alpha=0.7)
            axes[1, 1].set_title('Benchmark Summary Statistics', fontweight='bold')
            axes[1, 1].set_xticks(range(len(stats_names)))
            axes[1, 1].set_xticklabels(stats_names, rotation=45, ha='right')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars4, stats_values)):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(stats_values) * 0.01,
                               f'{value:.2f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Benchmark visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")


def main():
    """Run comprehensive research framework examples and benchmarking."""
    print("🔬 Research Framework Examples and Benchmarking")
    print("=" * 60)
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize benchmark
    benchmark = ResearchFrameworkBenchmark(n_samples=1500)
    
    # Run comprehensive benchmark
    print("\n🚀 Starting Comprehensive Benchmark...")
    results = benchmark.run_comprehensive_benchmark()
    
    # Display results summary
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    if 'summary' in results:
        summary = results['summary']
        
        if 'best_framework' in summary:
            print(f"🏆 Best Framework: {summary['best_framework_name']}")
            print(f"   Accuracy Improvement: {summary['best_improvement']:.4f}")
        
        if 'significant_frameworks' in summary:
            print(f"✅ Significant Improvements: {summary['significant_frameworks']}/4 frameworks")
            if summary['significant_framework_names']:
                print(f"   Frameworks: {', '.join(summary['significant_framework_names'])}")
        
        if 'total_benchmark_time' in summary:
            print(f"⏱️  Total Benchmark Time: {summary['total_benchmark_time']:.2f} seconds")
        
        if 'success_rate' in summary:
            print(f"✅ Success Rate: {summary['success_rate']:.1%}")
    
    # Print detailed results for each framework
    print(f"\n📋 DETAILED RESULTS")
    print("=" * 60)
    
    for framework_key, framework_results in results['framework_results'].items():
        if 'error' not in framework_results:
            print(f"\n{framework_results['framework']}:")
            
            if 'improvement_over_baseline' in framework_results:
                improvement = framework_results['improvement_over_baseline']
                significant = framework_results.get('is_significant', False)
                print(f"   Accuracy Improvement: {improvement:.4f} {'✅' if significant else '⚠️'}")
                
            if 'best_improvement' in framework_results:
                improvement = framework_results['best_improvement']
                significant = framework_results.get('is_significant', False)
                print(f"   Best Improvement: {improvement:.4f} {'✅' if significant else '⚠️'}")
            
            if 'benchmark_time' in framework_results:
                print(f"   Execution Time: {framework_results['benchmark_time']:.2f}s")
            
            # Special metrics for each framework
            if framework_key == 'causal' and 'causal_edges_discovered' in framework_results:
                print(f"   Causal Edges: {framework_results['causal_edges_discovered']}")
            
            if framework_key == 'uncertainty':
                if 'expected_calibration_error' in framework_results:
                    ece = framework_results['expected_calibration_error']
                    print(f"   Calibration Error: {ece:.4f}")
                if 'brier_score' in framework_results:
                    brier = framework_results['brier_score']
                    print(f"   Brier Score: {brier:.4f}")
        else:
            print(f"\n{framework_results['framework']}: ❌ Failed")
            print(f"   Error: {framework_results['error']}")
    
    # Create visualizations
    print(f"\n📈 Creating Benchmark Visualizations...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_path = f"/tmp/research_benchmark_viz_{timestamp}.png"
    benchmark.visualize_benchmark_results(results['framework_results'], viz_path)
    
    print(f"\n🎉 Research Framework Benchmark Complete!")
    print(f"📄 Full results available in MLflow")
    print(f"📊 Visualization saved to: {viz_path}")
    
    return results


if __name__ == "__main__":
    results = main()