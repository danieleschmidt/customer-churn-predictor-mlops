"""
Quick Start Examples for Research Frameworks.

This script provides simple, focused examples for each research framework
that can be run independently to demonstrate key capabilities.
"""

import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def create_sample_data(n_samples=200):
    """Create sample dataset for examples."""
    np.random.seed(42)
    
    data = {
        'customer_id': [f'C{i:04d}' for i in range(n_samples)],
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 100, n_samples),
        'TotalCharges': np.random.uniform(100, 5000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    }
    
    X = pd.DataFrame(data)
    
    # Generate target
    churn_prob = (
        0.3 * (X['tenure'] < 12).astype(int) +
        0.2 * (X['MonthlyCharges'] > 70).astype(int) +
        0.3 * (X['Contract'] == 'Month-to-month').astype(int) +
        0.1 * np.random.random(n_samples)
    )
    
    y = pd.Series((churn_prob > 0.4).astype(int), name='churn')
    
    return X, y

def example_causal_discovery():
    """Example: Causal Discovery Framework."""
    print("\n🧬 CAUSAL DISCOVERY EXAMPLE")
    print("-" * 40)
    
    try:
        from src.causal_discovery_framework import CausalGraphNeuralNetwork, CausalDiscoveryConfig
        
        # Create sample data
        X, y = create_sample_data(300)
        
        print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Churn rate: {y.mean():.3f}")
        
        # Configure causal discovery
        config = CausalDiscoveryConfig(
            significance_level=0.05,
            max_iterations=200
        )
        
        # Create and train model
        print("\nTraining causal model...")
        start_time = time.time()
        
        causal_model = CausalGraphNeuralNetwork(config)
        causal_model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        sample_customer = X.iloc[[0]]
        prediction = causal_model.predict(sample_customer)[0]
        probability = causal_model.predict_proba(sample_customer)[0, 1]
        
        print(f"\nSample Prediction:")
        print(f"  Customer ID: {sample_customer['customer_id'].iloc[0]}")
        print(f"  Prediction: {'Churn' if prediction else 'No Churn'}")
        print(f"  Probability: {probability:.3f}")
        
        # Get causal insights
        causal_importance = causal_model.get_causal_importance()
        if causal_importance:
            print(f"\nTop Causal Features:")
            sorted_importance = sorted(causal_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance[:5]:
                print(f"  {feature}: {importance:.3f}")
        
        causal_edges = len(causal_model.causal_graph.edges) if causal_model.causal_graph else 0
        print(f"\nCausal Relationships Discovered: {causal_edges}")
        
        return True
        
    except Exception as e:
        print(f"❌ Causal Discovery Example failed: {e}")
        return False

def example_temporal_graphs():
    """Example: Temporal Graph Neural Networks."""
    print("\n⏰ TEMPORAL GRAPH EXAMPLE")
    print("-" * 40)
    
    try:
        from src.temporal_graph_networks import TemporalGraphNeuralNetwork, TemporalGraphConfig
        
        # Create sample data
        X, y = create_sample_data(250)
        
        print(f"Dataset: {len(X)} samples with temporal data")
        print(f"Date range: {X['timestamp'].min()} to {X['timestamp'].max()}")
        
        # Configure temporal model
        config = TemporalGraphConfig(
            time_window_days=30,
            sequence_length=5,
            embedding_dim=32
        )
        
        # Create and train model
        print("\nTraining temporal graph model...")
        start_time = time.time()
        
        temporal_model = TemporalGraphNeuralNetwork(config)
        temporal_model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        sample_customer = X.iloc[[0]]
        prediction = temporal_model.predict(sample_customer)[0]
        probability = temporal_model.predict_proba(sample_customer)[0, 1]
        
        print(f"\nSample Prediction:")
        print(f"  Customer ID: {sample_customer['customer_id'].iloc[0]}")
        print(f"  Prediction: {'Churn' if prediction else 'No Churn'}")
        print(f"  Probability: {probability:.3f}")
        
        # Get temporal attention
        customer_id = sample_customer['customer_id'].iloc[0]
        attention_weights = temporal_model.get_temporal_attention_weights(customer_id)
        
        if attention_weights:
            print(f"\nTemporal Attention Weights:")
            for time_step, weight in list(attention_weights.items())[:3]:
                print(f"  {time_step}: {weight:.3f}")
        
        num_windows = len(temporal_model.temporal_graph.graphs) if temporal_model.temporal_graph else 0
        print(f"\nTemporal Windows Created: {num_windows}")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal Graph Example failed: {e}")
        return False

def example_multimodal_fusion():
    """Example: Multi-Modal Fusion."""
    print("\n🎭 MULTI-MODAL FUSION EXAMPLE")
    print("-" * 40)
    
    try:
        from src.multimodal_fusion_framework import (
            MultiModalFusionNetwork, MultiModalConfig, create_synthetic_multimodal_data
        )
        
        # Create sample data
        X, y = create_sample_data(200)
        
        # Generate synthetic multi-modal data
        text_data, behavioral_data = create_synthetic_multimodal_data(X, y)
        
        print(f"Dataset: {len(X)} samples")
        print(f"Text data: {len(text_data)} customer reviews/feedback")
        print(f"Behavioral data: {len(behavioral_data)} action sequences")
        
        # Configure multi-modal model
        config = MultiModalConfig(
            text_max_features=1000,
            sequence_length=10,
            fusion_strategy="attention"
        )
        
        # Create and train model
        print("\nTraining multi-modal fusion model...")
        start_time = time.time()
        
        multimodal_model = MultiModalFusionNetwork(config)
        multimodal_model.fit(X, y, text_data, behavioral_data)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        sample_idx = 0
        sample_customer = X.iloc[[sample_idx]]
        sample_text = [text_data[sample_idx]]
        sample_behavior = [behavioral_data[sample_idx]]
        
        prediction = multimodal_model.predict(sample_customer, sample_text, sample_behavior)[0]
        probability = multimodal_model.predict_proba(sample_customer, sample_text, sample_behavior)[0, 1]
        
        print(f"\nSample Prediction:")
        print(f"  Customer ID: {sample_customer['customer_id'].iloc[0]}")
        print(f"  Prediction: {'Churn' if prediction else 'No Churn'}")
        print(f"  Probability: {probability:.3f}")
        print(f"  Text: '{sample_text[0][:60]}...'")
        print(f"  Behavior: {sample_behavior[0][:5]}...")
        
        # Get modality importance
        modality_importance = multimodal_model.get_modality_importance()
        if modality_importance:
            print(f"\nModality Importance:")
            for modality, importance in modality_importance.items():
                print(f"  {modality.title()}: {importance:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-Modal Fusion Example failed: {e}")
        return False

def example_uncertainty_quantification():
    """Example: Uncertainty-Aware Ensembles."""
    print("\n🎯 UNCERTAINTY QUANTIFICATION EXAMPLE")
    print("-" * 40)
    
    try:
        from src.uncertainty_aware_ensembles import UncertaintyAwareEnsemble, UncertaintyConfig
        
        # Create sample data
        X, y = create_sample_data(300)
        
        print(f"Dataset: {len(X)} samples")
        print(f"Features: {list(X.select_dtypes(include=[np.number]).columns)[:5]}")
        
        # Configure uncertainty model
        config = UncertaintyConfig(
            n_estimators=5,
            ensemble_methods=['rf', 'gb', 'lr'],
            n_monte_carlo_samples=20,
            confidence_threshold=0.8
        )
        
        # Create and train model
        print("\nTraining uncertainty-aware ensemble...")
        start_time = time.time()
        
        uncertainty_model = UncertaintyAwareEnsemble(config)
        uncertainty_model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions with uncertainty
        sample_customer = X.iloc[[0]]
        uncertainty_results = uncertainty_model.predict_with_uncertainty(sample_customer)
        
        prediction = uncertainty_results['predictions'][0]
        probability = uncertainty_results['probabilities'][0]
        calibrated_prob = uncertainty_results['calibrated_probabilities'][0]
        epistemic_unc = uncertainty_results['epistemic_uncertainty'][0]
        aleatoric_unc = uncertainty_results['aleatoric_uncertainty'][0]
        total_unc = uncertainty_results['total_uncertainty'][0]
        confidence_interval = uncertainty_results['confidence_intervals'][0]
        
        print(f"\nSample Prediction with Uncertainty:")
        print(f"  Customer ID: {sample_customer['customer_id'].iloc[0]}")
        print(f"  Prediction: {'Churn' if prediction else 'No Churn'}")
        print(f"  Raw Probability: {probability:.3f}")
        print(f"  Calibrated Probability: {calibrated_prob:.3f}")
        print(f"  Epistemic Uncertainty: {epistemic_unc:.3f}")
        print(f"  Aleatoric Uncertainty: {aleatoric_unc:.3f}")
        print(f"  Total Uncertainty: {total_unc:.3f}")
        print(f"  Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        
        # Check high confidence predictions
        high_conf_results = uncertainty_model.get_high_confidence_predictions(X.iloc[:10])
        confidence_score = high_conf_results['confidence_score']
        
        print(f"\nHigh Confidence Analysis (first 10 samples):")
        print(f"  Confidence Score: {confidence_score:.3f}")
        print(f"  High Confidence Predictions: {len(high_conf_results['confident_indices'])}/10")
        
        return True
        
    except Exception as e:
        print(f"❌ Uncertainty Quantification Example failed: {e}")
        return False

def run_api_examples():
    """Example: Research API Usage."""
    print("\n🚀 RESEARCH API EXAMPLES")
    print("-" * 40)
    
    try:
        import requests
        import json
        
        # Note: This assumes the API server is running
        BASE_URL = "http://localhost:8000"
        
        print("📝 Note: Make sure the API server is running with research endpoints enabled")
        print("   Run: python -m src.api --enable-research")
        print()
        
        # Example API requests (documentation only, won't run without server)
        examples = {
            "Causal Prediction": {
                "endpoint": "/research/causal/predict",
                "payload": {
                    "customer_data": {
                        "tenure": 12,
                        "MonthlyCharges": 70.0,
                        "Contract": "Month-to-month",
                        "gender": "Female"
                    }
                }
            },
            "Temporal Prediction": {
                "endpoint": "/research/temporal/predict", 
                "payload": {
                    "customer_data": [
                        {"customer_id": "C001", "timestamp": "2023-01-01", "tenure": 6},
                        {"customer_id": "C001", "timestamp": "2023-02-01", "tenure": 7}
                    ]
                }
            },
            "Multi-Modal Prediction": {
                "endpoint": "/research/multimodal/predict",
                "payload": {
                    "customer_data": {"tenure": 12, "MonthlyCharges": 70.0},
                    "text_data": "Service quality declining, considering alternatives",
                    "behavioral_data": ["login", "view_billing", "support_contact"]
                }
            },
            "Uncertainty Prediction": {
                "endpoint": "/research/uncertainty/predict",
                "payload": {
                    "customer_data": {
                        "tenure": 12,
                        "MonthlyCharges": 70.0,
                        "Contract": "Month-to-month"
                    }
                }
            },
            "Research Experiment": {
                "endpoint": "/research/experiment",
                "payload": {
                    "experiment_type": "causal",
                    "n_samples": 500
                }
            }
        }
        
        print("Example API Requests:")
        for example_name, example_info in examples.items():
            print(f"\n{example_name}:")
            print(f"  POST {BASE_URL}{example_info['endpoint']}")
            print(f"  Payload: {json.dumps(example_info['payload'], indent=4)}")
        
        print("\n📖 See the API documentation at http://localhost:8000/docs for interactive testing")
        
        return True
        
    except Exception as e:
        print(f"❌ API Examples failed: {e}")
        return False

def main():
    """Run all research framework examples."""
    print("🔬 RESEARCH FRAMEWORK EXAMPLES")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track results
    results = {}
    
    # Run individual examples
    print("Running individual framework examples...")
    
    results['causal'] = example_causal_discovery()
    results['temporal'] = example_temporal_graphs()
    results['multimodal'] = example_multimodal_fusion()
    results['uncertainty'] = example_uncertainty_quantification()
    
    # API examples (documentation)
    results['api'] = run_api_examples()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 EXAMPLES SUMMARY")
    print("=" * 50)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"✅ Successful Examples: {successful}/{total}")
    
    for example_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {example_name.replace('_', ' ').title()}: {status}")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\n🚀 Next Steps:")
        print("  1. Run comprehensive benchmark: python examples/research_framework_examples.py")
        print("  2. Start API server: python -m src.api --enable-research")
        print("  3. Test API endpoints: http://localhost:8000/docs")
        print("  4. Run production experiments with your own data")
    else:
        print(f"\n⚠️  {total - successful} examples failed. Check error messages above.")
    
    return successful == total

if __name__ == "__main__":
    success = main()