#!/usr/bin/env python3
"""
Autonomous ML Pipeline Execution Script.

This script demonstrates the fully autonomous machine learning capabilities
of the customer churn prediction system, requiring minimal human intervention.
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.autonomous_ml_orchestrator import (
    create_autonomous_orchestrator, run_autonomous_ml_pipeline
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run autonomous ML pipeline for customer churn prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic autonomous training
  python scripts/run_autonomous_ml.py data/raw/customer_data.csv

  # Full pipeline with custom settings
  python scripts/run_autonomous_ml.py data/raw/customer_data.csv \\
    --target-column Churn \\
    --output-dir autonomous_results \\
    --continuous-learning \\
    --explainable-ai
    
  # Advanced ensemble with monitoring
  python scripts/run_autonomous_ml.py data/raw/customer_data.csv \\
    --advanced-ensemble \\
    --performance-threshold 0.85 \\
    --monitoring-interval 1800
        """
    )
    
    # Required arguments
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to training data CSV file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--target-column',
        type=str,
        default='Churn',
        help='Name of target variable column (default: Churn)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('autonomous_ml_results'),
        help='Directory for output files (default: autonomous_ml_results)'
    )
    
    parser.add_argument(
        '--continuous-learning',
        action='store_true',
        help='Enable continuous learning and monitoring'
    )
    
    parser.add_argument(
        '--advanced-ensemble',
        action='store_true',
        default=True,
        help='Use advanced ensemble methods (default: True)'
    )
    
    parser.add_argument(
        '--explainable-ai',
        action='store_true',
        default=True,
        help='Include explainable AI features (default: True)'
    )
    
    parser.add_argument(
        '--performance-threshold',
        type=float,
        default=0.8,
        help='Minimum performance threshold (default: 0.8)'
    )
    
    parser.add_argument(
        '--monitoring-interval',
        type=int,
        default=3600,
        help='Performance monitoring interval in seconds (default: 3600)'
    )
    
    parser.add_argument(
        '--max-training-time',
        type=int,
        default=1800,
        help='Maximum training time in seconds (default: 1800)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--demo-mode',
        action='store_true',
        help='Run in demonstration mode with detailed logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    if not (0.1 <= args.test_size <= 0.5):
        logger.error("Test size must be between 0.1 and 0.5")
        sys.exit(1)
    
    if not (0.5 <= args.performance_threshold <= 1.0):
        logger.error("Performance threshold must be between 0.5 and 1.0")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging for demo mode
    if args.demo_mode:
        import logging
        logging.getLogger().setLevel(logging.INFO)
        logger.info("🤖 AUTONOMOUS ML DEMONSTRATION MODE ACTIVATED")
        logger.info("=" * 60)
    
    try:
        logger.info("🚀 Starting Autonomous ML Pipeline...")
        logger.info(f"📊 Data: {data_path}")
        logger.info(f"🎯 Target: {args.target_column}")
        logger.info(f"📁 Output: {args.output_dir}")
        logger.info(f"📈 Performance Threshold: {args.performance_threshold}")
        
        # Run the autonomous ML pipeline
        result = run_autonomous_ml_pipeline(
            data_path=data_path,
            target_column=args.target_column,
            output_dir=args.output_dir
        )
        
        # Display results
        logger.info("✅ AUTONOMOUS ML PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {result.model_name}")
        logger.info(f"📊 Accuracy: {result.accuracy:.4f}")
        logger.info(f"🎯 F1 Score: {result.f1_score:.4f}")
        logger.info(f"📈 ROC-AUC: {result.roc_auc:.4f}")
        logger.info(f"⏱️  Training Time: {result.training_time:.2f} seconds")
        logger.info(f"💾 Model Size: {result.model_size_mb:.2f} MB")
        
        # Show top features
        if result.feature_importance:
            logger.info("🔍 TOP PREDICTIVE FEATURES:")
            sorted_features = sorted(
                result.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        # Performance assessment
        if result.f1_score >= args.performance_threshold:
            logger.info("🎉 MODEL MEETS PERFORMANCE REQUIREMENTS!")
        else:
            logger.warning("⚠️  Model performance below threshold. Consider retraining.")
        
        # Save detailed results
        results_file = args.output_dir / 'autonomous_training_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'result': result.__dict__,
                'configuration': {
                    'data_path': str(data_path),
                    'target_column': args.target_column,
                    'continuous_learning': args.continuous_learning,
                    'advanced_ensemble': args.advanced_ensemble,
                    'explainable_ai': args.explainable_ai,
                    'performance_threshold': args.performance_threshold,
                    'test_size': args.test_size
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Autonomous ML Pipeline Failed: {e}")
        if args.demo_mode:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)