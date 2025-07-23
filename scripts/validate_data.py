#!/usr/bin/env python3
"""
Data validation script for customer churn dataset.

This script provides comprehensive validation of customer churn data including:
- Schema validation against expected data structure
- Business rule validation for data integrity
- Data quality checks and outlier detection
- ML-specific validation for training and prediction data

Usage:
    python scripts/validate_data.py data/raw/customer_data.csv
    python scripts/validate_data.py data/processed/processed_features.csv --for-prediction
    python scripts/validate_data.py --help
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data_validation import (
        validate_customer_data, 
        validate_training_data, 
        validate_prediction_data,
        ValidationError
    )
    from logging_config import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

logger = get_logger(__name__)


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate customer churn data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/raw/customer_data.csv
  %(prog)s data/processed/processed_features.csv --for-prediction
  %(prog)s data/raw/customer_data.csv --detailed --output validation_report.txt
        """
    )
    
    parser.add_argument(
        "data_file",
        help="Path to CSV file containing customer data"
    )
    
    parser.add_argument(
        "--for-prediction",
        action="store_true",
        help="Validate data for prediction (target variable not required)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed validation report"
    )
    
    parser.add_argument(
        "--output",
        help="Save validation report to specified file"
    )
    
    parser.add_argument(
        "--check-distribution",
        action="store_true",
        help="Check feature distributions for ML validation"
    )
    
    parser.add_argument(
        "--no-business-rules",
        action="store_true",
        help="Skip business rule validation"
    )
    
    args = parser.parse_args()
    
    logger.info("🔍 Starting data validation")
    logger.info(f"Input file: {args.data_file}")
    logger.info(f"Validation mode: {'Prediction' if args.for_prediction else 'Training'}")
    
    try:
        # Validate the data
        if args.for_prediction:
            report = validate_prediction_data(args.data_file)
        else:
            kwargs = {
                'check_distribution': args.check_distribution,
                'check_business_rules': not args.no_business_rules
            }
            report = validate_customer_data(args.data_file, **kwargs)
        
        # Generate output
        if args.detailed:
            output = report.get_detailed_report()
        else:
            output = report.get_summary()
            if not report.is_valid:
                output += "\n\n🚨 ERRORS:\n"
                for i, error in enumerate(report.errors[:5], 1):  # Show first 5 errors
                    output += f"  {i}. {error}\n"
                if len(report.errors) > 5:
                    output += f"  ... and {len(report.errors) - 5} more errors\n"
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"📄 Validation report saved to: {args.output}")
        
        print(output)
        
        # Exit with appropriate code
        if report.is_valid:
            logger.info("✅ Data validation passed")
            return 0
        else:
            logger.error("❌ Data validation failed")
            return 1
    
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        return 1
    except FileNotFoundError:
        logger.error(f"❌ File not found: {args.data_file}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())