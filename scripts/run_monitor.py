import os
import sys
import argparse

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitor_performance import monitor_and_retrain


def main():
    """Evaluate current model performance and retrain if necessary."""
    parser = argparse.ArgumentParser(description="Monitor model and retrain if accuracy drops")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Accuracy threshold below which to retrain. Overrides CHURN_THRESHOLD env var.",
    )
    args = parser.parse_args()

    monitor_and_retrain(threshold=args.threshold)


if __name__ == '__main__':
    main()
