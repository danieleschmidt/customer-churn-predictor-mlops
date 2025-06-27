import argparse
import json
from src.monitor_performance import evaluate_model
from src.config import load_config


def run_evaluation(
    model_path: str | None = None,
    X_path: str | None = None,
    y_path: str | None = None,
    output=None,
    run_id=None,
    *,
    detailed: bool = False,
):
    """Evaluate the model and optionally save metrics to a JSON file."""
    cfg = load_config()
    model_path = model_path or cfg["model"]["path"]
    X_path = X_path or cfg["data"]["processed_features"]
    y_path = y_path or cfg["data"]["processed_target"]
    result = evaluate_model(
        model_path, X_path, y_path, run_id=run_id, detailed=detailed
    )
    if detailed:
        accuracy, f1, report = result
    else:
        accuracy, f1 = result
        report = None
    if output:
        metrics = {"accuracy": accuracy, "f1_score": f1}
        if report is not None:
            metrics["classification_report"] = report
        with open(output, "w") as f:
            json.dump(metrics, f)
        print(f"Saved metrics to {output}")
    if detailed:
        return accuracy, f1, report
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    cfg = load_config()
    parser.add_argument(
        "--model_path", default=cfg["model"]["path"], help="Path to model file"
    )
    parser.add_argument(
        "--X_path",
        default=cfg["data"]["processed_features"],
        help="Processed features CSV",
    )
    parser.add_argument(
        "--y_path", default=cfg["data"]["processed_target"], help="Processed target CSV"
    )
    parser.add_argument("--output", help="Optional JSON file to store metrics")
    parser.add_argument("--run_id", help="MLflow run ID to download artifacts")
    parser.add_argument(
        "--detailed", action="store_true", help="Include classification report"
    )
    args = parser.parse_args()

    result = run_evaluation(
        args.model_path,
        args.X_path,
        args.y_path,
        args.output,
        run_id=args.run_id,
        detailed=args.detailed,
    )
    if args.detailed:
        accuracy, f1, report = result
        print(json.dumps(report, indent=2))
    else:
        accuracy, f1 = result
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
