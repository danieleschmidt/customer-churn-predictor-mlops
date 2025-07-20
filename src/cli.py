import typer
from typing import Optional, Any

from scripts.run_preprocessing import run_preprocessing
from scripts.run_training import run_training
from scripts.run_evaluation import run_evaluation
from scripts.run_pipeline import run_pipeline
from src.monitor_performance import monitor_and_retrain
from scripts.run_prediction import run_predictions

app = typer.Typer(help="Customer churn prediction command-line interface")


@app.command()
def preprocess() -> None:
    """Preprocess raw data and save processed datasets."""
    run_preprocessing()


@app.command()
def train(
    x_path: Optional[str] = None,
    y_path: Optional[str] = None,
    solver: str = "liblinear",
    c: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
) -> None:
    """Train the churn model."""
    run_training(
        x_path,
        y_path,
        solver=solver,
        C=c,
        penalty=penalty,
        random_state=random_state,
        max_iter=max_iter,
        test_size=test_size,
    )


@app.command()
def evaluate(
    model_path: Optional[str] = None,
    x_path: Optional[str] = None,
    y_path: Optional[str] = None,
    run_id: Optional[str] = None,
    output: Optional[str] = None,
    detailed: bool = False,
) -> None:
    """Evaluate a trained model."""
    run_evaluation(
        model_path=model_path,
        X_path=x_path,
        y_path=y_path,
        output=output,
        run_id=run_id,
        detailed=detailed,
    )


@app.command()
def pipeline(
    raw_path: Optional[str] = None,
    solver: str = "liblinear",
    c: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
) -> None:
    """Run preprocessing, training and evaluation as one pipeline."""
    run_pipeline(
        raw_path,
        solver=solver,
        C=c,
        penalty=penalty,
        random_state=random_state,
        max_iter=max_iter,
        test_size=test_size,
    )


@app.command()
def monitor(
    threshold: Optional[float] = None,
    x_path: Optional[str] = None,
    y_path: Optional[str] = None,
    solver: str = "liblinear",
    c: float = 1.0,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 100,
    test_size: float = 0.2,
) -> None:
    """Monitor model performance and retrain if necessary."""
    kwargs = {
        "threshold": threshold,
        "solver": solver,
        "C": c,
        "penalty": penalty,
        "random_state": random_state,
        "max_iter": max_iter,
        "test_size": test_size,
    }
    if x_path is not None:
        kwargs["X_path"] = x_path
    if y_path is not None:
        kwargs["y_path"] = y_path
    clean_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
    monitor_and_retrain(**clean_kwargs)


@app.command()
def predict(
    input_csv: str,
    output_csv: str = "predictions.csv",
    run_id: Optional[str] = None,
) -> None:
    """Generate predictions for a CSV of features."""
    run_predictions(input_csv, output_csv, run_id=run_id)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
