import typer
from typing import Optional, Any

from scripts.run_preprocessing import run_preprocessing
from scripts.run_training import run_training
from scripts.run_evaluation import run_evaluation
from scripts.run_pipeline import run_pipeline
from src.monitor_performance import monitor_and_retrain
from scripts.run_prediction import run_predictions
from .validation import DEFAULT_PATH_VALIDATOR, ValidationError

app = typer.Typer(help="Customer churn prediction command-line interface")


@app.command()
def preprocess() -> None:
    """
    Preprocess raw data and save processed datasets.
    
    This command loads raw customer data, performs feature engineering,
    handles missing values, and splits the data into training and test sets.
    The processed datasets are saved for use in model training.
    
    The preprocessing includes:
    - Data cleaning and missing value imputation
    - Feature encoding (one-hot encoding for categorical variables)
    - Feature scaling for numerical variables
    - Train/test split with stratification
    
    Output files are saved to the configured data directory.
    """
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
    """
    Train the churn prediction model using logistic regression.
    
    This command trains a machine learning model to predict customer churn
    using processed feature and target datasets. The model is trained using
    logistic regression with configurable hyperparameters.
    
    Parameters
    ----------
    x_path : str, optional
        Path to the processed features CSV file. If None, uses default path.
    y_path : str, optional
        Path to the processed target CSV file. If None, uses default path.
    solver : str, default="liblinear"
        Algorithm to use in the optimization problem. Options: 'liblinear',
        'lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'.
    c : float, default=1.0
        Inverse of regularization strength; smaller values specify stronger
        regularization.
    penalty : str, default="l2"
        Penalty norm used in the penalization. Options: 'l1', 'l2', 'elasticnet'.
    random_state : int, default=42
        Random seed for reproducible results.
    max_iter : int, default=100
        Maximum number of iterations for the solver to converge.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0.0 to 1.0).
    
    Returns
    -------
    None
        The trained model is saved to disk and logged to MLflow if available.
    """
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
    """
    Evaluate a trained model's performance on test data.
    
    This command loads a trained model and evaluates its performance using
    accuracy, F1-score, and optionally a detailed classification report.
    Results can be saved to a file and logged to MLflow.
    
    Parameters
    ----------
    model_path : str, optional
        Path to the trained model file. If None, uses default path or
        downloads from MLflow using run_id.
    x_path : str, optional
        Path to the test features CSV file. If None, uses default path.
    y_path : str, optional
        Path to the test target CSV file. If None, uses default path.
    run_id : str, optional
        MLflow run ID to download model artifacts from if model_path
        is not available locally.
    output : str, optional
        Path to save evaluation results. If None, results are only logged.
    detailed : bool, default=False
        Whether to generate a detailed classification report with per-class
        precision, recall, and F1-scores.
    
    Returns
    -------
    None
        Evaluation results are printed to console and optionally saved to file.
    """
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
    """
    Execute the complete machine learning pipeline from raw data to trained model.
    
    This command runs the entire ML pipeline in sequence:
    1. Data preprocessing (cleaning, feature engineering, splitting)
    2. Model training (logistic regression with specified parameters)
    3. Model evaluation (accuracy and F1-score calculation)
    
    This is equivalent to running preprocess, train, and evaluate commands
    sequentially, but ensures consistency in data splits and parameters.
    
    Parameters
    ----------
    raw_path : str, optional
        Path to the raw customer data CSV file. If None, uses default path.
    solver : str, default="liblinear"
        Algorithm to use in the optimization problem.
    c : float, default=1.0
        Inverse of regularization strength.
    penalty : str, default="l2"
        Penalty norm used in the penalization.
    random_state : int, default=42
        Random seed for reproducible results across all pipeline steps.
    max_iter : int, default=100
        Maximum number of iterations for the solver to converge.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    
    Returns
    -------
    None
        Processed data, trained model, and evaluation results are saved.
    """
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
    """
    Monitor model performance and automatically retrain if performance degrades.
    
    This command evaluates the current model's performance against a
    configurable accuracy threshold. If the model's accuracy falls below
    the threshold, it automatically triggers retraining with the latest data.
    
    This is useful for production deployments where model performance may
    degrade over time due to data drift or changing customer behavior patterns.
    
    Parameters
    ----------
    threshold : float, optional
        Minimum accuracy threshold below which the model will be retrained.
        If None, uses the value from CHURN_THRESHOLD environment variable
        or the default threshold.
    x_path : str, optional
        Path to the current features CSV file for evaluation.
        If None, uses default path.
    y_path : str, optional
        Path to the current target CSV file for evaluation.
        If None, uses default path.
    solver : str, default="liblinear"
        Algorithm to use if retraining is triggered.
    c : float, default=1.0
        Inverse of regularization strength for retraining.
    penalty : str, default="l2"
        Penalty norm for retraining.
    random_state : int, default=42
        Random seed for retraining reproducibility.
    max_iter : int, default=100
        Maximum iterations for retraining.
    test_size : float, default=0.2
        Test split proportion for retraining.
    
    Returns
    -------
    None
        If retraining occurs, new model is saved and logged.
    """
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
    """
    Generate churn predictions for a batch of customers from a CSV file.
    
    This command loads a trained model and generates churn predictions
    for customers provided in a CSV file. Each row should contain the
    same features used during model training. The output includes both
    binary predictions (0/1) and churn probabilities.
    
    The input CSV should contain customer features in the same format
    as the training data. If a preprocessor was used during training,
    it will be automatically applied to transform the input data.
    
    Parameters
    ----------
    input_csv : str
        Path to the CSV file containing customer features for prediction.
        Each row represents one customer, columns represent features.
    output_csv : str, default="predictions.csv"
        Path where prediction results will be saved. The output includes
        customer IDs (if present), churn predictions, and churn probabilities.
    run_id : str, optional
        MLflow run ID to download the model from if it's not available
        locally. If None, uses the most recent run or local model.
    
    Returns
    -------
    None
        Predictions are saved to the specified output CSV file.
    
    Raises
    ------
    ValidationError
        If input file doesn't exist or output path is invalid.
    typer.Exit
        If validation fails, exits with code 1.
    """
    try:
        # Validate input paths before processing
        DEFAULT_PATH_VALIDATOR.validate_path(input_csv, must_exist=True)
        DEFAULT_PATH_VALIDATOR.validate_path(output_csv, allow_create=True)
        run_predictions(input_csv, output_csv, run_id=run_id)
    except ValidationError as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """
    Entry point for the customer churn prediction CLI application.
    
    This function serves as the main entry point when the CLI is invoked.
    It initializes the Typer application and processes command-line arguments
    to execute the appropriate subcommand.
    
    The CLI provides the following commands:
    - preprocess: Clean and prepare raw data for training
    - train: Train a machine learning model for churn prediction
    - evaluate: Assess model performance on test data
    - pipeline: Run the complete ML workflow end-to-end
    - monitor: Check model performance and retrain if necessary
    - predict: Generate predictions for new customer data
    
    Usage
    -----
    From command line:
        python -m src.cli <command> [options]
    
    Examples
    --------
    >>> python -m src.cli preprocess
    >>> python -m src.cli train --c 0.5 --solver liblinear
    >>> python -m src.cli predict input.csv --output predictions.csv
    """
    app()


if __name__ == "__main__":
    main()
