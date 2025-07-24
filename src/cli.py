import typer
from typing import Optional, Any

from scripts.run_preprocessing import run_preprocessing
from scripts.run_training import run_training
from scripts.run_evaluation import run_evaluation
from scripts.run_pipeline import run_pipeline
from src.monitor_performance import monitor_and_retrain
from scripts.run_prediction import run_predictions
from .validation import DEFAULT_PATH_VALIDATOR, ValidationError
from .data_validation import validate_customer_data, ValidationError as DataValidationError
from .health_check import get_health_status, get_comprehensive_health, get_readiness_status
from .model_cache import get_cache_stats, invalidate_model_cache
from .metrics import get_prometheus_metrics, get_metrics_collector

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


@app.command()
def validate(
    data_file: str,
    for_prediction: bool = typer.Option(False, "--for-prediction", help="Validate data for prediction (target variable not required)"),
    detailed: bool = typer.Option(False, "--detailed", help="Generate detailed validation report"),
    output: Optional[str] = typer.Option(None, "--output", help="Save validation report to specified file"),
    check_distribution: bool = typer.Option(False, "--check-distribution", help="Check feature distributions for ML validation"),
    no_business_rules: bool = typer.Option(False, "--no-business-rules", help="Skip business rule validation")
) -> None:
    """
    Validate customer churn data with comprehensive checks.
    
    This command performs extensive validation of customer churn data including:
    - Schema validation against expected data structure
    - Business rule validation for data integrity  
    - Data quality checks and outlier detection
    - ML-specific validation for training and prediction data
    
    The validation helps ensure data quality before training or prediction,
    preventing issues that could lead to poor model performance or failures.
    
    Parameters
    ----------
    data_file : str
        Path to CSV file containing customer data to validate.
    for_prediction : bool, default=False
        If True, validates data for prediction (target variable not required).
        Use this flag when validating data for batch predictions.
    detailed : bool, default=False
        If True, generates a detailed validation report with full statistics.
        Otherwise, shows a summary of validation results.
    output : str, optional
        Path to save the validation report. If not specified, results are
        printed to console only.
    check_distribution : bool, default=False
        If True, performs feature distribution checks for ML validation.
        Useful for detecting data drift in prediction data.
    no_business_rules : bool, default=False
        If True, skips business rule validation. Use with caution as this
        may allow inconsistent data to pass validation.
    
    Returns
    -------
    None
        Validation results are printed to console or saved to file.
        Exit code 0 indicates validation passed, 1 indicates failure.
    
    Examples
    --------
    >>> python -m src.cli validate data/raw/customer_data.csv
    >>> python -m src.cli validate data/processed/features.csv --for-prediction
    >>> python -m src.cli validate data/raw/customer_data.csv --detailed --output report.txt
    
    Raises
    ------
    ValidationError
        If file cannot be read or critical validation setup fails.
    typer.Exit
        If data validation fails, exits with code 1.
    """
    try:
        # Validate input file path
        DEFAULT_PATH_VALIDATOR.validate_path(data_file, must_exist=True)
        
        # Perform data validation
        kwargs = {
            'check_distribution': check_distribution,
            'check_business_rules': not no_business_rules
        }
        
        if for_prediction:
            from .data_validation import validate_prediction_data
            report = validate_prediction_data(data_file)
        else:
            report = validate_customer_data(data_file, **kwargs)
        
        # Generate output
        if detailed:
            output_text = report.get_detailed_report()
        else:
            output_text = report.get_summary()
            if not report.is_valid:
                output_text += "\n\n🚨 ERRORS:\n"
                for i, error in enumerate(report.errors[:5], 1):  # Show first 5 errors
                    output_text += f"  {i}. {error}\n"
                if len(report.errors) > 5:
                    output_text += f"  ... and {len(report.errors) - 5} more errors\n"
        
        # Output results
        if output:
            DEFAULT_PATH_VALIDATOR.validate_path(output, allow_create=True)
            with open(output, 'w') as f:
                f.write(output_text)
            typer.echo(f"📄 Validation report saved to: {output}")
        
        typer.echo(output_text)
        
        # Exit with appropriate code
        if not report.is_valid:
            typer.echo("❌ Data validation failed", err=True)
            raise typer.Exit(1)
        else:
            typer.echo("✅ Data validation passed")
    
    except ValidationError as e:
        typer.echo(f"File validation error: {e}", err=True)
        raise typer.Exit(1)
    except DataValidationError as e:
        typer.echo(f"Data validation error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def health() -> None:
    """
    Check application health status.
    
    This command performs a basic health check to verify that the application
    is running properly. It returns a simple status indicator suitable for
    monitoring systems and basic health checks.
    
    The health check verifies:
    - Application is responsive
    - Service uptime information
    - Basic system status
    
    Exit codes:
    - 0: Application is healthy
    - 1: Application has issues
    """
    import json
    
    try:
        health_status = get_health_status()
        typer.echo(json.dumps(health_status, indent=2))
        
        if health_status.get("status") == "healthy":
            typer.echo("✅ Application is healthy")
        else:
            typer.echo("❌ Application has health issues", err=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Health check failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def health_detailed() -> None:
    """
    Check comprehensive application health with detailed diagnostics.
    
    This command performs an extensive health check that validates all
    critical application components and dependencies. It provides detailed
    diagnostic information for troubleshooting and monitoring.
    
    The comprehensive health check includes:
    - Basic application status and uptime
    - Model availability and age
    - Data directory accessibility
    - Configuration file validation
    - Critical dependency availability
    - Overall system health assessment
    
    This command is useful for:
    - Detailed system diagnostics
    - Pre-deployment health verification
    - Troubleshooting application issues
    - Monitoring system integration
    
    Exit codes:
    - 0: Application is healthy or degraded (operational)
    - 1: Application is unhealthy (requires attention)
    """
    import json
    
    try:
        health_status = get_comprehensive_health()
        typer.echo(json.dumps(health_status, indent=2))
        
        overall_status = health_status.get("overall_status", "unknown")
        summary = health_status.get("summary", {})
        
        if overall_status == "healthy":
            typer.echo("✅ Application is fully healthy")
        elif overall_status == "degraded":
            typer.echo("⚠️  Application is operational but degraded")
            if summary.get("errors"):
                typer.echo("Issues found:")
                for error in summary["errors"][:5]:  # Show first 5 errors
                    typer.echo(f"  - {error}")
        else:
            typer.echo("❌ Application is unhealthy", err=True)
            if summary.get("errors"):
                typer.echo("Critical issues found:")
                for error in summary["errors"][:5]:  # Show first 5 errors
                    typer.echo(f"  - {error}")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Comprehensive health check failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def ready() -> None:
    """
    Check application readiness for serving requests.
    
    This command performs a readiness check specifically designed for container
    orchestration systems like Kubernetes. It verifies that the application
    is ready to serve requests and handle workloads.
    
    The readiness check validates:
    - Trained model availability
    - Data directory accessibility  
    - Critical dependencies loaded
    - Essential services operational
    
    This command is typically used as:
    - Kubernetes readiness probe
    - Load balancer health check
    - Container orchestration readiness gate
    - Pre-traffic routing verification
    
    Exit codes:
    - 0: Application is ready to serve requests
    - 1: Application is not ready (should not receive traffic)
    """
    import json
    
    try:
        readiness_status = get_readiness_status()
        typer.echo(json.dumps(readiness_status, indent=2))
        
        if readiness_status.get("ready", False):
            typer.echo("✅ Application is ready")
        else:
            typer.echo("❌ Application is not ready", err=True)
            checks = readiness_status.get("checks", {})
            failed_checks = [name for name, status in checks.items() if not status]
            if failed_checks:
                typer.echo(f"Failed readiness checks: {', '.join(failed_checks)}")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Readiness check failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cache_stats() -> None:
    """
    Display model cache statistics and entry information.
    
    This command shows detailed statistics about the model cache including:
    - Cache hit rate and performance metrics
    - Memory usage and utilization
    - Cached entries with access patterns
    - Cache efficiency indicators
    
    The cache stores frequently accessed models, preprocessors, and metadata
    to improve prediction performance by avoiding repeated file loading.
    
    Information displayed includes:
    - Hit rate percentage and total requests
    - Memory usage (current and maximum)
    - Number of cached entries
    - Individual entry details (age, access count, size)
    
    This command is useful for:
    - Monitoring cache performance
    - Tuning cache configuration
    - Debugging cache-related issues
    - Understanding memory usage patterns
    """
    import json
    
    try:
        cache_info = get_cache_stats()
        typer.echo("📊 Model Cache Statistics")
        typer.echo("=" * 50)
        
        stats = cache_info["stats"]
        
        # Summary statistics
        typer.echo(f"Entries: {stats['entries']}")
        typer.echo(f"Memory Usage: {stats['memory_used_mb']:.2f} MB / {stats['max_memory_mb']:.2f} MB")
        typer.echo(f"Memory Utilization: {stats['memory_utilization']:.1f}%")
        typer.echo(f"Hit Rate: {stats['hit_rate']:.1f}%")
        typer.echo(f"Total Requests: {stats['total_requests']}")
        typer.echo(f"Cache Hits: {stats['hits']}")
        typer.echo(f"Cache Misses: {stats['misses']}")
        typer.echo(f"Evictions: {stats['evictions']}")
        typer.echo(f"Invalidations: {stats['invalidations']}")
        
        # Entry details
        entries = cache_info["entries"]
        if entries:
            typer.echo("\n📁 Cached Entries")
            typer.echo("-" * 50)
            for entry in entries:
                typer.echo(f"Key: {entry['key']}")
                typer.echo(f"  Age: {entry['age_seconds']:.1f}s")
                typer.echo(f"  Accesses: {entry['access_count']}")
                typer.echo(f"  Last Access: {entry['last_access_ago']:.1f}s ago")
                typer.echo(f"  Size: {entry['size_mb']:.2f} MB")
                if entry['file_path']:
                    typer.echo(f"  File: {entry['file_path']}")
                typer.echo("")
        else:
            typer.echo("\n📁 No cached entries")
            
    except Exception as e:
        typer.echo(f"❌ Failed to get cache statistics: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cache_clear(
    confirm: bool = typer.Option(False, "--confirm", help="Confirm cache clearing without prompt")
) -> None:
    """
    Clear the model cache to free memory.
    
    This command removes all cached models, preprocessors, and metadata
    from memory. The next prediction requests will reload artifacts from
    disk, which may be slower but ensures fresh data is used.
    
    Use this command when:
    - Memory usage is too high
    - Models have been updated and cache needs refresh
    - Troubleshooting cache-related issues
    - Forcing reload of all artifacts
    
    Parameters
    ----------
    confirm : bool, default=False
        Skip confirmation prompt and immediately clear cache.
        Use with caution in production environments.
    
    Exit codes:
    - 0: Cache cleared successfully
    - 1: Operation cancelled or error occurred
    """
    import json
    
    try:
        # Get current cache info
        cache_info = get_cache_stats()
        stats = cache_info["stats"]
        
        if stats["entries"] == 0:
            typer.echo("ℹ️  Cache is already empty")
            return
        
        # Confirmation prompt unless --confirm used
        if not confirm:
            typer.echo(f"⚠️  About to clear cache with {stats['entries']} entries")
            typer.echo(f"   Memory to free: {stats['memory_used_mb']:.2f} MB")
            
            if not typer.confirm("Are you sure you want to clear the cache?"):
                typer.echo("❌ Cache clear cancelled")
                raise typer.Exit(1)
        
        # Clear the cache
        invalidate_model_cache()  # Clear entire cache
        
        typer.echo("✅ Model cache cleared successfully")
        typer.echo(f"   Freed {stats['memory_used_mb']:.2f} MB of memory")
        typer.echo(f"   Removed {stats['entries']} cached entries")
        
    except Exception as e:
        typer.echo(f"❌ Failed to clear cache: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def metrics() -> None:
    """
    Export Prometheus metrics for monitoring and observability.
    
    This command outputs application metrics in Prometheus exposition format,
    suitable for scraping by Prometheus monitoring systems. Metrics include:
    
    - Prediction performance and latency
    - Model accuracy and behavior
    - Cache performance statistics  
    - Health check durations and status
    - Error counts and active requests
    - System resource usage
    
    The output follows the Prometheus exposition format specification and
    can be consumed by:
    - Prometheus monitoring server
    - Grafana for visualization
    - AlertManager for alerting
    - Custom monitoring tools
    
    Usage examples:
    - Direct output: python -m src.cli metrics
    - HTTP endpoint: curl http://localhost:8000/metrics
    - Prometheus scraping: Configure as scrape target
    
    Exit codes:
    - 0: Metrics exported successfully
    - 1: Error generating metrics
    """
    try:
        # Get metrics in Prometheus format
        metrics_output = get_prometheus_metrics()
        
        # Output to stdout (standard for Prometheus exposition)
        typer.echo(metrics_output, nl=False)
        
    except Exception as e:
        typer.echo(f"❌ Failed to generate metrics: {e}", err=True)
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
    - validate: Validate customer data with comprehensive quality checks
    
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
