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
from .security import get_security_report, check_container_security, get_security_policies
from .rate_limiter import get_rate_limiter, RateLimitRule, get_rate_limit_stats

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


@app.command()
def security_scan(
    image: str = typer.Argument(..., help="Docker image to scan"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for security report"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, table)"),
    max_high: int = typer.Option(0, "--max-high", help="Maximum allowed high severity vulnerabilities"),
    max_medium: int = typer.Option(5, "--max-medium", help="Maximum allowed medium severity vulnerabilities")
) -> None:
    """
    Perform comprehensive security scan of Docker image.
    
    This command scans the specified Docker image for security vulnerabilities,
    misconfigurations, and other security issues. It provides detailed reporting
    and can be integrated into CI/CD pipelines for automated security checks.
    
    The security scan includes:
    - Vulnerability assessment using Trivy scanner
    - Container configuration analysis
    - Security policy validation
    - Image signature verification (if available)
    - Dockerfile security analysis
    
    Parameters
    ----------
    image : str
        Docker image to scan (e.g., "churn-predictor:latest")
    output : str, optional
        File path to save the security report
    format : str, default="json"
        Output format for the report (json, table)
    max_high : int, default=0
        Maximum allowed high severity vulnerabilities before failing
    max_medium : int, default=5
        Maximum allowed medium severity vulnerabilities before failing
    
    Exit codes:
    - 0: Image passes security checks
    - 1: Image fails security requirements or scan error
    
    Examples:
    - Basic scan: python -m src.cli security-scan churn-predictor:latest
    - With output: python -m src.cli security-scan --output report.json image:tag
    - Strict policy: python -m src.cli security-scan --max-high 0 --max-medium 2 image:tag
    """
    import json
    
    try:
        typer.echo(f"🔍 Starting security scan of: {image}")
        
        # Generate comprehensive security report
        report = get_security_report(image)
        
        # Check if scan was successful
        if "error" in report:
            typer.echo(f"❌ Security scan failed: {report['error']}", err=True)
            raise typer.Exit(1)
        
        # Evaluate security status
        scan_result = report.get("scan_result", {})
        is_secure = scan_result.get("is_secure", False)
        
        # Check vulnerability thresholds
        high_count = scan_result.get("high_severity_count", 0)
        medium_count = scan_result.get("medium_severity_count", 0)
        total_count = scan_result.get("total_vulnerabilities", 0)
        
        # Display results
        if format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2)
                typer.echo(f"📄 Security report saved to: {output}")
            else:
                typer.echo(json.dumps(report, indent=2))
        else:
            # Table format summary
            typer.echo("\n📊 Security Scan Summary")
            typer.echo("=" * 50)
            typer.echo(f"Image: {image}")
            typer.echo(f"Scan Time: {report['timestamp']}")
            typer.echo(f"Total Vulnerabilities: {total_count}")
            typer.echo(f"High/Critical: {high_count}")
            typer.echo(f"Medium: {medium_count}")
            typer.echo(f"Security Score: {scan_result.get('security_score', 'N/A')}/100")
            typer.echo(f"Signature Verified: {'✅' if report.get('signature_verified') else '❌'}")
            
            # Show recommendations
            if report.get("recommendations"):
                typer.echo("\n💡 Recommendations:")
                for rec in report["recommendations"][:5]:
                    typer.echo(f"  • {rec}")
        
        # Check thresholds and exit accordingly
        if high_count > max_high:
            typer.echo(f"❌ High severity vulnerabilities ({high_count}) exceed limit ({max_high})", err=True)
            raise typer.Exit(1)
        
        if medium_count > max_medium:
            typer.echo(f"❌ Medium severity vulnerabilities ({medium_count}) exceed limit ({max_medium})", err=True)
            raise typer.Exit(1)
        
        if is_secure:
            typer.echo("✅ Image passes security requirements")
        else:
            typer.echo("⚠️  Image has security concerns but within acceptable limits")
            
    except Exception as e:
        typer.echo(f"❌ Security scan failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def security_policies() -> None:
    """
    Display security policies and recommendations.
    
    This command shows the security policies that should be applied
    to container deployments, including runtime security measures,
    network policies, and image requirements.
    
    The policies cover:
    - Container runtime security
    - Network access controls
    - Image security requirements
    - Resource limitations
    - Security profiles (AppArmor, SELinux, seccomp)
    
    Use this command to:
    - Review security requirements
    - Configure Kubernetes security policies
    - Understand security best practices
    - Generate policy templates
    
    Exit codes:
    - 0: Policies displayed successfully
    - 1: Error retrieving policies
    """
    import json
    
    try:
        typer.echo("🔐 Container Security Policies")
        typer.echo("=" * 50)
        
        policies = get_security_policies()
        
        # Display policies in organized format
        for category, category_policies in policies.items():
            typer.echo(f"\n📋 {category.replace('_', ' ').title()}")
            typer.echo("-" * 30)
            
            for policy_name, policy_config in category_policies.items():
                enabled = policy_config.get("enabled", False)
                status = "✅ Enabled" if enabled else "❌ Disabled"
                typer.echo(f"  {policy_name.replace('_', ' ').title()}: {status}")
                typer.echo(f"    {policy_config.get('description', 'No description')}")
                
                enforcement = policy_config.get("enforcement")
                if enforcement:
                    typer.echo(f"    Enforcement: {enforcement}")
                
                # Show specific configurations
                if policy_name == "resource_limits" and "limits" in policy_config:
                    limits = policy_config["limits"]
                    typer.echo(f"    Memory: {limits.get('memory', 'Not set')}")
                    typer.echo(f"    CPU: {limits.get('cpu', 'Not set')}")
                
                if policy_name == "base_image_restrictions" and "allowed_registries" in policy_config:
                    registries = policy_config["allowed_registries"][:3]  # Show first 3
                    typer.echo(f"    Allowed registries: {', '.join(registries)}")
                
                typer.echo("")
        
        typer.echo("💡 To implement these policies:")
        typer.echo("  • Use Kubernetes Pod Security Policies/Standards")
        typer.echo("  • Configure OPA Gatekeeper policies")
        typer.echo("  • Apply Falco runtime security rules")
        typer.echo("  • Use admission controllers")
        
    except Exception as e:
        typer.echo(f"❌ Failed to retrieve security policies: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def rate_limit_stats() -> None:
    """
    Display rate limiting statistics and configuration.
    
    This command shows comprehensive statistics about the rate limiting system
    including active rules, backend information, and current usage metrics.
    
    Information displayed includes:
    - Rate limiting backend type (memory or Redis)
    - Number of active rate limiting rules
    - Rule configurations and limits
    - Backend-specific statistics (Redis connection info, memory usage)
    
    The rate limiting system protects the API endpoints from abuse by limiting
    the number of requests per client per time window. Different endpoints
    have different limits based on their resource requirements.
    
    Exit codes:
    - 0: Statistics displayed successfully
    - 1: Error retrieving statistics
    """
    import json
    
    try:
        typer.echo("📊 Rate Limiting Statistics")
        typer.echo("=" * 50)
        
        stats = get_rate_limit_stats()
        
        # Display backend information
        backend = stats.get("backend", "unknown")
        typer.echo(f"Backend: {backend}")
        
        if backend == "redis":
            typer.echo(f"Redis Version: {stats.get('redis_version', 'unknown')}")
            typer.echo(f"Connected Clients: {stats.get('connected_clients', 0)}")
            typer.echo(f"Memory Usage: {stats.get('used_memory', 'unknown')}")
        elif backend == "memory":
            typer.echo(f"Active Buckets: {stats.get('active_buckets', 0)}")
        
        typer.echo(f"Active Rules: {stats.get('rules', 0)}")
        
        # Display rule configurations
        rules_config = stats.get("rules_config", {})
        if rules_config:
            typer.echo("\n📋 Rate Limiting Rules")
            typer.echo("-" * 30)
            
            for rule_name, rule_config in rules_config.items():
                typer.echo(f"\n{rule_name}:")
                typer.echo(f"  Requests: {rule_config.get('requests', 'N/A')} per {rule_config.get('window_seconds', 'N/A')}s")
                typer.echo(f"  Burst Size: {rule_config.get('burst_size', 'N/A')}")
                typer.echo(f"  Per IP: {'Yes' if rule_config.get('per_ip', False) else 'No'}")
                typer.echo(f"  Per Endpoint: {'Yes' if rule_config.get('per_endpoint', False) else 'No'}")
                
                description = rule_config.get('description', '')
                if description:
                    typer.echo(f"  Description: {description}")
        else:
            typer.echo("\n📋 No rate limiting rules configured")
        
        typer.echo("\n💡 Rate limiting protects API endpoints from abuse")
        typer.echo("   Configure rules using: python -m src.cli rate-limit-add")
        
    except Exception as e:
        typer.echo(f"❌ Failed to get rate limiting statistics: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def rate_limit_add(
    rule_key: str = typer.Argument(..., help="Rule identifier (e.g., 'predict', 'health', 'admin')"),
    requests: int = typer.Option(..., "--requests", "-r", help="Maximum requests allowed"),
    window_seconds: int = typer.Option(..., "--window", "-w", help="Time window in seconds"),
    burst_size: Optional[int] = typer.Option(None, "--burst", "-b", help="Maximum burst size"),
    description: str = typer.Option("", "--description", "-d", help="Rule description"),
    per_ip: bool = typer.Option(True, "--per-ip/--no-per-ip", help="Apply rate limit per IP address"),
    per_endpoint: bool = typer.Option(True, "--per-endpoint/--no-per-endpoint", help="Apply rate limit per endpoint")
) -> None:
    """
    Add or update a rate limiting rule.
    
    This command creates or updates a rate limiting rule for API endpoints.
    Rate limiting rules control how many requests clients can make within
    a specified time window, helping prevent abuse and ensure fair usage.
    
    Parameters:
    - rule_key: Unique identifier for the rule (e.g., 'predict', 'health')
    - requests: Maximum number of requests allowed in the time window
    - window_seconds: Duration of the time window in seconds
    - burst_size: Maximum burst capacity (defaults to 25% of requests if not specified)
    - description: Human-readable description of the rule
    - per_ip: Whether to apply the limit per client IP address
    - per_endpoint: Whether to apply the limit per API endpoint
    
    Common rule examples:
    - High-traffic endpoints: 200 requests per 60 seconds
    - Prediction endpoints: 30 requests per 60 seconds
    - Admin endpoints: 10 requests per 60 seconds
    - Training endpoints: 5 requests per 300 seconds
    
    Exit codes:
    - 0: Rule added successfully
    - 1: Error adding rule
    
    Examples:
    --------
    >>> python -m src.cli rate-limit-add predict --requests 30 --window 60 --burst 10
    >>> python -m src.cli rate-limit-add admin --requests 10 --window 60 --description "Admin endpoints"
    """
    try:
        # Create rate limit rule
        rule = RateLimitRule(
            requests=requests,
            window_seconds=window_seconds,
            burst_size=burst_size,
            per_ip=per_ip,
            per_endpoint=per_endpoint,
            description=description
        )
        
        # Add rule to rate limiter
        rate_limiter = get_rate_limiter()
        rate_limiter.add_rule(rule_key, rule)
        
        typer.echo(f"✅ Rate limiting rule added: {rule_key}")
        typer.echo(f"   Requests: {requests} per {window_seconds} seconds")
        
        if burst_size:
            typer.echo(f"   Burst Size: {burst_size}")
        else:
            typer.echo(f"   Burst Size: {rule.burst_size} (auto-calculated)")
        
        typer.echo(f"   Per IP: {'Yes' if per_ip else 'No'}")
        typer.echo(f"   Per Endpoint: {'Yes' if per_endpoint else 'No'}")
        
        if description:
            typer.echo(f"   Description: {description}")
        
        typer.echo("\n💡 Rule is now active for API endpoints")
        
    except Exception as e:
        typer.echo(f"❌ Failed to add rate limiting rule: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    log_level: str = typer.Option("info", "--log-level", help="Log level (debug, info, warning, error)")
) -> None:
    """
    Start the FastAPI web server with rate limiting.
    
    This command starts the REST API server that exposes CLI functionality
    through HTTP endpoints with comprehensive rate limiting, monitoring,
    and security features.
    
    The API server provides:
    - Machine learning prediction endpoints
    - Health and readiness checks
    - Prometheus metrics export
    - Data validation endpoints
    - Admin and security management
    - Interactive OpenAPI documentation
    
    Parameters:
    - host: IP address to bind the server to (0.0.0.0 for all interfaces)
    - port: TCP port to listen on (default: 8000)
    - workers: Number of worker processes for production (default: 1)
    - reload: Enable automatic reloading for development (default: False)
    - log_level: Logging verbosity level
    
    The server includes automatic rate limiting for all endpoints:
    - Default: 100 requests per minute
    - Predictions: 30 requests per minute (burst: 10)
    - Health checks: 200 requests per minute
    - Training: 5 requests per 5 minutes
    - Admin: 10 requests per minute
    
    Access the interactive API documentation at:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    
    Exit codes:
    - 0: Server started successfully
    - 1: Server startup failed
    
    Examples:
    --------
    >>> python -m src.cli serve --port 8000
    >>> python -m src.cli serve --host 127.0.0.1 --port 8080 --reload
    >>> python -m src.cli serve --workers 4 --log-level debug
    """
    try:
        import uvicorn
        from .api import app as fastapi_app
        
        typer.echo(f"🚀 Starting Customer Churn Prediction API")
        typer.echo(f"   Host: {host}")
        typer.echo(f"   Port: {port}")
        typer.echo(f"   Workers: {workers}")
        typer.echo(f"   Reload: {reload}")
        typer.echo(f"   Log Level: {log_level}")
        typer.echo("")
        typer.echo(f"📚 API Documentation:")
        typer.echo(f"   Swagger UI: http://{host}:{port}/docs")
        typer.echo(f"   ReDoc: http://{host}:{port}/redoc")
        typer.echo("")
        typer.echo("🔒 Rate limiting is active for all endpoints")
        typer.echo("⚡ Starting server...")
        
        # Start the server
        uvicorn.run(
            "src.api:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except ImportError:
        typer.echo("❌ FastAPI dependencies not installed", err=True)
        typer.echo("   Install with: pip install fastapi uvicorn[standard]", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to start API server: {e}", err=True)
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
