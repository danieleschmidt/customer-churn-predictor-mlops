# Path Configuration Guide

This guide explains the new configurable path system that allows the churn prediction application to run in different environments (development, Docker, production) with customizable file paths.

## Overview

The path configuration system provides:
- **Environment-based configuration**: Set paths via environment variables
- **Flexible deployment**: Support for development, Docker, and production environments
- **Backwards compatibility**: Existing code continues to work unchanged
- **Centralized management**: All path logic in one place
- **Automatic directory creation**: Required directories are created automatically

## Quick Start

### 1. Basic Usage (No Configuration)
The system works out-of-the-box with sensible defaults:
```bash
# Uses default paths: data/, models/, logs/
python -m src.cli preprocess
python -m src.cli train
python -m src.cli predict data/processed/processed_features.csv
```

### 2. Environment Configuration
Set environment variables to customize paths:
```bash
# Development environment
export CHURN_BASE_DIR="./my_workspace"
export CHURN_DATA_DIR="./my_workspace/data"
export CHURN_MODELS_DIR="./my_workspace/models"

python -m src.cli preprocess  # Uses custom paths
```

### 3. Production Deployment
```bash
# Production environment
export CHURN_BASE_DIR="/opt/churn-predictor"
export CHURN_DATA_DIR="/data/churn"
export CHURN_MODELS_DIR="/models/churn"
export CHURN_LOGS_DIR="/var/log/churn"

python -m src.cli train  # Uses production paths
```

## Environment Variables

### Directory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHURN_BASE_DIR` | `.` | Base directory for all application files |
| `CHURN_DATA_DIR` | `data` | Directory for data files |
| `CHURN_MODELS_DIR` | `models` | Directory for model artifacts |
| `CHURN_LOGS_DIR` | `logs` | Directory for log files |
| `CHURN_PROCESSED_DIR` | `processed` | Subdirectory for processed data |

### Specific File Paths (Optional)

| Variable | Default Path | Description |
|----------|--------------|-------------|
| `CHURN_RAW_DATA_PATH` | `data/raw/customer_data.csv` | Raw input data file |
| `CHURN_PROCESSED_FEATURES_PATH` | `data/processed/processed_features.csv` | Processed features |
| `CHURN_PROCESSED_TARGET_PATH` | `data/processed/processed_target.csv` | Processed target variable |
| `CHURN_MODEL_PATH` | `models/churn_model.joblib` | Trained model file |
| `CHURN_FEATURE_COLUMNS_PATH` | `models/feature_columns.json` | Feature column metadata |
| `CHURN_PREPROCESSOR_PATH` | `models/preprocessor.joblib` | Data preprocessor |
| `CHURN_MLFLOW_RUN_ID_PATH` | `models/mlflow_run_id.txt` | MLflow run ID file |

## Deployment Scenarios

### Development Environment
```bash
# Set up development workspace
export CHURN_BASE_DIR="./dev_workspace"
export CHURN_DATA_DIR="./dev_workspace/data"
export CHURN_MODELS_DIR="./dev_workspace/models"
export CHURN_LOGS_DIR="./dev_workspace/logs"

# All files will be created under ./dev_workspace/
```

### Docker Container
```dockerfile
# Dockerfile
FROM python:3.12
WORKDIR /app
COPY . /app

# Set container paths
ENV CHURN_BASE_DIR=/app
ENV CHURN_DATA_DIR=/app/data
ENV CHURN_MODELS_DIR=/app/models
ENV CHURN_LOGS_DIR=/app/logs

# Mount volumes for persistence
VOLUME ["/app/data", "/app/models", "/app/logs"]
```

```bash
# Docker run with volume mounts
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           churn-predictor
```

### Production Server
```bash
# System-wide installation
export CHURN_BASE_DIR="/opt/churn-predictor"
export CHURN_DATA_DIR="/data/churn"
export CHURN_MODELS_DIR="/models/churn"
export CHURN_LOGS_DIR="/var/log/churn"

# Create service directories
sudo mkdir -p /data/churn /models/churn /var/log/churn
sudo chown churn-user:churn-group /data/churn /models/churn /var/log/churn
```

### Cloud Storage Integration
```bash
# Use cloud storage with absolute paths
export CHURN_RAW_DATA_PATH="/mnt/s3-bucket/customer_data.csv"
export CHURN_PROCESSED_FEATURES_PATH="/mnt/s3-bucket/processed_features.csv"
export CHURN_PROCESSED_TARGET_PATH="/mnt/s3-bucket/processed_target.csv"
export CHURN_MODEL_PATH="/mnt/efs-models/churn_model.joblib"
```

## Programming Interface

### Using Path Configuration in Code

```python
from src.path_config import get_model_path, get_data_path, get_log_path

# Get paths that respect environment configuration
model_path = get_model_path("my_model.joblib")
data_path = get_data_path("processed", "features.csv")
log_path = get_log_path("application.log")

# Use in file operations
import pandas as pd
df = pd.read_csv(data_path)
```

### Custom Path Configuration

```python
from src.path_config import PathConfig, set_path_config

# Create custom configuration
config = PathConfig(
    base_dir="/custom/base",
    data_dir="/custom/data",
    models_dir="/custom/models"
)

# Set as global configuration
set_path_config(config)

# Now all modules use custom paths
from src.constants import MODEL_PATH  # Uses custom model path
```

### Backwards Compatibility

Existing code continues to work without changes:

```python
# Old approach (still works)
from src.constants import MODEL_PATH, PROCESSED_FEATURES_PATH

# New approach (recommended for new code)
from src.path_config import get_model_path, get_processed_features_path

model_path = get_model_path()
features_path = get_processed_features_path()
```

## Configuration File (.env)

Create a `.env` file for your environment (copy from `.env.example`):

```bash
# Copy template
cp .env.example .env

# Edit for your environment
nano .env
```

Example `.env` file:
```bash
# Development configuration
CHURN_BASE_DIR=./workspace
CHURN_DATA_DIR=./workspace/data
CHURN_MODELS_DIR=./workspace/models
CHURN_LOGS_DIR=./workspace/logs

# Optional: Override specific file paths
CHURN_RAW_DATA_PATH=./workspace/data/raw/my_data.csv
```

## Validation and Security

The path configuration system integrates with the validation framework:

- All paths are validated for security (no path traversal)
- Only allowed directories are accessible
- File extensions are restricted to safe types
- Directories are created automatically with proper permissions

## Testing

Test your path configuration:

```bash
# Test with environment variables
export CHURN_BASE_DIR="/tmp/test"
python -c "
from src.path_config import get_model_path, get_data_path
print('Model path:', get_model_path())
print('Data path:', get_data_path('processed', 'test.csv'))
"

# Run tests
python -m pytest tests/test_path_config.py -v
```

## Migration Guide

### For Existing Deployments

1. **No immediate changes required**: Existing deployments continue working
2. **Gradual migration**: Set environment variables as needed
3. **Test thoroughly**: Verify paths are resolved correctly

### For New Deployments

1. **Copy environment template**: `cp .env.example .env`
2. **Configure paths**: Edit `.env` for your environment
3. **Test configuration**: Run a simple command to verify paths
4. **Deploy with confidence**: All paths are validated and secure

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure the user has write access to configured directories
2. **Path not found**: Check that environment variables are set correctly
3. **Import errors**: Verify that relative paths are correctly resolved

### Debug Path Resolution

```python
from src.path_config import get_path_config

config = get_path_config()
print(f"Base dir: {config.base_dir}")
print(f"Data dir: {config.data_dir}")
print(f"Models dir: {config.models_dir}")
print(f"Logs dir: {config.logs_dir}")
```

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python -m src.cli preprocess  # Will show path resolution details
```

## Best Practices

1. **Use environment variables**: Don't hardcode paths in production
2. **Test different environments**: Verify deployment-specific configurations
3. **Document custom paths**: Clearly document any non-standard path configurations
4. **Use absolute paths for production**: Avoid relative paths in production environments
5. **Secure permissions**: Ensure proper file/directory permissions in production
6. **Monitor disk usage**: Keep track of disk usage in configured directories

This configurable path system provides the flexibility needed for different deployment scenarios while maintaining security and simplicity.