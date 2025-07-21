# Input Validation Framework Guide

This document describes the comprehensive input validation framework implemented to secure file operations and ensure data integrity throughout the customer churn prediction system.

## Overview

The validation framework provides three main validator classes:
- **PathValidator**: Secures file path operations and prevents path traversal attacks
- **DataValidator**: Validates DataFrame structure, numeric ranges, and string formats  
- **MLValidator**: Validates machine learning hyperparameters and model inputs

## Security Features

### Path Traversal Prevention
- Blocks `../` patterns in file paths
- Restricts access to allowed directories only
- Validates file extensions against whitelist
- Automatic path resolution and normalization

### File Operation Security
- Validates all file paths before read/write operations
- Creates parent directories safely when needed
- Enforces file extension restrictions
- Logs all file operations for audit trails

### Data Integrity Validation
- Ensures DataFrames meet minimum/maximum row requirements
- Validates required columns are present
- Checks data types and numeric ranges
- Prevents processing of malformed data

## Usage Examples

### Basic Path Validation

```python
from src.validation import DEFAULT_PATH_VALIDATOR, ValidationError

try:
    # Validate a file path for reading
    safe_path = DEFAULT_PATH_VALIDATOR.validate_path(
        "data/input.csv", 
        must_exist=True
    )
    
    # Validate output path (creates parent dirs if needed)
    output_path = DEFAULT_PATH_VALIDATOR.validate_path(
        "output/predictions.csv",
        allow_create=True
    )
    
except ValidationError as e:
    print(f"Path validation failed: {e}")
```

### Safe CSV Operations

```python
from src.validation import safe_read_csv, safe_write_csv

# Safe CSV reading with automatic validation
try:
    df = safe_read_csv("data/processed/features.csv")
    print(f"Loaded {len(df)} rows safely")
    
    # Process data...
    
    # Safe CSV writing with validation
    safe_write_csv(df, "output/results.csv")
    
except ValidationError as e:
    print(f"CSV operation failed: {e}")
```

### Data Validation

```python
from src.validation import DataValidator

# Validate DataFrame structure
try:
    validated_df = DataValidator.validate_dataframe(
        df,
        required_columns=["feature1", "feature2", "target"],
        min_rows=100,
        max_rows=1000000
    )
    
    # Validate numeric ranges
    validated_c = DataValidator.validate_numeric_range(
        c_value,
        min_value=0.001,
        max_value=1000.0,
        name="regularization_C"
    )
    
except ValidationError as e:
    print(f"Data validation failed: {e}")
```

### ML Hyperparameter Validation

```python
from src.validation import MLValidator

# Validate model hyperparameters
hyperparams = {
    "C": 1.0,
    "max_iter": 100,
    "penalty": "l2",
    "solver": "liblinear",
    "random_state": 42,
    "test_size": 0.2
}

try:
    validated_params = MLValidator.validate_model_hyperparameters(hyperparams)
    print("All hyperparameters are valid")
    
except ValidationError as e:
    print(f"Invalid hyperparameter: {e}")
```

## Configuration

### Custom Path Validator

```python
from src.validation import PathValidator

# Create custom validator with specific restrictions
custom_validator = PathValidator(
    allowed_directories=["data/", "models/", "custom_output/"],
    allowed_extensions=[".csv", ".json", ".pkl"]
)

# Use for specific validation needs
safe_path = custom_validator.validate_path("custom_output/model.pkl")
```

### Allowed Directories (Default)

- `data/` - For all data files
- `models/` - For model artifacts  
- `logs/` - For log files
- `tests/` - For test data
- `/tmp/` - For temporary files
- `scripts/` - For script outputs

### Allowed Extensions (Default)

- `.csv` - Data files
- `.json` - Configuration and metadata
- `.joblib`, `.pkl` - Model artifacts
- `.txt` - Text files and documentation
- `.yaml`, `.yml` - Configuration files
- `.log` - Log files

## Integration Points

The validation framework is integrated throughout the codebase:

### Scripts
- `scripts/run_prediction.py` - Input/output CSV validation
- `scripts/run_training.py` - Data file and hyperparameter validation
- All script modules validate file paths before processing

### Core Modules  
- `src/train_model.py` - Data loading with validation
- `src/cli.py` - Command-line argument validation
- All modules use safe file operations

### Error Handling

All validation errors raise `ValidationError` with descriptive messages:

```python
try:
    result = some_validated_operation()
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle error appropriately
    raise
```

## Security Benefits

1. **Path Traversal Prevention**: Blocks attacks like `../../../etc/passwd`
2. **Directory Restriction**: Limits file access to safe directories only
3. **Extension Whitelisting**: Prevents execution of dangerous file types
4. **Data Integrity**: Ensures all processed data meets quality standards
5. **Parameter Validation**: Prevents injection of invalid ML parameters
6. **Audit Logging**: All file operations are logged for security monitoring

## Testing

Comprehensive tests are provided in `tests/test_validation.py`:

```bash
# Run validation tests (when dependencies are available)
python -m pytest tests/test_validation.py -v
```

Test coverage includes:
- Path traversal attack prevention
- Directory and extension restrictions  
- Data validation edge cases
- Hyperparameter validation
- Integration security tests

## Best Practices

1. **Always Use Safe Functions**: Use `safe_read_csv()` and `safe_write_csv()` instead of direct pandas operations
2. **Validate Early**: Validate all inputs at the entry points of functions
3. **Use Default Validators**: Leverage `DEFAULT_PATH_VALIDATOR` for consistent behavior
4. **Handle Errors Gracefully**: Catch `ValidationError` and provide meaningful error messages
5. **Log Security Events**: Log all validation failures for security monitoring
6. **Regular Updates**: Review and update allowed directories/extensions as needed

## Migration Notes

Existing code has been updated to use the validation framework:
- Direct `pd.read_csv()` calls replaced with `safe_read_csv()`
- File existence checks replaced with path validation
- Manual path construction replaced with validated path operations
- Parameter validation added to all ML functions

For any legacy code not yet updated, follow the patterns shown in the updated modules.