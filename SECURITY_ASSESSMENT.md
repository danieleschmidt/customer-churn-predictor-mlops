# Security Assessment: User Input Vectors and Risks

## Executive Summary

This security assessment identifies all user input vectors in the customer churn prediction codebase and evaluates their security risks. The analysis reveals several critical vulnerabilities related to path traversal, insufficient input validation, and potential injection attacks.

## 1. File Path Input Vectors

### 1.1 Command-Line Interface (CLI) File Paths

**Location**: `src/cli.py`

#### Input Vectors:
- `train` command:
  - `x_path`: Path to processed features CSV
  - `y_path`: Path to processed target CSV
- `evaluate` command:
  - `model_path`: Path to model file
  - `x_path`: Path to processed features CSV
  - `y_path`: Path to processed target CSV
  - `output`: Path for saving metrics JSON
- `pipeline` command:
  - `raw_path`: Path to raw customer data CSV
- `monitor` command:
  - `x_path`: Path to processed features CSV
  - `y_path`: Path to processed target CSV
- `predict` command:
  - `input_csv`: Path to input CSV file (REQUIRED)
  - `output_csv`: Path for saving predictions

**Current Validation**: NONE
**Security Risks**:
- **Path Traversal**: Users can specify paths like `../../sensitive/data.csv`
- **Arbitrary File Read**: No restrictions on reading files from any location
- **Arbitrary File Write**: No restrictions on writing files to any location
- **Directory Traversal**: Users can navigate to parent directories

**Recommended Validation**:
```python
import os
import pathlib

def validate_file_path(path: str, allowed_dirs: list[str], must_exist: bool = False) -> str:
    """Validate and sanitize file paths."""
    # Resolve to absolute path
    abs_path = os.path.abspath(path)
    
    # Check if path is within allowed directories
    allowed = False
    for allowed_dir in allowed_dirs:
        allowed_abs = os.path.abspath(allowed_dir)
        if abs_path.startswith(allowed_abs):
            allowed = True
            break
    
    if not allowed:
        raise ValueError(f"Path {path} is outside allowed directories")
    
    # Check for path traversal attempts
    if ".." in pathlib.Path(path).parts:
        raise ValueError("Path traversal detected")
    
    # Check existence if required
    if must_exist and not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {path}")
    
    return abs_path
```

### 1.2 Configuration File Paths

**Location**: `src/config.py`

#### Input Vectors:
- YAML configuration file path (default: `config.yml`)
- Paths within YAML configuration:
  - `data.raw`
  - `data.processed_features`
  - `data.processed_target`
  - `model.path`
  - `model.feature_columns`
  - `model.run_id`
  - `model.preprocessor`

**Current Validation**: NONE
**Security Risks**:
- **YAML Injection**: Using `yaml.safe_load()` mitigates code execution
- **Path Traversal**: Configuration can specify arbitrary file paths
- **Information Disclosure**: Can read sensitive files via configuration

**Recommended Validation**:
- Validate all paths in configuration against allowed directories
- Use a schema validator for YAML configuration
- Implement path sanitization for all file paths in config

### 1.3 Direct File Operations

**Location**: Multiple files

#### Key Operations:
- `pd.read_csv()` - Reading CSV files without validation
- `joblib.dump()` / `joblib.load()` - Model serialization
- `json.dump()` / `json.load()` - JSON file operations
- `open()` - Direct file operations

**Current Validation**: MINIMAL (only existence checks)
**Security Risks**:
- **Arbitrary File Read**: Can read any file accessible to the process
- **Deserialization Attack**: Loading untrusted joblib files can execute code
- **Memory Exhaustion**: Large files can cause OOM errors

## 2. User Data Inputs

### 2.1 CSV Data Input

**Location**: `src/preprocess_data.py`, prediction scripts

#### Input Vectors:
- Customer data CSV files
- Prediction input CSV files
- Feature values within CSV

**Current Validation**: MINIMAL
- Only `TotalCharges` is validated (converted to numeric)
- No validation on other fields

**Security Risks**:
- **CSV Injection**: Malicious formulas in CSV cells
- **Data Type Confusion**: Unexpected data types can cause errors
- **Memory Exhaustion**: Extremely large CSV files
- **Malformed Data**: Can cause pandas parsing errors

**Recommended Validation**:
```python
def validate_csv_input(df: pd.DataFrame, max_rows: int = 1000000) -> pd.DataFrame:
    """Validate CSV input data."""
    # Check size limits
    if len(df) > max_rows:
        raise ValueError(f"CSV exceeds maximum rows ({max_rows})")
    
    # Validate column names (alphanumeric + underscore only)
    for col in df.columns:
        if not col.replace('_', '').isalnum():
            raise ValueError(f"Invalid column name: {col}")
    
    # Check for formula injection
    for col in df.select_dtypes(include=['object']):
        if df[col].astype(str).str.startswith(('=', '+', '-', '@')).any():
            raise ValueError(f"Potential formula injection in column {col}")
    
    return df
```

### 2.2 Model Parameters

**Location**: CLI commands, training scripts

#### Input Vectors:
- `solver`: Logistic regression solver type
- `C`: Regularization parameter (float)
- `penalty`: Penalty type
- `random_state`: Random seed (int)
- `max_iter`: Maximum iterations (int)
- `test_size`: Test split size (float)

**Current Validation**: Type conversion only
**Security Risks**:
- **Resource Exhaustion**: Large `max_iter` values
- **Invalid Parameters**: Can cause model training failures
- **Numerical Overflow**: Extreme parameter values

**Recommended Validation**:
```python
VALID_SOLVERS = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
VALID_PENALTIES = ['l1', 'l2', 'elasticnet', 'none']

def validate_model_params(params: dict) -> dict:
    """Validate model training parameters."""
    # Validate solver
    if params.get('solver') not in VALID_SOLVERS:
        raise ValueError(f"Invalid solver: {params['solver']}")
    
    # Validate C (regularization)
    if not 0.001 <= params.get('C', 1.0) <= 1000:
        raise ValueError("C must be between 0.001 and 1000")
    
    # Validate penalty
    if params.get('penalty') not in VALID_PENALTIES:
        raise ValueError(f"Invalid penalty: {params['penalty']}")
    
    # Validate max_iter
    if not 1 <= params.get('max_iter', 100) <= 10000:
        raise ValueError("max_iter must be between 1 and 10000")
    
    # Validate test_size
    if not 0.05 <= params.get('test_size', 0.2) <= 0.95:
        raise ValueError("test_size must be between 0.05 and 0.95")
    
    return params
```

## 3. Environment Variables

**Location**: `src/env_config.py`

#### Input Vectors:
- `MLFLOW_RUN_ID`: MLflow run identifier
- `CHURN_THRESHOLD`: Performance threshold (float)
- `LOG_LEVEL`: Logging level
- `LOG_FILE`: Log file path

**Current Validation**: GOOD - Comprehensive validation implemented
- MLflow run ID: 32-char hex validation
- Churn threshold: Range validation (0.0-1.0)
- Log level: Enum validation
- Log file: Path and permission validation

**Minor Improvements Needed**:
- Add maximum path length validation for LOG_FILE
- Consider adding allowed directory restrictions for LOG_FILE

## 4. Web/API Inputs

**Current Status**: No web/API endpoints detected in the codebase

## 5. Critical Security Issues

### 5.1 Path Traversal Vulnerabilities

**Severity**: HIGH

All file path inputs lack proper validation, allowing users to:
- Read sensitive files: `python cli.py predict ../../../../etc/passwd output.csv`
- Write to arbitrary locations: `python cli.py predict input.csv /tmp/../../sensitive/output.csv`
- Access files outside the project directory

### 5.2 Deserialization Risks

**Severity**: HIGH

The use of `joblib.load()` without validation poses risks:
- Loading malicious model files can execute arbitrary code
- No integrity checking on model files
- No validation of model file sources

### 5.3 CSV Injection

**Severity**: MEDIUM

CSV files are processed without sanitization:
- Formula injection possible (=cmd|'/c calc'!A0)
- No validation of cell contents
- Could lead to code execution in spreadsheet applications

### 5.4 Resource Exhaustion

**Severity**: MEDIUM

No limits on:
- File sizes (can process multi-GB files)
- Number of rows in CSV files
- Model training iterations
- Memory usage during processing

## 6. Recommendations

### 6.1 Immediate Actions (Critical)

1. **Implement Path Validation**:
   - Create a centralized path validation function
   - Define allowed directories (e.g., `data/`, `models/`)
   - Reject paths with `..` components
   - Use `os.path.abspath()` and check against allowed paths

2. **Add Model File Validation**:
   - Implement cryptographic signatures for model files
   - Validate model file sources before loading
   - Consider using safer serialization formats

3. **Sanitize CSV Inputs**:
   - Check for formula injection patterns
   - Validate data types and ranges
   - Implement size limits

### 6.2 Short-term Improvements

1. **Input Validation Layer**:
   ```python
   # src/validators.py
   from typing import Any, Dict
   import os
   import re
   
   class InputValidator:
       ALLOWED_DIRS = ['data/', 'models/', 'output/']
       MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
       
       @classmethod
       def validate_file_path(cls, path: str, mode: str = 'r') -> str:
           """Validate file paths for security."""
           # Implementation here
           pass
       
       @classmethod
       def validate_csv_data(cls, df: pd.DataFrame) -> pd.DataFrame:
           """Validate CSV data for security."""
           # Implementation here
           pass
   ```

2. **Configuration Schema**:
   ```python
   # src/config_schema.py
   CONFIG_SCHEMA = {
       "type": "object",
       "properties": {
           "data": {
               "type": "object",
               "properties": {
                   "raw": {"type": "string", "pattern": "^data/.*\\.csv$"},
                   # ... other properties
               }
           }
       }
   }
   ```

3. **Logging Security Events**:
   - Log all file access attempts
   - Log validation failures
   - Monitor for suspicious patterns

### 6.3 Long-term Enhancements

1. **Implement Principle of Least Privilege**:
   - Run processes with minimal permissions
   - Use separate directories for input/output
   - Implement file access controls

2. **Add Security Testing**:
   - Create security-focused unit tests
   - Test path traversal prevention
   - Test input validation boundaries

3. **Security Documentation**:
   - Document all security controls
   - Create security guidelines for contributors
   - Maintain security changelog

## 7. Testing Recommendations

### Security Test Cases:

```python
# tests/test_security.py
def test_path_traversal_prevention():
    """Test that path traversal attempts are blocked."""
    malicious_paths = [
        "../../../etc/passwd",
        "data/../../sensitive.txt",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\sam"
    ]
    
    for path in malicious_paths:
        with pytest.raises(ValueError):
            validate_file_path(path)

def test_csv_injection_prevention():
    """Test that CSV injection attempts are blocked."""
    malicious_data = pd.DataFrame({
        'name': ['John', '=cmd|"/c calc"!A0'],
        'value': [100, 200]
    })
    
    with pytest.raises(ValueError):
        validate_csv_data(malicious_data)
```

## 8. Conclusion

The codebase currently has significant security vulnerabilities, primarily around path traversal and input validation. While environment variable validation is well-implemented, file path and data input validation are severely lacking. Implementing the recommended security controls will significantly improve the application's security posture and protect against common attack vectors.

Priority should be given to:
1. Path validation for all file operations
2. Input sanitization for CSV data
3. Model file integrity validation
4. Resource usage limits

These improvements will help ensure the application can be safely deployed in production environments.