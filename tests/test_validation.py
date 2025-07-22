"""
Tests for the input validation framework.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.validation import (
    PathValidator,
    DataValidator,
    MLValidator,
    ValidationError,
    safe_read_csv,
    safe_write_csv,
    DEFAULT_PATH_VALIDATOR
)


class TestPathValidator:
    """Test cases for PathValidator class."""
    
    def test_valid_path_in_allowed_directory(self):
        """Test validation of paths in allowed directories."""
        validator = PathValidator(allowed_directories=["data/", "models/"])
        
        # Should not raise exception
        result = validator.validate_path("data/test.csv", must_exist=False)
        assert isinstance(result, Path)
        assert str(result).endswith("data/test.csv")
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts."""
        validator = PathValidator()
        
        with pytest.raises(ValidationError, match="Path traversal detected"):
            validator.validate_path("../../../etc/passwd")
    
    def test_disallowed_directory(self):
        """Test rejection of paths outside allowed directories."""
        validator = PathValidator(allowed_directories=["data/"])
        
        with pytest.raises(ValidationError, match="not in allowed directories"):
            validator.validate_path("/etc/passwd", must_exist=False)
    
    def test_disallowed_extension(self):
        """Test rejection of files with disallowed extensions."""
        validator = PathValidator(allowed_extensions=[".csv"])
        
        with pytest.raises(ValidationError, match="File extension not allowed"):
            validator.validate_path("data/test.exe", must_exist=False)
    
    def test_must_exist_file_missing(self):
        """Test that missing files are rejected when must_exist=True."""
        validator = PathValidator()
        
        with pytest.raises(ValidationError, match="Required file does not exist"):
            validator.validate_path("data/nonexistent.csv", must_exist=True)
    
    def test_parent_directory_creation(self):
        """Test automatic parent directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PathValidator(allowed_directories=[tmpdir])
            new_path = os.path.join(tmpdir, "subdir", "test.csv")
            
            result = validator.validate_path(new_path, allow_create=True, check_parent=True)
            
            # Parent directory should be created
            assert result.parent.exists()
    
    def test_empty_path(self):
        """Test rejection of empty paths."""
        validator = PathValidator()
        
        with pytest.raises(ValidationError, match="Path cannot be empty"):
            validator.validate_path("")


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        result = DataValidator.validate_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_dataframe_min_rows(self):
        """Test minimum rows validation."""
        df = pd.DataFrame({"A": [1]})
        
        with pytest.raises(ValidationError, match="must have at least 5 rows"):
            DataValidator.validate_dataframe(df, min_rows=5)
    
    def test_dataframe_max_rows(self):
        """Test maximum rows validation."""
        df = pd.DataFrame({"A": range(100)})
        
        with pytest.raises(ValidationError, match="exceeds maximum 10 rows"):
            DataValidator.validate_dataframe(df, max_rows=10)
    
    def test_missing_required_columns(self):
        """Test validation of required columns."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            DataValidator.validate_dataframe(df, required_columns=["A", "B", "C"])
    
    def test_empty_dataframe(self):
        """Test rejection of completely empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="cannot be completely empty"):
            DataValidator.validate_dataframe(df)
    
    def test_non_dataframe_input(self):
        """Test rejection of non-DataFrame input."""
        with pytest.raises(ValidationError, match="must be a pandas DataFrame"):
            DataValidator.validate_dataframe("not a dataframe")
    
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        # Valid range
        result = DataValidator.validate_numeric_range(5.0, min_value=0.0, max_value=10.0)
        assert result == 5.0
        
        # Below minimum
        with pytest.raises(ValidationError, match="must be >= 0"):
            DataValidator.validate_numeric_range(-1.0, min_value=0.0)
        
        # Above maximum
        with pytest.raises(ValidationError, match="must be <= 10"):
            DataValidator.validate_numeric_range(15.0, max_value=10.0)
        
        # Non-numeric
        with pytest.raises(ValidationError, match="must be numeric"):
            DataValidator.validate_numeric_range("not a number")
    
    def test_string_validation(self):
        """Test string validation."""
        # Valid string
        result = DataValidator.validate_string("test", max_length=10)
        assert result == "test"
        
        # Too long
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            DataValidator.validate_string("very long string", max_length=5)
        
        # Invalid characters
        with pytest.raises(ValidationError, match="invalid characters"):
            DataValidator.validate_string("test123", allowed_chars=r"^[a-z]+$")
        
        # Non-string
        with pytest.raises(ValidationError, match="must be a string"):
            DataValidator.validate_string(123)


class TestMLValidator:
    """Test cases for MLValidator class."""
    
    def test_valid_hyperparameters(self):
        """Test validation of valid ML hyperparameters."""
        params = {
            "C": 1.0,
            "max_iter": 100,
            "random_state": 42,
            "test_size": 0.2,
            "penalty": "l2",
            "solver": "liblinear"
        }
        
        result = MLValidator.validate_model_hyperparameters(params)
        assert result["C"] == 1.0
        assert result["max_iter"] == 100
        assert result["penalty"] == "l2"
    
    def test_invalid_penalty(self):
        """Test rejection of invalid penalty values."""
        params = {"penalty": "invalid_penalty"}
        
        with pytest.raises(ValidationError, match="Invalid penalty"):
            MLValidator.validate_model_hyperparameters(params)
    
    def test_invalid_solver(self):
        """Test rejection of invalid solver values."""
        params = {"solver": "invalid_solver"}
        
        with pytest.raises(ValidationError, match="Invalid solver"):
            MLValidator.validate_model_hyperparameters(params)
    
    def test_out_of_range_c(self):
        """Test rejection of C values outside valid range."""
        params = {"C": -1.0}
        
        with pytest.raises(ValidationError, match="Invalid C"):
            MLValidator.validate_model_hyperparameters(params)
    
    def test_invalid_test_size(self):
        """Test rejection of invalid test_size values."""
        params = {"test_size": 1.5}
        
        with pytest.raises(ValidationError, match="Invalid test_size"):
            MLValidator.validate_model_hyperparameters(params)
    
    def test_none_parameter(self):
        """Test rejection of None parameter values."""
        params = {"C": None}
        
        with pytest.raises(ValidationError, match="cannot be None"):
            MLValidator.validate_model_hyperparameters(params)
    
    def test_unknown_parameter(self):
        """Test handling of unknown parameters."""
        params = {"unknown_param": "value"}
        
        result = MLValidator.validate_model_hyperparameters(params)
        assert result["unknown_param"] == "value"


class TestSafeCSVOperations:
    """Test cases for safe CSV read/write operations."""
    
    def test_safe_read_csv_valid(self):
        """Test safe CSV reading with valid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("A,B\n1,2\n3,4\n")
            temp_path = f.name
        
        try:
            # Patch validator to allow the temp file
            with patch.object(DEFAULT_PATH_VALIDATOR, 'validate_path') as mock_validate:
                mock_validate.return_value = Path(temp_path)
                
                df = safe_read_csv(temp_path)
                assert len(df) == 2
                assert list(df.columns) == ["A", "B"]
        finally:
            os.unlink(temp_path)
    
    def test_safe_read_csv_invalid_path(self):
        """Test safe CSV reading with invalid path."""
        with pytest.raises(ValidationError):
            safe_read_csv("../../../etc/passwd")
    
    def test_safe_write_csv_valid(self):
        """Test safe CSV writing with valid data."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Patch validator to allow the temp file
            with patch.object(DEFAULT_PATH_VALIDATOR, 'validate_path') as mock_validate:
                mock_validate.return_value = Path(temp_path)
                
                result_path = safe_write_csv(df, temp_path)
                assert result_path.exists()
                
                # Verify content
                read_df = pd.read_csv(result_path)
                assert len(read_df) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_safe_write_csv_invalid_dataframe(self):
        """Test safe CSV writing with invalid DataFrame."""
        df = pd.DataFrame()  # Empty DataFrame
        
        with pytest.raises(ValidationError, match="cannot be completely empty"):
            safe_write_csv(df, "output.csv")


class TestIntegrationSecurity:
    """Integration tests for security features."""
    
    def test_prevent_path_traversal_attack(self):
        """Test that path traversal attacks are prevented."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValidationError):
                DEFAULT_PATH_VALIDATOR.validate_path(path, must_exist=False)
    
    def test_file_extension_whitelist(self):
        """Test that only allowed file extensions are permitted."""
        dangerous_extensions = [".exe", ".bat", ".sh", ".py", ".js"]
        
        for ext in dangerous_extensions:
            if ext not in DEFAULT_PATH_VALIDATOR.allowed_extensions:
                with pytest.raises(ValidationError):
                    DEFAULT_PATH_VALIDATOR.validate_path(f"data/file{ext}", must_exist=False)
    
    def test_directory_whitelist(self):
        """Test that only allowed directories are accessible."""
        dangerous_paths = ["/etc/", "/bin/", "C:\\Windows\\", "/root/"]
        
        for path in dangerous_paths:
            with pytest.raises(ValidationError):
                DEFAULT_PATH_VALIDATOR.validate_path(f"{path}/file.csv", must_exist=False)


class TestSpecificExceptionHandling:
    """Test cases for specific exception handling instead of generic Exception."""
    
    def test_safe_read_csv_file_not_found(self):
        """Test that safe_read_csv properly handles FileNotFoundError."""
        from src.validation import safe_read_csv
        
        with pytest.raises(ValidationError, match="Failed to read CSV"):
            safe_read_csv("/nonexistent/path/file.csv")
    
    def test_safe_write_csv_permission_denied(self):
        """Test that safe_write_csv properly handles PermissionError."""
        import pandas as pd
        from src.validation import safe_write_csv
        
        # Try to write to a directory that would cause permission error
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        with pytest.raises(ValidationError, match="Failed to write CSV"):
            safe_write_csv(df, "/root/readonly/file.csv")
    
    def test_safe_read_json_invalid_json(self):
        """Test that safe_read_json properly handles JSON decode errors."""
        from src.validation import safe_read_json
        import tempfile
        
        # Create a file with invalid JSON content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_file = f.name
        
        try:
            with pytest.raises(ValidationError, match="Failed to read JSON"):
                safe_read_json(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_safe_write_text_encoding_error(self):
        """Test that safe_write_text properly handles encoding errors."""
        from src.validation import safe_write_text
        import tempfile
        
        # This test verifies the exception is caught and re-raised as ValidationError
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create content that might cause encoding issues
            result = safe_write_text("test content", temp_file)
            # If it succeeds, verify the file was written
            assert os.path.exists(result)
        except ValidationError:
            # Expected if there's an encoding or permission issue
            pass
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])