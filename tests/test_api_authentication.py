"""
Unit tests for API authentication system.

Tests the secure API key-based authentication implementation including:
- Valid authentication scenarios
- Invalid token handling
- Missing credentials handling
- Environment configuration errors
- Security features (constant-time comparison)
- Error logging and HTTP responses
"""

import os
import pytest
import hashlib
from unittest.mock import patch, MagicMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from src.api import verify_auth


class TestAPIAuthentication:
    """Test suite for API authentication functionality."""
    
    @pytest.fixture
    def valid_api_key(self):
        """Fixture providing a valid API key for testing."""
        return "test-api-key-1234567890abcdef"
    
    @pytest.fixture
    def valid_credentials(self, valid_api_key):
        """Fixture providing valid HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=valid_api_key
        )
    
    @pytest.fixture
    def invalid_credentials(self):
        """Fixture providing invalid HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer", 
            credentials="invalid-token-that-is-long-enough"
        )
    
    @pytest.fixture
    def short_credentials(self):
        """Fixture providing credentials that are too short."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="short"
        )

    @pytest.mark.asyncio
    async def test_successful_authentication(self, valid_api_key, valid_credentials):
        """Test successful authentication with valid API key."""
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            result = await verify_auth(valid_credentials)
            assert result == "authenticated_user"

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_401(self):
        """Test that missing credentials raise 401 Unauthorized."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_auth(None)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing authentication credentials" in str(exc_info.value.detail)
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    @pytest.mark.asyncio
    async def test_missing_api_key_env_var_raises_500(self, valid_credentials):
        """Test that missing API_KEY environment variable raises 500."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(valid_credentials)
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Authentication not properly configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self, valid_api_key, invalid_credentials):
        """Test that invalid tokens raise 401 Unauthorized."""
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(invalid_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication token" in str(exc_info.value.detail)
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    @pytest.mark.asyncio
    async def test_short_token_raises_401(self, valid_api_key, short_credentials):
        """Test that tokens shorter than 16 characters raise 401."""
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(short_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_empty_token_raises_401(self, valid_api_key):
        """Test that empty tokens raise 401."""
        empty_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=""
        )
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(empty_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_whitespace_token_raises_401(self, valid_api_key):
        """Test that whitespace-only tokens raise 401."""
        whitespace_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="   \t\n   "
        )
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(whitespace_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_token_with_leading_trailing_whitespace(self, valid_api_key):
        """Test that tokens with whitespace are properly trimmed."""
        padded_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=f"  {valid_api_key}  "
        )
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            result = await verify_auth(padded_credentials)
            assert result == "authenticated_user"

    @pytest.mark.asyncio
    @patch('src.api.logger')
    async def test_successful_auth_logging(self, mock_logger, valid_api_key, valid_credentials):
        """Test that successful authentication is logged properly."""
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            await verify_auth(valid_credentials)
            
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Successful authentication for token:" in log_message
            assert valid_api_key[:8] in log_message

    @pytest.mark.asyncio
    @patch('src.api.logger')
    async def test_failed_auth_logging(self, mock_logger, valid_api_key, invalid_credentials):
        """Test that failed authentication is logged properly."""
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException):
                await verify_auth(invalid_credentials)
            
            mock_logger.warning.assert_called_once()
            log_message = mock_logger.warning.call_args[0][0]
            assert "Authentication failed for token:" in log_message

    @pytest.mark.asyncio
    @patch('src.api.logger')
    async def test_missing_credentials_logging(self, mock_logger):
        """Test that missing credentials are logged properly."""
        with pytest.raises(HTTPException):
            await verify_auth(None)
        
        mock_logger.warning.assert_called_once_with(
            "Authentication attempted without credentials"
        )

    @pytest.mark.asyncio
    @patch('src.api.logger')
    async def test_missing_env_var_logging(self, mock_logger, valid_credentials):
        """Test that missing environment variable is logged properly."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(HTTPException):
                await verify_auth(valid_credentials)
        
        mock_logger.error.assert_called_once_with(
            "API_KEY environment variable not configured"
        )

    @pytest.mark.asyncio
    async def test_constant_time_comparison_security(self, valid_api_key):
        """Test that authentication uses constant-time comparison for security."""
        # This test verifies the security implementation uses secrets.compare_digest
        credentials1 = HTTPAuthorizationCredentials(scheme="Bearer", credentials="a" * 32)
        credentials2 = HTTPAuthorizationCredentials(scheme="Bearer", credentials="b" * 32)
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            # Both should fail but with consistent timing
            with pytest.raises(HTTPException):
                await verify_auth(credentials1)
            
            with pytest.raises(HTTPException):
                await verify_auth(credentials2)

    @pytest.mark.asyncio
    @patch('src.api.secrets.compare_digest')
    async def test_uses_secrets_compare_digest(self, mock_compare, valid_api_key, valid_credentials):
        """Test that the implementation uses secrets.compare_digest for security."""
        mock_compare.return_value = True
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            await verify_auth(valid_credentials)
            
            mock_compare.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.api.hashlib.sha256')
    async def test_exception_handling_in_auth(self, mock_sha256, valid_api_key, valid_credentials):
        """Test that unexpected exceptions are handled properly."""
        mock_sha256.side_effect = Exception("Hashing error")
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(valid_credentials)
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Authentication system error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('src.api.logger')
    @patch('src.api.hashlib.sha256')
    async def test_exception_logging(self, mock_sha256, mock_logger, valid_api_key, valid_credentials):
        """Test that authentication exceptions are logged properly."""
        error_message = "Hashing error"
        mock_sha256.side_effect = Exception(error_message)
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException):
                await verify_auth(valid_credentials)
        
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Authentication error:" in log_message
        assert error_message in log_message


class TestAuthenticationIntegration:
    """Integration tests for authentication system."""
    
    @pytest.fixture
    def valid_api_key(self):
        """Fixture providing a valid API key for testing."""
        return "valid-api-key-12345678901234567890"
    
    @pytest.mark.asyncio
    async def test_authentication_with_real_environment(self):
        """Test authentication with realistic environment setup."""
        test_key = "production-like-api-key-12345678901234567890"
        
        with patch.dict(os.environ, {"API_KEY": test_key}):
            # Test valid authentication
            valid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=test_key
            )
            result = await verify_auth(valid_creds)
            assert result == "authenticated_user"
            
            # Test invalid authentication
            invalid_creds = HTTPAuthorizationCredentials(
                scheme="Bearer", 
                credentials="wrong-key-12345678901234567890"
            )
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(invalid_creds)
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_authentication_security_headers(self, valid_api_key):
        """Test that authentication errors include proper security headers."""
        invalid_creds = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid-token-1234567890"
        )
        
        with patch.dict(os.environ, {"API_KEY": valid_api_key}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_auth(invalid_creds)
        
        # Verify security headers are present
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED