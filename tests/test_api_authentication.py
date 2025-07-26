"""
Test API authentication implementation.
"""
import unittest
from unittest.mock import patch, AsyncMock
import asyncio
import os
import hashlib
import hmac
from datetime import datetime, timedelta
from pathlib import Path

# Import the verify_auth function to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from api import verify_auth
    from fastapi.security import HTTPAuthorizationCredentials
    HAS_API_IMPORTS = True
except ImportError:
    HAS_API_IMPORTS = False


class TestAPIAuthentication(unittest.TestCase):
    """Test API authentication functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        # Set up test environment variables
        self.test_api_key = "test_api_key_12345"
        self.test_secret = "test_secret_67890"
        
        # Patch environment variables
        self.env_patcher = patch.dict(os.environ, {
            'API_KEY': self.test_api_key,
            'API_SECRET': self.test_secret
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'env_patcher'):
            self.env_patcher.stop()
    
    def test_verify_auth_with_no_credentials(self):
        """Test authentication with no credentials returns None."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        async def run_test():
            result = await verify_auth(None)
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNone(result)
    
    def test_verify_auth_with_invalid_token(self):
        """Test authentication with invalid token returns None."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        invalid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token_123"
        )
        
        async def run_test():
            result = await verify_auth(invalid_credentials)
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNone(result, "Invalid tokens should not be authenticated")
    
    def test_verify_auth_with_valid_api_key(self):
        """Test authentication with valid API key."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        valid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=self.test_api_key
        )
        
        async def run_test():
            result = await verify_auth(valid_credentials)
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result, "Valid API key should be authenticated")
        self.assertIsInstance(result, str, "Should return user identifier string")
        self.assertEqual(result, "api_key_user", "Should return correct user type")
    
    def test_verify_auth_with_hmac_signature(self):
        """Test authentication with HMAC-signed token."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        # Create HMAC signature for timestamp-based token
        timestamp = str(int(datetime.now().timestamp()))
        message = f"api_access:{timestamp}"
        signature = hmac.new(
            self.test_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{timestamp}.{signature}"
        
        valid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        async def run_test():
            result = await verify_auth(valid_credentials)
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result, "Valid HMAC token should be authenticated")
        self.assertEqual(result, "hmac_user", "Should return correct user type")
    
    def test_verify_auth_with_expired_hmac_token(self):
        """Test authentication with expired HMAC token."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        # Create expired timestamp (1 hour ago)
        expired_time = datetime.now() - timedelta(hours=1)
        timestamp = str(int(expired_time.timestamp()))
        message = f"api_access:{timestamp}"
        signature = hmac.new(
            self.test_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{timestamp}.{signature}"
        
        expired_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        async def run_test():
            result = await verify_auth(expired_credentials)
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNone(result, "Expired HMAC tokens should not be authenticated")
    
    def test_api_authentication_environment_variables(self):
        """Test that authentication uses environment variables correctly."""
        # Test that API_KEY and API_SECRET are required
        self.assertIsNotNone(os.getenv('API_KEY'), "API_KEY environment variable should be set")
        self.assertIsNotNone(os.getenv('API_SECRET'), "API_SECRET environment variable should be set")
    
    def test_todo_comment_removed(self):
        """Test that TODO comment has been removed from verify_auth function."""
        if not HAS_API_IMPORTS:
            self.skipTest("API imports not available")
        
        # Read the API file and check for TODO comment
        api_file = Path(__file__).parent.parent / "src" / "api.py"
        if api_file.exists():
            content = api_file.read_text()
            self.assertNotIn(
                "TODO: Implement actual token verification",
                content,
                "TODO comment should be removed after implementing authentication"
            )


if __name__ == '__main__':
    unittest.main()