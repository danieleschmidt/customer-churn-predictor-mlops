# API Authentication Guide

The Customer Churn Prediction API supports two authentication methods for secured endpoints.

## Authentication Methods

### 1. Static API Key Authentication

The simplest authentication method using a bearer token.

**Setup:**
```bash
export API_KEY="your-secret-api-key-here"
```

**Usage:**
```bash
curl -H "Authorization: Bearer your-secret-api-key-here" \
     https://api.example.com/admin/clear-cache
```

**Security Notes:**
- Store API keys securely (environment variables, secrets management)
- Rotate keys regularly
- Use HTTPS only in production

### 2. HMAC Signature Authentication

Time-based authentication using HMAC-SHA256 signatures for enhanced security.

**Setup:**
```bash
export API_SECRET="your-secret-signing-key"
```

**Token Generation:**
```python
import hmac
import hashlib
from datetime import datetime

def generate_hmac_token(secret: str) -> str:
    timestamp = str(int(datetime.now().timestamp()))
    message = f"api_access:{timestamp}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{timestamp}.{signature}"

# Usage
token = generate_hmac_token("your-secret-signing-key")
```

**Usage:**
```bash
TOKEN=$(python3 -c "
import hmac, hashlib
from datetime import datetime
timestamp = str(int(datetime.now().timestamp()))
message = f'api_access:{timestamp}'
signature = hmac.new(b'your-secret-signing-key', message.encode(), hashlib.sha256).hexdigest()
print(f'{timestamp}.{signature}')
")

curl -H "Authorization: Bearer $TOKEN" \
     https://api.example.com/admin/security/scan
```

**Security Features:**
- Tokens expire after 15 minutes (900 seconds)
- Cryptographically secure signatures
- Replay attack protection via timestamps
- No persistent tokens to compromise

## Protected Endpoints

The following endpoints require authentication:

| Endpoint | Method | Authentication Required | Description |
|----------|--------|------------------------|-------------|
| `/admin/clear-cache` | POST | Yes | Clear model cache |
| `/admin/security/scan` | POST | Yes | Security scan of Docker images |
| `/admin/reset-rate-limits` | POST | Yes | Reset rate limiting counters |

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Successfully authenticated |
| 401 | Unauthorized - missing or invalid credentials |
| 403 | Forbidden - valid credentials but insufficient permissions |

## Error Responses

```json
{
  "detail": "Could not validate credentials",
  "status_code": 401
}
```

## Implementation Notes

- Authentication is implemented using FastAPI's HTTPBearer security scheme
- All authentication attempts are logged for audit purposes
- HMAC comparison uses constant-time comparison to prevent timing attacks
- Both authentication methods can be used simultaneously

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Optional | Static API key for bearer token auth |
| `API_SECRET` | Optional | Secret key for HMAC signature generation |

**Note:** At least one of `API_KEY` or `API_SECRET` must be configured for authentication to work.

## Example Client Implementation

```python
import requests
import hmac
import hashlib
from datetime import datetime

class ChurnAPIClient:
    def __init__(self, base_url: str, api_key: str = None, api_secret: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _get_headers(self):
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        elif self.api_secret:
            token = self._generate_hmac_token()
            return {"Authorization": f"Bearer {token}"}
        else:
            return {}
    
    def _generate_hmac_token(self):
        timestamp = str(int(datetime.now().timestamp()))
        message = f"api_access:{timestamp}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}.{signature}"
    
    def clear_cache(self):
        response = requests.post(
            f"{self.base_url}/admin/clear-cache",
            headers=self._get_headers()
        )
        return response.json()

# Usage examples
client_api_key = ChurnAPIClient("https://api.example.com", api_key="your-key")
client_hmac = ChurnAPIClient("https://api.example.com", api_secret="your-secret")

result = client_api_key.clear_cache()
```

## Security Best Practices

1. **Use HTTPS only** in production environments
2. **Store secrets securely** using environment variables or secrets management systems
3. **Rotate credentials regularly** especially API keys
4. **Monitor authentication logs** for suspicious activity
5. **Use HMAC tokens** for higher security scenarios
6. **Implement rate limiting** on authentication endpoints
7. **Validate input** thoroughly to prevent injection attacks