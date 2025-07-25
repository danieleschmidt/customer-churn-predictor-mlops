# API Overview

The Customer Churn Predictor provides a comprehensive REST API for integrating churn prediction capabilities into your applications. The API is built with FastAPI and provides high-performance, async endpoints with comprehensive documentation.

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your actual domain.

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## API Features

### 🔐 Authentication
- **API Key Authentication**: Secure access using API keys
- **Rate Limiting**: Per-IP and per-endpoint rate limiting
- **Security Headers**: Comprehensive HTTP security headers

### 📊 Endpoints
- **Health Checks**: Monitor service health and readiness
- **Predictions**: Single and batch prediction endpoints
- **Metrics**: Prometheus metrics for monitoring
- **Admin**: Administrative endpoints for management

### 🛡️ Security
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses
- **CORS Support**: Configurable cross-origin requests
- **Request Logging**: Audit trail for all requests

## Quick Example

Here's a simple example of making a prediction:

```python
import requests

# Configure authentication
headers = {
    "Authorization": "Bearer your-api-key-here",
    "Content-Type": "application/json"
}

# Make a prediction request
data = {
    "customer_data": {
        "tenure": 12,
        "monthly_charges": 65.50,
        "total_charges": 786.00,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No"
    }
}

response = requests.post(
    "http://localhost:8000/predict",
    headers=headers,
    json=data
)

if response.status_code == 200:
    result = response.json()
    print(f"Churn probability: {result['churn_probability']}")
    print(f"Prediction: {result['prediction']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
    "success": true,
    "data": {
        // Response data here
    },
    "timestamp": "2025-07-25T10:30:00Z",
    "request_id": "req_abc123"
}
```

### Error Response
```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data",
        "details": {
            "field": "monthly_charges",
            "issue": "Must be a positive number"
        }
    },
    "timestamp": "2025-07-25T10:30:00Z",
    "request_id": "req_abc123"
}
```

## Rate Limiting

The API implements intelligent rate limiting:

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Health checks | 1000 requests | 1 hour |
| Predictions | 100 requests | 1 hour |
| Admin endpoints | 50 requests | 1 hour |
| General | 500 requests | 1 hour |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Status Codes

The API uses standard HTTP status codes:

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Next Steps

- [Authentication →](authentication.md) - Learn about API authentication
- [Endpoints →](endpoints.md) - Explore available endpoints
- [Rate Limiting →](rate-limiting.md) - Understand rate limiting details
- [Error Handling →](errors.md) - Handle errors gracefully