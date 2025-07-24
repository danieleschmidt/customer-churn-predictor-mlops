# API Rate Limiting and Documentation

This document provides comprehensive documentation for the Customer Churn Prediction API with advanced rate limiting capabilities.

## Overview

The Customer Churn Prediction API provides RESTful endpoints for machine learning operations with comprehensive rate limiting, monitoring, and security features. The API exposes CLI functionality through HTTP endpoints while protecting against abuse through intelligent rate limiting.

## Features

### 🚀 Core Functionality
- **Machine Learning Predictions**: Single and batch customer churn predictions
- **Health Monitoring**: Comprehensive health checks and readiness probes
- **Metrics Export**: Prometheus-compatible metrics endpoint
- **Data Validation**: Customer data quality validation
- **Model Caching**: Intelligent model and preprocessor caching
- **Security Scanning**: Docker image vulnerability assessment

### 🔒 Rate Limiting
- **Token Bucket Algorithm**: Smooth rate limiting with burst capability
- **Per-IP and Per-Endpoint**: Granular rate limiting controls
- **Multiple Backends**: Memory-based and Redis-based storage
- **Configurable Rules**: Flexible rate limiting configuration
- **Real-time Monitoring**: Rate limiting statistics and metrics

### 📚 Documentation
- **OpenAPI/Swagger**: Interactive API documentation
- **ReDoc**: Alternative documentation interface
- **Comprehensive Examples**: Usage examples and integration guides

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Using CLI command
python -m src.cli serve --host 0.0.0.0 --port 8000

# Using Docker
docker run -p 8000:8000 churn-predictor:latest server
```

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/` | GET | API information | 100/min |
| `/health` | GET | Basic health check | 200/min |
| `/health/detailed` | GET | Comprehensive health check | 200/min |
| `/ready` | GET | Kubernetes readiness probe | 200/min |
| `/metrics` | GET | Prometheus metrics | 60/min |

### Prediction Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/predict` | POST | Single customer prediction | 30/min (burst: 10) |
| `/predict/batch` | POST | Batch predictions from CSV | 30/min (burst: 10) |

### Data & Validation

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/validate` | POST | Data validation | 30/min |
| `/cache/stats` | GET | Cache statistics | 100/min |
| `/cache/clear` | POST | Clear cache (auth required) | 10/min |

### Admin & Security

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/admin/security/scan` | POST | Security scan (auth required) | 10/min |
| `/admin/security/policies` | GET | Security policies | 10/min |
| `/admin/rate-limit/stats` | GET | Rate limit statistics | 10/min |
| `/admin/rate-limit/rules/{key}` | POST | Add rate limit rule (auth required) | 10/min |

## Rate Limiting Details

### Algorithm

The API uses the **Token Bucket Algorithm** for smooth rate limiting:

- **Capacity**: Maximum number of tokens (burst size)
- **Refill Rate**: Tokens added per second
- **Consumption**: Each request consumes tokens
- **Blocking**: Requests blocked when insufficient tokens

### Default Rules

```python
{
    "default": "100 requests per 60 seconds",
    "predict": "30 requests per 60 seconds (burst: 10)",
    "health": "200 requests per 60 seconds", 
    "metrics": "60 requests per 60 seconds",
    "train": "5 requests per 300 seconds",
    "admin": "10 requests per 60 seconds"
}
```

### Headers

All responses include rate limiting headers:

```http
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995260
Retry-After: 30
```

### Error Response

When rate limited (HTTP 429):

```json
{
    "error": "Rate limit exceeded",
    "detail": "Too many requests. Try again in 30 seconds.",
    "requests_remaining": 0,
    "reset_time": 1640995260.0
}
```

## Configuration

### Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# Rate Limiting
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_BACKEND=memory  # or 'redis'

# Authentication (for admin endpoints)
API_SECRET_KEY=your-secret-key
```

### Custom Rate Limit Rules

```bash
# Add custom rule via CLI
python -m src.cli rate-limit-add custom_endpoint \
    --requests 50 \
    --window 60 \
    --burst 15 \
    --description "Custom endpoint rate limit"

# View current rules
python -m src.cli rate-limit-stats
```

### Redis Backend

For distributed deployments, use Redis:

```bash
# Install Redis
pip install redis

# Configure Redis URL
export REDIS_URL=redis://localhost:6379/0

# Start with Redis backend
python -m src.cli serve --redis-backend
```

## Usage Examples

### 1. Single Prediction

```python
import requests

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer_data": {
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic"
        }
    }
)

# Check rate limiting headers
print(f"Remaining: {response.headers.get('X-RateLimit-Remaining')}")
print(f"Reset: {response.headers.get('X-RateLimit-Reset')}")

# Get prediction result
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

### 2. Batch Predictions

```python
import requests

# Upload CSV file for batch predictions
with open('customers.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/predict/batch",
        files={"file": f}
    )

result = response.json()
print(f"Processed {result['count']} customers")
print(f"Predictions: {result['predictions'][:5]}...")  # First 5
```

### 3. Health Monitoring

```python
import requests

# Basic health check
health = requests.get("http://localhost:8000/health").json()
print(f"Status: {health['status']}")

# Detailed health check
detailed = requests.get("http://localhost:8000/health/detailed").json()
print(f"Overall Status: {detailed['overall_status']}")

# Prometheus metrics
metrics = requests.get("http://localhost:8000/metrics").text
print("Metrics exported for Prometheus")
```

### 4. Rate Limit Management

```python
import requests

# Get rate limiting statistics
stats = requests.get("http://localhost:8000/admin/rate-limit/stats").json()
print(f"Backend: {stats['backend']}")
print(f"Active Rules: {stats['rules']}")

# Add custom rate limit rule (requires auth)
headers = {"Authorization": "Bearer your-token"}
rule_data = {
    "requests": 100,
    "window_seconds": 60,
    "burst_size": 25,
    "description": "Custom API limit"
}

response = requests.post(
    "http://localhost:8000/admin/rate-limit/rules/custom",
    json=rule_data,
    headers=headers
)
```

## Monitoring and Observability

### Prometheus Metrics

The `/metrics` endpoint exports comprehensive metrics:

```
# Request counts by endpoint
api_requests_total{endpoint="predict",method="POST"} 1234

# Request latency
api_request_duration_seconds{endpoint="predict"} 0.125

# Rate limiting metrics
rate_limit_requests_blocked_total{rule="predict"} 56
rate_limit_active_rules 8

# Cache performance
model_cache_hits_total 789
model_cache_misses_total 123
```

### Health Checks

```bash
# Kubernetes liveness probe
curl http://localhost:8000/health

# Kubernetes readiness probe  
curl http://localhost:8000/ready

# Detailed diagnostics
curl http://localhost:8000/health/detailed
```

### Logging

Structured logging with rate limiting events:

```json
{
    "timestamp": "2023-01-01T12:00:00Z",
    "level": "INFO",
    "message": "Rate limit rule added",
    "rule_key": "predict",
    "requests": 30,
    "window_seconds": 60
}
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t churn-predictor:latest .

# Run with rate limiting
docker run -d \
    -p 8000:8000 \
    -e REDIS_URL=redis://redis:6379/0 \
    --name churn-api \
    churn-predictor:latest server
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - WORKERS=4
    depends_on:
      - redis
    command: server

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
      - name: api
        image: churn-predictor:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: WORKERS
          value: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "500m"
```

## Security Considerations

### Rate Limiting Security

- **DoS Protection**: Prevents denial of service attacks
- **Resource Protection**: Limits resource consumption per client
- **Fair Usage**: Ensures equitable access to API resources
- **Abuse Prevention**: Blocks malicious automated requests

### Authentication

Admin endpoints require authentication:

```python
# Add Authorization header
headers = {"Authorization": "Bearer your-api-token"}
response = requests.post(
    "http://localhost:8000/admin/rate-limit/rules/custom",
    json=rule_data,
    headers=headers
)
```

### Input Validation

All endpoints validate input data:

- **Schema Validation**: Pydantic models ensure correct data types
- **Size Limits**: File uploads have size restrictions
- **Content Validation**: CSV files validated before processing

## Troubleshooting

### Common Issues

#### 1. Rate Limit Exceeded

```bash
# Check current limits
curl -I http://localhost:8000/predict

# Response headers:
# X-RateLimit-Remaining: 0
# X-RateLimit-Reset: 1640995260
# Retry-After: 30
```

**Solution**: Wait for reset time or contact admin to increase limits.

#### 2. Redis Connection Failed

```bash
# Error: Failed to connect to Redis
# Fallback: Using memory backend
```

**Solution**: Check Redis connection and configuration.

#### 3. Model Not Found

```bash
# Error: Model not found at models/churn_model.joblib
```

**Solution**: Train model first or provide MLflow run ID.

### CLI Diagnostics

```bash
# Check rate limiting status
python -m src.cli rate-limit-stats

# Test API health
python -m src.cli health-detailed

# View cache statistics
python -m src.cli cache-stats

# Check security policies
python -m src.cli security-policies
```

### Performance Tuning

```bash
# Increase workers for production
python -m src.cli serve --workers 4

# Use Redis for distributed deployments
export REDIS_URL=redis://cluster:6379/0

# Tune rate limits based on usage
python -m src.cli rate-limit-add predict --requests 100 --window 60
```

## API Client Libraries

### Python Client

```python
class ChurnPredictionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def predict(self, customer_data):
        response = self.session.post(
            f"{self.base_url}/predict",
            json={"customer_data": customer_data}
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(f"Rate limited. Retry after {retry_after}s")
        
        response.raise_for_status()
        return response.json()
    
    def health(self):
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = ChurnPredictionClient()
result = client.predict({
    "tenure": 12,
    "MonthlyCharges": 50.0
})
```

### JavaScript Client

```javascript
class ChurnPredictionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async predict(customerData) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ customer_data: customerData })
        });
        
        if (response.status === 429) {
            const retryAfter = response.headers.get('Retry-After');
            throw new Error(`Rate limited. Retry after ${retryAfter}s`);
        }
        
        return response.json();
    }
    
    async health() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
}

// Usage
const client = new ChurnPredictionClient();
const result = await client.predict({
    tenure: 12,
    MonthlyCharges: 50.0
});
```

## Support and Contact

For issues and questions:

- **Documentation**: http://localhost:8000/docs
- **Health Status**: http://localhost:8000/health
- **Rate Limit Stats**: Use `python -m src.cli rate-limit-stats`
- **Logs**: Check application logs for detailed error information

## Changelog

### Version 1.0.0
- Initial API implementation with rate limiting
- FastAPI framework with OpenAPI documentation
- Token bucket rate limiting algorithm
- Memory and Redis backend support
- Comprehensive test suite
- Docker containerization
- Kubernetes deployment support
- Prometheus metrics integration
- CLI management commands