# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Customer Churn Predictor MLOps platform.

## Quick Diagnostics

### Health Check Commands

```bash
# Basic health check
python -m src.cli health

# Detailed health check
python -m src.cli health-detailed

# API health check
curl http://localhost:8000/health
```

### Log Analysis

```bash
# View application logs
tail -f logs/application.log

# Check specific component logs
grep "ERROR" logs/application.log
grep "health_check" logs/application.log

# Docker container logs
docker logs churn-predictor
```

## Common Issues

### 🚫 Authentication Issues

#### Problem: API returns 401 Unauthorized

**Symptoms:**
- API requests return `401 Unauthorized`
- Error message: "Missing authentication credentials"

**Solutions:**

1. **Check API Key Configuration**
   ```bash
   # Verify API_KEY is set
   echo $API_KEY
   
   # Must be at least 16 characters
   export API_KEY="your-secure-api-key-here"
   ```

2. **Verify Request Headers**
   ```bash
   # Correct format
   curl -H "Authorization: Bearer your-api-key" http://localhost:8000/predict
   
   # Check header format in your client
   headers = {"Authorization": "Bearer your-api-key"}
   ```

3. **Docker Environment Variables**
   ```bash
   # Check container environment
   docker exec churn-predictor env | grep API_KEY
   
   # Restart with correct environment
   docker run -e API_KEY="your-key" churn-predictor:latest
   ```

#### Problem: Token format validation error

**Symptoms:**
- Error: "Invalid token format"
- API key is shorter than 16 characters

**Solution:**
```bash
# Generate secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Use generated key
export API_KEY="generated-secure-key-here"
```

### 📊 Model Loading Issues

#### Problem: Model not found errors

**Symptoms:**
- Error: "No model files found"
- Health check shows model unavailable

**Solutions:**

1. **Check Model Directory**
   ```bash
   # Verify models directory exists
   ls -la models/
   
   # Check for model files
   find models/ -name "*.pkl" -o -name "*.joblib"
   ```

2. **Train a Model**
   ```bash
   # Train initial model
   python -m src.cli train
   
   # Or run full pipeline
   python -m src.cli pipeline
   ```

3. **Download from MLflow**
   ```bash
   # Set MLflow tracking URI
   export MLFLOW_TRACKING_URI="your-mlflow-server"
   
   # Download latest model
   python -c "
   from src.mlflow_utils import MLflowArtifactManager
   with MLflowArtifactManager() as manager:
       manager.download_latest_model('models/')
   "
   ```

#### Problem: Model loading takes too long

**Symptoms:**
- Long API response times
- Timeout errors during prediction

**Solutions:**

1. **Enable Model Caching**
   ```bash
   # Configure cache settings
   export MODEL_CACHE_MAX_ENTRIES=10
   export MODEL_CACHE_MAX_MEMORY_MB=500
   export MODEL_CACHE_TTL_SECONDS=3600
   ```

2. **Pre-warm Cache**
   ```bash
   # Make a test prediction to warm cache
   python -m src.cli predict data/processed/processed_features.csv
   ```

### 🗄️ Data Issues

#### Problem: Data validation failures

**Symptoms:**
- Error: "Data validation failed"
- Invalid column names or data types

**Solutions:**

1. **Check Data Schema**
   ```bash
   # Validate input data
   python scripts/validate_data.py data/raw/customer_data.csv
   
   # View expected schema
   python -c "
   from src.data_validation import ChurnDataValidator
   validator = ChurnDataValidator()
   print(validator.get_schema_info())
   "
   ```

2. **Fix Common Data Issues**
   ```python
   import pandas as pd
   
   # Load and inspect data
   df = pd.read_csv('data/raw/customer_data.csv')
   print(df.info())
   print(df.head())
   
   # Fix common issues
   df.columns = df.columns.str.lower().str.replace(' ', '_')
   df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
   df.dropna(inplace=True)
   
   # Save corrected data
   df.to_csv('data/raw/customer_data_fixed.csv', index=False)
   ```

3. **Preprocessing Issues**
   ```bash
   # Re-run preprocessing with validation
   python -m src.cli preprocess --validate
   
   # Check preprocessing logs
   grep "preprocessing" logs/application.log
   ```

### 🐳 Docker Issues

#### Problem: Container won't start

**Symptoms:**
- Container exits immediately
- Docker build failures

**Solutions:**

1. **Check Docker Logs**
   ```bash
   # View container logs
   docker logs churn-predictor
   
   # Follow logs in real-time
   docker logs -f churn-predictor
   ```

2. **Common Startup Issues**
   ```bash
   # Issue: Missing environment variables
   docker run -e API_KEY="test-key" churn-predictor:latest
   
   # Issue: Volume permission problems
   sudo chown -R 1000:1000 data/ models/ logs/
   
   # Issue: Port conflicts
   docker run -p 8001:8000 churn-predictor:latest
   ```

3. **Debug Container**
   ```bash
   # Run interactive container
   docker run -it churn-predictor:latest bash
   
   # Test health check manually
   python -c "from src.health_check import is_healthy; print(is_healthy())"
   ```

#### Problem: Container health check failing

**Symptoms:**
- Container marked as unhealthy
- Health endpoint returns errors

**Solutions:**

1. **Check Health Components**
   ```bash
   # Test individual health checks
   docker exec churn-predictor python -c "
   from src.health_check import HealthChecker
   checker = HealthChecker()
   print('Model:', checker.check_model_availability())
   print('Data:', checker.check_data_directories())
   print('Dependencies:', checker.check_dependencies())
   "
   ```

2. **Fix Common Health Issues**
   ```bash
   # Create missing directories
   docker exec churn-predictor mkdir -p data/raw data/processed models
   
   # Fix permissions
   docker exec churn-predictor chown -R appuser:appuser /app
   ```

### 🌐 API Issues

#### Problem: API server won't start

**Symptoms:**
- Port binding errors
- Uvicorn startup failures

**Solutions:**

1. **Check Port Availability**
   ```bash
   # Check if port is in use
   netstat -tulpn | grep :8000
   
   # Use different port
   python -m src.cli api --port 8001
   ```

2. **Server Configuration**
   ```bash
   # Check server logs
   grep "uvicorn" logs/application.log
   
   # Start with debug mode
   python -m src.cli api --host 0.0.0.0 --port 8000 --log-level debug
   ```

#### Problem: Rate limiting issues

**Symptoms:**
- 429 Too Many Requests errors
- Rate limit headers show zero remaining

**Solutions:**

1. **Check Rate Limit Status**
   ```bash
   # View rate limit headers
   curl -I http://localhost:8000/health
   
   # Check rate limit stats
   curl http://localhost:8000/admin/rate-limit/stats
   ```

2. **Adjust Rate Limits**
   ```python
   # Temporarily increase limits
   from src.rate_limiter import get_rate_limiter
   limiter = get_rate_limiter()
   limiter.add_rule("health", RateLimitRule(requests=1000, window_seconds=3600))
   ```

### 📈 Performance Issues

#### Problem: Slow prediction responses

**Symptoms:**
- High API response times
- Timeout errors

**Solutions:**

1. **Enable Performance Monitoring**
   ```bash
   # Check metrics endpoint
   curl http://localhost:8000/metrics
   
   # Look for performance metrics
   grep "prediction_latency" logs/application.log
   ```

2. **Optimize Model Cache**
   ```bash
   # Increase cache size
   export MODEL_CACHE_MAX_ENTRIES=20
   export MODEL_CACHE_MAX_MEMORY_MB=1000
   
   # Monitor cache performance
   python -c "
   from src.model_cache import get_model_cache
   cache = get_model_cache()
   print(cache.get_stats())
   "
   ```

3. **Resource Monitoring**
   ```bash
   # Monitor system resources
   python -c "
   from src.health_check import HealthChecker
   checker = HealthChecker()
   print(checker.check_resource_usage())
   "
   ```

### 🔧 MLflow Issues

#### Problem: MLflow connection failures

**Symptoms:**
- Cannot connect to MLflow server
- Experiment tracking not working

**Solutions:**

1. **Check MLflow Configuration**
   ```bash
   # Verify MLflow URI
   echo $MLFLOW_TRACKING_URI
   
   # Test connection
   python -c "
   import mlflow
   print(mlflow.get_tracking_uri())
   experiments = mlflow.search_experiments()
   print(f'Found {len(experiments)} experiments')
   "
   ```

2. **Start Local MLflow**
   ```bash
   # Start local MLflow server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0
   
   # Or use Docker Compose
   docker-compose up mlflow
   ```

## Advanced Diagnostics

### System Health Dashboard

```python
#!/usr/bin/env python3
"""Comprehensive system health dashboard."""

import json
from src.health_check import get_comprehensive_health
from src.metrics import get_metrics_collector

def generate_health_dashboard():
    """Generate comprehensive health report."""
    
    # Get health status
    health = get_comprehensive_health()
    
    # Get metrics
    try:
        metrics = get_metrics_collector()
        metrics_data = {
            "active_requests": metrics._active_requests.value,
            "total_predictions": metrics._prediction_count.value,
            "error_count": metrics._error_count.value,
        }
    except Exception as e:
        metrics_data = {"error": str(e)}
    
    # Combine data
    dashboard = {
        "timestamp": health["checks"]["basic"]["timestamp"],
        "overall_status": health["overall_status"],
        "health_summary": health["summary"],
        "metrics": metrics_data,
        "component_health": {
            "model": health["checks"]["model"]["model_available"],
            "data": health["checks"]["data_directories"]["data_dirs_accessible"],
            "dependencies": health["checks"]["dependencies"]["dependencies_available"],
            "resources": health["checks"]["resource_usage"]["resources_healthy"]
        }
    }
    
    return dashboard

if __name__ == "__main__":
    dashboard = generate_health_dashboard()
    print(json.dumps(dashboard, indent=2))
```

### Log Analysis Script

```bash
#!/bin/bash
# analyze_logs.sh - Automated log analysis

echo "=== Error Analysis ==="
grep -c "ERROR" logs/application.log
echo

echo "=== Recent Errors ==="
grep "ERROR" logs/application.log | tail -5
echo

echo "=== Performance Analysis ==="
grep "prediction_latency" logs/application.log | tail -5
echo

echo "=== Health Check Summary ==="
grep "Health check" logs/application.log | tail -3
echo

echo "=== Rate Limiting ==="
grep "Rate limit" logs/application.log | tail -5
```

## Getting Help

### Collecting Debug Information

Before seeking help, collect this information:

```bash
# System information
python --version
docker --version
docker-compose --version

# Application health
python -m src.cli health-detailed > health_report.json

# Recent logs
tail -100 logs/application.log > recent_logs.txt

# Container status (if using Docker)
docker ps -a > container_status.txt
docker logs churn-predictor > container_logs.txt

# Environment variables (sanitized)
env | grep -E "(LOG_LEVEL|MODEL_CACHE|MLFLOW)" > environment.txt
```

### Support Channels

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check the full documentation for detailed guides
- **Community**: Join discussions in the project community

### Emergency Procedures

#### Service Degradation

```bash
# 1. Check health status
curl http://localhost:8000/health

# 2. Restart service
docker-compose restart churn-predictor

# 3. Check logs for errors
docker logs churn-predictor | grep ERROR

# 4. Scale down if needed
docker-compose scale churn-predictor=1
```

#### Data Corruption

```bash
# 1. Stop service
docker-compose stop churn-predictor

# 2. Backup current data
cp -r data/ data_backup_$(date +%Y%m%d)

# 3. Restore from backup
cp -r data_backup_YYYYMMDD/ data/

# 4. Restart service
docker-compose start churn-predictor
```

Remember: Always test solutions in a development environment before applying to production!