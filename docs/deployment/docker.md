# Docker Deployment

This guide covers deploying the Customer Churn Predictor using Docker containers for production environments.

## Overview

The application provides multi-stage Docker builds optimized for:

- **Production**: Minimal, secure containers for production deployment
- **Development**: Extended containers with development tools
- **Training**: Specialized containers for model training workloads

## Quick Start

### Single Container

```bash
# Build the production image
docker build --target production -t churn-predictor:latest .

# Run the container
docker run -d \
  --name churn-predictor \
  -p 8000:8000 \
  -e API_KEY="your-secure-api-key" \
  -v $(pwd)/data:/app/data:rw \
  -v $(pwd)/models:/app/models:rw \
  churn-predictor:latest
```

### Docker Compose (Recommended)

```bash
# Start the full stack
docker-compose up -d

# View logs
docker-compose logs -f churn-predictor

# Stop the stack
docker-compose down
```

## Docker Image Details

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

```dockerfile
# Build stage - Contains build tools and compilers
FROM python:3.12-slim as builder

# Production stage - Minimal runtime environment  
FROM python:3.12-slim as production

# Development stage - Includes dev tools
FROM production as development
```

### Image Sizes

| Stage | Size | Use Case |
|-------|------|----------|
| Production | ~200MB | Production deployment |
| Development | ~350MB | Local development |
| Builder | ~450MB | Build artifacts only |

### Security Features

- **Non-root user**: Runs as `appuser` (UID 1000)
- **Minimal base**: Uses slim Python image
- **No package managers**: Runtime image has no pip/apt
- **Read-only filesystem**: Application code is read-only
- **Health checks**: Built-in container health monitoring

## Environment Configuration

### Required Environment Variables

```bash
# Security
API_KEY=your-secure-api-key-minimum-16-chars

# Application Configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
WORKERS=1                        # Number of worker processes
PYTHONPATH=/app                  # Python module path

# Model Cache Settings
MODEL_CACHE_MAX_ENTRIES=10       # Maximum cached models
MODEL_CACHE_MAX_MEMORY_MB=500    # Memory limit in MB
MODEL_CACHE_TTL_SECONDS=3600     # Cache TTL in seconds

# MLflow Configuration (optional)
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_RUN_ID=                   # Specific run ID (optional)
```

### Development Variables

```bash
# Development Mode
ENVIRONMENT=development
LOG_LEVEL=DEBUG
MODEL_CACHE_MAX_ENTRIES=5
MODEL_CACHE_MAX_MEMORY_MB=200
```

## Volume Mounts

### Required Volumes

```bash
# Data directories
./data:/app/data:rw              # Training and input data
./models:/app/models:rw          # Model artifacts
./logs:/app/logs:rw              # Application logs

# Configuration (read-only)
./config.yml:/app/config.yml:ro  # Application configuration
```

### Volume Permissions

The container runs as non-root user (`appuser:1000`). Ensure host directories have proper permissions:

```bash
# Set proper ownership
sudo chown -R 1000:1000 data/ models/ logs/

# Or use current user
chown -R $(id -u):$(id -g) data/ models/ logs/
```

## Health Checks

### Container Health Check

The Dockerfile includes a built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.health_check import is_healthy; import sys; sys.exit(0 if is_healthy() else 1)"
```

### Health Check Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Basic health status |
| `/health/detailed` | Comprehensive health report |
| `/ready` | Kubernetes readiness probe |

### Health Check Example

```bash
# Check container health
docker exec churn-predictor python -c "from src.health_check import is_healthy; print(is_healthy())"

# HTTP health check
curl http://localhost:8000/health
```

## Resource Limits

### Recommended Limits

```yaml
# docker-compose.yml
services:
  churn-predictor:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

### Memory Configuration

Adjust model cache based on available memory:

```bash
# For 1GB container
MODEL_CACHE_MAX_MEMORY_MB=400

# For 2GB container  
MODEL_CACHE_MAX_MEMORY_MB=800

# For 4GB container
MODEL_CACHE_MAX_MEMORY_MB=1600
```

## Production Deployment

### Production Dockerfile

```dockerfile
FROM python:3.12-slim as production

# Security: Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -s /bin/bash -m appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Set permissions
RUN mkdir -p data/raw data/processed models logs && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.health_check import is_healthy; import sys; sys.exit(0 if is_healthy() else 1)"

CMD ["python", "-m", "src.cli", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Environment

```bash
# Production environment file
# .env.production

# Security
API_KEY=production-api-key-very-secure-minimum-32-characters

# Performance
WORKERS=4
LOG_LEVEL=WARNING
MODEL_CACHE_MAX_ENTRIES=20
MODEL_CACHE_MAX_MEMORY_MB=1000

# External services
MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com
DATABASE_URL=postgresql://user:pass@db:5432/churn_db

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

## Monitoring Integration

### Prometheus Metrics

The container exposes Prometheus metrics at `/metrics`:

```bash
# Scrape metrics
curl http://localhost:8000/metrics
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Dashboard JSON available at
./monitoring/grafana/dashboards/churn-predictor.json
```

### Log Aggregation

Configure log forwarding:

```yaml
# docker-compose.yml
services:
  churn-predictor:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs churn-predictor

# Common issues:
# 1. Missing API_KEY environment variable
# 2. Permission issues with volumes
# 3. Port already in use
```

#### Health Check Failures

```bash
# Test health check manually
docker exec churn-predictor python -c "from src.health_check import is_healthy; print(is_healthy())"

# Check specific health components
docker exec churn-predictor python -c "
from src.health_check import HealthChecker
checker = HealthChecker()
print(checker.get_comprehensive_health())
"
```

#### Memory Issues

```bash
# Check memory usage
docker stats churn-predictor

# Reduce cache size
docker exec churn-predictor \
  -e MODEL_CACHE_MAX_MEMORY_MB=200 \
  churn-predictor:latest
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
docker run -it \
  -e LOG_LEVEL=DEBUG \
  -e PYTHONPATH=/app \
  churn-predictor:latest \
  python -c "from src.health_check import get_comprehensive_health; import json; print(json.dumps(get_comprehensive_health(), indent=2))"
```

## Next Steps

- [Docker Compose →](docker-compose.md) - Multi-service orchestration
- [Production Setup →](production.md) - Production deployment guide
- [Monitoring →](monitoring.md) - Monitoring and observability
- [Health Checks →](health-checks.md) - Health check configuration