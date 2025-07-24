# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Customer Churn Prediction application using Docker and Docker Compose.

## Quick Start

### Prerequisites

- Docker 20.10+ 
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Basic Deployment

```bash
# Build and start the application
./scripts/docker-build.sh
./scripts/docker-deploy.sh

# Or using Docker Compose directly
docker-compose up -d

# Check application health
curl http://localhost:8000/health
```

## Deployment Modes

### 1. Production Deployment

Basic production setup with health checks and monitoring:

```bash
# Build production image
./scripts/docker-build.sh --target production

# Deploy production services
./scripts/docker-deploy.sh --env production

# Access application
curl http://localhost:8000
```

### 2. Development Environment

Extended development setup with development tools:

```bash
# Deploy with development profile
./scripts/docker-deploy.sh --profile development --env development

# Access development container
docker exec -it churn-predictor-dev bash

# Run tests inside container
docker exec churn-predictor-dev python -m pytest
```

### 3. Complete MLOps Stack

Full deployment with MLflow tracking and monitoring:

```bash
# Deploy complete stack
./scripts/docker-deploy.sh --profile mlflow,monitoring

# Access services
# - Application: http://localhost:8000
# - MLflow: http://localhost:5000  
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### 4. Training Pipeline

Batch training with MLflow integration:

```bash
# Start MLflow server
./scripts/docker-deploy.sh --profile mlflow

# Run training pipeline
./scripts/docker-deploy.sh --profile training
```

## Service Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Application log level |
| `MODEL_CACHE_MAX_ENTRIES` | `10` | Maximum cached models |
| `MODEL_CACHE_MAX_MEMORY_MB` | `500` | Cache memory limit |
| `MODEL_CACHE_TTL_SECONDS` | `3600` | Cache entry TTL |
| `WORKERS` | `1` | Number of worker processes |
| `MLFLOW_TRACKING_URI` | - | MLflow server URL |
| `GRAFANA_PASSWORD` | `admin` | Grafana admin password |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Training and test data |
| `./models` | `/app/models` | Model artifacts |
| `./logs` | `/app/logs` | Application logs |
| `./config.yml` | `/app/config.yml` | Configuration file |

## Docker Images

### Multi-Stage Build

The Dockerfile supports multiple build targets:

- **production**: Optimized runtime image (~500MB)
- **development**: Extended image with dev tools (~800MB)

### Image Tags

```bash
# Version-specific tags
churn-predictor:1.0.0
churn-predictor:1.0.0-production
churn-predictor:1.0.0-development

# Latest tags
churn-predictor:latest
churn-predictor:dev
```

## Service Orchestration

### Docker Compose Profiles

| Profile | Services | Purpose |
|---------|----------|---------|
| (default) | churn-predictor | Basic application |
| `development` | churn-predictor-dev | Development environment |
| `mlflow` | mlflow | Model tracking |
| `training` | churn-trainer | Batch training |
| `monitoring` | prometheus, grafana | Metrics and visualization |

### Service Dependencies

```
churn-predictor
├── data volumes
├── model volumes
└── logging

churn-trainer
├── mlflow (optional)
├── data volumes
└── model volumes

monitoring stack
├── prometheus
└── grafana
```

## Operations

### Health Checks

All services include health checks:

```bash
# Application health
docker run --rm churn-predictor:latest health

# Detailed health check
docker run --rm churn-predictor:latest health detailed

# Readiness check
docker run --rm churn-predictor:latest ready
```

### Logging

Centralized logging configuration:

```bash
# View application logs
docker-compose logs -f churn-predictor

# View all service logs  
docker-compose logs -f

# Filter by log level
docker-compose logs -f | grep ERROR
```

### Monitoring

Access monitoring dashboards:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Metrics**: http://localhost:8000/metrics

### Cache Management

Manage model cache:

```bash
# View cache statistics
docker exec churn-predictor python -m src.cli cache-stats

# Clear cache
docker exec churn-predictor python -m src.cli cache-clear --confirm
```

## Container Commands

### Available Entrypoint Commands

| Command | Description |
|---------|-------------|
| `server` | Start web server |
| `worker` | Start background worker |
| `train` | Run training pipeline |
| `predict` | Run batch predictions |
| `health` | Health check |
| `cache` | Cache management |
| `shell` | Interactive shell |

### Examples

```bash
# Start web server
docker run -p 8000:8000 churn-predictor:latest server

# Run training
docker run -v $(pwd)/data:/app/data churn-predictor:latest train

# Interactive debugging
docker run -it churn-predictor:latest shell

# Run specific CLI command
docker run churn-predictor:latest python -m src.cli --help
```

## Production Deployment

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Application | 1 core | 1GB | 2GB |
| MLflow | 0.5 core | 512MB | 5GB |
| Prometheus | 0.5 core | 512MB | 10GB |
| Grafana | 0.25 core | 256MB | 1GB |

### Security Considerations

1. **Non-root user**: Containers run as non-root user
2. **Read-only filesystem**: Application code is read-only
3. **Secrets management**: Use Docker secrets or external secret stores
4. **Network isolation**: Services use isolated bridge network
5. **Resource limits**: Set memory and CPU limits

### High Availability

For production HA deployment:

```yaml
# docker-compose.prod.yml
services:
  churn-predictor:
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        max_attempts: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Backup and Recovery

Critical data backup:

```bash
# Backup data volumes
docker run --rm -v churn-prediction_mlflow-data:/data \
  -v $(pwd)/backup:/backup alpine \
  tar czf /backup/mlflow-$(date +%Y%m%d).tar.gz -C /data .

# Backup model artifacts
tar czf models-backup-$(date +%Y%m%d).tar.gz models/
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports 8000, 5000, 3000, 9090 are available
2. **Memory limits**: Increase Docker memory allocation
3. **Volume permissions**: Ensure host directories are writable
4. **Network issues**: Check firewall and Docker network settings

### Debug Commands

```bash
# Check container logs
docker-compose logs churn-predictor

# Inspect container
docker inspect churn-predictor

# Execute commands in container
docker exec -it churn-predictor bash

# Check resource usage
docker stats

# Network debugging
docker network ls
docker network inspect churn-prediction_churn-net
```

### Performance Tuning

1. **Increase worker processes**: Set `WORKERS` environment variable
2. **Optimize cache settings**: Tune cache memory and TTL
3. **Resource allocation**: Increase Docker memory/CPU limits
4. **Volume optimization**: Use named volumes for better performance

## Kubernetes Deployment

For Kubernetes deployment, see the generated manifests:

```bash
# Generate Kubernetes manifests
kompose convert -f docker-compose.yml

# Apply to cluster
kubectl apply -f churn-predictor-*.yaml
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: ./scripts/docker-build.sh --target production
    
    - name: Run tests
      run: docker run --rm churn-predictor:latest python -m pytest
    
    - name: Push to registry
      run: ./scripts/docker-build.sh --push --registry ${{ secrets.REGISTRY }}
```

## Support

For additional support:

- Check service logs: `docker-compose logs`
- Review health checks: `docker-compose exec churn-predictor python -m src.cli health-detailed`
- Monitor resource usage: `docker stats`
- Validate configuration: `docker-compose config`