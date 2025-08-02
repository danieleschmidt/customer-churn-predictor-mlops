# Build Guide - Customer Churn Predictor

This guide covers building, packaging, and deploying the Customer Churn Predictor application.

## Overview

The project uses a multi-stage Docker build process for optimal production deployment, with comprehensive build automation through Make and CI/CD pipelines.

## Build Requirements

### System Requirements
- Docker 20.10+ and Docker Compose v2+
- Python 3.10+ (for local development)
- Make (for build automation)
- Git (for version information)

### Development Requirements
- 4GB+ RAM available to Docker
- 10GB+ disk space for images and data
- Network access for dependency downloads

## Quick Start

### Production Build
```bash
# Build production image
make build

# Run production container
make run

# Or using Docker Compose
docker-compose up churn-predictor
```

### Development Build
```bash
# Build development image
make build-dev

# Run development container with live reload
make run-dev

# Or using Docker Compose with profiles
docker-compose --profile development up churn-predictor-dev
```

## Docker Architecture

### Multi-Stage Build

The Dockerfile uses a multi-stage build for efficiency:

1. **Builder Stage**: Compiles dependencies and installs build tools
2. **Production Stage**: Minimal runtime environment
3. **Development Stage**: Extended with development tools

### Image Variants

| Target | Size | Use Case | Features |
|--------|------|----------|----------|
| `production` | ~200MB | Production deployment | Minimal, optimized |
| `development` | ~500MB | Local development | Dev tools, debugging |

### Build Arguments

```bash
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VERSION=1.2.3 \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --target production \
  -t churn-predictor:1.2.3 .
```

## Build Automation

### Make Targets

```bash
# Core build commands
make build              # Build production image
make build-dev         # Build development image
make clean             # Clean artifacts
make clean-docker      # Clean Docker artifacts

# Testing builds
make ci-build          # CI/CD build process
make test-docker       # Test Docker builds

# Release builds
make push              # Push to registry
make release           # Full release process
```

### Environment Variables

```bash
# Build configuration
export VERSION=1.2.3
export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
export VCS_REF=$(git rev-parse --short HEAD)

# Registry configuration
export DOCKER_REGISTRY=ghcr.io
export DOCKER_REPO=yourorg/customer-churn-predictor
```

## Docker Compose Services

### Core Services

#### Application Service (`churn-predictor`)
```yaml
services:
  churn-predictor:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data:rw
      - ./models:/app/models:rw
```

#### Development Service (`churn-predictor-dev`)
```yaml
services:
  churn-predictor-dev:
    build:
      target: development
    ports:
      - "8001:8000"
      - "8888:8888"  # Jupyter
    volumes:
      - .:/app:rw    # Live code reload
```

### Supporting Services

#### MLflow Tracking (`mlflow`)
- Experiment tracking and model registry
- Persistent data storage
- Web UI on port 5000

#### Monitoring Stack (`prometheus`, `grafana`)
- Metrics collection and visualization
- Custom dashboards for ML metrics
- Alerting capabilities

#### Training Service (`churn-trainer`)
- Batch model training
- Automated retraining workflows
- Integration with MLflow

### Service Profiles

```bash
# Development profile
docker-compose --profile development up

# Full monitoring stack
docker-compose --profile monitoring up

# Training workflow
docker-compose --profile training up
```

## Container Security

### Security Features

1. **Non-root User**: Application runs as non-privileged user
2. **Minimal Base**: Uses Python slim image
3. **Multi-stage**: Removes build tools from final image
4. **Read-only Filesystem**: Where possible
5. **Health Checks**: Built-in health monitoring

### Security Scanning

```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image churn-predictor:latest

# Scan base images
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image python:3.12-slim
```

### Security Best Practices

- Regular base image updates
- Vulnerability scanning in CI/CD
- Secrets via environment variables
- Network isolation with custom networks
- Resource limits and health checks

## Performance Optimization

### Build Optimization

1. **Layer Caching**: Dependencies copied before source code
2. **Multi-stage**: Separate build and runtime stages
3. **Minimal Base**: Python slim instead of full image
4. **Dependency Locking**: Use requirements.lock files

### Runtime Optimization

1. **Resource Limits**: Memory and CPU constraints
2. **Health Checks**: Proper monitoring
3. **Graceful Shutdown**: Signal handling
4. **Persistent Volumes**: Data and model storage

### Build Performance Tips

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from churn-predictor:latest .

# Parallel builds
docker-compose build --parallel
```

## Deployment Strategies

### Local Development

```bash
# Quick start
make setup && make run-dev

# With full stack
docker-compose --profile development --profile monitoring up
```

### Staging Environment

```bash
# Build and deploy
make build
docker tag churn-predictor:latest registry.example.com/churn-predictor:staging
docker push registry.example.com/churn-predictor:staging

# Deploy to staging
ENVIRONMENT=staging docker-compose up -d
```

### Production Deployment

```bash
# Production build with optimization
make build
make push

# Deploy with monitoring
docker-compose --profile monitoring up -d
```

## Container Registry

### Supported Registries

- **GitHub Container Registry** (ghcr.io) - Recommended
- **Docker Hub** - Public images
- **AWS ECR** - Enterprise deployment
- **Azure Container Registry** - Azure deployment

### Image Tagging Strategy

```bash
# Version tags
churn-predictor:1.2.3
churn-predictor:1.2
churn-predictor:1

# Environment tags  
churn-predictor:latest
churn-predictor:staging
churn-predictor:dev

# Git-based tags
churn-predictor:sha-abc123f
churn-predictor:pr-456
```

## Software Bill of Materials (SBOM)

### Generating SBOM

```bash
# Generate SBOM
python scripts/generate_sbom.py

# Include in build
make sbom
```

### SBOM Contents

- Python dependencies with versions
- System packages and libraries
- Build metadata and Git information
- License information
- Supply chain artifacts

## Monitoring and Observability

### Container Metrics

- CPU and memory usage
- Network and disk I/O
- Container health status
- Application metrics

### Logging

```bash
# View application logs
docker-compose logs -f churn-predictor

# Structured logging with JSON format
docker-compose logs churn-predictor | jq .
```

### Health Checks

```bash
# Manual health check
docker exec churn-predictor python -c "from src.health_check import is_healthy; print(is_healthy())"

# Health check endpoint
curl http://localhost:8000/health
```

## Troubleshooting

### Common Build Issues

#### Dependency Installation Failures
```bash
# Check package availability
docker run --rm python:3.12-slim pip install package-name

# Use build cache
docker build --no-cache .
```

#### Resource Constraints
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Check current limits
docker system info | grep -i memory
```

#### Network Issues
```bash
# Test network connectivity
docker run --rm alpine ping google.com

# Use different DNS
docker build --build-arg DNS=8.8.8.8 .
```

### Runtime Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs churn-predictor

# Debug with shell
docker run -it --entrypoint /bin/bash churn-predictor:latest
```

#### Performance Problems
```bash
# Monitor resource usage
docker stats churn-predictor

# Profile application
docker exec churn-predictor python -m cProfile -o profile.stats src/api.py
```

#### Volume Mount Issues
```bash
# Check permissions
ls -la data/ models/ logs/

# Fix ownership
sudo chown -R 1000:1000 data/ models/ logs/
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build and Test
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: make build
      - name: Test image
        run: make test-docker
```

### Build Matrix

Test across multiple configurations:
- Python versions: 3.10, 3.11, 3.12
- Operating systems: Ubuntu, Windows, macOS
- Docker versions: Latest stable

### Automated Deployment

```yaml
name: Deploy
on:
  push:
    tags: ['v*']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Build and push
        run: |
          make build
          make push
      - name: Deploy to staging
        run: make deploy-staging
```

## Advanced Topics

### Custom Base Images

Create organization-specific base images:

```dockerfile
FROM python:3.12-slim as org-base
RUN apt-get update && apt-get install -y \
    common-tools \
    security-updates
# Add organization certificates, tools, etc.

FROM org-base as churn-predictor
# Application-specific layers
```

### Multi-Architecture Builds

```bash
# Enable buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t churn-predictor:multiarch .
```

### Build Secrets

```bash
# Use build secrets for private dependencies
docker build \
  --secret id=pypi-token,src=./pypi-token.txt \
  --build-arg PIP_EXTRA_INDEX_URL=https://user:${PYPI_TOKEN}@pypi.example.com/simple/ \
  .
```

## Best Practices Summary

### Security
- ✅ Use specific base image tags
- ✅ Run as non-root user
- ✅ Regular vulnerability scanning
- ✅ Minimal attack surface
- ✅ Secrets via environment variables

### Performance
- ✅ Multi-stage builds
- ✅ Layer caching optimization
- ✅ Minimal base images
- ✅ Resource limits
- ✅ Health checks

### Maintainability
- ✅ Consistent naming conventions
- ✅ Comprehensive documentation
- ✅ Automated testing
- ✅ Version tagging strategy
- ✅ SBOM generation

### Operations
- ✅ Container health monitoring
- ✅ Structured logging
- ✅ Metrics collection
- ✅ Graceful shutdown
- ✅ Persistent data management