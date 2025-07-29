# Customer Churn Prediction MLOps Application
# Multi-stage Docker build for production deployment

# Build stage
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add labels for container metadata
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.title="churn-predictor" \
      org.opencontainers.image.description="Production-ready ML system for customer churn prediction" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/yourorg/customer-churn-predictor"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==23.0.0 uvicorn[standard]==0.24.0

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yml ./

# Production stage
FROM python:3.13-slim as production

# Set runtime arguments
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG APP_USER=appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -g ${GROUP_ID} ${APP_USER} && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m ${APP_USER}

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Copy and set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories with proper permissions
RUN mkdir -p data/raw data/processed models logs && \
    chown -R ${APP_USER}:${APP_USER} /app

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure application defaults
ENV MODEL_CACHE_MAX_ENTRIES=10
ENV MODEL_CACHE_MAX_MEMORY_MB=500
ENV MODEL_CACHE_TTL_SECONDS=3600
ENV LOG_LEVEL=INFO
ENV WORKERS=1

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.health_check import is_healthy; import sys; sys.exit(0 if is_healthy() else 1)"

# Switch to non-root user
USER ${APP_USER}

# Expose application port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command - can be overridden
CMD ["health"]

# Development stage
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER ${APP_USER}

# Override default command for development
CMD ["health", "detailed"]