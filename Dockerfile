# Customer Churn Prediction MLOps Application
# Multi-stage Docker build for production deployment

# Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add labels for container metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="churn-predictor" \
      org.label-schema.description="Customer churn prediction MLOps application" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0" \
      maintainer="Terragon Labs"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==23.0.0

# Production stage
FROM python:3.12-slim as production

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