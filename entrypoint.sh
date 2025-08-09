#!/bin/bash
set -e

# Production entrypoint script for ML service

echo "==================================="
echo "Starting ML Service - Production"
echo "==================================="

# Set default values
export ML_ENVIRONMENT=${ML_ENVIRONMENT:-production}
export ML_LOG_LEVEL=${ML_LOG_LEVEL:-INFO}
export ML_WORKERS=${ML_WORKERS:-4}

# Wait for database to be ready
echo "Waiting for database connection..."
python -c "
import os
import time
import psutil
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        engine = create_engine(os.getenv('DATABASE_URL'))
        engine.connect()
        print('Database connection successful!')
        break
    except OperationalError as e:
        print(f'Database connection failed (attempt {retry_count + 1}/{max_retries}): {e}')
        retry_count += 1
        time.sleep(2)
else:
    print('Failed to connect to database after maximum retries')
    exit(1)
"

# Wait for Redis to be ready
echo "Waiting for Redis connection..."
python -c "
import os
import time
import redis

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))
        r.ping()
        print('Redis connection successful!')
        break
    except Exception as e:
        print(f'Redis connection failed (attempt {retry_count + 1}/{max_retries}): {e}')
        retry_count += 1
        time.sleep(2)
else:
    print('Failed to connect to Redis after maximum retries')
    exit(1)
"

# Run database migrations
echo "Running database migrations..."
python -c "
try:
    from src.database import run_migrations
    run_migrations()
    print('Database migrations completed successfully!')
except Exception as e:
    print(f'Migration failed: {e}')
    exit(1)
"

# Initialize system components
echo "Initializing system components..."
python -c "
try:
    # Initialize caching system
    from src.advanced_caching_optimization import get_global_cache
    cache = get_global_cache()
    print('✓ Caching system initialized')
    
    # Initialize monitoring
    from src.advanced_monitoring import get_global_monitoring
    monitoring = get_global_monitoring()
    print('✓ Monitoring system initialized')
    
    # Initialize error handling
    from src.error_handling_recovery import error_handler
    print('✓ Error handling system initialized')
    
    print('All system components initialized successfully!')
except Exception as e:
    print(f'System initialization failed: {e}')
    exit(1)
"

# Pre-warm cache if enabled
if [ "${ENABLE_CACHE_WARMING:-false}" = "true" ]; then
    echo "Pre-warming cache..."
    python -c "
    try:
        from src.cache_warming import warm_cache
        warm_cache()
        print('Cache warming completed!')
    except Exception as e:
        print(f'Cache warming failed: {e}')
        # Don't exit on cache warming failure
    "
fi

# Load initial models if available
echo "Loading initial models..."
python -c "
try:
    from src.model_loader import load_initial_models
    load_initial_models()
    print('Initial models loaded successfully!')
except Exception as e:
    print(f'Model loading failed: {e}')
    # Don't exit on model loading failure in case models are trained later
"

# Set up signal handlers for graceful shutdown
trap 'echo \"Received SIGTERM, shutting down gracefully...\"; kill -TERM \$PID; wait \$PID' TERM
trap 'echo \"Received SIGINT, shutting down gracefully...\"; kill -INT \$PID; wait \$PID' INT

# Start the main application
echo "Starting ML service..."
echo "Environment: $ML_ENVIRONMENT"
echo "Log Level: $ML_LOG_LEVEL"
echo "Workers: $ML_WORKERS"
echo "==================================="

# Determine the command to run based on environment
if [ "$1" = "web" ] || [ -z "$1" ]; then
    # Main web service
    exec python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers $ML_WORKERS \
        --log-level $(echo $ML_LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
        --access-log \
        --loop uvloop \
        --http httptools &
    PID=$!
elif [ "$1" = "worker" ]; then
    # Background worker
    exec python -m src.workers.background_worker &
    PID=$!
elif [ "$1" = "streaming" ]; then
    # Streaming worker
    exec python -m src.workers.streaming_worker &
    PID=$!
elif [ "$1" = "scheduler" ]; then
    # Task scheduler
    exec python -m src.scheduler.task_scheduler &
    PID=$!
else
    # Custom command
    exec "$@" &
    PID=$!
fi

# Wait for the process to finish
wait $PID