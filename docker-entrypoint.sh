#!/bin/bash
set -e

# Docker entrypoint script for churn prediction application
# Provides flexible initialization and command handling

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Initialize application
init_app() {
    log "Initializing churn prediction application..."
    
    # Create necessary directories
    mkdir -p data/raw data/processed models logs
    
    # Set proper permissions
    chmod 755 data/raw data/processed models logs
    
    # Validate Python environment
    if ! python -c "import src.cli" 2>/dev/null; then
        error "Failed to import application modules"
        exit 1
    fi
    
    # Run basic health check
    if python -c "from src.health_check import is_healthy; exit(0 if is_healthy() else 1)" 2>/dev/null; then
        success "Application health check passed"
    else
        warn "Application health check failed - some dependencies may be missing"
    fi
    
    success "Application initialized successfully"
}

# Validate environment variables
validate_env() {
    log "Validating environment configuration..."
    
    # Check required Python path
    if [[ -z "${PYTHONPATH}" ]]; then
        export PYTHONPATH=/app
        log "Set PYTHONPATH to /app"
    fi
    
    # Validate cache configuration
    if [[ -n "${MODEL_CACHE_MAX_MEMORY_MB}" ]]; then
        if ! [[ "${MODEL_CACHE_MAX_MEMORY_MB}" =~ ^[0-9]+$ ]]; then
            error "MODEL_CACHE_MAX_MEMORY_MB must be a positive integer"
            exit 1
        fi
    fi
    
    # Validate log level
    if [[ -n "${LOG_LEVEL}" ]]; then
        case "${LOG_LEVEL}" in
            DEBUG|INFO|WARNING|ERROR|CRITICAL)
                log "Log level set to ${LOG_LEVEL}"
                ;;
            *)
                warn "Invalid LOG_LEVEL '${LOG_LEVEL}', using INFO"
                export LOG_LEVEL=INFO
                ;;
        esac
    fi
    
    success "Environment validation completed"
}

# Wait for dependencies
wait_for_dependencies() {
    if [[ -n "${MLFLOW_TRACKING_URI}" && "${MLFLOW_TRACKING_URI}" == http* ]]; then
        log "Waiting for MLflow tracking server..."
        
        # Extract host and port from URI
        HOST=$(echo "${MLFLOW_TRACKING_URI}" | sed 's|http://||' | cut -d: -f1)
        PORT=$(echo "${MLFLOW_TRACKING_URI}" | sed 's|http://||' | cut -d: -f2)
        
        # Wait for MLflow to be available
        for i in {1..30}; do
            if curl -s "${MLFLOW_TRACKING_URI}" > /dev/null 2>&1; then
                success "MLflow tracking server is available"
                break
            fi
            log "Waiting for MLflow... (attempt $i/30)"
            sleep 2
        done
    fi
}

# Run data validation if data exists
validate_data() {
    if [[ -f "data/raw/customer_data.csv" ]]; then
        log "Validating raw data..."
        
        if python -m src.cli validate data/raw/customer_data.csv --detailed 2>/dev/null; then
            success "Data validation passed"
        else
            warn "Data validation failed - proceed with caution"
        fi
    else
        log "No raw data found - skipping validation"
    fi
}

# Handle different startup modes
case "${1}" in
    "server")
        # Production server mode
        init_app
        validate_env
        wait_for_dependencies
        validate_data
        
        log "Starting application server..."
        shift
        exec gunicorn \
            --bind 0.0.0.0:8000 \
            --workers "${WORKERS:-1}" \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile - \
            --error-logfile - \
            --log-level "${LOG_LEVEL,,}" \
            "$@"
        ;;
        
    "worker")
        # Background worker mode
        init_app
        validate_env
        wait_for_dependencies
        
        log "Starting background worker..."
        shift
        exec python -m src.cli monitor "$@"
        ;;
        
    "train")
        # Training mode
        init_app
        validate_env
        wait_for_dependencies
        validate_data
        
        log "Starting model training..."
        shift
        exec python -m src.cli pipeline "$@"
        ;;
        
    "predict")
        # Prediction mode
        init_app
        validate_env
        wait_for_dependencies
        
        log "Running predictions..."
        shift
        exec python -m src.cli predict "$@"
        ;;
        
    "health")
        # Health check mode
        validate_env
        
        if [[ "${2}" == "detailed" ]]; then
            exec python -m src.cli health-detailed
        else
            exec python -m src.cli health
        fi
        ;;
        
    "cache")
        # Cache management mode
        init_app
        validate_env
        
        case "${2}" in
            "stats")
                exec python -m src.cli cache-stats
                ;;
            "clear")
                shift 2
                exec python -m src.cli cache-clear "$@"
                ;;
            *)
                error "Invalid cache command. Use 'stats' or 'clear'"
                exit 1
                ;;
        esac
        ;;
        
    "shell")
        # Interactive shell mode
        init_app
        validate_env
        
        log "Starting interactive shell..."
        exec /bin/bash
        ;;
        
    "python")
        # Python interpreter mode
        init_app
        validate_env
        
        shift
        exec python "$@"
        ;;
        
    *)
        # Default: run CLI commands
        init_app
        validate_env
        
        if [[ $# -eq 0 ]]; then
            # No arguments - run health check
            exec python -m src.cli health
        else
            # Pass through to CLI
            exec python -m src.cli "$@"
        fi
        ;;
esac