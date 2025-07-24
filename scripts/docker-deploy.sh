#!/bin/bash
set -e

# Docker deployment script for churn prediction application
# Supports different deployment environments and configurations

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="churn-prediction"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
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

# Parse command line arguments
ACTION="up"
PROFILES=""
DETACH=true
BUILD=false
FORCE_RECREATE=false

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [ACTION]

Actions:
  up              Start services (default)
  down            Stop services
  restart         Restart services
  logs            Show service logs
  status          Show service status
  clean           Clean up containers and volumes

Options:
  --env ENV               Environment (production, development, staging) [default: production]
  --profile PROFILE       Docker Compose profiles to include (development, mlflow, training, monitoring)
  --foreground           Run in foreground (don't detach)
  --build                Build images before starting
  --force-recreate       Force recreate containers
  -h, --help             Show this help message

Examples:
  $0                                    # Start production services
  $0 --profile development              # Start with development profile
  $0 --profile mlflow,monitoring        # Start with MLflow and monitoring
  $0 --env staging --build              # Build and start staging environment
  $0 down                              # Stop all services
  $0 clean                             # Clean up everything

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --profile)
            PROFILES="$2"
            shift 2
            ;;
        --foreground)
            DETACH=false
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --force-recreate)
            FORCE_RECREATE=true
            shift
            ;;
        up|down|restart|logs|status|clean)
            ACTION="$1"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    production|development|staging)
        ;;
    *)
        error "Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Set environment variables
export ENVIRONMENT
export VERSION="${VERSION:-latest}"
export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
export VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')

# Create .env file for Docker Compose
create_env_file() {
    log "Creating environment configuration..."
    
    cat > .env << EOF
# Generated environment file for Docker deployment
ENVIRONMENT=${ENVIRONMENT}
VERSION=${VERSION}
BUILD_DATE=${BUILD_DATE}
VCS_REF=${VCS_REF}

# Application configuration
LOG_LEVEL=${LOG_LEVEL:-INFO}
MODEL_CACHE_MAX_ENTRIES=${MODEL_CACHE_MAX_ENTRIES:-10}
MODEL_CACHE_MAX_MEMORY_MB=${MODEL_CACHE_MAX_MEMORY_MB:-500}
MODEL_CACHE_TTL_SECONDS=${MODEL_CACHE_TTL_SECONDS:-3600}

# Resource configuration
WORKERS=${WORKERS:-1}

# External services
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-}
MLFLOW_RUN_ID=${MLFLOW_RUN_ID:-}

# Monitoring
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin}
EOF
    
    success "Environment configuration created"
}

# Prepare Docker Compose command
prepare_compose_cmd() {
    local cmd="docker-compose"
    
    # Add project name
    cmd="$cmd -p $PROJECT_NAME"
    
    # Add compose file
    cmd="$cmd -f $COMPOSE_FILE"
    
    # Add profiles if specified
    if [[ -n "$PROFILES" ]]; then
        IFS=',' read -ra PROFILE_ARRAY <<< "$PROFILES"
        for profile in "${PROFILE_ARRAY[@]}"; do
            cmd="$cmd --profile $profile"
        done
    fi
    
    echo "$cmd"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Execute action
execute_action() {
    local compose_cmd=$(prepare_compose_cmd)
    
    case $ACTION in
        up)
            log "Starting services..."
            
            local up_args=""
            
            if [[ "$DETACH" == "true" ]]; then
                up_args="$up_args -d"
            fi
            
            if [[ "$BUILD" == "true" ]]; then
                up_args="$up_args --build"
            fi
            
            if [[ "$FORCE_RECREATE" == "true" ]]; then
                up_args="$up_args --force-recreate"
            fi
            
            if $compose_cmd up $up_args; then
                success "Services started successfully"
                
                # Show service status
                echo
                $compose_cmd ps
                
                # Show URLs
                echo
                log "Service URLs:"
                echo "  Application: http://localhost:8000"
                if [[ "$PROFILES" == *"mlflow"* ]]; then
                    echo "  MLflow:      http://localhost:5000"
                fi
                if [[ "$PROFILES" == *"monitoring"* ]]; then
                    echo "  Grafana:     http://localhost:3000 (admin/admin)"
                    echo "  Prometheus:  http://localhost:9090"
                fi
            else
                error "Failed to start services"
                exit 1
            fi
            ;;
            
        down)
            log "Stopping services..."
            
            if $compose_cmd down; then
                success "Services stopped successfully"
            else
                error "Failed to stop services"
                exit 1
            fi
            ;;
            
        restart)
            log "Restarting services..."
            
            if $compose_cmd restart; then
                success "Services restarted successfully"
                $compose_cmd ps
            else
                error "Failed to restart services"
                exit 1
            fi
            ;;
            
        logs)
            log "Showing service logs..."
            $compose_cmd logs -f
            ;;
            
        status)
            log "Service status:"
            $compose_cmd ps
            ;;
            
        clean)
            warn "This will remove all containers, networks, and volumes"
            read -p "Are you sure? (y/N) " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log "Cleaning up..."
                
                $compose_cmd down -v --remove-orphans
                docker system prune -f
                
                success "Cleanup completed"
            else
                log "Cleanup cancelled"
            fi
            ;;
            
        *)
            error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    log "Docker deployment script for churn prediction"
    log "Environment: $ENVIRONMENT"
    log "Action: $ACTION"
    if [[ -n "$PROFILES" ]]; then
        log "Profiles: $PROFILES"
    fi
    
    check_prerequisites
    create_env_file
    execute_action
}

# Run main function
main