#!/bin/bash
set -e

# Docker build script for churn prediction application
# Supports multi-stage builds and various deployment targets

# Configuration
IMAGE_NAME="churn-predictor"
REGISTRY="${REGISTRY:-}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')

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

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Parse command line arguments
TARGET="production"
PUSH=false
CACHE=true
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --target TARGET      Build target (production, development) [default: production]"
            echo "  --push               Push image to registry after build"
            echo "  --no-cache           Disable Docker build cache"
            echo "  --platform PLATFORM  Target platform (e.g., linux/amd64,linux/arm64)"
            echo "  --version VERSION    Image version tag [default: git short hash]"
            echo "  --registry REGISTRY  Registry prefix for image name"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate target
case $TARGET in
    production|development)
        ;;
    *)
        error "Invalid target: $TARGET. Must be 'production' or 'development'"
        exit 1
        ;;
esac

# Set image tag
if [[ -n "$REGISTRY" ]]; then
    IMAGE_TAG="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    IMAGE_TAG_TARGET="${REGISTRY}/${IMAGE_NAME}:${VERSION}-${TARGET}"
else
    IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
    IMAGE_TAG_TARGET="${IMAGE_NAME}:${VERSION}-${TARGET}"
fi

log "Building Docker image..."
log "Target: $TARGET"
log "Version: $VERSION"
log "Image tag: $IMAGE_TAG"
log "Build date: $BUILD_DATE"
log "VCS ref: $VCS_REF"

# Prepare build arguments
BUILD_ARGS=(
    --build-arg "BUILD_DATE=${BUILD_DATE}"
    --build-arg "VERSION=${VERSION}"
    --build-arg "VCS_REF=${VCS_REF}"
    --target "$TARGET"
    --tag "$IMAGE_TAG"
    --tag "$IMAGE_TAG_TARGET"
)

# Add platform if specified
if [[ -n "$PLATFORM" ]]; then
    BUILD_ARGS+=(--platform "$PLATFORM")
fi

# Add cache options
if [[ "$CACHE" == "false" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

# Run Docker build
log "Executing docker build..."
if docker build "${BUILD_ARGS[@]}" .; then
    success "Docker build completed successfully"
else
    error "Docker build failed"
    exit 1
fi

# Display image information
log "Image details:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Run basic image test
log "Testing image..."
if docker run --rm "$IMAGE_TAG" health; then
    success "Image health check passed"
else
    error "Image health check failed"
    exit 1
fi

# Push to registry if requested
if [[ "$PUSH" == "true" ]]; then
    if [[ -z "$REGISTRY" ]]; then
        error "Registry must be specified when pushing"
        exit 1
    fi
    
    log "Pushing image to registry..."
    
    if docker push "$IMAGE_TAG" && docker push "$IMAGE_TAG_TARGET"; then
        success "Image pushed successfully"
        log "Published: $IMAGE_TAG"
        log "Published: $IMAGE_TAG_TARGET"
    else
        error "Failed to push image"
        exit 1
    fi
fi

success "Build process completed successfully"

# Output usage instructions
echo
echo "Usage instructions:"
echo "  Run application:     docker run --rm -p 8000:8000 $IMAGE_TAG"
echo "  Health check:        docker run --rm $IMAGE_TAG health"
echo "  Interactive shell:   docker run --rm -it $IMAGE_TAG shell"
echo "  With Docker Compose: docker-compose up"