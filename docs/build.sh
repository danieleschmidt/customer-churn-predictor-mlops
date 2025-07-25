#!/bin/bash
# Documentation build script for Customer Churn Predictor MLOps
# 
# This script builds the documentation website using MkDocs with Material theme.
# It can build for development (with live reload) or production (static files).

set -e

# Configuration
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DOCS_DIR")"
BUILD_DIR="$DOCS_DIR/site"
REQUIREMENTS_FILE="$DOCS_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        log_error "pip is required but not installed"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

install_requirements() {
    log_info "Installing documentation requirements..."
    
    # Use pip3 if available, otherwise pip
    PIP_CMD="pip3"
    if ! command -v pip3 &> /dev/null; then
        PIP_CMD="pip"
    fi
    
    # Install requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        $PIP_CMD install -r "$REQUIREMENTS_FILE"
        log_success "Requirements installed successfully"
    else
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
}

build_docs() {
    local mode="$1"
    
    cd "$DOCS_DIR"
    
    if [ "$mode" == "serve" ]; then
        log_info "Starting development server with live reload..."
        log_info "Documentation will be available at: http://127.0.0.1:8000"
        log_info "Press Ctrl+C to stop the server"
        echo
        mkdocs serve --dev-addr=127.0.0.1:8000
    else
        log_info "Building static documentation..."
        
        # Clean previous build
        if [ -d "$BUILD_DIR" ]; then
            rm -rf "$BUILD_DIR"
            log_info "Cleaned previous build directory"
        fi
        
        # Build documentation
        mkdocs build --strict
        
        log_success "Documentation built successfully"
        log_info "Build output: $BUILD_DIR"
    fi
}

deploy_docs() {
    log_info "Deploying documentation to GitHub Pages..."
    
    cd "$DOCS_DIR"
    
    # Deploy to gh-pages branch
    mkdocs gh-deploy --force
    
    log_success "Documentation deployed to GitHub Pages"
}

validate_docs() {
    log_info "Validating documentation..."
    
    cd "$DOCS_DIR"
    
    # Build with strict mode to catch errors
    mkdocs build --strict --clean
    
    # Check for broken links (if linkchecker is installed)
    if command -v linkchecker &> /dev/null; then
        log_info "Checking for broken links..."
        linkchecker "$BUILD_DIR"
    else
        log_warning "linkchecker not found, skipping link validation"
    fi
    
    log_success "Documentation validation completed"
}

show_help() {
    cat << EOF
Documentation Build Script for Customer Churn Predictor MLOps

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build       Build static documentation (default)
    serve       Start development server with live reload
    deploy      Deploy documentation to GitHub Pages
    validate    Validate documentation and check for issues
    install     Install documentation requirements
    clean       Clean build artifacts
    help        Show this help message

Options:
    --skip-install    Skip installing requirements
    --verbose         Enable verbose output

Examples:
    $0 build                    # Build static documentation
    $0 serve                    # Start development server
    $0 deploy                   # Deploy to GitHub Pages
    $0 validate                 # Validate documentation
    $0 install                  # Install requirements only

EOF
}

clean_build() {
    log_info "Cleaning build artifacts..."
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_success "Removed build directory: $BUILD_DIR"
    fi
    
    # Clean MkDocs cache
    if [ -d "$DOCS_DIR/.mkdocs" ]; then
        rm -rf "$DOCS_DIR/.mkdocs"
        log_success "Removed MkDocs cache"
    fi
    
    log_success "Clean completed"
}

# Main script
main() {
    local command="${1:-build}"
    local skip_install=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-install)
                skip_install=true
                shift
                ;;
            --verbose)
                verbose=true
                set -x
                shift
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            *)
                command="$1"
                shift
                ;;
        esac
    done
    
    log_info "Customer Churn Predictor MLOps - Documentation Builder"
    log_info "Command: $command"
    echo
    
    # Check dependencies
    check_dependencies
    
    # Install requirements unless skipped
    if [ "$skip_install" = false ]; then
        install_requirements
    fi
    
    # Execute command
    case "$command" in
        build)
            build_docs "build"
            ;;
        serve)
            build_docs "serve"
            ;;
        deploy)
            deploy_docs
            ;;
        validate)
            validate_docs
            ;;
        install)
            log_success "Requirements installation completed"
            ;;
        clean)
            clean_build
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Trap to handle interrupts gracefully
trap 'log_info "\nBuild interrupted by user"; exit 130' INT

# Run main function
main "$@"