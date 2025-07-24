#!/bin/bash
set -e

# Security scanning script for Docker images
# Integrates with CI/CD pipelines for automated security checks

# Configuration
IMAGE_NAME="${1:-churn-predictor:latest}"
SEVERITY_THRESHOLD="${SEVERITY_THRESHOLD:-HIGH}"
MAX_HIGH_VULNS="${MAX_HIGH_VULNS:-0}"
MAX_MEDIUM_VULNS="${MAX_MEDIUM_VULNS:-5}"
REPORT_FORMAT="${REPORT_FORMAT:-json}"
OUTPUT_DIR="${OUTPUT_DIR:-./security-reports}"
FAIL_ON_VULNERABILITIES="${FAIL_ON_VULNERABILITIES:-true}"

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

# Check prerequisites
check_prerequisites() {
    log "Checking security scanning prerequisites..."
    
    # Check if Trivy is installed
    if ! command -v trivy &> /dev/null; then
        error "Trivy is not installed. Installing..."
        install_trivy
    else
        log "Trivy found: $(trivy --version | head -1)"
    fi
    
    # Check if Cosign is available (optional)
    if command -v cosign &> /dev/null; then
        log "Cosign found: $(cosign version)"
    else
        warn "Cosign not found - image signature verification will be skipped"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    success "Prerequisites check completed"
}

# Install Trivy if not present
install_trivy() {
    log "Installing Trivy..."
    
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case $ARCH in
        x86_64) ARCH="64bit" ;;
        arm64|aarch64) ARCH="ARM64" ;;
        *) error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    
    # Download and install Trivy
    TRIVY_VERSION="0.48.3"
    DOWNLOAD_URL="https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_${OS^}_${ARCH}.tar.gz"
    
    log "Downloading Trivy from: $DOWNLOAD_URL"
    
    curl -sL "$DOWNLOAD_URL" | tar xz -C /tmp
    sudo mv /tmp/trivy /usr/local/bin/
    
    log "Trivy installed successfully"
}

# Scan image for vulnerabilities
scan_vulnerabilities() {
    local image="$1"
    local report_file="$OUTPUT_DIR/vulnerability-report.json"
    
    log "Scanning $image for vulnerabilities..."
    
    # Run Trivy vulnerability scan
    trivy image \
        --format json \
        --output "$report_file" \
        --severity "${SEVERITY_THRESHOLD},CRITICAL" \
        --quiet \
        --no-progress \
        "$image"
    
    if [[ $? -eq 0 ]]; then
        success "Vulnerability scan completed"
        
        # Parse results
        total_vulns=$(jq '[.Results[]?.Vulnerabilities // empty] | length' "$report_file")
        high_vulns=$(jq '[.Results[]?.Vulnerabilities // empty | .[] | select(.Severity == "HIGH" or .Severity == "CRITICAL")] | length' "$report_file")
        medium_vulns=$(jq '[.Results[]?.Vulnerabilities // empty | .[] | select(.Severity == "MEDIUM")] | length' "$report_file")
        
        log "Vulnerability Summary:"
        log "  Total vulnerabilities: $total_vulns"
        log "  High/Critical: $high_vulns"
        log "  Medium: $medium_vulns"
        
        # Check thresholds
        if [[ $high_vulns -gt $MAX_HIGH_VULNS ]]; then
            error "High/Critical vulnerabilities ($high_vulns) exceed limit ($MAX_HIGH_VULNS)"
            return 1
        fi
        
        if [[ $medium_vulns -gt $MAX_MEDIUM_VULNS ]]; then
            error "Medium vulnerabilities ($medium_vulns) exceed limit ($MAX_MEDIUM_VULNS)"
            return 1
        fi
        
        success "Image passes vulnerability threshold checks"
        return 0
    else
        error "Vulnerability scan failed"
        return 1
    fi
}

# Scan for secrets and sensitive data
scan_secrets() {
    local image="$1"
    local report_file="$OUTPUT_DIR/secrets-report.json"
    
    log "Scanning $image for secrets..."
    
    # Run Trivy secret scan
    trivy image \
        --scanners secret \
        --format json \
        --output "$report_file" \
        --quiet \
        --no-progress \
        "$image"
    
    if [[ $? -eq 0 ]]; then
        secret_count=$(jq '[.Results[]?.Secrets // empty] | length' "$report_file")
        
        if [[ $secret_count -gt 0 ]]; then
            error "Found $secret_count secrets in image"
            return 1
        else
            success "No secrets found in image"
            return 0
        fi
    else
        error "Secret scan failed"
        return 1
    fi
}

# Check for misconfigurations
scan_misconfigurations() {
    local dockerfile="${1:-Dockerfile}"
    local report_file="$OUTPUT_DIR/misconfig-report.json"
    
    if [[ ! -f "$dockerfile" ]]; then
        warn "Dockerfile not found - skipping misconfiguration scan"
        return 0
    fi
    
    log "Scanning $dockerfile for misconfigurations..."
    
    # Run Trivy config scan
    trivy config \
        --format json \
        --output "$report_file" \
        --quiet \
        "$dockerfile"
    
    if [[ $? -eq 0 ]]; then
        misconfig_count=$(jq '[.Results[]?.Misconfigurations // empty] | length' "$report_file")
        high_misconfigs=$(jq '[.Results[]?.Misconfigurations // empty | .[] | select(.Severity == "HIGH" or .Severity == "CRITICAL")] | length' "$report_file")
        
        log "Misconfiguration Summary:"
        log "  Total issues: $misconfig_count"
        log "  High/Critical: $high_misconfigs"
        
        if [[ $high_misconfigs -gt 0 ]]; then
            error "Found $high_misconfigs high/critical misconfigurations"
            return 1
        else
            success "No critical misconfigurations found"
            return 0
        fi
    else
        error "Misconfiguration scan failed"
        return 1
    fi
}

# Verify image signature (if Cosign available)
verify_signature() {
    local image="$1"
    
    if ! command -v cosign &> /dev/null; then
        warn "Cosign not available - skipping signature verification"
        return 0
    fi
    
    log "Verifying signature for $image..."
    
    if cosign verify "$image" &> /dev/null; then
        success "Image signature verified"
        return 0
    else
        warn "Image signature verification failed or image is unsigned"
        return 0  # Don't fail on signature issues for now
    fi
}

# Generate security report
generate_report() {
    local image="$1"
    local summary_file="$OUTPUT_DIR/security-summary.json"
    local html_report="$OUTPUT_DIR/security-report.html"
    
    log "Generating security summary report..."
    
    # Create summary JSON
    cat > "$summary_file" << EOF
{
    "image": "$image",
    "scan_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "scanner_version": "$(trivy --version | head -1)",
    "thresholds": {
        "max_high_vulnerabilities": $MAX_HIGH_VULNS,
        "max_medium_vulnerabilities": $MAX_MEDIUM_VULNS
    },
    "reports": {
        "vulnerabilities": "vulnerability-report.json",
        "secrets": "secrets-report.json",
        "misconfigurations": "misconfig-report.json"
    }
}
EOF
    
    # Generate HTML report if vulnerabilities exist
    if [[ -f "$OUTPUT_DIR/vulnerability-report.json" ]]; then
        log "Generating HTML vulnerability report..."
        trivy image \
            --format table \
            --output "$html_report" \
            --severity "${SEVERITY_THRESHOLD},CRITICAL" \
            "$image" || true
    fi
    
    success "Security reports generated in: $OUTPUT_DIR"
}

# Main execution
main() {
    local image="$1"
    
    if [[ -z "$image" ]]; then
        error "Usage: $0 <image-name>"
        error "Example: $0 churn-predictor:latest"
        exit 1
    fi
    
    log "Starting security scan for: $image"
    log "Configuration:"
    log "  Max high vulnerabilities: $MAX_HIGH_VULNS"
    log "  Max medium vulnerabilities: $MAX_MEDIUM_VULNS"
    log "  Fail on vulnerabilities: $FAIL_ON_VULNERABILITIES"
    log "  Output directory: $OUTPUT_DIR"
    
    # Run security checks
    check_prerequisites
    
    local exit_code=0
    
    # Vulnerability scan
    if ! scan_vulnerabilities "$image"; then
        exit_code=1
    fi
    
    # Secret scan
    if ! scan_secrets "$image"; then
        exit_code=1
    fi
    
    # Misconfiguration scan
    if ! scan_misconfigurations "Dockerfile"; then
        exit_code=1
    fi
    
    # Signature verification
    verify_signature "$image"
    
    # Generate reports
    generate_report "$image"
    
    # Final result
    if [[ $exit_code -eq 0 ]]; then
        success "Security scan passed for $image"
        log "All security checks completed successfully"
    else
        error "Security scan failed for $image"
        if [[ "$FAIL_ON_VULNERABILITIES" == "true" ]]; then
            error "Failing build due to security issues"
            exit 1
        else
            warn "Security issues found but not failing build (FAIL_ON_VULNERABILITIES=false)"
        fi
    fi
}

# Execute main function
main "$@"