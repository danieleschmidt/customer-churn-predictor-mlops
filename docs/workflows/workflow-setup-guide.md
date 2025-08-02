# GitHub Actions Workflow Setup Guide

This guide provides comprehensive instructions for setting up GitHub Actions workflows for the Customer Churn Predictor MLOps system.

## Overview

The project includes four main workflow categories:
1. **Continuous Integration (CI)** - Code quality, testing, and validation
2. **Continuous Deployment (CD)** - Automated deployment to staging and production
3. **MLOps Pipeline** - Model training, validation, and deployment
4. **Security Scanning** - Comprehensive security and compliance checks

## Prerequisites

### Required Secrets

Configure these secrets in your GitHub repository settings:

#### Basic Secrets
```bash
# API Keys and Tokens
GITHUB_TOKEN                    # Automatically provided by GitHub
API_KEY                        # Application API key for testing

# Container Registry
REGISTRY_USERNAME              # Container registry username
REGISTRY_PASSWORD              # Container registry password/token

# Security Scanning
GITGUARDIAN_API_KEY           # GitGuardian secret scanning (optional)
CODECOV_TOKEN                 # Code coverage reporting (optional)
```

#### MLOps Secrets
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI           # MLflow tracking server URL
MLFLOW_USERNAME               # MLflow authentication username
MLFLOW_PASSWORD               # MLflow authentication password

# Cloud Provider (if using)
AWS_ACCESS_KEY_ID             # AWS credentials for data/model storage
AWS_SECRET_ACCESS_KEY         # AWS secret access key
AWS_REGION                    # AWS region

# Database (future)
DATABASE_URL                  # Database connection string
DATABASE_PASSWORD             # Database password
```

#### Deployment Secrets
```bash
# Staging Environment
STAGING_HOST                  # Staging server hostname
STAGING_SSH_KEY               # SSH key for staging deployment
STAGING_API_KEY               # Staging environment API key

# Production Environment
PRODUCTION_HOST               # Production server hostname
PRODUCTION_SSH_KEY            # SSH key for production deployment
PRODUCTION_API_KEY            # Production environment API key

# Notification Services
SLACK_WEBHOOK_URL             # Slack notifications webhook
EMAIL_SMTP_PASSWORD           # Email notifications password
```

### Required Permissions

Ensure your GitHub token has these permissions:
- `actions: write` - For workflow management
- `contents: write` - For repository access
- `packages: write` - For container registry
- `security-events: write` - For security scan results
- `issues: write` - For creating deployment approval issues
- `pull-requests: write` - For PR comments

## Workflow Setup Instructions

### Step 1: Create Workflow Directory

```bash
# Create GitHub Actions directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/ml-ops.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
```

### Step 2: Configure Environments

Create GitHub environments in your repository settings:

#### Staging Environment
- **Name**: `staging`
- **Protection Rules**: None (automatic deployment)
- **Environment Secrets**:
  - `STAGING_HOST`
  - `STAGING_API_KEY`
  - `DATABASE_URL` (staging database)

#### Production Environment
- **Name**: `production`
- **Protection Rules**:
  - Required reviewers: 2+ team members
  - Wait timer: 5 minutes
  - Deployment branches: `main` and tags only
- **Environment Secrets**:
  - `PRODUCTION_HOST`
  - `PRODUCTION_API_KEY`
  - `DATABASE_URL` (production database)

### Step 3: Configure Branch Protection

Set up branch protection rules for the `main` branch:

```yaml
Required status checks:
  - "Code Quality & Security"
  - "Tests (3.12)"  # At least one Python version
  - "Docker Build & Test"
  - "ML Model Validation"

Require branches to be up to date: true
Require pull request reviews: true
Required approving reviews: 2
Dismiss stale reviews: true
Require review from CODEOWNERS: true
Restrict pushes to matching branches: true
```

### Step 4: Set Up CODEOWNERS

Create `.github/CODEOWNERS` file:

```
# Global ownership
* @admin @devops-team

# Python code
*.py @ml-team @backend-team
src/ @ml-team @backend-team
tests/ @qa-team @ml-team

# Infrastructure and deployment
Dockerfile @devops-team
docker-compose*.yml @devops-team
.github/workflows/ @devops-team @admin

# Documentation
docs/ @tech-writers @ml-team
*.md @tech-writers

# ML-specific files
src/train_model.py @ml-team @data-scientists
src/preprocess_data.py @ml-team @data-engineers
models/ @ml-team

# Security and compliance
.github/workflows/security-scan.yml @security-team @devops-team
```

## Workflow Customization

### CI Workflow Customization

Edit `.github/workflows/ci.yml` to customize:

#### Python Version Matrix
```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]  # Adjust versions as needed
```

#### Test Configuration
```yaml
- name: Run unit tests
  run: pytest tests/ -m "unit" -v --cov=src --cov-report=xml
  # Add custom test markers, coverage thresholds, etc.
```

#### Performance Test Thresholds
```yaml
- name: Run performance tests
  run: |
    locust -f tests/performance/locustfile.py \
      --headless \
      --users 50 \              # Adjust load
      --spawn-rate 5 \
      --run-time 60s \          # Adjust duration
      --host http://localhost:8000
```

### CD Workflow Customization

Edit `.github/workflows/cd.yml` to customize:

#### Deployment Strategy
```yaml
# Blue-Green Deployment
- name: Deploy to production (blue-green)
  run: |
    # Deploy to blue environment
    kubectl apply -f k8s/blue/
    # Health check
    # Switch traffic
    # Cleanup green environment

# Canary Deployment
- name: Deploy to production (canary)
  run: |
    # Deploy canary version (10% traffic)
    kubectl apply -f k8s/canary/
    # Monitor metrics
    # Gradually increase traffic
```

#### Approval Process
```yaml
- name: Approve production deployment
  uses: trstringer/manual-approval@v1
  with:
    secret: ${{ github.TOKEN }}
    approvers: admin,devops-lead,product-owner  # Customize approvers
    minimum-approvals: 2                        # Customize threshold
    timeout-minutes: 60                         # Deployment window
```

### MLOps Workflow Customization

Edit `.github/workflows/ml-ops.yml` to customize:

#### Training Schedule
```yaml
on:
  schedule:
    - cron: '0 2 * * 1'    # Weekly on Monday
    # - cron: '0 2 * * *'  # Daily
    # - cron: '0 2 1 * *'  # Monthly
```

#### Model Performance Thresholds
```yaml
- name: Check model performance
  run: |
    python -c "
    needs_retraining = (
        accuracy < 0.8 or           # Adjust accuracy threshold
        drift_score > 0.3 or        # Adjust drift threshold
        days_since_training > 7     # Adjust time threshold
    )
    "
```

#### Hyperparameter Tuning
```yaml
strategy:
  matrix:
    model_config:
      - { solver: 'liblinear', C: 1.0, penalty: 'l2' }
      - { solver: 'saga', C: 0.5, penalty: 'l1' }
      - { solver: 'lbfgs', C: 2.0, penalty: 'l2' }
      # Add more configurations as needed
```

### Security Workflow Customization

Edit `.github/workflows/security-scan.yml` to customize:

#### Vulnerability Thresholds
```yaml
- name: Check for critical vulnerabilities
  run: |
    # Adjust severity thresholds
    if critical_count > 0:          # Zero tolerance for critical
        sys.exit(1)
    elif high_count > 5:            # Allow some high severity
        sys.exit(1)
    elif medium_count > 20:         # Allow more medium severity
        sys.exit(1)
```

#### License Compliance
```yaml
# Customize allowed licenses
allowed_licenses = {
    'MIT License', 'MIT', 'Apache Software License',
    'Apache 2.0', 'BSD License', 'BSD',
    # Add your organization's approved licenses
}

# Customize forbidden licenses
forbidden_licenses = {
    'GPL v3', 'GNU General Public License v3',
    'AGPL', 'Affero GPL', 'LGPL',
    # Add licenses your organization prohibits
}
```

## Advanced Configuration

### Matrix Builds

Test across multiple configurations:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.10", "3.11", "3.12"]
    include:
      - os: ubuntu-latest
        python-version: "3.12"
        extra-tests: true
    exclude:
      - os: macos-latest
        python-version: "3.10"
```

### Conditional Execution

Run jobs based on conditions:

```yaml
# Run only on specific paths
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - '.github/workflows/**'

# Skip workflows based on commit message
- name: Check if should skip
  if: "!contains(github.event.head_commit.message, '[skip ci]')"
```

### Caching Strategies

Optimize build times with caching:

```yaml
# Python dependencies
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

# Docker layers
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
  with:
    buildkitd-flags: --debug

- name: Build with cache
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Artifact Management

Handle build artifacts effectively:

```yaml
# Upload test results
- name: Upload test results
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-results-${{ matrix.python-version }}
    path: |
      test-results/
      coverage.xml
      htmlcov/
    retention-days: 30

# Download artifacts in dependent jobs
- name: Download test artifacts
  uses: actions/download-artifact@v3
  with:
    name: test-results-3.12
    path: test-results/
```

## Monitoring and Maintenance

### Workflow Monitoring

Monitor workflow health:

1. **Success Rates**: Track workflow success/failure rates
2. **Execution Time**: Monitor build duration trends
3. **Resource Usage**: Check runner utilization
4. **Cost Tracking**: Monitor GitHub Actions minutes usage

### Regular Maintenance Tasks

#### Weekly Tasks
- Review failed workflow runs
- Update dependency versions
- Check security scan results
- Verify deployment health

#### Monthly Tasks
- Review and update workflow configurations
- Analyze performance trends
- Update runner images and actions versions
- Review and rotate secrets

#### Quarterly Tasks
- Comprehensive security audit
- Performance optimization review
- Cost analysis and optimization
- Team access and permissions review

### Troubleshooting Common Issues

#### Build Failures

```bash
# Debug workflow locally
act -j test  # Using nektos/act

# Check logs
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed
```

#### Permission Issues

```yaml
# Add explicit permissions
permissions:
  contents: read
  packages: write
  security-events: write
  actions: write
```

#### Resource Constraints

```yaml
# Use larger runners for resource-intensive jobs
runs-on: ubuntu-latest-4-cores  # GitHub-hosted larger runner
# or
runs-on: self-hosted             # Self-hosted runner
```

#### Timeout Issues

```yaml
# Adjust timeouts
timeout-minutes: 30  # Default is 360 minutes

# Cancel previous runs
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Security Best Practices

### Secret Management

1. **Use Repository Secrets**: Never hardcode secrets in workflows
2. **Environment-Specific Secrets**: Use environment secrets for deployment
3. **Least Privilege**: Grant minimal required permissions
4. **Secret Rotation**: Regularly rotate secrets and tokens
5. **Audit Access**: Monitor secret access and usage

### Workflow Security

1. **Pin Action Versions**: Use specific commit hashes for actions
2. **Review Dependencies**: Audit third-party actions
3. **Limit Permissions**: Use explicit permission declarations
4. **Secure Runners**: Use trusted runner environments
5. **Input Validation**: Validate all external inputs

### Code Security

1. **Branch Protection**: Enforce branch protection rules
2. **Required Reviews**: Require code review before merge
3. **Status Checks**: Mandate security scans pass
4. **Dependency Scanning**: Regular vulnerability scanning
5. **Supply Chain Security**: SBOM generation and validation

## Integration with External Tools

### MLflow Integration

```yaml
- name: Log experiment to MLflow
  run: |
    python -c "
    import mlflow
    mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')
    with mlflow.start_run():
        mlflow.log_param('github_run_id', '${{ github.run_id }}')
        mlflow.log_param('github_sha', '${{ github.sha }}')
        # Log other experiment data
    "
```

### Slack Notifications

```yaml
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
    text: "🚨 Deployment failed for ${{ github.ref_name }}"
```

### Jira Integration

```yaml
- name: Create Jira ticket for failed deployment
  if: failure()
  uses: atlassian/gajira-create@v3
  with:
    project: PROJ
    issuetype: Bug
    summary: "Deployment failure: ${{ github.ref_name }}"
    description: "Deployment failed in workflow ${{ github.workflow }}"
```

## Resources and References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening Guide](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [MLOps with GitHub Actions](https://github.com/machine-learning-apps/MLOps-GitHub-Actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)

## Support and Troubleshooting

For issues with workflow setup:

1. Check the [troubleshooting section](#troubleshooting-common-issues)
2. Review GitHub Actions logs and error messages
3. Consult the team documentation or support channels
4. Create an issue in the repository for team assistance

Remember to test workflows in a development branch before merging to main!