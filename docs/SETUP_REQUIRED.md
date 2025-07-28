# Manual Setup Requirements

## Repository Configuration

### Branch Protection Rules
- **Main Branch**: Require PR reviews, dismiss stale reviews
- **Status Checks**: Require CI checks to pass
- **Admin Enforcement**: Include administrators

### GitHub Actions Setup  
1. Enable Actions in repository settings
2. Copy workflows from `workflow-files-for-manual-addition/`
3. Configure repository secrets:
   - `PYPI_TOKEN` - For package publishing
   - `DOCKERHUB_TOKEN` - For container registry
   - `CODECOV_TOKEN` - For coverage reporting

### Security Configuration
- Enable Dependabot alerts and security updates
- Configure code scanning with CodeQL
- Set up secret scanning for sensitive data

### Repository Settings
- **Topics**: machine-learning, mlops, python, fastapi
- **Description**: Production-ready ML system for customer churn prediction
- **Homepage**: Link to documentation site

### Team Permissions
- **Maintainers**: Admin access for releases
- **Contributors**: Write access for development
- **Reviewers**: Triage access for issue management

## External Integrations

### Required Services
- **MLflow**: Model tracking and registry
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Optional Integrations
- **Slack**: Development notifications
- **Sentry**: Error tracking and monitoring
- **DataDog**: Application performance monitoring

## Documentation

All manual setup procedures are documented in:
- [GitHub Actions Setup Guide](workflows/README.md)
- [Security Configuration Guide](../SECURITY.md)
- [Development Environment Setup](DEVELOPMENT.md)