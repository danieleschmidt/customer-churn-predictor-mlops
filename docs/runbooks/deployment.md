# Deployment Runbook

## Overview
This runbook covers deployment procedures for the ML Churn Prediction service.

## Pre-deployment Checklist
- [ ] All tests pass in CI/CD
- [ ] Security scans completed
- [ ] Performance tests passed
- [ ] Documentation updated
- [ ] Database migrations tested
- [ ] Rollback plan prepared

## Deployment Environments

### Development
- **Purpose**: Feature development and testing
- **URL**: http://localhost:8000
- **Database**: Local SQLite/PostgreSQL
- **Deployment**: Automatic on push to feature branches

### Staging
- **Purpose**: Integration testing and user acceptance
- **URL**: https://staging.example.com
- **Database**: Staging database (separate from prod)
- **Deployment**: Automatic on merge to main branch

### Production
- **Purpose**: Live service for end users
- **URL**: https://api.example.com
- **Database**: Production database
- **Deployment**: Manual trigger on release tags

## Deployment Procedures

### Standard Deployment
1. **Pre-deployment verification**
   ```bash
   # Run full test suite
   python -m pytest tests/ --cov=src

   # Security scan
   bandit -r src/

   # Build and test container
   docker build -t churn-predictor:latest .
   docker run --rm churn-predictor:latest python -m unittest
   ```

2. **Deploy to staging**
   ```bash
   # Tag and push
   git tag v1.x.x
   git push origin v1.x.x

   # Monitor deployment
   kubectl get pods -n staging
   kubectl logs -f deployment/churn-predictor -n staging
   ```

3. **Staging validation**
   - Run smoke tests
   - Verify API endpoints
   - Check model performance metrics
   - Validate monitoring and alerts

4. **Production deployment**
   ```bash
   # Deploy to production
   kubectl apply -f k8s/production/

   # Monitor rollout
   kubectl rollout status deployment/churn-predictor -n production
   ```

### Rollback Procedures
1. **Immediate rollback**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/churn-predictor -n production

   # Verify rollback
   kubectl rollout status deployment/churn-predictor -n production
   ```

2. **Database rollback** (if applicable)
   ```bash
   # Run down migrations
   alembic downgrade -1

   # Restore from backup if needed
   pg_restore -d production_db backup_file.sql
   ```

## Monitoring Post-Deployment
- Application metrics (response times, error rates)
- Business metrics (prediction accuracy, throughput)
- Infrastructure metrics (CPU, memory, disk)
- User feedback and support tickets

## Deployment Checklist
- [ ] Application starts successfully
- [ ] Health checks pass
- [ ] API endpoints respond correctly
- [ ] Model predictions are accurate
- [ ] Monitoring and alerting functional
- [ ] Performance within expected ranges
- [ ] No increase in error rates
- [ ] Database connections stable

## Emergency Procedures
If critical issues are detected:
1. Immediately rollback deployment
2. Notify incident response team
3. Begin incident response procedures
4. Investigate root cause
5. Prepare hotfix if needed

## Environment Variables
Key environment variables for deployment:
- `DATABASE_URL`: Database connection string
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `API_KEY`: Authentication key
- `LOG_LEVEL`: Logging verbosity
- `CACHE_TTL`: Model cache timeout