# Testing Guide - Customer Churn Predictor

This guide covers the comprehensive testing strategy for the Customer Churn Predictor MLOps system.

## Testing Philosophy

Our testing approach follows the testing pyramid:
- **Unit Tests (70%)**: Fast, isolated tests for individual components
- **Integration Tests (20%)**: Tests for component interactions
- **End-to-End Tests (10%)**: Complete workflow validation

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared pytest fixtures
├── test_config.yaml           # Test configuration
├── fixtures/                  # Test data and utilities
│   ├── __init__.py
│   └── sample_data.py         # Sample data generators
├── unit/                      # Unit tests (individual components)
├── integration/               # Integration tests
├── e2e/                       # End-to-end tests
├── performance/               # Performance and load tests
│   ├── locustfile.py          # Load testing configuration
│   └── test_api_performance.py
└── security/                  # Security tests
    └── test_security.py
```

## Running Tests

### Quick Test Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance
make test-security

# Run with coverage
make coverage

# Run tests in parallel
pytest -n auto

# Run tests with specific markers
pytest -m "unit"
pytest -m "integration" 
pytest -m "e2e"
pytest -m "performance"
pytest -m "security"
```

### Detailed Test Commands

```bash
# Unit tests only
pytest tests/ -m "unit" -v

# Integration tests with Docker
pytest tests/integration/ -v --docker

# End-to-end tests (requires full setup)
pytest tests/e2e/ -v --slow

# Performance tests
pytest tests/performance/ -v
locust -f tests/performance/locustfile.py --headless -u 50 -r 5 -t 60s

# Security tests
pytest tests/security/ -v
bandit -r src/ -f json
safety check

# Coverage reporting
pytest --cov=src --cov-report=html --cov-report=term-missing
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation.

**Location**: Root of `tests/` directory and `tests/unit/` (future)
**Markers**: `@pytest.mark.unit`
**Coverage Target**: >90%

**Examples**:
- `test_preprocess_data.py`: Data preprocessing functions
- `test_train_model.py`: Model training logic
- `test_predict_churn.py`: Prediction functions
- `test_config.py`: Configuration management

**Best Practices**:
- Mock external dependencies
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests fast (<1 second each)

### Integration Tests

Test component interactions and API endpoints.

**Location**: `tests/integration/`
**Markers**: `@pytest.mark.integration`
**Coverage Target**: >80%

**Examples**:
- API endpoint testing
- Database integration (future)
- External service integration
- Docker container testing

**Best Practices**:
- Use test containers for dependencies
- Test realistic data flows
- Validate API contracts
- Test error propagation

### End-to-End Tests

Test complete workflows from start to finish.

**Location**: `tests/e2e/`
**Markers**: `@pytest.mark.e2e`
**Coverage Target**: Key user journeys

**Test Scenarios**:
- Complete ML pipeline (preprocess → train → predict)
- API workflow (authentication → prediction → response)
- Model persistence and loading
- Error handling and recovery

**Best Practices**:
- Test realistic user scenarios
- Use production-like data
- Validate business requirements
- Test performance benchmarks

### Performance Tests

Validate system performance under various loads.

**Location**: `tests/performance/`
**Markers**: `@pytest.mark.performance`

**Test Types**:
- **Load Testing**: Normal expected load
- **Stress Testing**: Beyond normal capacity
- **Spike Testing**: Sudden load increases
- **Volume Testing**: Large data sets

**Tools**:
- **Locust**: HTTP load testing
- **pytest-benchmark**: Function-level benchmarking
- **Memory profiling**: Resource usage testing

**Metrics**:
- Response time (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Resource utilization

### Security Tests

Validate security controls and vulnerability protection.

**Location**: `tests/security/`
**Markers**: `@pytest.mark.security`

**Test Areas**:
- Authentication and authorization
- Input validation and sanitization
- API security headers
- Rate limiting
- Data protection

**Tools**:
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning
- **Custom tests**: Application-specific security

## Test Configuration

### Environment Variables

```bash
# Test environment
export ENVIRONMENT=test
export API_KEY=test-api-key
export LOG_LEVEL=debug

# Database (future)
export DATABASE_URL=sqlite:///:memory:

# External services
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Configuration Files

- `tests/test_config.yaml`: Test-specific configuration
- `conftest.py`: Shared pytest fixtures and setup
- `pytest.ini` (via `pyproject.toml`): Pytest configuration

### Test Data Management

**Fixtures**: Use `tests/fixtures/` for reusable test data
**Factories**: Create data generators for different scenarios
**Cleanup**: Automatic cleanup of test artifacts
**Isolation**: Each test should be independent

## Test Fixtures and Utilities

### Sample Data Generation

```python
from tests.fixtures import (
    create_sample_customer_data,
    create_processed_features,
    create_api_request_data
)

# Generate sample data for testing
data = create_sample_customer_data(n_samples=100)
api_data = create_api_request_data()
```

### Common Fixtures

```python
@pytest.fixture
def sample_model():
    """Provide a trained model for testing."""
    
@pytest.fixture
def api_client():
    """Provide authenticated API client."""
    
@pytest.fixture
def temp_data_dir():
    """Provide temporary directory for test data."""
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Scheduled runs (nightly)

### Test Matrix

Tests run across:
- Python versions: 3.10, 3.11, 3.12
- Operating systems: Ubuntu, Windows, macOS
- Dependency versions: Minimum and latest

### Quality Gates

**Pull Request Requirements**:
- All tests pass
- Coverage ≥ 80%
- No security vulnerabilities
- Performance benchmarks met

**Deployment Requirements**:
- Full test suite passes
- Integration tests pass
- Security scans clean
- Performance tests within thresholds

## Test Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate XML for CI/CD
pytest --cov=src --cov-report=xml

# Terminal coverage summary
pytest --cov=src --cov-report=term-missing
```

### Performance Reports

```bash
# Locust performance report
locust -f tests/performance/locustfile.py --headless -u 50 -r 5 -t 60s --html=performance-report.html

# Benchmark comparison
pytest-benchmark compare
```

### Test Results

- **JUnit XML**: For CI/CD integration
- **HTML Reports**: For human review
- **JSON**: For programmatic analysis
- **Allure**: Advanced reporting (optional)

## Best Practices

### Writing Good Tests

1. **Clear Names**: Test names should describe what they test
2. **Single Responsibility**: One assertion per test when possible
3. **Independent**: Tests should not depend on each other
4. **Deterministic**: Tests should produce consistent results
5. **Fast**: Unit tests should complete quickly

### Test Data Management

1. **Use Factories**: Generate test data programmatically
2. **Avoid Hardcoding**: Use configuration for test parameters
3. **Clean Isolation**: Each test should clean up after itself
4. **Realistic Data**: Use data that mimics production

### Debugging Tests

```bash
# Run single test with verbose output
pytest tests/test_example.py::test_function -v -s

# Drop into debugger on failure
pytest --pdb

# Run with logging
pytest --log-cli-level=DEBUG

# Profile test performance
pytest --profile
```

### Performance Considerations

1. **Parallel Execution**: Use `pytest-xdist` for parallel tests
2. **Test Selection**: Use markers to run only relevant tests
3. **Caching**: Cache expensive setup operations
4. **Mocking**: Mock external services to reduce dependencies

## Test Maintenance

### Regular Tasks

- **Update Dependencies**: Keep testing libraries current
- **Review Coverage**: Identify untested code paths
- **Performance Monitoring**: Track test execution trends
- **Cleanup**: Remove obsolete tests

### Code Review Checklist

- [ ] Tests cover new functionality
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Tests are independent and isolated
- [ ] Test names are descriptive
- [ ] No hardcoded values
- [ ] Appropriate test markers used

## Troubleshooting

### Common Issues

**Tests Failing Locally But Passing in CI**:
- Check environment differences
- Verify dependency versions
- Check for race conditions

**Slow Test Execution**:
- Use parallel execution
- Mock external services
- Optimize test data generation

**Flaky Tests**:
- Check for timing dependencies
- Add proper wait conditions
- Ensure test isolation

**Coverage Issues**:
- Check for untested branches
- Add tests for error conditions
- Test configuration variations

### Getting Help

1. Check test logs and error messages
2. Review test configuration
3. Run tests in isolation
4. Check for environment issues
5. Consult team documentation or contacts

## Future Enhancements

### Planned Improvements

- **Database Testing**: When database integration is added
- **Contract Testing**: API contract validation
- **Chaos Engineering**: Fault injection testing
- **A/B Testing**: Model comparison frameworks
- **Visual Testing**: UI testing (if frontend is added)

### Advanced Testing Patterns

- **Property-Based Testing**: Using Hypothesis
- **Mutation Testing**: Code quality validation
- **Snapshot Testing**: Output validation
- **Approval Testing**: Complex output verification