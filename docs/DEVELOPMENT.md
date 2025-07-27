# Development Guide

## Overview

This guide provides comprehensive instructions for setting up, developing, and contributing to the Customer Churn Predictor project. It covers everything from initial setup to advanced development workflows.

## Quick Start

### Prerequisites

- **Python 3.12+**: Required for the application
- **Docker**: For containerized development and deployment
- **Git**: Version control system
- **Make**: Build automation (optional but recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/yourorg/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Development Setup

```bash
# Option 1: Using Make (recommended)
make setup

# Option 2: Manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests to ensure everything works
make test

# Or manually
pytest tests/ -v
```

### 4. Start Development Server

```bash
# Using Make
make serve

# Or manually
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## Development Environment

### Directory Structure

```
customer-churn-predictor/
├── .devcontainer/          # VS Code dev container configuration
├── .github/                # GitHub Actions workflows and templates
├── .vscode/                # VS Code settings
├── data/                   # Data files
│   ├── raw/               # Raw input data
│   └── processed/         # Processed data
├── docs/                   # Documentation
│   ├── adr/              # Architecture Decision Records
│   ├── api/              # API documentation
│   └── guides/           # User guides
├── models/                 # Trained model artifacts
├── monitoring/             # Monitoring configuration
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── __init__.py
│   ├── api.py            # FastAPI application
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   └── ...               # Other modules
├── tests/                  # Test files
│   ├── integration/      # Integration tests
│   ├── performance/      # Performance tests
│   ├── security/         # Security tests
│   └── conftest.py       # Test configuration
├── docker-compose.yml      # Multi-service orchestration
├── Dockerfile             # Container definition
├── Makefile              # Build automation
├── pyproject.toml        # Project configuration
└── requirements*.txt     # Dependencies
```

### Development Tools

#### VS Code Integration

The project includes VS Code configuration for optimal development experience:

```bash
# Open in VS Code
code .

# With dev container
code . && F1 > "Dev Containers: Reopen in Container"
```

#### Pre-commit Hooks

Automated code quality checks run on every commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

#### Docker Development

```bash
# Build development image
make build-dev

# Run in container
make run-dev

# Full development environment
docker-compose up -d
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit code ...

# Run tests
make test

# Check code quality
make lint

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Security scan
make security

# Type checking
mypy src/

# Test coverage
make coverage
```

### 3. Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-performance
make test-security

# Run with coverage
make coverage

# Mutation testing
make mutation-test
```

## Code Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort (black profile)
- **Linting**: Flake8 with extensions
- **Type checking**: mypy with strict settings

### Code Quality Rules

1. **All code must be tested** (minimum 80% coverage)
2. **Type hints required** for all public functions
3. **Docstrings required** for all modules, classes, and public functions
4. **No security vulnerabilities** (bandit + safety)
5. **Performance requirements** must be met

### Example Code Style

```python
"""Module docstring describing the purpose."""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CustomerChurnPredictor:
    """Predicts customer churn using machine learning.
    
    This class encapsulates the model training and prediction logic
    for customer churn prediction.
    
    Attributes:
        model_path: Path to the trained model file.
        feature_columns: List of feature column names.
    """
    
    def __init__(self, model_path: str, feature_columns: list[str]) -> None:
        """Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file.
            feature_columns: List of feature column names.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If feature columns are invalid.
        """
        self.model_path = model_path
        self.feature_columns = feature_columns
        
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn probability for given features.
        
        Args:
            features: Dictionary of feature values.
            
        Returns:
            Dictionary containing prediction and probability.
            
        Raises:
            ValueError: If features are invalid.
        """
        # Implementation here
        pass
```

## ML Development

### Data Pipeline

```bash
# Preprocess data
make preprocess

# Train model
make train

# Evaluate model
make evaluate

# Run full pipeline
make pipeline
```

### Model Development Workflow

1. **Data Exploration**: Use Jupyter notebooks for analysis
2. **Feature Engineering**: Implement in `src/preprocess_data.py`
3. **Model Training**: Update `src/train_model.py`
4. **Model Evaluation**: Add metrics to `src/evaluate_model.py`
5. **Model Deployment**: Update API endpoints

### MLflow Integration

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View experiments
open http://localhost:5000
```

## API Development

### FastAPI Structure

```python
# src/api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI(
    title="Customer Churn Predictor",
    description="Production-ready ML API for churn prediction",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    probability: float
    
@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest) -> PredictionResponse:
    """Predict customer churn."""
    # Implementation here
    pass
```

### API Testing

```bash
# Start API server
make serve

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure": 12, "monthly_charges": 50.0}}'

# Interactive API docs
open http://localhost:8000/docs
```

## Database Development (Future)

### Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Database Testing

```bash
# Setup test database
make db-test-setup

# Run database tests
pytest tests/database/ -v
```

## Performance Optimization

### Profiling

```bash
# Profile API endpoints
py-spy top --pid $(pgrep -f uvicorn)

# Memory profiling
memory_profiler src/api.py

# Load testing
make load-test
```

### Optimization Checklist

- [ ] Database query optimization
- [ ] API response caching
- [ ] Model inference optimization
- [ ] Memory usage optimization
- [ ] Container resource limits

## Debugging

### Local Debugging

```python
# Add to code for debugging
import pdb; pdb.set_trace()

# Or use iPython debugger
import IPython; IPython.embed()
```

### Container Debugging

```bash
# Debug running container
docker exec -it container_name /bin/bash

# View logs
docker logs container_name -f

# Debug with VS Code
# Use .vscode/launch.json configuration
```

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Model Loading**: Check file paths and permissions
3. **API Errors**: Verify authentication and input validation
4. **Test Failures**: Check test data and mock configurations

## Contributing

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass
6. **Submit** a pull request

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] No security vulnerabilities
- [ ] Performance impact assessed

### Code Review Guidelines

- Focus on correctness, security, and maintainability
- Provide constructive feedback
- Suggest improvements where applicable
- Check for proper error handling
- Verify test coverage

## Advanced Topics

### Custom Metrics

```python
# src/metrics.py
from prometheus_client import Counter, Histogram

prediction_requests = Counter('prediction_requests_total', 'Total prediction requests')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
def predict_with_metrics(features):
    prediction_requests.inc()
    return predict(features)
```

### Async Processing

```python
# src/async_tasks.py
import asyncio
from celery import Celery

app = Celery('churn_predictor')

@app.task
async def batch_prediction(data_path: str) -> str:
    """Process batch predictions asynchronously."""
    # Implementation here
    pass
```

### Plugin Development

```python
# src/plugins/base.py
from abc import ABC, abstractmethod

class ModelPlugin(ABC):
    """Base class for model plugins."""
    
    @abstractmethod
    def train(self, data) -> None:
        """Train the model."""
        pass
        
    @abstractmethod
    def predict(self, features) -> Any:
        """Make predictions."""
        pass
```

## Resources

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)

### Development Tools

- [Black Code Formatter](https://black.readthedocs.io/)
- [isort Import Sorter](https://pycqa.github.io/isort/)
- [mypy Type Checker](https://mypy.readthedocs.io/)
- [pre-commit Hooks](https://pre-commit.com/)

### External Resources

- [Python Package Index](https://pypi.org/)
- [GitHub Actions Marketplace](https://github.com/marketplace)
- [Docker Hub](https://hub.docker.com/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python)

## Getting Help

### Internal Resources

- **Documentation**: Check `docs/` directory
- **Code Examples**: See `examples/` directory
- **Tests**: Reference `tests/` for usage examples

### External Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions in GitHub Discussions
- **Slack**: Join the development Slack channel
- **Email**: Contact the development team

### Troubleshooting

If you encounter issues:

1. Check the [troubleshooting guide](troubleshooting.md)
2. Search existing GitHub issues
3. Run diagnostics: `make diagnose`
4. Create a new issue with detailed information

---

**Happy coding! 🚀**