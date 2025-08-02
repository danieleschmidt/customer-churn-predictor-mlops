# Developer Guide - Customer Churn Predictor

## Development Environment Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git
- Make (optional, for build scripts)

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd customer-churn-predictor-mlops
   pip install -r requirements-dev.txt
   pre-commit install
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run Tests**
   ```bash
   pytest -v
   make test  # Alternative using Makefile
   ```

4. **Start Development Server**
   ```bash
   docker-compose up -d
   python -m src.api
   ```

## Project Architecture

### Core Components

```
src/
├── api.py              # FastAPI REST API
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── train_model.py      # Model training logic
├── predict_churn.py    # Prediction logic
├── preprocess_data.py  # Data preprocessing
├── monitoring/         # Model monitoring
└── security/          # Authentication & security
```

### Data Flow

1. **Raw Data** → `data/raw/customer_data.csv`
2. **Preprocessing** → `data/processed/processed_features.csv`
3. **Training** → `models/churn_model.joblib`
4. **Prediction** → API responses or batch files

### Configuration System

The system uses `config.yml` for default configurations:

```yaml
data:
  raw_data: data/raw/customer_data.csv
  processed_features: data/processed/processed_features.csv
  processed_target: data/processed/processed_target.csv

model:
  path: models/churn_model.joblib
  preprocessor_path: models/preprocessor.joblib

api:
  host: 0.0.0.0
  port: 8000
  debug: false
```

Environment variables override config file settings:
- `CHURN_MODEL_PATH`
- `API_KEY`
- `MLFLOW_TRACKING_URI`

## Development Workflow

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement Feature**
   - Write code following existing patterns
   - Add comprehensive tests
   - Update documentation

3. **Quality Checks**
   ```bash
   pytest                 # Run tests
   black src/ tests/      # Format code
   ruff check src/ tests/ # Lint code
   mypy src/             # Type checking
   ```

4. **Create Pull Request**
   - Ensure all tests pass
   - Add descriptive PR description
   - Request code review

### Testing Strategy

#### Unit Tests
```bash
# Run specific test file
pytest tests/test_train_model.py -v

# Run with coverage
pytest --cov=src tests/

# Run performance tests
pytest tests/performance/ -v
```

#### Integration Tests
```bash
# End-to-end API tests
pytest tests/integration/test_end_to_end.py -v

# Docker integration tests
pytest tests/test_docker_integration.py -v
```

#### Security Tests
```bash
# Security validation
pytest tests/security/ -v

# Dependency scanning
safety check

# Docker security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image customer-churn-predictor:latest
```

## Code Standards

### Style Guidelines

- **Formatting**: Use Black with line length 88
- **Imports**: Sort with isort
- **Linting**: Follow Ruff rules
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all modules, classes, and functions

### Example Function

```python
def train_churn_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """Train a logistic regression model for churn prediction.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained model, accuracy score)
        
    Raises:
        ValueError: If input data is invalid
    """
    if X.empty or y.empty:
        raise ValueError("Input data cannot be empty")
        
    # Implementation here
    return model, accuracy
```

### Error Handling

```python
from src.exceptions import ChurnPredictorError, ModelNotFoundError

try:
    model = load_model(model_path)
except FileNotFoundError:
    raise ModelNotFoundError(f"Model not found at {model_path}")
except Exception as e:
    raise ChurnPredictorError(f"Failed to load model: {e}")
```

## API Development

### Adding New Endpoints

1. **Define Route**
   ```python
   @app.post("/predict/advanced")
   @require_auth
   async def advanced_prediction(
       request: AdvancedPredictionRequest
   ) -> AdvancedPredictionResponse:
       """Advanced churn prediction with feature importance."""
       # Implementation
   ```

2. **Add Request/Response Models**
   ```python
   class AdvancedPredictionRequest(BaseModel):
       customer_data: CustomerData
       include_features: bool = False
       explain_prediction: bool = False
   ```

3. **Add Tests**
   ```python
   def test_advanced_prediction_endpoint(client, auth_headers):
       response = client.post(
           "/predict/advanced",
           json={"customer_data": sample_data},
           headers=auth_headers
       )
       assert response.status_code == 200
   ```

### Authentication

The API uses token-based authentication:

```python
from src.security import require_auth

@app.get("/protected-endpoint")
@require_auth
async def protected_endpoint():
    return {"message": "Access granted"}
```

## Model Development

### Adding New Models

1. **Create Model Class**
   ```python
   class XGBoostChurnModel(ChurnModel):
       def __init__(self, **kwargs):
           self.model = XGBClassifier(**kwargs)
           
       def train(self, X, y):
           self.model.fit(X, y)
           
       def predict(self, X):
           return self.model.predict(X)
   ```

2. **Register Model**
   ```python
   # In config.py
   MODEL_REGISTRY = {
       'logistic': LogisticRegressionModel,
       'xgboost': XGBoostChurnModel,
   }
   ```

3. **Add Model Tests**
   ```python
   def test_xgboost_model_training():
       model = XGBoostChurnModel()
       model.train(X_train, y_train)
       predictions = model.predict(X_test)
       assert len(predictions) == len(X_test)
   ```

### Experiment Tracking

All experiments are logged to MLflow:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({
        "model_type": "xgboost",
        "max_depth": 6,
        "learning_rate": 0.1
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Database Integration

### Adding Database Support

1. **Define Models**
   ```python
   from sqlalchemy import Column, Integer, String, Float
   from src.database import Base
   
   class PredictionLog(Base):
       __tablename__ = "prediction_logs"
       
       id = Column(Integer, primary_key=True)
       customer_id = Column(String, nullable=False)
       prediction = Column(Integer, nullable=False)
       probability = Column(Float, nullable=False)
       timestamp = Column(DateTime, default=datetime.utcnow)
   ```

2. **Add Migration**
   ```bash
   alembic revision --autogenerate -m "Add prediction logging"
   alembic upgrade head
   ```

## Performance Optimization

### Profiling

```bash
# Profile API endpoints
python -m cProfile -o profile.stats scripts/benchmark_api.py

# Analyze with snakeviz
snakeviz profile.stats
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_preprocessed_features(customer_data: str) -> np.ndarray:
    """Cache preprocessed features for faster predictions."""
    data = json.loads(customer_data)
    return preprocess_customer_data(data)
```

## Deployment

### Docker Development

```bash
# Build development image
docker build -t churn-predictor:dev .

# Run with hot reload
docker run -v $(pwd):/app -p 8000:8000 churn-predictor:dev

# Run tests in container
docker run --rm churn-predictor:dev pytest
```

### Environment Configuration

```bash
# Development
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=debug

# Production
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=info
```

## Debugging

### Local Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()
```

### Container Debugging

```bash
# Debug running container
docker exec -it container_id /bin/bash

# Check logs
docker logs container_id -f

# Debug with pdb
docker run -it --rm churn-predictor:dev python -m pdb scripts/debug_script.py
```

### Common Issues

1. **Model Loading Errors**
   - Check file paths and permissions
   - Verify model was saved correctly
   - Check MLflow connectivity

2. **API Authentication Issues**
   - Verify API_KEY environment variable
   - Check token format and expiration
   - Review security configuration

3. **Performance Problems**
   - Profile slow endpoints
   - Check database query performance
   - Review caching strategies

## Contributing Guidelines

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Security implications considered
- [ ] Performance impact assessed
- [ ] Backward compatibility maintained

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(api): add batch prediction endpoint`
- `fix(model): handle missing feature columns`
- `docs(api): update authentication guide`

### Release Process

1. **Version Bump**
   ```bash
   bump2version patch  # or minor, major
   ```

2. **Update Changelog**
   ```bash
   conventional-changelog -p angular -i CHANGELOG.md -s
   ```

3. **Create Release**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)