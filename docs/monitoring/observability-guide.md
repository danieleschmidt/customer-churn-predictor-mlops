# Observability Guide - Customer Churn Predictor

This guide covers the comprehensive observability strategy for monitoring, alerting, and troubleshooting the Customer Churn Predictor MLOps system.

## Overview

Our observability stack provides complete visibility into:
- **Application Performance**: Response times, throughput, errors
- **Infrastructure Health**: CPU, memory, disk, network
- **ML Model Performance**: Accuracy, drift, data quality
- **Business Metrics**: Churn rates, predictions, revenue impact

## Architecture

### Monitoring Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│     Grafana     │
│   (Metrics)     │    │   (Storage)     │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  OpenTelemetry  │    │   Alertmanager  │    │     Jaeger      │
│   (Tracing)     │    │   (Alerting)    │    │    (Tracing)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards and visualization
- **Alertmanager**: Alert routing and management
- **Jaeger**: Distributed tracing
- **OpenTelemetry**: Unified observability framework

## Getting Started

### Quick Start

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
open http://localhost:16686 # Jaeger
```

### Configuration

```bash
# Environment variables
export GRAFANA_PASSWORD=secure_password
export ENVIRONMENT=production
export PROMETHEUS_RETENTION=30d
```

## Metrics Collection

### Application Metrics

#### HTTP Metrics
- `http_requests_total`: Total HTTP requests by method, status, endpoint
- `http_request_duration_seconds`: Request duration histogram
- `http_requests_in_flight`: Current number of HTTP requests being served

#### ML Model Metrics
- `model_accuracy`: Current model accuracy score
- `model_predictions_total`: Total predictions made
- `model_prediction_duration_seconds`: Time taken for predictions
- `model_last_updated_timestamp`: When model was last retrained

#### Business Metrics
- `predicted_churn_rate`: Current predicted churn rate
- `customer_segments_total`: Number of customers by segment
- `revenue_impact_dollars`: Estimated revenue impact of predictions

#### System Metrics
- `process_memory_bytes`: Memory usage
- `process_cpu_seconds_total`: CPU usage
- `python_gc_collections_total`: Garbage collection statistics

### Custom Metrics Example

```python
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
PREDICTION_COUNTER = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction_type']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds',
    'Time spent on predictions',
    ['model_version']
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_version']
)

# Use in application
@PREDICTION_LATENCY.time()
def make_prediction(data):
    result = model.predict(data)
    PREDICTION_COUNTER.labels(
        model_version='v1.0',
        prediction_type='churn'
    ).inc()
    return result
```

## Dashboards

### Main Dashboard Components

#### System Overview
- Service health status
- Request rate and latency
- Error rates and status codes
- Resource utilization (CPU, memory)

#### ML Model Performance  
- Model accuracy over time
- Prediction latency percentiles
- Feature importance changes
- Data quality metrics

#### Business KPIs
- Predicted churn rate trends
- Customer segment distribution
- Revenue impact metrics
- Operational efficiency

### Dashboard Access

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: Set via `GRAFANA_PASSWORD`

### Custom Dashboards

Create custom dashboards for specific needs:

```json
{
  "dashboard": {
    "title": "Custom ML Dashboard",
    "panels": [
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Alert Categories

#### Critical Alerts (Immediate Response)
- Application down
- Critical error rate (>10%)
- Database connection failed
- Training job failed

#### Warning Alerts (Investigation Required)
- High error rate (>5%)
- High response time
- Low model accuracy
- Model drift detected

#### Informational Alerts (Monitoring)
- Model not updated recently
- High memory usage
- Queue backup

### Alert Configuration

Alerts are defined in `monitoring/alert-rules.yml`:

```yaml
groups:
  - name: churn_predictor_critical
    rules:
      - alert: ApplicationDown
        expr: up{job="churn-predictor"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Application is down"
          description: "Application has been down for {{ $for }}"
```

### Alert Routing

Configure alert destinations in Alertmanager:

```yaml
route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
  - name: 'warning-alerts'
    email_configs:
      - to: 'team@example.com'
        subject: 'Warning Alert'
```

## Distributed Tracing

### Jaeger Integration

Distributed tracing helps understand request flows:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use in application
@tracer.start_as_current_span("make_prediction")
def make_prediction(customer_data):
    with tracer.start_as_current_span("preprocess_data"):
        processed_data = preprocess(customer_data)
    
    with tracer.start_as_current_span("model_inference"):
        prediction = model.predict(processed_data)
    
    return prediction
```

### Trace Analysis

- **Request Flow**: See complete request journey
- **Performance Bottlenecks**: Identify slow operations
- **Error Propagation**: Track error sources
- **Dependencies**: Understand service interactions

## Log Management

### Structured Logging

Use structured logging for better searchability:

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "Prediction made",
    customer_id="12345",
    prediction=1,
    probability=0.85,
    model_version="v1.0",
    duration_ms=45.2
)
```

### Log Aggregation

For production, consider log aggregation:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Grafana Loki**: Lightweight log aggregation
- **Cloud Solutions**: AWS CloudWatch, Azure Monitor

### Log Levels

- **ERROR**: System errors, exceptions
- **WARN**: Non-critical issues, performance concerns
- **INFO**: Normal application flow, business events
- **DEBUG**: Detailed diagnostic information

## ML-Specific Monitoring

### Model Performance Monitoring

```python
from prometheus_client import Gauge, Histogram

# Model metrics
model_accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Monitor model performance
def monitor_model_performance(y_true, y_pred, prediction_time):
    accuracy = accuracy_score(y_true, y_pred)
    model_accuracy_gauge.set(accuracy)
    prediction_latency.observe(prediction_time)
    
    # Alert if accuracy drops
    if accuracy < 0.8:
        logger.warning(
            "Model accuracy dropped",
            accuracy=accuracy,
            threshold=0.8
        )
```

### Data Quality Monitoring

```python
def monitor_data_quality(data):
    # Check for missing values
    missing_rate = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    
    # Check for outliers
    outlier_rate = detect_outliers(data)
    
    # Check data freshness
    last_update = get_data_timestamp(data)
    staleness = (datetime.now() - last_update).total_seconds()
    
    # Log metrics
    logger.info(
        "Data quality check",
        missing_rate=missing_rate,
        outlier_rate=outlier_rate,
        staleness_seconds=staleness
    )
```

### Feature Drift Detection

```python
from scipy import stats

def detect_feature_drift(reference_data, current_data):
    drift_scores = {}
    
    for feature in reference_data.columns:
        # Statistical test for drift
        statistic, p_value = stats.ks_2samp(
            reference_data[feature],
            current_data[feature]
        )
        
        drift_scores[feature] = {
            'statistic': statistic,
            'p_value': p_value,
            'has_drift': p_value < 0.05
        }
    
    # Log drift detection results
    for feature, scores in drift_scores.items():
        if scores['has_drift']:
            logger.warning(
                "Feature drift detected",
                feature=feature,
                p_value=scores['p_value']
            )
    
    return drift_scores
```

## Performance Optimization

### Metrics Cardinality

Control metrics cardinality to avoid performance issues:

```python
# Good: Limited cardinality
prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['model_version', 'prediction_type']  # Limited labels
)

# Bad: High cardinality
prediction_counter = Counter(
    'predictions_total',
    'Total predictions', 
    ['customer_id', 'timestamp']  # Too many unique values
)
```

### Sampling

Use sampling for high-volume traces:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
```

### Retention Policies

Configure appropriate retention:

```yaml
# Prometheus retention
retention: 30d
retention.size: 10GB

# Jaeger retention
dependencies:
  schedule: "0 6 * * *"
  ttl: 168h  # 7 days
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check container memory
docker stats churn-predictor

# Check application memory
curl http://localhost:8000/metrics | grep process_memory
```

#### Missing Metrics
```bash
# Check metric endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### Slow Queries
```bash
# Check query performance
curl -g 'http://localhost:9090/api/v1/query?query=up'

# Enable query logging
--log.level=debug
```

### Debugging Steps

1. **Check Application Health**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify Metrics Endpoint**
   ```bash
   curl http://localhost:8000/metrics
   ```

3. **Check Prometheus Targets**
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

4. **Review Container Logs**
   ```bash
   docker-compose logs churn-predictor
   ```

5. **Check Resource Usage**
   ```bash
   docker stats
   ```

## Best Practices

### Monitoring

- ✅ Monitor business metrics, not just technical metrics
- ✅ Use appropriate alert thresholds
- ✅ Implement health checks
- ✅ Monitor external dependencies
- ✅ Use structured logging

### Alerting

- ✅ Alert on symptoms, not causes
- ✅ Make alerts actionable
- ✅ Include runbook links
- ✅ Use appropriate severity levels
- ✅ Test alert routing

### Performance

- ✅ Control metrics cardinality
- ✅ Use appropriate sampling rates
- ✅ Set retention policies
- ✅ Monitor monitoring system health
- ✅ Optimize dashboard queries

### Security

- ✅ Secure monitoring endpoints
- ✅ Use authentication for dashboards
- ✅ Encrypt data in transit
- ✅ Audit access to monitoring data
- ✅ Sanitize sensitive information

## Advanced Topics

### Custom Exporters

Create custom exporters for specific needs:

```python
from prometheus_client import CollectorRegistry, generate_latest

class MLModelExporter:
    def __init__(self, model):
        self.model = model
        self.registry = CollectorRegistry()
        
    def collect(self):
        # Custom metrics collection logic
        yield GaugeMetricFamily(
            'ml_model_size_bytes',
            'Model size in bytes',
            value=self.get_model_size()
        )
```

### Multi-Environment Monitoring

Configure monitoring for different environments:

```yaml
# docker-compose.override.yml for staging
services:
  prometheus:
    command:
      - '--config.file=/etc/prometheus/prometheus-staging.yml'
      - '--retention.time=7d'
```

### Integration with CI/CD

Monitor deployment health:

```yaml
# GitHub Actions
- name: Check deployment health
  run: |
    curl -f http://staging.example.com/health
    curl -f http://staging.example.com/metrics
```

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [ML Monitoring Best Practices](https://ml-ops.org/content/monitoring)