# User Guide - Customer Churn Predictor

## Getting Started

This guide will help you understand and use the Customer Churn Predictor system effectively.

## What is Customer Churn Prediction?

Customer churn prediction helps businesses identify customers who are likely to stop using their services. This system uses machine learning to analyze customer data and predict the probability of churn.

## System Overview

The Customer Churn Predictor provides:
- **REST API**: For real-time predictions
- **Batch Processing**: For bulk prediction jobs
- **Model Monitoring**: To track prediction accuracy
- **Data Validation**: To ensure input data quality

## Using the API

### Authentication

All API requests require authentication using an API key:

```bash
export API_KEY="your-secure-api-key-here"
```

### Making Predictions

#### Single Customer Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_data": {
      "tenure": 12,
      "MonthlyCharges": 65.50,
      "TotalCharges": 786.00,
      "gender": "Female",
      "SeniorCitizen": 0
    }
  }'
```

#### Batch Predictions

For multiple customers, use the batch prediction endpoint:

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"tenure": 12, "MonthlyCharges": 65.50, "TotalCharges": 786.00, "gender": "Female", "SeniorCitizen": 0},
      {"tenure": 24, "MonthlyCharges": 85.25, "TotalCharges": 2046.00, "gender": "Male", "SeniorCitizen": 1}
    ]
  }'
```

## Input Data Format

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tenure` | Integer | Months as customer | 12 |
| `MonthlyCharges` | Float | Monthly bill amount | 65.50 |
| `TotalCharges` | Float | Total amount paid | 786.00 |
| `gender` | String | Customer gender | "Female" or "Male" |
| `SeniorCitizen` | Integer | Senior citizen flag | 0 or 1 |

### Optional Fields

The system supports additional fields that may improve prediction accuracy. See the [API documentation](../api/overview.md) for the complete field list.

## Understanding Predictions

### Response Format

```json
{
  "prediction": 0,
  "probability": 0.23,
  "confidence": "high",
  "customer_id": "12345"
}
```

- **prediction**: 0 = No churn, 1 = Likely to churn
- **probability**: Churn probability (0.0 to 1.0)
- **confidence**: System confidence in prediction (low/medium/high)

### Interpretation Guidelines

| Probability Range | Risk Level | Recommended Action |
|------------------|------------|-------------------|
| 0.0 - 0.3 | Low | Standard retention activities |
| 0.3 - 0.7 | Medium | Enhanced engagement programs |
| 0.7 - 1.0 | High | Immediate intervention required |

## Command Line Interface

### Installation

```bash
pip install -r requirements.txt
```

### Basic Commands

```bash
# Preprocess data
python -m src.cli preprocess

# Train model
python -m src.cli train --solver saga --C 0.5

# Evaluate model
python -m src.cli evaluate --detailed

# Make predictions
python -m src.cli predict data/input.csv --output_csv results.csv
```

### Advanced Usage

#### Custom Model Parameters

```bash
python -m src.cli train \
  --solver saga \
  --C 0.5 \
  --penalty l1 \
  --random_state 42 \
  --max_iter 200 \
  --test_size 0.3
```

#### Monitoring and Retraining

```bash
# Monitor model performance
python scripts/run_monitor.py --threshold 0.85

# Force retraining
python scripts/run_training.py --force-retrain
```

## Troubleshooting

### Common Issues

#### Authentication Errors
- **Problem**: "401 Unauthorized"
- **Solution**: Verify API key is set correctly and has proper format

#### Prediction Errors
- **Problem**: "Invalid input data"
- **Solution**: Check input format matches required schema

#### Performance Issues
- **Problem**: Slow response times
- **Solution**: Use batch predictions for multiple customers

### Getting Help

1. Check the [troubleshooting guide](../operations/troubleshooting.md)
2. Review [API documentation](../api/overview.md)
3. Contact support with detailed error messages

## Best Practices

### Data Quality
- Ensure all required fields are present
- Validate data ranges and formats
- Handle missing values appropriately

### Performance Optimization
- Use batch predictions for bulk operations
- Cache results when appropriate
- Monitor API rate limits

### Security
- Keep API keys secure and rotate regularly
- Use HTTPS in production
- Validate all input data

## Next Steps

- Explore the [Developer Guide](developer-guide.md) for customization
- Review [API Documentation](../api/overview.md) for detailed specifications
- Check [Deployment Guide](../deployment/docker.md) for production setup