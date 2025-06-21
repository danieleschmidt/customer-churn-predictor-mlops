# Development Plan

This checklist outlines steps to develop and maintain the customer churn prediction project with MLOps best practices.

## Setup
- [x] Install Python 3.8 and create a virtual environment
- [x] Install project requirements with `pip install -r requirements.txt`
- [x] Configure MLflow tracking URI (local filesystem is sufficient)

## Data Preparation
- [x] Review `data/raw/customer_data.csv` for missing or inconsistent values
- [x] Run `scripts/run_preprocessing.py` to generate processed datasets
- [ ] Commit processed data to version control or data versioning tool (e.g., DVC)

## Modeling
- [x] Train an initial model using `scripts/run_training.py`
  - [x] Evaluate metrics: accuracy and F1 score
  - [x] Log parameters and metrics in MLflow
- [x] Save and version the trained model in the `models/` directory

## Experiment Tracking
- [x] Record each experiment's parameters, metrics, and artifacts in MLflow
- [x] Review and compare runs to select the best-performing model

## Continuous Integration / Continuous Deployment
- [x] Ensure unit tests in `tests/` cover preprocessing and modeling functions
- [x] Verify GitHub Actions workflow runs tests on each push and pull request
- [x] Automate preprocessing and training jobs on pull requests

## Prediction Pipeline
- [x] Implement a script to load the trained model and predict churn for new data (`src/predict_churn.py`)
- [x] Document the expected input format after preprocessing

## Maintenance
- [ ] Periodically update dependencies in `requirements.txt`
- [ ] Monitor model performance and retrain with fresh data when necessary

