# Customer Churn Predictor with MLOps

This project focuses on building a model to predict customer churn and integrating MLOps practices throughout the development lifecycle. We will cover data preprocessing, model training, evaluation, experiment tracking, model versioning, and a basic CI/CD pipeline.

## Project Goals
- Develop an effective binary classification model to predict customer churn.
- Implement a robust data preprocessing pipeline (handling missing values, encoding, scaling).
- Track experiments using MLflow (or a similar tool).
- Version control data and models (e.g., using DVC, or simply git-lfs for models).
- Set up a basic CI/CD pipeline using GitHub Actions to automate testing and model training/validation.
- Ensure code quality with linting and automated tests.

## Tech Stack (Planned)
- Python
- Scikit-learn
- Pandas, NumPy
- MLflow
- DVC (optional)
- GitHub Actions

## Initial File Structure
customer-churn-predictor-mlops/
├── data/
│   └── raw/
│       └── customer_data.csv # Sample customer data
│   └── processed/
├── notebooks/
│   └── churn_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_churn.py
├── tests/
│   ├── __init__.py
│   └── test_preprocess_data.py
├── .github/workflows/ # For GitHub Actions
│   └── main.yml
├── scripts/ # Utility scripts (e.g., run_training.sh)
├── models/ # For trained model artifacts
├── requirements.txt
├── .gitignore
└── README.md

## Processed Input Format

The preprocessing step produces `data/processed/processed_features.csv`.
Categorical variables are one-hot encoded (e.g. `gender_Female`,
`gender_Male`), while `SeniorCitizen`, `tenure`, `MonthlyCharges`, and
`TotalCharges` remain numeric. The `make_prediction` function expects
input data with these columns.

## How to Contribute (and test Jules)
Jules, our Async Development Agent, will assist in building out features, tests, and MLOps components. Please create clear issues.
