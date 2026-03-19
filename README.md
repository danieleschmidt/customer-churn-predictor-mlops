# Customer Churn Predictor — MLOps Pipeline

End-to-end machine learning pipeline for predicting customer churn. Built with a focus on clean code, reproducibility, and practical MLOps patterns — not enterprise scaffolding bloat.

## What's here

```
churn/
  generate_data.py   # synthetic churn dataset generator (5k rows, realistic signals)
  features.py        # sklearn feature pipeline (scaling, OHE, ColumnTransformer)
  evaluate.py        # ROC-AUC, precision-recall, confusion matrix, feature importance
  tracker.py         # lightweight JSON experiment tracker (no MLflow)
train.py             # full training script (LogReg baseline + XGBoost)
predict.py           # batch prediction / demo script
tests/               # pytest suite (22 tests)
data/raw/            # generated CSV lives here
models/              # trained pipelines saved as .pkl
experiments/         # run logs (JSON) + charts (PNG)
```

## Features

| Feature | Implementation |
|---|---|
| Synthetic data | Realistic Telco-style churn signals (tenure, charges, contract, support calls) |
| Feature pipeline | `StandardScaler` + `OneHotEncoder` via `ColumnTransformer` |
| Models | Logistic Regression (baseline) + XGBoost |
| Evaluation | ROC-AUC, avg precision, F1, confusion matrix, feature importance |
| Experiment tracking | JSON-based tracker — no external dependencies |
| Serving | `predict.py` — load saved pipeline, score new records |

## Quick start

```bash
# Install deps (conda python recommended)
pip install -r requirements.txt

# 1. Generate synthetic data
python -m churn.generate_data --n-samples 5000

# 2. Train both models
python train.py

# 3. Score new customers
python predict.py --demo          # 10 synthetic records
python predict.py --input data/new_customers.csv --output results.csv
```

## Example output

```
[1/2] Training Logistic Regression baseline …
  ROC-AUC:       0.8912
  Avg Precision: 0.8079
  F1 (binary):   0.7301

[2/2] Training XGBoost …
  ROC-AUC:       0.8771
  Avg Precision: 0.7890
  F1 (binary):   0.7147

Top-10 feature importances (XGBoost):
  Contract_Month-to-month    ██████████████████████████████
  InternetService_Fiber optic ████████████████████████
  tenure                     █████████████
  MonthlyCharges             ███████████
  ...

Best run: LogisticRegression  ROC-AUC=0.8912
```

Charts saved to `experiments/charts/` (ROC curve + confusion matrix per model).

## Experiment tracker

All runs are logged to `experiments/runs.json`:

```json
[
  {
    "run_id": "60b4929f",
    "timestamp": "2026-03-19T14:46:50+00:00",
    "model": "LogisticRegression",
    "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    "metrics": {"roc_auc": 0.8912, "avg_precision": 0.8079, "f1": 0.7301, "f1_macro": 0.7899},
    "tags": {"type": "baseline"}
  }
]
```

## Tests

```bash
pytest tests/ -v
# 22 passed in 0.76s
```

## Dataset schema

| Column | Type | Description |
|---|---|---|
| tenure | int | Months as customer (1–72) |
| MonthlyCharges | float | Monthly bill ($18–$120) |
| TotalCharges | float | Cumulative charges |
| SupportCalls | int | Number of support interactions |
| Contract | categorical | Month-to-month / One year / Two year |
| InternetService | categorical | DSL / Fiber optic / None |
| OnlineSecurity, TechSupport, StreamingTV | categorical | Yes / No / No internet service |
| PaperlessBilling, PaymentMethod | categorical | Billing preferences |
| SeniorCitizen | int | 0 or 1 |
| Churn | target | Yes / No |

Churn rate in generated data: ~36% (class imbalance is realistic, not oversampled).

## MLOps patterns demonstrated

- **Reproducibility**: fixed random seeds, versioned data files, logged hyperparams
- **Pipeline-first**: preprocessing baked into the saved model — no training/serving skew
- **Experiment tracking**: every run logged; `best_run()` selects champion
- **Separation of concerns**: data gen, features, evaluation, training, serving are separate modules
- **No MLflow tax**: the tracker is ~80 lines of Python; add MLflow later if you need the UI

## Requirements

- Python 3.10+
- scikit-learn ≥ 1.3
- pandas ≥ 2.0
- numpy ≥ 1.24
- xgboost ≥ 2.0 (optional — falls back to sklearn GradientBoosting)
- joblib, matplotlib, seaborn (optional charts)
