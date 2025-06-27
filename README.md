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

## Project Configuration

Most scripts read default paths from `config.yml`. You can customize where
processed data and model artifacts are stored by editing this file. Each
section overrides the defaults defined in the code. For example:

```yaml
data:
  processed_features: custom/features.csv
  processed_target: custom/target.csv
model:
  path: output/model.joblib
```

## Processed Input Format

The preprocessing step produces `data/processed/processed_features.csv`.
Categorical variables are one-hot encoded (e.g. `gender_Female`,
`gender_Male`), while `SeniorCitizen`, `tenure`, `MonthlyCharges`, and
`TotalCharges` remain numeric. The `make_prediction` function expects
input data with these columns.
Processed datasets are committed to version control in the `data/processed/`
folder so that experiments can be reproduced easily.

## Preprocessing the Data
Run `python scripts/run_preprocessing.py` to generate the processed feature and
target CSV files in the `data/processed/` directory. This step cleans the raw
dataset and performs one-hot encoding so that training and prediction scripts
use consistent inputs. The fitted preprocessing pipeline is saved to
`models/preprocessor.joblib` and reused during prediction.

## Training the Model
Run `python scripts/run_training.py` to train a churn model using the processed
datasets. The script expects `data/processed/processed_features.csv` and
`data/processed/processed_target.csv` by default, but you can override these
paths with `--X_path` and `--y_path`.  Additional logistic regression
hyperparameters are exposed as CLI flags, for example:

```bash
python scripts/run_training.py \
  --solver saga --C 0.5 --penalty l1 --random_state 7 --max_iter 200 --test_size 0.3
```
## Monitoring and Retraining
Run `python -m src.monitor_performance` to evaluate the current model on the processed dataset. If accuracy drops below 0.8 the model will automatically retrain. A convenience wrapper is also available via `python scripts/run_monitor.py`.
The accuracy threshold can be overridden with the `CHURN_THRESHOLD` environment variable or the `--threshold` argument of `run_monitor.py`. The same hyperparameters as `run_training.py` can also be supplied when retraining through this script. You can also override the processed data paths with `--X_path` and `--y_path`:

```bash
python scripts/run_monitor.py \
  --threshold 0.85 \
  --solver saga --C 0.8 --penalty l1 --random_state 21 --max_iter 250 --test_size 0.25
```

## Evaluating a Trained Model
Use `python scripts/run_evaluation.py` to compute accuracy and F1-score of the current model on the processed dataset. Pass `--output metrics.json` to save the metrics to a JSON file. Include the `--detailed` flag to also generate a full classification report.
If the model file is missing, the evaluation script will automatically download it from MLflow using the saved run ID (or the `MLFLOW_RUN_ID` environment variable). You can also specify the run directly with `--run_id <RUN_ID>`.
The evaluation step also logs these metrics to MLflow so you can monitor performance over time.
For a more comprehensive breakdown of precision and recall per class, run:

```bash
python scripts/run_evaluation.py --detailed --output detailed_metrics.json
```

## Batch Prediction
Use `python scripts/run_prediction.py <input_csv> --output_csv predictions.csv` to generate churn predictions for a CSV file of processed features. The script adds `prediction` and `probability` columns and saves the result to the specified output file.

During training, the feature column order is saved to `models/feature_columns.json` and logged to MLflow as an artifact. The prediction utilities automatically load this file so that incoming data can be aligned to the expected columns. If a column is missing in the input, it will be filled with zero during prediction.
The training step also records the MLflow run ID in `models/mlflow_run_id.txt`. If `feature_columns.json` is missing, the prediction code uses this run ID to download the artifact from MLflow.
If `models/churn_model.joblib` is missing, the prediction code will also use the saved run ID to download the trained model from MLflow automatically.
The `run_prediction.py` script relies on the same mechanism, so batch predictions work even if the local model file is absent as long as the run ID file is present. You can also set the `MLFLOW_RUN_ID` environment variable or pass `--run_id <RUN_ID>` to specify the run ID without the file.

## End-to-End Pipeline
Run `python scripts/run_pipeline.py` to execute preprocessing, training, and evaluation in one step. This recreates the processed datasets, trains the model, evaluates it, and prints the resulting metrics.
You can pass the same hyperparameters used by `run_training.py` to experiment with different model settings:

```bash
python scripts/run_pipeline.py \
  --solver saga --C 0.5 --penalty l1 --random_state 7 --max_iter 200 --test_size 0.3
```

## Developer Setup
1. Install project dependencies using `pip install -r requirements.txt`.
2. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. Run `pytest -q` to ensure everything works.

## Command-Line Interface
All tasks can be executed through a consolidated CLI powered by [Typer](https://typer.tiangolo.com/).
Run the desired command via `python -m src.cli <command>`:

```bash
python -m src.cli preprocess       # prepare datasets
python -m src.cli train            # train the model
python -m src.cli evaluate --detailed
python -m src.cli predict data/processed/processed_features.csv --output_csv preds.csv
```

Use `--help` after any command to view available options.
## How to Contribute (and test Jules)
Jules, our Async Development Agent, will assist in building out features, tests, and MLOps components. Please create clear issues.
