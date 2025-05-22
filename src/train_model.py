import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb # Import XGBoost
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.xgboost # For logging XGBoost models
import joblib
import os
import numpy as np # For confusion matrix logging

# Define model configurations
MODEL_CONFIG = [
    {
        'name': 'LogisticRegression',
        'estimator': LogisticRegression(random_state=42, max_iter=1000), # Increased max_iter for convergence
        'params': {
            'solver': ['liblinear', 'saga'],
            'C': [0.01, 0.1, 1, 10]
        }
    },
    {
        'name': 'RandomForest',
        'estimator': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    {
        'name': 'XGBoost',
        'estimator': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
]

def train_churn_model(X_path, y_path, experiment_name="ChurnPredictionExperiment"):
    """
    Trains multiple churn prediction models using GridSearchCV, logs them with MLflow, and saves them.
    """
    print(f"Loading data from {X_path} and {y_path}...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting or creating MLflow experiment '{experiment_name}': {e}. Using default experiment.")
    
    best_overall_model_score = -1
    best_overall_model_path = None
    best_overall_model_name = None
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    for config in MODEL_CONFIG:
        model_name = config['name']
        estimator = config['estimator']
        param_grid = config['params']

        with mlflow.start_run(run_name=f"Train_{model_name}") as run:
            run_id = run.info.run_id
            print(f"Starting run for {model_name} with MLflow Run ID: {run_id}")
            mlflow.log_param("model_name", model_name)

            print(f"Performing GridSearchCV for {model_name}...")
            grid_search = GridSearchCV(estimator, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_ # Using 'best_score' as in prompt

            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best CV F1-score for {model_name}: {best_score:.4f}")

            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_f1_score", best_score) # Logged with consistent name

            y_pred_test = best_estimator.predict(X_test)
            y_proba_test = best_estimator.predict_proba(X_test)[:, 1]

            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_roc_auc = roc_auc_score(y_test, y_proba_test)
            
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()
            mlflow.log_metric("test_tn", tn)
            mlflow.log_metric("test_fp", fp)
            mlflow.log_metric("test_fn", fn)
            mlflow.log_metric("test_tp", tp)

            print(f"Test Set Metrics for {model_name}:")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  F1-score: {test_f1:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            print(f"  ROC AUC: {test_roc_auc:.4f}")

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_roc_auc", test_roc_auc)

            print(f"Logging {model_name} model to MLflow...")
            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(best_estimator, artifact_path=model_name)
            else:
                mlflow.sklearn.log_model(best_estimator, artifact_path=model_name)

            model_filename = f"{model_name.lower()}_model.joblib" # Aligned with prompt
            specific_model_path = os.path.join(model_dir, model_filename)
            print(f"Saving best {model_name} model to {specific_model_path}...")
            joblib.dump(best_estimator, specific_model_path)
            mlflow.log_artifact(specific_model_path, artifact_path="saved_models")

            if test_f1 > best_overall_model_score:
                best_overall_model_score = test_f1
                best_overall_model_path = specific_model_path
                best_overall_model_name = model_name
                generic_model_path = os.path.join(model_dir, 'churn_model.joblib')
                print(f"New best overall model ({model_name}) found with F1-score: {test_f1:.4f}. Saving to {generic_model_path}")
                joblib.dump(best_estimator, generic_model_path)

    print("\nTraining complete for all models.")
    if best_overall_model_name:
        print(f"Best overall model saved: {best_overall_model_name} to {best_overall_model_path} (and as models/churn_model.joblib)")
        mlflow.set_tag("best_overall_model_name", best_overall_model_name)
        mlflow.set_tag("best_overall_model_f1_score", best_overall_model_score)
    
    return best_overall_model_path


if __name__ == '__main__':
    print("Starting model training and hyperparameter tuning script directly (for testing purposes)...")
    default_X_path = 'data/processed/processed_features.csv'
    default_y_path = 'data/processed/processed_target.csv'

    if not (os.path.exists(default_X_path) and os.path.exists(default_y_path)):
        print(f"Error: Processed data not found at {default_X_path} or {default_y_path}.")
        print("Please run the preprocessing script first (e.g., scripts/run_preprocessing.py).")
    else:
        best_model_path = train_churn_model(default_X_path, default_y_path)
        if best_model_path:
            print(f"Best overall model from the run saved at: {best_model_path}")
        else:
            print("No models were trained successfully.")
