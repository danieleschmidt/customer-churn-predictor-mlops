import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.train_model import train_churn_model

MODEL_PATH = 'models/churn_model.joblib'
PROCESSED_FEATURES_PATH = 'data/processed/processed_features.csv'
PROCESSED_TARGET_PATH = 'data/processed/processed_target.csv'

THRESHOLD_ACCURACY = 0.8


def evaluate_model(model_path=MODEL_PATH, X_path=PROCESSED_FEATURES_PATH, y_path=PROCESSED_TARGET_PATH):
    """Evaluate the existing model on the processed dataset."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1


def monitor_and_retrain():
    """Monitor performance and retrain the model if accuracy falls below threshold."""
    try:
        accuracy, f1 = evaluate_model()
        print(f"Current model accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    except FileNotFoundError as e:
        print(e)
        print("Training model from scratch...")
        train_churn_model(PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH)
        return

    if accuracy < THRESHOLD_ACCURACY:
        print(
            f"Accuracy {accuracy:.4f} is below threshold {THRESHOLD_ACCURACY}. Retraining model..."
        )
        train_churn_model(PROCESSED_FEATURES_PATH, PROCESSED_TARGET_PATH)
    else:
        print("Model performance is acceptable. No retraining required.")


if __name__ == '__main__':
    monitor_and_retrain()
