import os
import pandas as pd
import joblib
from src.preprocess_data import preprocess
from src.constants import PREPROCESSOR_PATH


def test_preprocessor_is_saved(tmp_path):
    data = {
        "customerID": ["1", "2"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 5],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "188.95"],
        "Churn": ["No", "Yes"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "df.csv"
    df.to_csv(csv_path, index=False)

    if os.path.exists(PREPROCESSOR_PATH):
        os.remove(PREPROCESSOR_PATH)

    X, y, pre = preprocess(csv_path, return_preprocessor=True, save_preprocessor=True)
    assert os.path.exists(PREPROCESSOR_PATH)

    loaded_pre = joblib.load(PREPROCESSOR_PATH)
    X_loaded = loaded_pre.transform(df.drop(["customerID", "Churn"], axis=1))
    assert X_loaded.shape == (2, X.shape[1])
