import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from typing import Union, Tuple

from .constants import PREPROCESSOR_PATH
import joblib


def preprocess(
    df_path: str, *, return_preprocessor: bool = False, save_preprocessor: bool = False
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, ColumnTransformer]]:
    """
    Preprocesses the raw customer data.

    Args:
        df_path (str): Path to the raw CSV data file.

    Parameters
    ----------
    df_path : str
        Path to the raw CSV file.
    return_preprocessor : bool, optional
        Whether to return the fitted ``ColumnTransformer``.
    save_preprocessor : bool, optional
        If ``True`` the fitted preprocessor is saved to
        ``PREPROCESSOR_PATH``.

    Returns
    -------
    tuple
        Processed features (``X``) and target (``y``). If
        ``return_preprocessor`` is ``True`` an additional third element,
        the fitted preprocessor, is returned.
    """
    df: pd.DataFrame = pd.read_csv(df_path)

    # Convert 'TotalCharges' to numeric, coercing errors, and fill NaNs
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target variable 'Churn'
    label_encoder_churn: LabelEncoder = LabelEncoder()
    df["Churn"] = label_encoder_churn.fit_transform(df["Churn"])
    y: pd.Series = df["Churn"]

    # Separate features (X)
    X: pd.DataFrame = df.drop("Churn", axis=1)
    X = X.drop("customerID", axis=1)  # Drop customerID as it's an identifier

    # Identify categorical and numerical features
    categorical_features: pd.Index = X.select_dtypes(include=["object"]).columns
    numerical_features: pd.Index = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for categorical and numerical features
    # One-hot encode categorical features
    # Numerical features are passed through for now; add scaling here if desired.
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",  # Keep numerical columns not explicitly transformed
    )

    X_processed: np.ndarray = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    # Get feature names from OneHotEncoder
    ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )

    # Concatenate OHE feature names with remaining numerical feature names
    # Numerical columns appear last in the order they existed prior to encoding
    # because remainder='passthrough' appends them after the encoded features.
    processed_feature_names = list(ohe_feature_names) + list(numerical_features)

    X_processed_df = pd.DataFrame(
        X_processed, columns=processed_feature_names, index=X.index
    )

    if save_preprocessor:
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

    if return_preprocessor:
        return X_processed_df, y, preprocessor
    return X_processed_df, y
