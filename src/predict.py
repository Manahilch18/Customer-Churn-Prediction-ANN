import pandas as pd
import numpy as np
from src.utils import load_all_artifacts
from src import config

# Pre-load artifacts to avoid reloading them on every prediction
ARTIFACTS = load_all_artifacts()


def _preprocess(df):
    """
    Applies the same preprocessing pipeline as notebook-04 to a raw customer
    dataframe (unscaled, with original column names from the raw CSV minus the
    ID/Surname columns). Returns a scaled NumPy array ready for model.predict().

    Expected raw columns:
        CreditScore, Gender (str or 0/1), Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        Geography_Germany (int 0/1), Geography_Spain (int 0/1)

    Note: Geography is expected to already be one-hot encoded as
    Geography_Germany and Geography_Spain because the Streamlit UI and the
    batch CSV both supply it that way. Gender must be 0/1 (Female=0, Male=1).
    """
    feature_cols = ARTIFACTS.get('feature_columns', [])
    if not feature_cols:
        raise RuntimeError("feature_columns not loaded. Ensure models/feature_columns.pkl exists.")

    # Validate all required columns are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Required: {feature_cols}"
        )

    # Reorder to match training column order; keep as DataFrame so scaler
    # receives named features (suppresses sklearn feature-name UserWarning).
    X_df = df[feature_cols].astype(float)

    scaler = ARTIFACTS.get('scaler')
    if scaler is None:
        raise RuntimeError("Scaler not loaded. Ensure models/scaler.pkl exists.")

    # Apply the same StandardScaler that was fit on raw X_train (notebook-04)
    X_scaled = scaler.transform(X_df)
    return X_scaled


def predict_single(customer_dict):
    """
    Predicts churn for a single customer given raw (unscaled) feature values.

    Args:
        customer_dict (dict): Raw customer features keyed by column name.
            Required keys match feature_columns.pkl exactly.
    Returns:
        dict: Contains 'churn_probability' (float), 'predicted_churn' (int),
              and 'prediction_threshold_used' (float).
    """
    df = pd.DataFrame([customer_dict])
    return predict_batch(df).iloc[0].to_dict()


def predict_batch(dataframe):
    """
    Predicts churn for a batch of customers given raw (unscaled) feature values.
    Both predict_single and predict_batch go through the same _preprocess() path
    to guarantee identical behavior.

    Args:
        dataframe (pd.DataFrame): Raw customer features. Must contain all columns
            in feature_columns.pkl.
    Returns:
        pd.DataFrame: Original dataframe with appended columns:
            'churn_probability', 'predicted_churn', 'prediction_threshold_used'.
    """
    df = dataframe.copy()

    X_scaled = _preprocess(df)

    model = ARTIFACTS.get('model')
    if model is None:
        raise RuntimeError("Model not loaded. Ensure models/ann_churn_production.h5 exists.")

    threshold = ARTIFACTS.get('best_threshold', config.DEFAULT_THRESHOLD)

    probs = model.predict(X_scaled, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    df['churn_probability'] = probs
    df['predicted_churn'] = preds
    df['prediction_threshold_used'] = threshold

    return df
