import os
import warnings
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_CPP_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from src import config

def load_all_artifacts():
    """
    Loads all saved models and artifacts from disk.
    Returns a dictionary containing the artifacts.
    """
    artifacts = {}
    
    # Load ANN Production Model
    if os.path.exists(config.PRODUCTION_MODEL_PATH):
        artifacts['model'] = tf.keras.models.load_model(config.PRODUCTION_MODEL_PATH, compile=False)
        
    # Load Scaler (SCALER_PATH = models/scaler.pkl, verified against X_train_processed.csv)
    if os.path.exists(config.SCALER_PATH):
        artifacts['scaler'] = joblib.load(config.SCALER_PATH)
        
    # Load Label Encoder only if needed, suppress sklearn version mismatch warnings
    if os.path.exists(config.LABEL_ENCODER_PATH):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            artifacts['label_encoder'] = joblib.load(config.LABEL_ENCODER_PATH)
        
    # Load Feature Columns
    if os.path.exists(config.FEATURE_COLUMNS_PATH):
        artifacts['feature_columns'] = joblib.load(config.FEATURE_COLUMNS_PATH)
        
    # Load Business Capacity
    if os.path.exists(config.BUSINESS_CAPACITY_PATH):
        artifacts['capacity'] = joblib.load(config.BUSINESS_CAPACITY_PATH)
    else:
        artifacts['capacity'] = config.DEFAULT_CAPACITY
        
    # Load Best Threshold
    if os.path.exists(config.BEST_THRESHOLD_PATH):
        artifacts['best_threshold'] = joblib.load(config.BEST_THRESHOLD_PATH)
    else:
        artifacts['best_threshold'] = config.DEFAULT_THRESHOLD
        
    return artifacts

def get_top_k_risk_customers(df, capacity):
    """
    Sorts a dataframe of predicted customers by churn probability
    and returns the top K (capacity) most at-risk customers.
    """
    if 'churn_probability' not in df.columns:
        raise ValueError("Dataframe must contain a 'churn_probability' column.")
        
    # Sort descending by probability
    sorted_df = df.sort_values('churn_probability', ascending=False).reset_index(drop=True)
    
    # Return top K
    return sorted_df.head(capacity)
