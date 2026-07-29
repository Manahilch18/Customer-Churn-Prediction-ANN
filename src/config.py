import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Churn_Modelling.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "proccessed") # Note: using original spelling 'proccessed' from dir tree

# Processed Data Paths
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train_processed.csv")
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test_processed.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train_processed.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test_processed.csv")

# Models Directory and Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "ann_churn_production.h5")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "ann_churn_baseline.h5")
# SCALER_PATH: the verified scaler fit on raw X_train (notebook-04 pipeline).
# final_scaler.pkl was incorrectly fit on already-scaled data; do not use it for inference.
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FINAL_SCALER_PATH = os.path.join(MODELS_DIR, "final_scaler.pkl")  # kept for reference; do not use
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
BUSINESS_CAPACITY_PATH = os.path.join(MODELS_DIR, "business_capacity.pkl")
BEST_THRESHOLD_PATH = os.path.join(MODELS_DIR, "best_threshold.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

# Reports Directory and Paths
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_CSV_PATH = os.path.join(REPORTS_DIR, "metrics.csv")
CONFUSION_MATRIX_CSV_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.csv")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Default Fallback Values (Can be overridden by loading PKLs)
DEFAULT_THRESHOLD = 0.5
DEFAULT_CAPACITY = 1000
COST_FN = 500
COST_FP = 50
