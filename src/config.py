import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "telco_churn.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned_churn.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model and Pipeline paths
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_v1.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# Experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Feature lists
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
