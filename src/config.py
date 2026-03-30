import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "transit_trips.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned_trips.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model and Pipeline paths
MODEL_PATH = os.path.join(MODELS_DIR, "delay_model.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# Experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "delay_minutes"

# Feature lists
CATEGORICAL_COLS = [
    "route_id", "stop_id", "weather_condition"
]

NUMERICAL_COLS = ["scheduled_hour", "route_avg_delay_30d", "temperature"]
