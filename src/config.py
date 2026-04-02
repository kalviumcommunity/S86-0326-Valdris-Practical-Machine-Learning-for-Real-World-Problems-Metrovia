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

# --- Feature and Target Definition ---

# Target definition
# Represents the actual delay in minutes for a given transit trip at a specific stop.
# This is a continuous numerical value (regression).
TARGET_COLUMN = "delay_minutes"

# Numerical features
# Information available at prediction time from historical logs and real-time sensors.
NUMERICAL_FEATURES = [
    "scheduled_hour",       # Hour of the day (0-23)
    "route_avg_delay_30d",  # Historical performance of the route
    "temperature"           # Real-time weather data
]

# Categorical features
# Discrete labels representing the route, location, and conditions.
CATEGORICAL_FEATURES = [
    "route_id", 
    "stop_id", 
    "weather_condition"
]

# Excluded columns
# Columns that carry no predictive signal or would cause leakage if included.
EXCLUDED_COLUMNS = [
    "trip_id"  # Unique identifier, no generalizable predictive value
]

# Derived
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Validation (optional but recommended for robustness)
# assert TARGET_COLUMN not in ALL_FEATURES, "Target leaked into features!"
# -------------------------------------
