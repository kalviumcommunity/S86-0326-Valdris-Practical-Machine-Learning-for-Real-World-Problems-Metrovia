import joblib
import os
from src.config import MODELS_DIR

def save_model(model, filename):
    """
    Saves a trained model to the models directory.
    """
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(filename):
    """
    Loads a model from the models directory.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model

def save_pipeline(pipeline, filename):
    """
    Saves a preprocessing pipeline to the models directory.
    """
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to: {path}")

def load_pipeline(filename):
    """
    Loads a preprocessing pipeline from the models directory.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found at: {path}")
    pipeline = joblib.load(path)
    print(f"Pipeline loaded from: {path}")
    return pipeline
