import pandas as pd
import joblib
from typing import Tuple

def save_artifacts(
    model, 
    pipeline, 
    model_path: str, 
    pipeline_path: str
):
    """
    Serialize and save model and preprocessing pipeline objects.
    
    Args:
        model: Trained model object.
        pipeline: Fitted preprocessing pipeline.
        model_path: String path for the model object.
        pipeline_path: String path for the pipeline object.
    """
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)

def load_artifacts(
    model_path: str, 
    pipeline_path: str
) -> Tuple[object, object]:
    """
    Load serialized model and preprocessing pipeline objects from disk.
    
    Args:
        model_path: File path of the model.
        pipeline_path: File path of the pipeline.
        
    Returns:
        A tuple (model, pipeline).
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline

def predict(
    new_data: pd.DataFrame, 
    model, 
    pipeline
) -> pd.DataFrame:
    """
    Apply pre-trained models to generate transit delay predictions on fresh data.
    
    Args:
        new_data: Input raw trip records.
        model: Loaded regressor object.
        pipeline: Loaded preprocessing pipeline transformer.
        
    Returns:
        pandas DataFrame containing original features and predicted delay_minutes.
    """
    # Transform raw incoming data
    X_processed = pipeline.transform(new_data)
    
    # Generate predictions
    predictions = model.predict(X_processed)
    
    # Return a copy of the input data with prepended predictions
    result_df = new_data.copy()
    result_df["predicted_delay_minutes"] = predictions
    
    return result_df
