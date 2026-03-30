import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """
    Evaluate the fitted regressor on transit data and return key performance metrics.
    
    Args:
        model: Trained regressor object.
        X_test: Transformed test feature DataFrame.
        y_test: True delay minutes.
        
    Returns:
        A dictionary containing (MAE, RMSE, R2, Median Absolute Error).
    """
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Compute regression metrics
    metrics = {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2_score": r2_score(y_test, y_pred)
    }
    
    return metrics
