import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Fit a regressor on the transit trip data to predict delay minutes.
    
    Args:
        X_train: Transformed feature DataFrame.
        y_train: Target labels (target: delay_minutes).
        random_state: Random state for reproducibility.
        
    Returns:
        Fitted RandomForest regression model.
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=7, 
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    return model
