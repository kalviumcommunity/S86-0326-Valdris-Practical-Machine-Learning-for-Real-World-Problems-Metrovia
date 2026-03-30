import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Fit a model on the provided training data.
    
    Args:
        X_train: Transformed feature DataFrame.
        y_train: Target labels.
        random_state: Random state for reproducibility.
        
    Returns:
        Fitted RandomForest model object.
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    return model
