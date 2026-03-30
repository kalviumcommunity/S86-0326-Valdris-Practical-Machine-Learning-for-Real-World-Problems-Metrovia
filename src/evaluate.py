import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """
    Evaluate the fitted model on test data and return key performance metrics.
    
    Args:
        model: Trained classifier (supports predict and predict_proba).
        X_test: Transformed test feature DataFrame.
        y_test: Encoded test target labels.
        
    Returns:
        A dictionary containing (accuracy, precision, recall, f1, roc_auc).
    """
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs)
    }
    
    return metrics
