import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning tasks like formatting and missing values.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Example: Handle 'TotalCharges' being string but numeric
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        # Impute missing with median for basic cleaning
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())
    
    # Drop customerID as it's not a predictor
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
        
    return df_clean

def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df: Input cleaned DataFrame.
        target_column: The name of the target variable.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple containing (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column].apply(lambda x: 1 if x == "Yes" else 0) # Encoding Yes/No target to 1/0
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
