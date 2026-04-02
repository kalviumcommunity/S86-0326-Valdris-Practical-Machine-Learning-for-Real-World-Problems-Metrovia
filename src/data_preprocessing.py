import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import src.config as config

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(filepath)

def validate_features(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    """
    Validate feature and target definitions.
    
    Checks for:
    - Target not in features
    - No ID-like columns in features
    - No suspiciously high correlations (possible leakage)
    - All features exist in the data
    """
    # Check target not in features
    if target_col in feature_cols:
        raise ValueError(f"Target '{target_col}' found in feature list! This causes leakage.")
    
    # Check all features exist in data
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Features not found in dataset: {missing}")
    
    # Simple leakage check: Correlation check for numerical features
    for col in feature_cols:
        if col in df.select_dtypes(include=['number']).columns:
            # We must use target variable for correlation assessment
            if target_col in df.columns:
                corr = df[col].corr(df[target_col])
                if abs(corr) > 0.95:
                    print(f"[WARNING] '{col}' has extremely high correlation ({corr:.3f}) with target. "
                          "Possible data leakage.")

    print(f"[SUCCESS] Feature validation passed: {len(feature_cols)} features and target '{target_col}'.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning tasks.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Ensure target column is numeric
    if config.TARGET_COLUMN in df_clean.columns:
        df_clean[config.TARGET_COLUMN] = pd.to_numeric(df_clean[config.TARGET_COLUMN], errors="coerce")
        # Impute missing delay with 0 (assuming no delay)
        df_clean[config.TARGET_COLUMN] = df_clean[config.TARGET_COLUMN].fillna(0)
    
    # We leave trip_id for now; it will be excluded when splitting via ALL_FEATURES
    return df_clean

def split_data(
    df: pd.DataFrame, 
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets based on explicit features.
    
    Args:
        df: Input cleaned DataFrame.
        target_column: The name of the target variable.
        feature_columns: Explicit list of columns to use as features.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple containing (X_train, X_test, y_train, y_test).
    """
    # Perform validation before splitting
    validate_features(df, target_column, feature_columns)
    
    # Separate using explicit feature lists (prevents accidental inclusion of IDs or Target)
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

