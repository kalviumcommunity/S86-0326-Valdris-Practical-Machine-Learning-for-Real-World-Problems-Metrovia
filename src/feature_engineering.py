import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessing_pipeline(
    categorical_cols: list[str], 
    numerical_cols: list[str]
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline for categorical encoding and numerical scaling.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        
    Returns:
        Fitted Pipeline or ColumnTransformer object.
    """
    
    # Preprocessing for numerical data: impute mean and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: impute mode and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor
