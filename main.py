import os
import pandas as pd
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, PIPELINE_PATH, 
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CATEGORICAL_COLS, NUMERICAL_COLS
)
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import save_artifacts

def main():
    """
    Main orchestration function to run the modular ML pipeline.
    """
    # 1. Ingestion
    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        df = load_data(RAW_DATA_PATH)
    except FileNotFoundError:
        print("Data file not found. Ensure raw data exists at the specified path.")
        return

    # 2. Cleaning
    print("Cleaning raw dataset...")
    df_clean = clean_data(df)
    
    # 3. Splitting
    print(f"Splitting data with test size {TEST_SIZE} and random state {RANDOM_STATE}...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )

    # 4. Feature Engineering
    print("Building and fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    
    # Fit and transform training data
    X_train_processed = pipeline.fit_transform(X_train)
    # Only transform test data (no fitting to prevent leakage)
    X_test_processed = pipeline.transform(X_test)

    # 5. Training
    print("Training Transit Delay Regressor...")
    model = train_model(X_train_processed, y_train, RANDOM_STATE)

    # 6. Evaluation
    print("Evaluating prediction accuracy...")
    metrics = evaluate_model(model, X_test_processed, y_test)
    
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # 7. Persistence
    print(f"\nSaving model to {MODEL_PATH}...")
    print(f"Saving pipeline to {PIPELINE_PATH}...")
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    
    print("Workflow complete.")

if __name__ == "__main__":
    main()
