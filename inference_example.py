import pandas as pd
from src.config import MODEL_PATH, PIPELINE_PATH
from src.predict import load_artifacts, predict

def run_prediction_example():
    """
    Simulates an inference scenario by loading artifacts and 
    predicting on new data.
    """
    # 1. Prepare new data (inference-time input)
    # This data would typically come from an API request or a new batch file
    new_data = pd.DataFrame({
        "route_id": ["R101", "R102"],
        "stop_id": ["S001", "S005"],
        "scheduled_hour": [8, 17],
        "route_avg_delay_30d": [2.6, 9.8],
        "temperature": [16, 4],
        "weather_condition": ["Sunny", "Rainy"]
    })

    print("--- Inference Mode ---")
    print("New data received:")
    print(new_data)

    # 2. Load artifacts (Inference mode never fits!)
    print(f"\nLoading artifacts from {MODEL_PATH} and {PIPELINE_PATH}...")
    try:
        model, pipeline = load_artifacts(MODEL_PATH, PIPELINE_PATH)
    except FileNotFoundError:
        print("Error: Model or pipeline files not found. Run training (main.py) first.")
        return

    # 3. Predict
    print("Generating predictions...")
    results = predict(new_data, model, pipeline)

    print("\nPrediction Results:")
    print(results[["route_id", "stop_id", "predicted_delay_minutes"]])

if __name__ == "__main__":
    run_prediction_example()
