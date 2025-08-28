import joblib
import pandas as pd
import json


def run_prediction():
    """
    Loads the saved AI model, reads new data from input.json,
    and makes a prediction.
    """
    # --- 1. Load the AI Model ---
    try:
        model = joblib.load('predictive_maintenance_pipeline.joblib')
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'predictive_maintenance_pipeline.joblib' not found.")
        return

    # --- 2. Load New Data for Prediction ---
    try:
        with open('input.json', 'r') as f:
            input_data = json.load(f)
        print("✅ New data loaded from input.json.")
    except FileNotFoundError:
        print("❌ Error: 'input.json' not found.")
        return

    # --- 3. Prepare Data and Predict ---
    # Convert the single dictionary of data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # === FIX: Add the missing columns with placeholder values ===
    # The model expects to see these columns, even if it ignores them.
    # We will add them here before making the prediction.
    input_df['Product ID'] = 'Placeholder'  # Add Product ID column
    input_df['Failure Type'] = 'No Failure'   # Add Failure Type column
    # ==========================================================

    # Use the loaded model to make a prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = probability[prediction] * 100

    # --- 4. Display the Result ---
    result_status = "Failure Predicted" if prediction == 1 else "Nominal"

    print("\n--- AI PREDICTION RESULTS ---")
    print(f"Status: {result_status}")
    print(f"Confidence: {confidence:.2f}%")
    print("-----------------------------\n")

    final_output = {
        "status": result_status,
        "is_failure": bool(prediction),
        "confidence": round(confidence, 2),
        "inputs": input_data
    }
    print("JSON output for automation tools:")
    print(json.dumps(final_output, indent=4))


if __name__ == "__main__":
    run_prediction()
