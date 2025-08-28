from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

# Initialize the Flask application
app = Flask(__name__)

# --- Load the AI Model once when the server starts ---
try:
    model = joblib.load('predictive_maintenance_pipeline.joblib')
except FileNotFoundError:
    print("FATAL ERROR: predictive_maintenance_pipeline.joblib not found. Server cannot start.")
    model = None

# --- Define the prediction endpoint ---


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded, check server logs."}), 500

    try:
        # Get the JSON data sent from n8n
        input_data = request.get_json()

        # Convert it to a DataFrame
        input_df = pd.DataFrame([input_data])

        # FIX: Add the missing placeholder columns the model expects
        input_df['Product ID'] = 'Placeholder'
        input_df['Failure Type'] = 'No Failure'

        # Make the prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence = probability[prediction] * 100

        # Prepare the result
        result_status = "Failure Predicted" if prediction == 1 else "Nominal"

        final_output = {
            "status": result_status,
            "is_failure": bool(prediction),
            "confidence": round(confidence, 2),
            "inputs": input_data
        }

        # Send the result back to n8n as JSON
        return jsonify(final_output)

    except Exception as e:
        # If something goes wrong, send back an error message
        return jsonify({"error": str(e)}), 400


# This allows you to run the server by typing "python app.py"
if __name__ == '__main__':
    print("AI Model Server starting on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
