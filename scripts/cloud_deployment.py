from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('outputs/model.pkl')
scaler = joblib.load('outputs/scaler.pkl')  # Load the scaler used during preprocessing

# Default route for instructions
@app.route('/', methods=['GET'])
def home():
    return (
        "Welcome to the Sales Prediction API!<br>"
        "Use the following endpoints:<br>"
        "/test - To test if the server is running.<br>"
        "/predict - To make predictions (POST request)."
    )

# Test route to verify the server is running
@app.route('/test', methods=['GET'])
def test():
    return "Test route is working!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(input_data)

        # Ensure the correct feature columns are present
        features = ['Region', 'Category', 'Units Sold', 'Discount', 'Month', 'Quarter']
        if not all(feature in df.columns for feature in features):
            return jsonify({"error": "Missing one or more required features!"}), 400

        # Select only the features required for prediction
        X = df[features]

        # Make predictions
        scaled_predictions = model.predict(X)

        # Inverse transform predictions to original scale
        unscaled_predictions = scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).flatten()

        # Round predictions to two decimal places
        rounded_predictions = [round(value, 2) for value in unscaled_predictions]

        # Return predictions as JSON
        return jsonify({'Predicted_Revenue': rounded_predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
