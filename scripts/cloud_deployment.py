from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('outputs/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    input_data = request.get_json()
    df = pd.DataFrame(input_data)

    # Make predictions
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
