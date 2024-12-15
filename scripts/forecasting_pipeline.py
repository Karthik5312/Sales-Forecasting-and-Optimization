import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np  # Import NumPy


def forecast_sales(model_path, input_path, output_path):
    # Load the model
    model = joblib.load(model_path)

    # Load new data for forecasting
    new_data = pd.read_csv(input_path)

    # Encode categorical columns using the same mappings as training
    label_enc_region = LabelEncoder()
    label_enc_category = LabelEncoder()

    # Convert class lists to NumPy arrays
    label_enc_region.classes_ = np.array(['East', 'North', 'South', 'West'])
    label_enc_category.classes_ = np.array(['Electronics', 'Fashion', 'Groceries'])

    # Transform the categorical columns
    new_data['Region'] = label_enc_region.transform(new_data['Region'])
    new_data['Category'] = label_enc_category.transform(new_data['Category'])

    # Select only the features used during training
    features = ['Region', 'Category', 'Units Sold', 'Discount', 'Month', 'Quarter']
    X = new_data[features]

    # Make predictions
    new_data['Predicted_Revenue'] = model.predict(X)

    # Save the forecast results
    new_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# Execute
forecast_sales('outputs/model.pkl', 'data/forecast_input.csv', 'data/forecast_output.csv')
