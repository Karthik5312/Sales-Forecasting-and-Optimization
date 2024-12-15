import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    # Load raw data
    data = pd.read_csv(input_path)

    # Handle missing values
    data.fillna({'Discount': 0, 'Revenue': data['Revenue'].mean()}, inplace=True)

    # Extract date-based features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter

    # Encode categorical variables
    label_enc = LabelEncoder()
    data['Region'] = label_enc.fit_transform(data['Region'])
    data['Category'] = label_enc.fit_transform(data['Category'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Units Sold', 'Discount', 'Revenue']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


# Execute
preprocess_data('data/raw_sales_data.csv', 'data/processed_data.csv')
