import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(input_path, output_path):
    data = pd.read_csv(input_path)
    model = IsolationForest(contamination=0.05, random_state=42)
    data['Anomaly'] = model.fit_predict(data[['Units Sold', 'Revenue', 'Discount']])

    anomalies = data[data['Anomaly'] == -1]
    anomalies.to_csv(output_path, index=False)
    print(f"Anomalies saved to {output_path}")

# Execute
detect_anomalies('data/processed_data.csv', 'data/anomaly_report.csv')
