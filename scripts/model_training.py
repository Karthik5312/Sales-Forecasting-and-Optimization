import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(input_path, model_path):
    # Load processed data
    data = pd.read_csv(input_path)
    X = data[['Region', 'Category', 'Units Sold', 'Discount', 'Month', 'Quarter']]
    y = data['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"Model RMSE: {rmse}")

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Execute
train_model('data/processed_data.csv', 'outputs/model.pkl')
