import shap
import joblib
import pandas as pd


def explain_predictions():
    # Load data and model
    data = pd.read_csv('data/processed_data.csv')
    model = joblib.load('outputs/model.pkl')
    
    # Select features used during training
    features = ['Region', 'Category', 'Units Sold', 'Discount', 'Month', 'Quarter']
    X = data[features]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values (disable additivity check if needed)
    shap_values = explainer(X, check_additivity=False)  # Disable additivity check if the discrepancy is small
    
    # Plot SHAP summary
    shap.summary_plot(shap_values, X)


# Execute
explain_predictions()
