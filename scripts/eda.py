import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda(input_path):
    # Load the processed data
    data = pd.read_csv(input_path)

    # Monthly Revenue Trends
    plt.figure(figsize=(10, 6))
    data.groupby('Month')['Revenue'].sum().plot(kind='bar', color='skyblue')
    plt.title('Monthly Revenue Trends')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.show()

    # Revenue by Category
    plt.figure(figsize=(10, 6))
    data.groupby('Category')['Revenue'].sum().plot(kind='bar', color='coral')
    plt.title('Revenue by Category')
    plt.xlabel('Category')
    plt.ylabel('Revenue')
    plt.show()

    # Correlation heatmap (filter only numeric columns)
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.show()


# Execute
eda('data/processed_data.csv')
