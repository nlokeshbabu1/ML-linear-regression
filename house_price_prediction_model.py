import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("House Price Prediction Model")
    print("="*40)
    
    # Create sample dataset: house size (in square feet) vs price (in thousands)
    # In a real scenario, you would load this from a CSV file
    sizes = np.array([800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2150, 2300, 2450, 2600, 2750, 3000]).reshape(-1, 1)
    prices = np.array([120, 140, 160, 190, 220, 240, 265, 290, 315, 340, 360, 385, 410, 435, 470])
    
    # Create a pandas DataFrame for better data handling
    df = pd.DataFrame({'House_Size_SqFt': sizes.flatten(), 'Price_Thousands': prices})
    print("Sample dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    
    # Prepare features (X) and target (y)
    X = df[['House_Size_SqFt']]  # Features
    y = df['Price_Thousands']   # Target variable
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Print model parameters
    print(f"\nModel Parameters:")
    print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Equation: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")
    
    # Test with new data points
    print(f"\nPrediction Examples:")
    test_sizes = np.array([[1000], [1500], [2000], [2500], [2800]])
    predicted_prices = model.predict(test_sizes)
    
    for size, price in zip(test_sizes.flatten(), predicted_prices):
        print(f"House size: {size:.0f} sq ft -> Predicted price: ${price:.1f}k")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data and regression line
    plt.subplot(1, 3, 1)
    plt.scatter(df['House_Size_SqFt'], df['Price_Thousands'], color='blue', label='Actual Data')
    plt.plot(df['House_Size_SqFt'], model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.xlabel('House Size (Sq Ft)')
    plt.ylabel('Price (Thousands $)')
    plt.title('House Size vs Price - Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training data vs predicted data
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, color='green', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, color='purple', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Predict for a range of sizes
    print(f"\nPredictions for various house sizes:")
    sizes_range = np.arange(800, 3100, 100).reshape(-1, 1)
    predicted_range = model.predict(sizes_range)
    
    # Show predictions for every 500 sq ft
    for i in range(0, len(sizes_range), 5):  # Every 5th element since step is 100
        s = sizes_range[i][0]
        pred = predicted_range[i]
        print(f"Size: {s:.0f} sq ft -> Predicted Price: ${pred:.1f}k")
    
    return model

if __name__ == "__main__":
    model = main()