import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def generate_sample_data(n_samples=100):
    """
    Generate sample data for house prices based on size
    Following a realistic pattern with some noise
    """
    # Generate random house sizes between 800 and 3000 square feet
    house_sizes = np.random.uniform(800, 3000, n_samples)
    
    # Generate prices based on size with some realistic relationship
    # Larger houses generally cost more, but with variation
    base_price = 100 + (house_sizes * 0.12)  # Base relationship ($120 per 1000 sq ft)
    noise = np.random.normal(0, 10, n_samples)    # Add some randomness
    prices = base_price + noise
    
    # Ensure prices are positive
    prices = np.clip(prices, 50, 500)  # Prices between $50k and $500k
    
    return house_sizes.reshape(-1, 1), prices

def create_and_train_model(X, y):
    """
    Create and train the linear regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

def main():
    print("House Price Prediction Model")
    print("="*40)
    
    # Generate sample data
    X, y = generate_sample_data(100)
    print(f"Generated {len(X)} sample data points")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = create_and_train_model(X_train, y_train)
    print(f"Model trained successfully!")
    
    # Print model parameters
    print(f"Slope (coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Equation: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")
    
    # Evaluate the model
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Test with new data points
    print(f"\nPrediction Examples:")
    test_sizes = np.array([[1000], [1500], [2000], [2500], [2800]])
    predicted_prices = model.predict(test_sizes)
    
    for size, price in zip(test_sizes.flatten(), predicted_prices):
        print(f"House size: {size:.0f} sq ft -> Predicted price: ${price:.1f}k")
    
    # Visualize the results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
    plt.xlabel('House Size (Sq Ft)')
    plt.ylabel('Price (Thousands $)')
    plt.title('Training Data and Regression Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (for test set)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, color='green', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print(f"\nModel Interpretation:")
    print(f"- For every additional square foot, price increases by approximately ${model.coef_[0]:.2f}k")
    print(f"- Base price (for a house with 0 sq ft) is predicted to be ${model.intercept_:.2f}k")
    
    return model

if __name__ == "__main__":
    model = main()