import numpy as np
from sklearn.linear_model import LinearRegression

def test_linear_model():
    """
    Test script to validate the linear regression concept
    with a simplified version of the house price model
    """
    print("Testing Linear Model for House Price Prediction")
    print("="*50)
    
    # Sample data: house sizes and corresponding prices
    # Format: [size] -> [price]
    house_sizes = np.array([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]).reshape(-1, 1)
    prices = np.array([120, 150, 180, 210, 240, 270, 300, 330, 360, 390])
    
    print("Sample data:")
    print("Size (sq ft) -> Price (thousands $)")
    for s, p in zip(house_sizes.flatten(), prices):
        print(f"{s} -> {p}")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(house_sizes, prices)
    
    print(f"\nTrained model:")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Equation: Price = {model.coef_[0]:.4f} * Size + {model.intercept_:.2f}")
    
    # Make predictions
    test_sizes = np.array([900, 1500, 2100]).reshape(-1, 1)
    predicted_prices = model.predict(test_sizes)
    
    print(f"\nPredictions:")
    for size, price in zip(test_sizes.flatten(), predicted_prices):
        print(f"Size: {size} sq ft -> Predicted Price: ${price:.2f}k")
        
    # Verify with a simple manual calculation
    print(f"\nManual verification for 1000 sq ft:")
    manual_calc = model.coef_[0] * 1000 + model.intercept_
    print(f"Manual: {model.coef_[0]:.4f} * 1000 + {model.intercept_:.2f} = {manual_calc:.2f}")
    print(f"Model prediction: {model.predict([[1000]])[0]:.2f}")
    
    return model

if __name__ == "__main__":
    model = test_linear_model()