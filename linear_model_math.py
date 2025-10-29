# Simple Linear Regression Example for House Prices

# This demonstrates the mathematical concept behind linear regression
# where we predict house prices based on size (square footage)

# Sample data points:
# House Size (sq ft) (x) -> Price (thousands $) (y)
# (800, 120), (1000, 150), (1200, 180), (1400, 210), (1600, 240), (1800, 270), (2000, 300), (2200, 330), (2400, 360), (2600, 390)

# Linear regression finds the best fitting line: y = mx + b
# where:
# - m is the slope (change in price per square foot)
# - b is the y-intercept (predicted price when size = 0 sq ft)

# Using the dataset above, we can calculate:
# - Slope (m) ≈ 0.15 (price increases by ~$0.15k or $150 per additional sq ft)
# - Intercept (b) ≈ 0.00 (base price for a house with 0 sq ft)

# So the equation becomes:
# Predicted Price = 0.15 * House Size + 0.00

# Example predictions:
# - 0 sq ft → $0.00k price
# - 1000 sq ft → 0.15*1000 + 0.00 = $150k
# - 2000 sq ft → 0.15*2000 + 0.00 = $300k

print("Linear Regression Model Example")
print("===============================")
print("y = mx + b")
print("where m = 0.15 (slope), b = 0.00 (intercept)")
print()

def predict_price(size):
    """Predict house price based on size in square feet"""
    slope = 0.15
    intercept = 0.00
    return slope * size + intercept

# Test the function
test_sizes = [500, 1000, 1500, 2000, 2500]
print("House Size (sq ft) -> Predicted Price (thousands $)")
for size in test_sizes:
    price = predict_price(size)
    print(f"{size:6.0f} sq ft -> ${price:6.2f}k")