import numpy as np
import matplotlib.pyplot as plt

# Simple Linear Regression from Scratch
def simple_linear_regression(x, y):
    """
    Calculate the slope (m) and y-intercept (b) for linear regression
    Formula: y = mx + b
    """
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope (m)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    m = numerator / denominator
    
    # Calculate y-intercept (b)
    b = y_mean - m * x_mean
    
    return m, b

# Example: Predicting test score based on hours studied
hours_studied = np.array([1, 2, 3, 4, 5, 6])
test_scores = np.array([50, 55, 65, 70, 80, 85])

print("Hours Studied:", hours_studied)
print("Test Scores:", test_scores)

# Calculate the regression line
slope, y_intercept = simple_linear_regression(hours_studied, test_scores)

print(f"\nSlope (m): {slope:.2f}")
print(f"Y-intercept (b): {y_intercept:.2f}")
print(f"Regression equation: Score = {slope:.2f} * Hours + {y_intercept:.2f}")

# Make predictions
def predict(x, m, b):
    return m * x + b

# Predict score for 3.5 hours of study
predicted_score = predict(3.5, slope, y_intercept)
print(f"\nPredicted score for 3.5 hours of study: {predicted_score:.2f}")

# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(hours_studied, test_scores, color='blue', label='Actual Data')
plt.plot(hours_studied, predict(hours_studied, slope, y_intercept), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Simple Linear Regression: Hours Studied vs Test Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()