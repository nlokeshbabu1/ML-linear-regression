# Simple Linear Regression Example for Student Marks

# This demonstrates the mathematical concept behind linear regression
# where we predict marks based on hours studied

# Sample data points:
# Hours Studied (x) -> Marks (y)
# (1, 25), (2, 30), (3, 35), (4, 40), (5, 50), (6, 55), (7, 60), (8, 70), (9, 75), (10, 80)

# Linear regression finds the best fitting line: y = mx + b
# where:
# - m is the slope (change in marks per hour studied)
# - b is the y-intercept (predicted marks when hours = 0)

# Using the dataset above, we can calculate:
# - Slope (m) ≈ 6.36 (marks increase by ~6.36 per additional hour)
# - Intercept (b) ≈ 17.00 (base marks without studying)

# So the equation becomes:
# Predicted Marks = 6.36 * Hours Studied + 17.00

# Example predictions:
# - 0 hours → 17.00 marks
# - 5 hours → 6.36*5 + 17.00 = 48.8 marks
# - 10 hours → 6.36*10 + 17.00 = 80.6 marks

print("Linear Regression Model Example")
print("===============================")
print("y = mx + b")
print("where m = 6.36 (slope), b = 17.00 (intercept)")
print()

def predict_marks(hours):
    """Predict marks based on hours studied"""
    slope = 6.36
    intercept = 17.00
    return slope * hours + intercept

# Test the function
test_hours = [0, 2.5, 5, 7.5, 10]
print("Hours Studied -> Predicted Marks")
for hours in test_hours:
    marks = predict_marks(hours)
    print(f"{hours:6.1f} hours -> {marks:6.2f} marks")