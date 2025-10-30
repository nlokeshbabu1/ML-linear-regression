# Linear Regression Guide

This guide explains linear regression concepts with practical Python implementations. Linear regression is a fundamental machine learning algorithm used to predict a continuous target variable based on one or more input features.

## Table of Contents
1. [Simple Linear Regression](#simple-linear-regression)
2. [Multiple Linear Regression](#multiple-linear-regression)
3. [Key Concepts](#key-concepts)
4. [Implementation Details](#implementation-details)

## Simple Linear Regression

Simple linear regression uses a single feature to predict the target variable. The model follows the equation:

`y = b₀ + b₁x`

Where:
- `y` is the target variable (e.g., Score)
- `x` is the input feature (e.g., Hours studied)
- `b₀` is the y-intercept
- `b₁` is the slope

### Implementation Example

The `linear_regression.py` file demonstrates simple linear regression using student study hours and scores:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample data
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

# Create DataFrame
df = pd.DataFrame(data)
print("📊 Dataset:\n", df, "\n")

# Split data into training and testing sets
X = df[['Hours']]
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print results
print("🧮 Model Parameters:")
print(f"Intercept (b₀): {model.intercept_:.2f}")
print(f"Slope (b₁): {model.coef_[0]:.2f}\n")

print("📈 Model Performance:")
print(f"R² Score (Accuracy): {r2:.3f}")
print(f"Mean Squared Error: {mse:.3f}\n")

# Predict for new data
hours = int(input("Enter the hours : "))
predicted_score = model.predict([[hours]])[0]
print(f"🔮 Predicted Score for {hours} hours studied: {predicted_score:.2f}%\n")

# Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score (%)')
plt.title('Linear Regression: Hours vs Score')
plt.legend()
plt.grid(True)
plt.show()
```

## Multiple Linear Regression

Multiple linear regression extends simple linear regression to use multiple features for prediction. The model follows the equation:

`y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`

Where:
- `y` is the target variable (e.g., Marks)
- `x₁, x₂, ..., xₙ` are input features (e.g., Hours, Sleep, Attendance)
- `b₀` is the y-intercept
- `b₁, b₂, ..., bₙ` are coefficients for each feature

### Implementation Example

The `linear_regression_multiple_feature.py` file demonstrates multiple linear regression using student study hours, sleep, and attendance to predict marks:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 2: Create the dataset
data = {
    'Hours': [5, 6, 4, 7, 8, 3, 9],
    'Sleep': [7, 6, 8, 5, 6, 7, 6],
    'Attendance': [80, 85, 70, 90, 95, 60, 100],
    'Marks': [75, 80, 70, 85, 90, 65, 95]
}

df = pd.DataFrame(data)

# Step 3: Split features and target
X = df[['Hours', 'Sleep', 'Attendance']]  # multiple features
y = df['Marks']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 20% of the data will be used for testing, 80% for training
# Fixes the random split so that every time you run it, you get the same train/test split

# Step 4: Train Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients and intercept
print("Intercept (b):", model.intercept_)
print("Coefficients (w1, w2, w3):", model.coef_)

y_pred = model.predict(X_test)

# Step 5: Predict and evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 6: Predict for a new student
# New student: 7 hours study, 6 hours sleep, 90% attendance
new_student = [[7, 6, 90]]
predicted_marks = model.predict(new_student)
print(f"Predicted Marks: {predicted_marks[0]:.2f}")
```

## Key Concepts

### R² Score (Coefficient of Determination)
- Measures how well the regression predictions approximate the real data points
- Ranges from 0 to 1, where 1 indicates perfect prediction
- Values closer to 1 represent better model performance

### Mean Squared Error (MSE)
- Measures the average squared difference between predicted and actual values
- Lower values indicate better model performance

### Train-Test Split
- Splits the dataset into training and testing sets
- Common split ratios include 80/20 or 70/30
- Training set is used to train the model
- Testing set is used to evaluate model performance

### Model Parameters
- **Intercept (b₀)**: The value of y when x is 0
- **Slope/Coefficient (b₁...bₙ)**: Represents how much y changes when x changes by one unit

## Implementation Details

Both implementations use the Scikit-Learn library for machine learning operations. The key steps include:

1. **Data Preparation**: Creating and organizing the dataset
2. **Feature Selection**: Identifying input and target variables
3. **Data Splitting**: Separating training and testing data
4. **Model Training**: Fitting the linear regression model to the training data
5. **Prediction**: Using the trained model to make predictions
6. **Evaluation**: Measuring the model's performance using metrics
7. **Visualization**: Plotting results for simple linear regression

The simple linear regression model includes visualization capabilities to show the relationship between the feature and target variables, while the multiple linear regression model focuses on prediction accuracy using multiple features simultaneously.

## Use Cases

- **Simple Linear Regression**: Best for scenarios with one clear predictor variable
- **Multiple Linear Regression**: More realistic for real-world scenarios where multiple factors influence the outcome

Both approaches provide valuable insights into relationships between variables and can be used for prediction and trend analysis.