import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def generate_sample_data(n_samples=100):
    """
    Generate sample data for student marks based on hours studied
    Following a realistic pattern with some noise
    """
    # Generate random study hours between 1 and 15 hours
    hours_studied = np.random.uniform(1, 15, n_samples)
    
    # Generate marks based on hours with some realistic relationship
    # More hours generally leads to higher marks, but with variation
    base_performance = 20 + (hours_studied * 4)  # Base relationship
    noise = np.random.normal(0, 5, n_samples)    # Add some randomness
    marks = base_performance + noise
    
    # Ensure marks are within realistic bounds (0-100)
    marks = np.clip(marks, 0, 100)
    
    return hours_studied.reshape(-1, 1), marks

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
    print("Student Mark Prediction Model")
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
    print(f"Equation: Marks = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")
    
    # Evaluate the model
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Test with new data points
    print(f"\nPrediction Examples:")
    test_hours = np.array([[2], [5], [8], [12], [15]])
    predicted_marks = model.predict(test_hours)
    
    for hours, mark in zip(test_hours.flatten(), predicted_marks):
        print(f"Hours studied: {hours:.1f} -> Predicted mark: {mark:.1f}")
    
    # Visualize the results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks')
    plt.title('Training Data and Regression Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (for test set)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, color='green', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Marks')
    plt.ylabel('Predicted Marks')
    plt.title('Actual vs Predicted Marks (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print(f"\nModel Interpretation:")
    print(f"- For every additional hour of study, marks increase by approximately {model.coef_[0]:.2f} points")
    print(f"- Students with 0 hours of study are predicted to score {model.intercept_:.2f} marks")
    
    return model

if __name__ == "__main__":
    model = main()