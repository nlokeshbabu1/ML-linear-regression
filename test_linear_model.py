import numpy as np
from sklearn.linear_model import LinearRegression

def test_linear_model():
    """
    Test script to validate the linear regression concept
    with a simplified version of the student mark model
    """
    print("Testing Linear Model for Student Marks Prediction")
    print("="*50)
    
    # Sample data: hours studied and corresponding marks
    # Format: [hours] -> [marks]
    hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    marks = np.array([25, 30, 35, 40, 50, 55, 60, 70, 75, 80])
    
    print("Sample data:")
    print("Hours -> Marks")
    for h, m in zip(hours_studied.flatten(), marks):
        print(f"{h} -> {m}")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(hours_studied, marks)
    
    print(f"\nTrained model:")
    print(f"Slope: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Equation: Marks = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")
    
    # Make predictions
    test_hours = np.array([2.5, 5.5, 8.5]).reshape(-1, 1)
    predicted_marks = model.predict(test_hours)
    
    print(f"\nPredictions:")
    for hours, mark in zip(test_hours.flatten(), predicted_marks):
        print(f"Hours: {hours} -> Predicted Mark: {mark:.2f}")
        
    # Verify with a simple manual calculation
    print(f"\nManual verification for 5 hours:")
    manual_calc = model.coef_[0] * 5 + model.intercept_
    print(f"Manual: {model.coef_[0]:.2f} * 5 + {model.intercept_:.2f} = {manual_calc:.2f}")
    print(f"Model prediction: {model.predict([[5]])[0]:.2f}")
    
    return model

if __name__ == "__main__":
    model = test_linear_model()