import numpy as np
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
