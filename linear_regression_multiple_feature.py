import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#Step 2: Create the dataset
data = {
    'Hours': [5, 6, 4, 7, 8, 3, 9],
    'Sleep': [7, 6, 8, 5, 6, 7, 6],
    'Attendance': [80, 85, 70, 90, 95, 60, 100],
    'Marks': [75, 80, 70, 85, 90, 65, 95]
}

df = pd.DataFrame(data)


#Step 3: Split features and target
X = df[['Hours', 'Sleep', 'Attendance']]  # multiple features
y = df['Marks']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#20% of the data will be used for testing, 80% for training
#Fixes the random split so that every time you run it, you get the same train/test split

#Step 4: Train Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients and intercept
print("Intercept (b):", model.intercept_)
print("Coefficients (w1, w2, w3):", model.coef_)

y_pred = model.predict(X_test)

#Step 5: Predict and evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

#Step 6: Predict for a new student
# New student: 7 hours study, 6 hours sleep, 90% attendance
new_student = [[7, 6, 90]]
predicted_marks = model.predict(new_student)
print(f"Predicted Marks: {predicted_marks[0]:.2f}")
