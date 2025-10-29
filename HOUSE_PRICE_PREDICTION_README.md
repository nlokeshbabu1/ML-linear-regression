# House Price Prediction Model

This script implements a linear regression model to predict house prices based on size (square footage) using numpy, pandas, sklearn, and matplotlib.

## Files

- `house_price_prediction_model.py`: Main implementation with visualization using numpy, pandas, sklearn, and matplotlib
- `STUDENT_MARK_MODEL_EXPLANATION.md`: Explanation of the linear model (adapted for house prices)
- `BEGINNERS_GUIDE_LINEAR_REGRESSION.md`: Comprehensive guide to linear regression

## Model Description

The model predicts house prices based on the size of the house in square feet. It follows a linear relationship:

`Price = Slope * Size + Intercept`

## Requirements

- Python 3
- numpy
- pandas
- matplotlib
- scikit-learn

## How to Run

```bash
python3 house_price_prediction_model.py
```

## Example Output

The model will:
- Load or generate sample data of house sizes vs prices
- Create a pandas DataFrame for better data handling
- Train the linear regression model
- Display model parameters (slope and intercept)
- Show performance metrics (Mean Squared Error and R² Score)
- Provide sample predictions for different house sizes
- Generate visualizations:
  1. Original data and regression line
  2. Actual vs predicted prices
  3. Residual plot

## Key Features

1. Uses numpy for numerical operations
2. Uses pandas for data handling and manipulation
3. Uses sklearn for the linear regression model
4. Uses matplotlib for data visualization
5. Includes model evaluation metrics
6. Shows predictions for various house sizes

## Interpretation

- The slope indicates how much additional price a house can expect per additional square foot
- The intercept represents the predicted price for a house with 0 square feet (theoretical base price)
- R² Score indicates how well the model fits the data (1.0 is perfect)