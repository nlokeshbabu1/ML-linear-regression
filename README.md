# House Price Prediction Model

This project implements a linear regression model to predict house prices based on size (square footage).

## Files

- `student_mark_model.py`: Main implementation with visualization (house price version)
- `house_price_prediction_model.py`: House price prediction model using numpy, pandas, sklearn, and matplotlib
- `test_linear_model.py`: Simple test script to validate the concept
- `HOUSE_PRICE_PREDICTION_README.md`: Documentation for the house price prediction model
- `HOUSE_PRICE_MODEL_EXPLANATION.md`: Detailed explanation of the house price model
- `linear_model_math.py`: Mathematical explanation and example

## Model Description

The model predicts house prices based on the size of the house in square feet. It follows a linear relationship:

`Price = Slope * Size + Intercept`

## Requirements

- Python 3
- numpy
- matplotlib
- scikit-learn
- pandas

## How to Run

```bash
python3 student_mark_model.py
```

## Example Output

The model will generate sample data, train the linear regression model, and display:

- Model parameters (slope and intercept)
- Performance metrics (Mean Squared Error and R² Score)
- Sample predictions for different house sizes
- Visualizations of the data and model

## Interpretation

- The slope indicates how much additional price (in thousands) a house can expect per additional square foot
- The intercept represents the theoretical price for a house with 0 square feet
- R² Score indicates how well the model fits the data (1.0 is perfect)