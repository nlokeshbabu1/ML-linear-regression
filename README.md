# Student Mark Prediction Model

This project implements a linear regression model to predict student marks based on hours studied.

## Files

- `student_mark_model.py`: Main implementation with visualization
- `test_linear_model.py`: Simple test script to validate the concept

## Model Description

The model predicts student marks based on the number of hours they studied. It follows a linear relationship:

`Marks = Slope * Hours + Intercept`

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
- Sample predictions for different study hours
- Visualizations of the data and model

## Interpretation

- The slope indicates how many additional marks a student can expect per additional hour of study
- The intercept represents the predicted marks for a student who studied 0 hours
- R² Score indicates how well the model fits the data (1.0 is perfect)