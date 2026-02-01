# CSC525-Mod3-Lab3.5
# 3.5.1: LAB: Linear regression using scikit-learn
# 
# The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds.
# This data can be used to predict the price of a diamond based on its characteristics.
# Use sklearn's LinearRegression() function to predict the price of a diamond from the diamond's carat and table values.
# 
# Import needed packages for regression.
# Initialize and fit a multiple linear regression model.
# Get the estimated intercept weight.
# Get the estimated weights of the carat and table features.
# Predict the price of a diamond with the user-input carat and table values.
# 
# Ex: If the input is:
# 0.5
# 60
# the output should be:
# Intercept is 1961.992
# Weights for carat and table features are [7820.038  -74.301]
# Predicted price is [1413.97]

# Import needed packages for regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

# Input feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv('diamonds.csv')

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds['price']

# Initialize a multiple linear regression model
model = LinearRegression()

# Fit the multiple linear regression model to the input and output features
model.fit(X, y)

# Get estimated intercept weight
intercept = model.intercept_
print('Intercept is', round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = model.coef_
print('Weights for carat and table features are', np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
prediction = model.predict([[carat, table]])
print('Predicted price is', np.round(prediction, 2))