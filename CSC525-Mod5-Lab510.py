# CSC-525-Lab510
# 5.10.1 Evaluating models using plots in scikit-learn
# 
# The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds.
# This data can be used to predict the price of a diamond based on its characteristics.
# Using a sample of the dataset and scikit-learn's LinearRegression() function, predict the price of a diamond from the user-input features.
# Using scikit-learn and matplotlib, create plots to evaluate the regression model.
# 
# Initialize and fit a multiple linear regression model.
# Use the model to predict the prices of instances in X.
# Compute the prediction errors.
# Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'.
# Include a dashed line at y=0.
# Generate a partial dependence display for both input features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics

# Load diamonds sample into dataframe
diamonds = pd.read_csv('diamonds.csv').sample(n=50, random_state=42)

# Get user-input features
feature1 = input()
feature2 = input()

# Define input and output features
X = diamonds[[feature1, feature2]]
y = diamonds['price']

# Initialize and fit a multiple linear regression model
mlrModel = LinearRegression()
mlrModel.fit(X, y)

# Use the model to predict the classification of instances in X
mlrPredY = mlrModel.predict(X)

# Compute prediction errors
mlrPredError = y - mlrPredY

# Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'
fig = plt.figure()
plt.scatter(mlrPredY, mlrPredError)
plt.xlabel('Predicted')
plt.ylabel('Prediction error')

# Add dashed line at y=0
plt.axhline(y=0, color='r', linestyle='--')

plt.savefig('predictionError.png')

# Generate a partial dependence display for both input features
PartialDependenceDisplay.from_estimator(mlrModel, X, features=[0, 1])
plt.savefig('partialDependence.png')

plt.savefig('partialDependence.png')

# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(y, mlrPredY)
print("MAE:", round(mae, 3))