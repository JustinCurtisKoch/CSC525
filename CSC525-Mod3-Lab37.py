# CSC525-Mod3-Lab3.7
# 3.7 LAB: k-nearest neighbors regression using scikit-learn
# 
# The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds.
# This data can be used to predict the price of a diamond based on its characteristics.
# Use sklearn's KNeighborsRegressor() function to predict the price of a diamond from the diamond's carat and table values.
# 
# Import needed packages for regression.
# Initialize and fit a k-nearest neighbor regression model using a Euclidean distance metric and k=12.
# Predict the price of a diamond with the user-input carat and table values.
# Find the distances and indices of the 12 nearest neighbors for the user-input instance
# Ex: If the input is:
# 
# 0.5
# 60
# the output should be:
# 
# Predicted price is [976.]
# Distances and indices of the 12 nearest neighbors are (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 
# array([[36978, 39309, 40216, 41316, 29444, 36080, 38823, 33766, 31706, 34796, 36889, 41402]]))

# Import needed packages for regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

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

# Initialize a k-nearest neighbors regression model using a Euclidean distance and k=12 
knnRegressor = KNeighborsRegressor(n_neighbors=12, metric='euclidean')

# Fit the kNN regression model to the input and output features
knnRegressor.fit(X, y)

# Create array with new carat and table values
Xnew = [[carat, table]]

# Predict the price of a diamond with the user-input carat and table values
prediction = knnRegressor.predict(Xnew)
print('Predicted price is', np.round(prediction, 2))

# Find the distances and indices of the 12 nearest neighbors for the new instance
neighbors = knnRegressor.kneighbors(Xnew)
print('Distances and indices of the 12 nearest neighbors are', neighbors)