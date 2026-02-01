# CSC525-Mod4-Labs4.7
# Multilayer perceptrons using scikit-learn
# 
# The diamonds.csv dataset contains the price, cut, color, and other characteristics of a sample of diamonds.
# Create a dataframe X containing all the features except cut, color, clarity, and price.
# Create a dataframe y containing the feature price.
# Initialize and fit a multilayer perceptron regressor with two hidden layers with 50 nodes each, the 'identity' activation function, max_iter=500, and random_state=123.
# Print the price predictions for the first five rows of the training data.
# Print the R-squared scores for the training data and the testing data, rounded to the fourth decimal place.
# Ex: If random_state=12 is used instead of random_state=123, the output is:
# 
# Price predictions: [-1358.27730125  1201.00841019  4221.32812498  -446.71476115
#   3499.11770257]
# Actual prices: 
#         price
# 8711     586
# 38897   1052
# 792     2862
# 31730    772
# 51461   2376
# Score for the training data:  0.7985
# Score for the testing data:  0.8029

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

diamonds = pd.read_csv('diamonds.csv')

diamonds = diamonds.sample(n=800, random_state=10)

# Create a dataframe X containing all the features except cut, color, clarity, and price
X = diamonds.drop(columns=['cut', 'color', 'clarity', 'price'])
# Create a dataframe y containing the feature price
y = diamonds['price']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=123)

# Initialize a multilayer perceptron regressor with two hidden layers with 50 nodes each, 
# the 'identity' activation function, max_iter=500, and random_state=123
mlpModel = MLPRegressor(hidden_layer_sizes=(50, 50), activation='identity', max_iter=500, random_state=123)

# Fit the model
mlpModel.fit(Xtrain, ytrain)

# Print the price predictions and actual prices for the first five rows of the dataframe
print("Price predictions:", mlpModel.predict(Xtrain[0:5]))
print("Actual prices: \n", ytrain[0:5])

# Print the R-squared score for the training data
print("Score for the training data: ", round(mlpModel.score(Xtrain, ytrain), 4))
# Print the R-squared score for the testing data
print("Score for the testing data: ", round(mlpModel.score(Xtest, ytest), 4))