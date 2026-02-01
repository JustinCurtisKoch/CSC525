# CSC525-Mod5-Labs58
# 5.8.1: Classification metrics using scikit-learn
# 
# The nbaallelo_log.csv file contains data on 126314 NBA games from 1947 to 2015.
# The dataset includes the features pts, elo_i, win_equiv, and game_result.
# Using a sample of the csv file nbaallelo_log.csv and scikit-learn's LogisticRegression() function, construct a logistic regression model to classify whether a team will win or lose a game based on the team's elo_i score.
# Using scikit-learn's metrics module, calculate the various classification metrics for the model.
# 
# Build a logistic model with default parameters, fit to the input and output features X and y.
# Use the model to predict the classification of instances in X.
# Calculate the confusion matrix for the model.
# Calculate the accuracy for the model.
# Calculate the precision for the model.
# Calculate the recall for the model.
# Calculate kappa for the model.
    
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe. 
NBA = pd.read_csv("nbaallelo_log.csv").sample(n=500, random_state=rand)

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = NBA['game_result'].replace(to_replace = ['L','W'], value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Build logistic model with default parameters, fit to X and y
logModel = LogisticRegression()
logModel.fit(X, y.values.ravel())

# Use the model to predict the classification of instances in X
logPredY = logModel.predict(X)

# Calculate the confusion matrix for the model
confMatrix = metrics.confusion_matrix(y, logPredY)
print("Confusion matrix:\n", confMatrix)

# Calculate the accuracy for the model
accuracy = metrics.accuracy_score(y, logPredY)
print("Accuracy:", round(accuracy,3))

# Calculate the precision for the model
precision = precision(y, logPredY)
print("Precision:", round(precision,3))

# Calculate the recall for the model
recall = metrics.recall_score(y, logPredY)
print("Recall:", round(recall, 3))

# Calculate kappa for the model
kappa = metrics.cohen_kappa_score(y, logPredY)
print("Kappa:", round(kappa, 3))