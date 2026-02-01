#The file SDSS contains 17 observational features and one class feature for 10000 deep sky objects observed by the Sloan Digital Sky Survey.
# Use sklearn's GaussianNB() function to perform Gaussian naive Bayes classification to classify each object by the object's redshift and u-g color.

# Import the necessary modules for Gaussian naive Bayes classification
# Create dataframe X with features redshift and u_g
# Create dataframe y with feature class
# Initialize a Gaussian naive Bayes model with the default parameters
# Fit the model
# Calculate the accuracy score
# Note: Use ravel() from numpy to flatten the second argument of GaussianNB.fit() into a 1-D array.

# Ex: If the feature u is used rather than u_g, the output is:
# Accuracy score is 0.987

# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from mlxtend.plotting import plot_decision_regions

# Load the dataset
skySurvey = pd.read_csv('skySurvey.csv')

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift', 'u_g']]

# Create dataframe y with feature class
y = skySurvey['class']

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()

# Fit the model
skySurveyNBModel.fit(X, y.ravel())

# Calculate the proportion of instances correctly classified
score = skySurveyNBModel.score(X, np.ravel(y))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)