# CSC525-Mod2-Lab2.6
# The dataset SDSS contains 17 observational features and one class feature for 10000 deep sky objects observed by the Sloan Digital Sky Survey.
#  Use sklearn's KNeighborsClassifier() function to perform kNN classification to classify each object by the object's redshift and u-g color.

# Import the necessary modules for kNN classification
# Create dataframe X with features redshift and u_g
# Create dataframe y with feature class
# Initialize a kNN model with k=3
# Fit the model using the training data
# Find the predicted classes for the test data
# Calculate the accuracy score using the test data
# Ex: If the feature u is used rather than u_g, the output is:

# Import needed packages for classification
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv')

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift', 'u_g']]

# Create dataframe y with feature class
y = skySurvey['class']

np.random.seed(42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize model with k=3
skySurveyKnn = KNeighborsClassifier(n_neighbors=3)

# Fit model using X_train and y_train
skySurveyKnn.fit(X_train, y_train)

# Find the predicted classes for X_test
y_pred = skySurveyKnn.predict(X_test)

# Calculate accuracy score
score = skySurveyKnn.score(X_test, y_test)

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)