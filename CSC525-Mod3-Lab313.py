# CSC525-Mod3-Lab3-13
# 3.13.1: LAB: Regression trees using scikit-learn

# The msleep_clean dataset contains information on sleep habits for 47 mammals. 
# Features include length of REM sleep, time spent awake, brain weight, and body weight.

# Create a dataframe X containing the features awake, brainwt, and bodywt, in that order.
# Create a dataframe y containing sleep_rem.
# Initialize and fit a regression tree with max_depth=3, ccp_alpha=0.02, and random_state=123 to the training data.
# Print the R-squared value for the testing set.
# Ex: If ccp_alpha=0 is used, the output should be:

# 0.36867242467112815
# |--- feature_0 <= 6.25
# |   |--- feature_1 <= 0.04
# |   |   |--- value: [3.90]
# |   |--- feature_1 >  0.04
# |   |   |--- value: [6.10]
# |--- feature_0 >  6.25
# |   |--- feature_0 <= 16.85
# |   |   |--- feature_1 <= 0.08
# |   |   |   |--- value: [2.15]
# |   |   |--- feature_1 >  0.08
# |   |   |   |--- value: [1.50]
# |   |--- feature_0 >  16.85
# |   |   |--- feature_0 <= 18.20
# |   |   |   |--- value: [1.13]
# |   |   |--- feature_0 >  18.20
# |   |   |   |--- value: [0.5

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

sleep = pd.read_csv('msleep_clean.csv')

# Create a dataframe X containing the features awake, brainwt, and bodywt, in that order
X = sleep[['awake', 'brainwt', 'bodywt']]

# Output feature: sleep_rem
y = sleep['sleep_rem']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize the model with max_depth=3, ccp_alpha=0.02, and random_state=123
DTRModel = DecisionTreeRegressor(max_depth=3, ccp_alpha=0.02, random_state=123)

# Fit the model
DTRModel.fit(X_train, y_train)

# Print the R-squared value for the testing set
print(DTRModel.score(X_test, y_test))

# Print text summary of tree
DTR_tree = export_text(DTRModel)
print(DTR_tree)