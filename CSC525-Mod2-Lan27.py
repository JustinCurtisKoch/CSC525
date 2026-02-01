# CSC525-Mod2-Lab2.7
# The nbaallelo_log file contains data on 126314 NBA games from 1947 to 2015. 
# The dataset includes the features pts, elo_i, win_equiv, and game_result. 
# Using the csv file nbaallelo_log.csv and scikit-learn's LogisticRegression() function, construct a logistic regression model to classify whether a team will win or lose a game based on the team's elo_i score.

# Create a binary feature win for game_result with 0 for L and 1 for W
# Use the LogisticRegression() function with penalty='l2' to construct a logistic regression model with win as the target and elo_i as the predictor
# Print the weights and intercept of the fitted model
# Find the proportion of instances correctly classified
# Note: Use ravel() from numpy to flatten the second argument of LogisticRegression.fit()into a 1-D array.

# Import the necessary libraries
# Your code here

# Load nbaallelo_log.csv into a dataframe
NBA = pd.read_csv('nbaallelo_log.csv')

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = NBA['game_result'].map({'L': 0, 'W': 1})

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Initialize and fit the logistic model using the LogisticRegression() function
# Your code here

# Print the weights for the fitted model
print('w1:', # Your code here)

# Print the intercept of the fitted model
print('w0:', # Your code here)

# Find the proportion of instances correctly classified
score = # Your code here
print(round(score, 3))