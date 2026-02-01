# CSC525 Principles of Machine Learning Module 2 Critical Thinking

# Iris Dataset Classification
# K-Nearest Neighbors (KNN) Classifier Implementation
 
# KNN cluster classification works by finding the distances between a query and all examples in its data. 
# The specified number of examples (K) closest to the query are selected. 
# The classifier then votes for the most frequent label found.
 
# There are several advantages of KNN classification, one of them being simple implementation. 
# The search space is robust as classes do not need to be linearly separable. 
# It can also be updated online easily as new instances with known classes are presented.
 
# A KNN model can be implemented using the following steps:
 
# Load the data;
# Initialize the value of k;
# For getting the predicted class, iterate from 1 to the total number of training data points;
# Calculate the distance between the test data and each row of training data;
# Sort the calculated distances in ascending order based on distance values;
# Get top k rows from the sorted array;
# Get the most frequent class of these rows; and
# Return the predicted class.

# ------------------------------------------------------------
# K-Nearest Neighbors (KNN) Classifier Implementation
# Iris Dataset Classification
# ------------------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Step 1: Load the dataset
# ------------------------------------------------------------
# The dataset contains four numerical features:
# Sepal length - Sepal width - Petal length - Petal width
# The class label represents the iris species.

iris_data = pd.read_csv(r'C:\Users\justi\Downloads\iris.csv')
print(iris_data.head())
print(iris_data.describe())

# Separate features and class labels
X = iris_data.iloc[:, :-1].values   # Feature matrix
y = iris_data.iloc[:, -1].values    # Target labels


# ------------------------------------------------------------
# Step 2: Initialize the value of k
# ------------------------------------------------------------
# The value of k determines how many nearest neighbors to consider

k = 5


# ------------------------------------------------------------
# Step 3: Define the distance calculation function
# ------------------------------------------------------------
# Euclidean distance is used to measure similarity between
# the test data point and each training data point.

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# ------------------------------------------------------------
# Steps 4–8: KNN Classification Algorithm
# ------------------------------------------------------------
# Step 4: Iterate through all training data points
# Step 5: Compute distance between test point and training data
# Step 6: Sort distances in ascending order
# Step 7: Select the k nearest neighbors
# Step 8: Determine the most frequent class among neighbors

def knn_classifier(X_train, y_train, test_point, k):
    """
    Predicts the class of a test data point using the
    K-Nearest Neighbors algorithm.
    """

    distances = []

    # Calculate distance from test point to each training point
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))

    # Sort distances based on distance value
    distances.sort(key=lambda x: x[0])

    # Select the top k nearest neighbors
    nearest_neighbors = distances[:k]

    # Extract class labels of nearest neighbors
    class_labels = [label for _, label in nearest_neighbors]

    # Determine the most frequent class label
    predicted_class = Counter(class_labels).most_common(1)[0][0]

    return predicted_class


# ------------------------------------------------------------
# Step 9: Test the classifier with a sample data point
# ------------------------------------------------------------
# Example input in centimeters: 
# sepal length, sepal width, petal length, petal width

test_sample = np.array([5.1, 3.5, 1.4, 0.2])

predicted_species = knn_classifier(X, y, test_sample, k)

# Display the predicted iris species
print("Predicted Iris Species:", predicted_species)


# ------------------------------------------------------------
# Step 10: Evaluate classification accuracy
# ------------------------------------------------------------
# The dataset is divided into training and testing subsets.
# Each test data point is classified using the KNN algorithm,
# and accuracy is computed as the proportion of correct predictions.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Generate predictions for the test set
correct_predictions = 0

for i in range(len(X_test)):
    prediction = knn_classifier(X_train, y_train, X_test[i], k)
    if prediction == y_test[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(X_test)

# Display classification accuracy
print("Classification Accuracy:", accuracy)

