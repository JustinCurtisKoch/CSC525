# ------------------------------------------------------------
# Linear vs. Polynomial Regression Comparison (Template)
# ------------------------------------------------------------

# Import required libraries for data handling, visualization,
# and machine learning model development

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# ------------------------------------------------------------
# Step 1: Load the dataset
# ------------------------------------------------------------
# The dataset should contain at least one independent variable
# and one dependent variable suitable for regression analysis.

data = pd.read_csv(r'C:\Users\justi\Downloads\Salary_Data.csv')


# ------------------------------------------------------------
# Step 2: Inspect the dataset structure
# ------------------------------------------------------------
# Display the first few rows of the dataset to verify that
# the data has been loaded correctly.

#print(data.head())


# ------------------------------------------------------------
# Step 3: Examine descriptive statistics
# ------------------------------------------------------------
# Summary statistics provide insight into the distribution,
# scale, and range of the variables used in regression.

#print(data.describe())

# Relationship between Salary and Level
plt.scatter(data['YearsExperience'], data['Salary'], color = 'lightcoral')
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

# ------------------------------------------------------------
# Step 4: Split the data into independent and dependent variables
# ------------------------------------------------------------
# The independent variable (X) represents years of experience.
# The dependent variable (y) represents salary.

X = data[['YearsExperience']].values
y = data['Salary'].values


# ------------------------------------------------------------
# Step 5: Split the dataset into training and testing sets
# ------------------------------------------------------------
# The data is divided to allow model training and evaluation
# on unseen observations.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# Step 6: Build and train the Linear Regression model
# ------------------------------------------------------------
# Linear regression models the relationship between experience
# and salary as a straight-line relationship.

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Generate predictions using the linear regression model
y_pred_linear = linear_model.predict(X_test)


# ------------------------------------------------------------
# Step 7: Build and train the Polynomial Regression model
# ------------------------------------------------------------
# Polynomial regression extends linear regression by introducing
# nonlinear terms, allowing the model to capture curvature
# in the relationship between experience and salary.

poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)

# Generate predictions using the polynomial regression model
y_pred_polynomial = polynomial_model.predict(X_test_poly)

# ------------------------------------------------------------
# Step 8: Evaluate model performance
# ------------------------------------------------------------
# Model performance is assessed using the coefficient of
# determination (R² score), which measures how well the
# model explains the variance in the dependent variable.

# Calculate R² score for Linear Regression
linear_r2 = linear_model.score(X_test, y_test)

# Calculate R² score for Polynomial Regression
polynomial_r2 = polynomial_model.score(X_test_poly, y_test)

# Display results
print("Linear Regression R² Score:", linear_r2)
print("Polynomial Regression R² Score:", polynomial_r2)

# ------------------------------------------------------------
# Step 9: Visualize regression results
# ------------------------------------------------------------
# This visualization compares the observed data points with
# the fitted linear and polynomial regression models.
# All results are displayed on a single chart for clarity.

# Create a smooth range of experience values for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Generate predictions for plotting
y_plot_linear = linear_model.predict(X_plot)

X_plot_poly = poly_features.transform(X_plot)
y_plot_polynomial = polynomial_model.predict(X_plot_poly)

# Create the plot
plt.figure(figsize=(8, 6))

# Plot actual data points
plt.scatter(X, y, label="Actual Data")

# Plot linear regression line
plt.plot(X_plot, y_plot_linear, label="Linear Regression")

# Plot polynomial regression curve
plt.plot(X_plot, y_plot_polynomial, label="Polynomial Regression (Degree 3)")

# Add labels and title
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear vs. Polynomial Regression Comparison")

# Display legend
plt.legend()

# Show the plot
plt.show()
