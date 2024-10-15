import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# For simplicity, we'll just use one feature (e.g., MedInc - median income)
X = X[['MedInc']]
y = y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def polynomial_features(X, degree):
    """Generate polynomial features for the input data."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))  # Start with a column of ones for bias
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, (X ** d).values))  # Add each polynomial feature
    return X_poly

def fit_polynomial_regression(X, y, alpha=1e-10):
    """Fit a polynomial regression model using the normal equation with regularization."""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    regularization_matrix = alpha * np.eye(X_b.shape[1])
    theta_best = np.linalg.inv(X_b.T.dot(X_b) + regularization_matrix).dot(X_b.T).dot(y)
    return theta_best

def predict(X, theta):
    """Make predictions using the polynomial model."""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return X_b.dot(theta)

# Set the degree for the polynomial regression
degree = 2

# Generate polynomial features
X_poly_train = polynomial_features(X_train, degree)
X_poly_test = polynomial_features(X_test, degree)

# Fit the model
theta_best = fit_polynomial_regression(X_poly_train, y_train)

# Make predictions
y_pred = predict(X_poly_test, theta_best)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('House Price')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.legend()
plt.show()
