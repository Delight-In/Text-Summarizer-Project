import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None  # Coefficients for each feature
        self.intercept = None  # Intercept

    def fit(self, X, y):
        # Add a column of ones to include the intercept in the model
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        # Calculate the coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        # Add the intercept to the predictions
        return X @ self.coefficients + self.intercept

    def plot(self, X, y):
        # For visualization, we can plot against the first feature
        plt.scatter(X[:, 0], y, color='blue', label='Data points')  # Use the first feature for plotting
        plt.xlabel('Feature 1')
        plt.ylabel('Target')
        plt.title('Multiple Linear Regression (First Feature vs Target)')
        plt.legend()
        plt.show()

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, header=None, names=column_names)

# Selecting features (x) and target (y)
X = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']].values  # Using multiple features
y = data['MEDV'].values

# Splitting the data into training, validation, and test sets
train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Create and fit the model
model = MultipleLinearRegression()
model.fit(X_train, y_train)

# Make predictions
val_predictions = model.predict(X_val)
test_prediction = model.predict(X_test)

# Calculate validation R-squared
ss_total = np.sum((y_val - np.mean(y_val)) ** 2)
ss_residual = np.sum((y_val - val_predictions) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print("Validation R-squared:", r_squared)

# Calculate test R-squared
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - test_prediction) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print("Test R-squared:", r_squared)

# Plot the results
model.plot(X_train, y_train)  # Show training fit using the first feature
