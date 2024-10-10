import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.m = 0  # slope
        self.b = 0  # intercept

    def fit(self, x, y):
        # Convert x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Calculate the means of x and y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate the slope (m) and intercept (b)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean

    def predict(self, x):
        return self.m * np.array(x) + self.b

    def plot(self, x, y):
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x, self.predict(x), color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, header=None, names=column_names)
data.head()

# Selecting the features (x) and the target (y)
x = data['RM']  # Average number of rooms
y = data['LSTAT']  # percentage of the lower status of the population

# Splitting the data
train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))

x_train, y_train = x[:train_size], y[:train_size]
x_val, y_val = x[train_size:(train_size + val_size)], y[train_size:(train_size + val_size)]
x_test, y_test = x[(train_size + val_size):], y[(train_size + val_size):]

# Create and fit the model
model = SimpleLinearRegression()
model.fit(x_train, y_train)

# Make predictions
val_predictions = model.predict(x_val)
test_prediction = model.predict(x_test)

# Print predictions for validation set
# print("Validation predictions:", val_predictions)
# print("Validation actual:", y_val)

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
model.plot(x_train, y_train)  # Show training fit
