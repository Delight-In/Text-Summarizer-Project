import numpy as np  # Import the NumPy library for numerical operations


def r2_score(y_true, y_pred):  # Function to calculate R² score
    corr_matrix = np.corrcoef(y_true, y_pred)  # Compute correlation matrix
    corr = corr_matrix[0, 1]  # Extract correlation coefficient
    return corr ** 2  # Return R² score (squared correlation)


class LinearRegression:  # Define the LinearRegression class
    def __init__(self, learning_rate=0.001, n_iters=1000):  # Initialize with learning rate and number of iterations
        self.lr = learning_rate  # Set learning rate
        self.n_iters = n_iters  # Set number of iterations
        self.weights = None  # Initialize weights
        self.bias = None  # Initialize bias

    def fit(self, X, y):  # Method to train the model
        n_samples, n_features = X.shape  # Get number of samples and features

        # Initialize parameters
        self.weights = np.zeros(n_features)  # Set weights to zeros
        self.bias = 0  # Set bias to zero

        # Gradient descent loop
        for _ in range(self.n_iters):  # Iterate for the specified number of iterations
            y_predicted = np.dot(X, self.weights) + self.bias  # Compute predicted values
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient for weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Gradient for bias

            # Update parameters
            self.weights -= self.lr * dw  # Update weights
            self.bias -= self.lr * db  # Update bias

    def predict(self, X):  # Method to make predictions
        y_approximated = np.dot(X, self.weights) + self.bias  # Calculate predicted values
        return y_approximated  # Return predicted values


# Testing the implementation
if __name__ == "__main__":  # Ensure this runs only if the script is executed directly
    # Imports for plotting and dataset creation
    import matplotlib.pyplot as plt  # Import Matplotlib for plotting
    from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting
    from sklearn import datasets  # Import datasets from sklearn

    def mean_squared_error(y_true, y_pred):  # Function to compute Mean Squared Error
        return np.mean((y_true - y_pred) ** 2)  # Calculate and return MSE

    X, y = datasets.make_regression(  # Create a regression dataset
        n_samples=100, n_features=1, noise=20, random_state=4  # Specify parameters for dataset
    )

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=1234  # Specify test size and random state
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)  # Instantiate the regressor
    regressor.fit(X_train, y_train)  # Train the model on training data
    predictions = regressor.predict(X_test)  # Make predictions on test data

    mse = mean_squared_error(y_test, predictions)  # Calculate Mean Squared Error
    print("MSE:", mse)  # Print MSE

    accu = r2_score(y_test, predictions)  # Calculate R² score
    print("Accuracy:", accu)  # Print R² score

    y_pred_line = regressor.predict(X)  # Get predictions for the entire dataset
    cmap = plt.get_cmap("viridis")  # Get a colormap for plotting
    fig = plt.figure(figsize=(8, 6))  # Create a figure for plotting
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)  # Scatter plot for training data
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)  # Scatter plot for testing data
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")  # Plot prediction line
    plt.show()  # Display the plot
