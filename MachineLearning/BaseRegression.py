import numpy as np  # Import the NumPy library for numerical operations


class BaseRegression:  # Define the base regression class
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):  # Initialize with learning rate and iterations
        # Assign the variables
        self.learning_rate = learning_rate  # Set learning rate
        self.n_iters = n_iters  # Set number of iterations

        # Weights and bias
        self.weights, self.bias = None, None  # Initialize weights and bias

    def fit(self, X, y):  # Method to train the model
        n_samples, n_features = X.shape  # Get number of samples and features

        self.weights, self.bias = np.zeros(n_features), 0  # Initialize weights and bias

        # Minimizing loss and finding the correct weights and biases using Gradient Descent
        for _ in range(self.n_iters):  # Iterate for the specified number of iterations
            y_predicted = self._approximation(X, self.weights, self.bias)  # Get predicted values

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient for weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Gradient for bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw  # Update weights
            self.bias -= self.learning_rate * db  # Update bias

    def predict(self, X):  # Method to make predictions
        return self._predict(X, self.weights, self.bias)  # Call the _predict method

    def _predict(self, X, w, b):  # Method to be implemented by subclasses
        raise NotImplementedError  # Raise error if not implemented

    def _approximation(self, X, w, b):  # Method to be implemented by subclasses
        raise NotImplementedError  # Raise error if not implemented


class LinearRegression(BaseRegression):  # Linear regression class inheriting from BaseRegression
    def _approximation(self, X, w, b):  # Method to compute predictions
        return np.dot(X, w) + b  # Return linear model predictions

    def _predict(self, X, w, b):  # Method to make predictions
        return np.dot(X, w) + b  # Return linear model predictions


class LogisticRegression(BaseRegression):  # Logistic regression class inheriting from BaseRegression
    def _approximation(self, X, w, b):  # Method to compute predictions
        linear_model = np.dot(X, w) + b  # Compute linear model
        return self._sigmoid(linear_model)  # Return sigmoid of linear model

    def _predict(self, X, w, b):  # Method to make predictions
        linear_model = np.dot(X, w) + b  # Compute linear model
        y_predicted = self._sigmoid(linear_model)  # Get predicted probabilities
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Convert probabilities to class labels
        return np.array(y_predicted_cls)  # Return class labels as a NumPy array

    def _sigmoid(self, x):  # Sigmoid activation function
        return 1 / (np.exp(-x) + 1)  # Compute and return sigmoid value


# Testing the implementation
if __name__ == "__main__":  # Ensure this runs only if the script is executed directly
    # Imports for dataset loading and model evaluation
    from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting
    from sklearn import datasets  # Import datasets from sklearn

    # Utils
    def r2_score(y_true, y_pred):  # Function to compute R² score
        corr_matrix = np.corrcoef(y_true, y_pred)  # Compute correlation matrix
        corr = corr_matrix[0, 1]  # Extract correlation coefficient
        return corr ** 2  # Return R² score (squared correlation)

    def mean_squared_error(y_true, y_pred):  # Function to compute Mean Squared Error
        return np.mean((y_true - y_pred) ** 2)  # Calculate and return MSE

    def accuracy(y_true, y_pred):  # Function to compute accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Calculate accuracy as the ratio of correct predictions
        return accuracy  # Return computed accuracy

    # Linear Regression
    X, y = datasets.make_regression(  # Create a regression dataset
        n_samples=100, n_features=1, noise=20, random_state=4  # Specify parameters for dataset
    )

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=1234  # Specify test size and random state
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)  # Instantiate the LinearRegression class
    regressor.fit(X_train, y_train)  # Train the model on training data
    predictions = regressor.predict(X_test)  # Make predictions on test data

    accu = r2_score(y_test, predictions)  # Calculate R² score for predictions
    print("Linear reg Accuracy:", accu)  # Print R² score

    # Logistic Regression
    bc = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = bc.data, bc.target  # Get features and target labels

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=1234  # Specify test size and random state
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)  # Instantiate the LogisticRegression class
    regressor.fit(X_train, y_train)  # Train the model on training data
    predictions = regressor.predict(X_test)  # Make predictions on test data

    print("Logistic reg classification accuracy:", accuracy(y_test, predictions))  # Print classification accuracy
