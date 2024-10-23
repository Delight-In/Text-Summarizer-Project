import numpy as np

class BaseRegression:
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        # Assign the learning rate and number of iterations for training
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Initialize weights and bias
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get the number of samples and features
        self.weights, self.bias = np.zeros(n_features), 0  # Initialize weights and bias to zero

        # Gradient Descent loop to minimize loss and optimize weights and bias
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)  # Get predicted values

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient for weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Gradient for bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw  # Adjust weights
            self.bias -= self.learning_rate * db  # Adjust bias

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)  # Get predictions using current weights and bias

    def _predict(self, X, w, b):
        raise NotImplementedError  # Method to be implemented in subclasses

    def _approximation(self, X, w, b):
        raise NotImplementedError  # Method to be implemented in subclasses


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b  # Linear approximation (dot product + bias)

    def _predict(self, X, w, b):
        return np.dot(X, w) + b  # Predictions are also linear


class LogisticRegression(BaseRegression):
    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b  # Calculate linear combination
        return self._sigmoid(linear_model)  # Apply sigmoid to get probabilities

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b  # Calculate linear combination
        y_predicted = self._sigmoid(linear_model)  # Get probabilities
        # Convert probabilities to binary class labels
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)  # Return as numpy array

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)  # Sigmoid function


# Testing
if __name__ == "__main__":
    # Imports for datasets and model selection
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Utility function to calculate R-squared score
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)  # Correlation matrix
        corr = corr_matrix[0, 1]  # Extract correlation coefficient
        return corr ** 2  # R-squared is the square of the correlation

    # Utility function to calculate mean squared error
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  # Mean of squared differences

    # Utility function to calculate accuracy
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Ratio of correct predictions
        return accuracy

    # Linear Regression Example
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4  # Generate synthetic regression data
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Initialize and train the Linear Regression model
    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)  # Fit the model to the training data
    predictions = regressor.predict(X_test)  # Make predictions on the test data

    # Calculate and print R-squared accuracy
    accu = r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)

    # Logistic Regression Example
    bc = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = bc.data, bc.target  # Extract features and target labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Initialize and train the Logistic Regression model
    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)  # Fit the model to the training data
    predictions = regressor.predict(X_test)  # Make predictions on the test data

    # Calculate and print classification accuracy
    print("Logistic reg classification accuracy:", accuracy(y_test, predictions))
