import numpy as np  # Import the NumPy library for numerical operations


class LogisticRegression:  # Define the LogisticRegression class
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
            # Approximate y with linear combination of weights and X, plus bias
            linear_model = np.dot(X, self.weights) + self.bias  # Compute linear model
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)  # Calculate predicted probabilities

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient for weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Gradient for bias
            # Update parameters
            self.weights -= self.lr * dw  # Update weights
            self.bias -= self.lr * db  # Update bias

    def predict(self, X):  # Method to make predictions
        linear_model = np.dot(X, self.weights) + self.bias  # Compute linear model
        y_predicted = self._sigmoid(linear_model)  # Calculate predicted probabilities
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Convert probabilities to class labels
        return np.array(y_predicted_cls)  # Return class labels as a NumPy array

    def _sigmoid(self, x):  # Sigmoid activation function
        return 1 / (1 + np.exp(-x))  # Compute and return sigmoid value


# Testing the implementation
if __name__ == "__main__":  # Ensure this runs only if the script is executed directly
    # Imports for dataset loading and model evaluation
    from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting
    from sklearn import datasets  # Import datasets from sklearn

    def accuracy(y_true, y_pred):  # Function to compute accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Calculate accuracy as the ratio of correct predictions
        return accuracy  # Return computed accuracy

    bc = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = bc.data, bc.target  # Get features and target labels

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=1234  # Specify test size and random state
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)  # Instantiate the regressor
    regressor.fit(X_train, y_train)  # Train the model on training data
    predictions = regressor.predict(X_test)  # Make predictions on test data

    print("LR classification accuracy:", accuracy(y_test, predictions))  # Print classification accuracy
