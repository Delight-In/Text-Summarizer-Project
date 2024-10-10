import numpy as np

class LogisticRegressor:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Fit the model to the training data."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for epoch in range(self.n_epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """Predict class labels for the input data."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        class_labels = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(class_labels)

# Example usage
if __name__ == "__main__":
    # Generate some synthetic binary classification data
    np.random.seed(0)
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

    # Train the Logistic Regressor
    model = LogisticRegressor(learning_rate=0.1, n_epochs=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Output the first 5 predictions
    print("Predictions:", predictions[:5])
    print("Actual:", y[:5])
