import numpy as np

class SGDRegressor:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Stochastic Gradient Descent
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Compute linear prediction
                linear_pred = np.dot(X[i], self.weights) + self.bias
                
                # Compute gradients
                dw = (1 / n_samples) * (linear_pred - y[i]) * X[i]
                db = (1 / n_samples) * (linear_pred - y[i])
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return linear_pred

# Example usage
if __name__ == "__main__":
    # Generate some random data for demonstration
    np.random.seed(0)
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    true_weights = np.array([2, 3])
    y = np.dot(X, true_weights) + np.random.normal(0, 0.1, size=100)  # Adding some noise

    # Train the SGD Regressor
    model = SGDRegressor(learning_rate=0.01, n_epochs=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Output the first 5 predictions
    print("Predictions:", predictions[:5])
    print("Actual:", y[:5])
