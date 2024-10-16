import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to +1 and -1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                if y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1:
                    # Correctly classified
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1, 2], [2, 3], [3, 1], [5, 1], [6, 4]])
    y = np.array([1, 1, 1, -1, -1])

    # Create SVM instance and fit the model
    svm = SVM()
    svm.fit(X, y)

    # Predict on new data
    predictions = svm.predict(X)
    print("Predictions:", predictions)
