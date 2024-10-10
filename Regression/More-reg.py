import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Fit method for ordinary least squares
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b @ self.coefficients

class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n, m = X_b.shape
        identity_matrix = np.eye(m)
        identity_matrix[0, 0] = 0  # Do not regularize the bias term
        self.coefficients = np.linalg.inv(X_b.T @ X_b + self.alpha * identity_matrix) @ X_b.T @ y

class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0, iterations=1000, learning_rate=0.01):
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coefficients = None

    def fit(self, X, y):
        n, m = X.shape
        self.coefficients = np.zeros(m)

        for _ in range(self.iterations):
            predictions = self.predict(X)
            errors = predictions - y
            gradient = (2/n) * (X.T @ errors) + self.alpha * np.sign(self.coefficients)
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        return X @ self.coefficients

class ElasticNet(LinearRegression):
    def __init__(self, alpha=1.0, l1_ratio=0.5, iterations=1000, learning_rate=0.01):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coefficients = None

    def fit(self, X, y):
        n, m = X.shape
        self.coefficients = np.zeros(m)

        for _ in range(self.iterations):
            predictions = self.predict(X)
            errors = predictions - y
            gradient = (2/n) * (X.T @ errors) + self.alpha * (self.l1_ratio * np.sign(self.coefficients) + (1 - self.l1_ratio) * self.coefficients)
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        return X @ self.coefficients

# Example usage
if __name__ == "__main__":
    # Create some synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    true_coefficients = np.array([1.5, -2.0, 3.0])
    y = X @ true_coefficients + np.random.randn(100) * 0.5  # Adding noise

    # Fit models
    ridge_model = RidgeRegression(alpha=1.0)
    ridge_model.fit(X, y)
    print("Ridge Coefficients:", ridge_model.coefficients)

    lasso_model = LassoRegression(alpha=1.0)
    lasso_model.fit(X, y)
    print("Lasso Coefficients:", lasso_model.coefficients)

    elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elastic_model.fit(X, y)
    print("Elastic Net Coefficients:", elastic_model.coefficients)
