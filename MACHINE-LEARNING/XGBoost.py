import numpy as np
import pandas as pd

class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_reg=1, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.trees = []
    
    class TreeNode:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        # Initialize predictions to a constant value (mean)
        self.y_mean = np.mean(y)
        y_pred = np.full(y.shape, self.y_mean)
        
        for _ in range(self.n_estimators):
            residuals = self._gradient(y, y_pred)
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            # Update predictions
            y_pred += self.learning_rate * self._predict_tree(tree, X)

    def _gradient(self, y, y_pred):
        # Calculate the gradient (first derivative of loss)
        return y_pred - y  # For Mean Squared Error

    def _hessian(self, y, y_pred):
        # Calculate the hessian (second derivative of loss)
        return np.ones_like(y_pred)  # For Mean Squared Error

    def _build_tree(self, X, residuals):
        # Build a decision tree
        m, n = X.shape
        best_split = None
        best_gain = -np.inf

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if np.any(left_indices) and np.any(right_indices):
                    gain = self._calculate_gain(residuals, left_indices, right_indices)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, threshold)

        if best_split is None:
            return self.TreeNode(value=np.mean(residuals))

        left_indices = X[:, best_split[0]] <= best_split[1]
        right_indices = X[:, best_split[0]] > best_split[1]
        
        left_tree = self._build_tree(X[left_indices], residuals[left_indices])
        right_tree = self._build_tree(X[right_indices], residuals[right_indices])
        
        return self.TreeNode(best_split[0], best_split[1], left_tree, right_tree)

    def _calculate_gain(self, residuals, left_indices, right_indices):
        left_residuals = residuals[left_indices]
        right_residuals = residuals[right_indices]
        
        # Calculate gain based on the residuals
        left_loss = np.sum(left_residuals ** 2) / len(left_residuals) if len(left_residuals) > 0 else 0
        right_loss = np.sum(right_residuals ** 2) / len(right_residuals) if len(right_residuals) > 0 else 0
        
        total_loss = (len(left_residuals) * left_loss + len(right_residuals) * right_loss) / (len(left_indices) + len(right_indices))
        
        return total_loss - self.gamma - (self.lambda_reg / 2) * (len(left_residuals) + len(right_residuals))

    def _predict_tree(self, node, X):
        if node.value is not None:
            return np.full(X.shape[0], node.value)
        
        left_indices = X[:, node.feature_index] <= node.threshold
        right_indices = X[:, node.feature_index] > node.threshold
        
        predictions = np.zeros(X.shape[0])
        predictions[left_indices] = self._predict_tree(node.left, X[left_indices])
        predictions[right_indices] = self._predict_tree(node.right, X[right_indices])
        
        return predictions
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.y_mean)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return y_pred

# Example usage
if __name__ == "__main__":
    # Create some example data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = X.ravel() ** 2 + np.random.randn(100) * 5  # Quadratic relationship with noise
    
    # Fit the XGBoost model
    xgb = XGBoost(n_estimators=100, learning_rate=0.1, lambda_reg=1, gamma=0)
    xgb.fit(X, y)
    
    # Make predictions
    y_pred = xgb.predict(X)

    # Plot results
    import matplotlib.pyplot as plt

    plt.scatter(X, y, color='blue', label='True Values')
    plt.scatter(X, y_pred, color='red', label='Predictions', alpha=0.5)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('XGBoost Regression')
    plt.legend()
    plt.show()
