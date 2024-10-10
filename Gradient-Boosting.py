import numpy as np

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
    
    class TreeNode:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def fit(self, X, y):
        # Initialize the predictions with the mean of y
        y_pred = np.full(y.shape, np.mean(y))
        
        for _ in range(self.n_estimators):
            # Compute the pseudo-residuals
            residuals = y - y_pred
            
            # Train a new tree on the residuals
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            y_pred += self.learning_rate * self._predict_tree(tree, X)
    
    def _build_tree(self, X, residuals):
        # Here we implement a simple decision tree for regression
        # Splitting on the feature that minimizes the MSE
        m, n = X.shape
        best_split = None
        best_mse = float('inf')
        
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                
                if np.any(left_indices) and np.any(right_indices):
                    left_residuals = residuals[left_indices]
                    right_residuals = residuals[right_indices]
                    
                    # Calculate the mean squared error for this split
                    left_mse = np.mean(left_residuals**2)
                    right_mse = np.mean(right_residuals**2)
                    total_mse = (left_indices.sum() * left_mse + right_indices.sum() * right_mse) / m
                    
                    if total_mse < best_mse:
                        best_mse = total_mse
                        best_split = (feature_index, threshold)

        if best_split is None:
            return self.TreeNode(np.mean(residuals))
        
        left_indices = X[:, best_split[0]] <= best_split[1]
        right_indices = X[:, best_split[0]] > best_split[1]
        
        left_tree = self._build_tree(X[left_indices], residuals[left_indices])
        right_tree = self._build_tree(X[right_indices], residuals[right_indices])
        
        node = self.TreeNode(best_split)
        node.left = left_tree
        node.right = right_tree
        
        return node

    def _predict_tree(self, node, X):
        if node.left is None and node.right is None:
            return np.full(X.shape[0], node.value)
        
        feature_index, threshold = node.value
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        
        predictions = np.zeros(X.shape[0])
        predictions[left_indices] = self._predict_tree(node.left, X[left_indices])
        predictions[right_indices] = self._predict_tree(node.right, X[right_indices])
        
        return predictions
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], np.mean(y))  # Initialize predictions
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return y_pred

# Example usage
if __name__ == "__main__":
    # Create some example data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = X.ravel() ** 2 + np.random.randn(100) * 5  # Quadratic relationship with noise
    
    # Fit the Gradient Boosting model
    gb = GradientBoosting(n_estimators=100, learning_rate=0.1)
    gb.fit(X, y)
    
    # Make predictions
    y_pred = gb.predict(X)

    # Plot results
    import matplotlib.pyplot as plt

    plt.scatter(X, y, color='blue', label='True Values')
    plt.scatter(X, y_pred, color='red', label='Predictions', alpha=0.5)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Gradient Boosting Regression')
    plt.legend()
    plt.show()
