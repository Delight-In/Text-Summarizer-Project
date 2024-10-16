import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # If all samples belong to the same class, return the class
        if len(unique_classes) == 1:
            return unique_classes[0]

        # If we reached max depth or have no features left, return the majority class
        if self.max_depth is not None and depth >= self.max_depth:
            return unique_classes[np.argmax(class_counts)]

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return unique_classes[np.argmax(class_counts)]

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        
        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log(probabilities + 1e-9))  # Small value to prevent log(0)

    def _predict(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_tree, right_tree = tree
        if sample[feature] < threshold:
            return self._predict(sample, left_tree)
        else:
            return self._predict(sample, right_tree)

# Example usage
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset (e.g., Iris dataset)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Decision Tree model
    dt = DecisionTree(max_depth=3)
    dt.fit(X_train, y_train)

    # Predict
    predictions = dt.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    print(f'Accuracy: {acc * 100:.2f}%')
