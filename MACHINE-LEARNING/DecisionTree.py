from collections import Counter
import numpy as np

def entropy(y):
    # Calculate the entropy of the label distribution
    hist = np.bincount(y)  # Count occurrences of each label
    ps = hist / len(y)  # Calculate probabilities
    return -np.sum([p * np.log2(p) for p in ps if p > 0])  # Entropy formula


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        # Initialize a node in the decision tree
        self.feature = feature  # Feature index for the split
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted value if it's a leaf node

    def is_leaf_node(self):
        # Check if the node is a leaf node
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        # Initialize the decision tree parameters
        self.min_samples_split = min_samples_split  # Minimum samples required to split
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_feats = n_feats  # Number of features to consider for each split
        self.root = None  # Root of the tree

    def fit(self, X, y):
        # Fit the decision tree model to the training data
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)  # Grow the tree starting from the root

    def predict(self, X):
        # Predict labels for the input data
        return np.array([self._traverse_tree(x, self.root) for x in X])  # Traverse the tree for each input

    def _grow_tree(self, X, y, depth=0):
        # Recursively grow the decision tree
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))  # Number of unique labels

        # Stopping Criteria
        if (
            depth >= self.max_depth  # Maximum depth reached
            or n_labels == 1  # Only one class present
            or n_samples < self.min_samples_split  # Not enough samples to split
        ):
            leaf_value = self._most_common_label(y)  # Assign the most common label to the leaf
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)  # Randomly select features

        # Greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)  # Split the data
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)  # Grow left subtree
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)  # Grow right subtree
        return Node(best_feat, best_thresh, left, right)  # Return the node

    def _best_criteria(self, X, y, feat_idxs):
        # Find the best feature and threshold for splitting
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]  # Current feature column
            thresholds = np.unique(X_column)  # Unique threshold values
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)  # Calculate information gain

                if gain > best_gain:  # Update best gain if current gain is better
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh  # Return the best feature and threshold

    def _information_gain(self, y, X_column, split_thresh):
        # Calculate the information gain from a split
        parent_entropy = entropy(y)  # Calculate parent entropy

        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0  # Return 0 gain if the split is invalid

        # Compute the weighted average of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])  # Calculate children's entropy
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r  # Weighted average of child entropy

        # Information Gain is difference in loss before vs after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # Split the data into left and right based on the threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()  # Indices for left split
        right_idxs = np.argwhere(X_column > split_thresh).flatten()  # Indices for right split
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        # Traverse the tree to find the prediction for a single sample
        if node.is_leaf_node():
            return node.value  # Return the leaf value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # Traverse left
        return self._traverse_tree(x, node.right)  # Traverse right

    def _most_common_label(self, y):
        # Find the most common label in the array
        counter = Counter(y)  # Count occurrences of each label
        most_common = counter.most_common(1)[0][0]  # Get the most common label
        return most_common


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        # Calculate accuracy as the ratio of correct predictions
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = data.data, data.target  # Features and labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234  # Split the dataset into training and testing sets
    )

    clf = DecisionTree(max_depth=10)  # Create a decision tree classifier
    clf.fit(X_train, y_train)  # Fit the model to the training data

    y_pred = clf.predict(X_test)  # Predict labels for the test set
    acc = accuracy(y_test, y_pred)  # Calculate accuracy

    print("Accuracy:", acc)  # Print the accuracy
