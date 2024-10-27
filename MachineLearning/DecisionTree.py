from collections import Counter  # Import Counter to count occurrences of labels

import numpy as np  # Import NumPy for numerical operations


def entropy(y):  # Function to calculate entropy
    hist = np.bincount(y)  # Count occurrences of each label
    ps = hist / len(y)  # Calculate probabilities of each label
    return -np.sum([p * np.log2(p) for p in ps if p > 0])  # Return entropy value


class Node:  # Define a Node class for the decision tree
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index for the split
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value if it's a leaf node

    def is_leaf_node(self):  # Method to check if the node is a leaf
        return self.value is not None  # Returns True if it's a leaf node


class DecisionTree:  # Define the DecisionTree class
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split  # Minimum samples needed to split
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_feats = n_feats  # Number of features to consider for splitting
        self.root = None  # Root node of the tree

    def fit(self, X, y):  # Method to fit the model to training data
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])  # Determine number of features
        self.root = self._grow_tree(X, y)  # Build the tree

    def predict(self, X):  # Method to make predictions
        return np.array([self._traverse_tree(x, self.root) for x in X])  # Traverse tree for each sample

    def _grow_tree(self, X, y, depth=0):  # Method to recursively grow the tree
        n_samples, n_features = X.shape  # Get number of samples and features
        n_labels = len(np.unique(y))  # Get number of unique labels

        # Stopping criteria
        if (
            depth >= self.max_depth  # If max depth is reached
            or n_labels == 1  # If only one label remains
            or n_samples < self.min_samples_split  # If not enough samples to split
        ):
            leaf_value = self._most_common_label(y)  # Get most common label for the leaf node
            return Node(value=leaf_value)  # Create and return a leaf node

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)  # Randomly select features for split

        # Greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)  # Find best feature and threshold

        # Grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)  # Split indices
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)  # Grow left subtree
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)  # Grow right subtree
        return Node(best_feat, best_thresh, left, right)  # Return new node

    def _best_criteria(self, X, y, feat_idxs):  # Method to find the best split criteria
        best_gain = -1  # Initialize best gain
        split_idx, split_thresh = None, None  # Initialize split index and threshold
        for feat_idx in feat_idxs:  # Iterate over selected feature indices
            X_column = X[:, feat_idx]  # Get current feature column
            thresholds = np.unique(X_column)  # Get unique values for possible thresholds
            for threshold in thresholds:  # Iterate over each threshold
                gain = self._information_gain(y, X_column, threshold)  # Calculate information gain

                if gain > best_gain:  # If the gain is better than the best gain found
                    best_gain = gain  # Update best gain
                    split_idx = feat_idx  # Update best feature index
                    split_thresh = threshold  # Update best threshold

        return split_idx, split_thresh  # Return the best split information

    def _information_gain(self, y, X_column, split_thresh):  # Method to compute information gain
        # Parent loss
        parent_entropy = entropy(y)  # Calculate entropy of parent node

        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)  # Get indices for split

        if len(left_idxs) == 0 or len(right_idxs) == 0:  # If split results in empty child nodes
            return 0  # Return zero gain

        # Compute the weighted average of the loss for the children
        n = len(y)  # Total number of samples
        n_l, n_r = len(left_idxs), len(right_idxs)  # Number of samples in left and right nodes
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])  # Entropy of children
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r  # Weighted average of child entropy

        # Information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy  # Calculate information gain
        return ig  # Return information gain

    def _split(self, X_column, split_thresh):  # Method to split data based on threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()  # Get indices for left split
        right_idxs = np.argwhere(X_column > split_thresh).flatten()  # Get indices for right split
        return left_idxs, right_idxs  # Return both sets of indices

    def _traverse_tree(self, x, node):  # Method to traverse the tree for a prediction
        if node.is_leaf_node():  # If the current node is a leaf
            return node.value  # Return the value of the leaf

        # Traverse left or right based on the feature and threshold
        if x[node.feature] <= node.threshold:  # If the feature value is less than or equal to the threshold
            return self._traverse_tree(x, node.left)  # Traverse left subtree
        return self._traverse_tree(x, node.right)  # Traverse right subtree

    def _most_common_label(self, y):  # Method to find the most common label
        counter = Counter(y)  # Count occurrences of each label
        most_common = counter.most_common(1)[0][0]  # Get the most common label
        return most_common  # Return the most common label


if __name__ == "__main__":  # Ensure this runs only if the script is executed directly
    # Imports
    from sklearn import datasets  # Import datasets module from sklearn
    from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting

    def accuracy(y_true, y_pred):  # Function to compute accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Calculate accuracy
        return accuracy  # Return computed accuracy

    data = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = data.data, data.target  # Get features and target labels

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=1234  # Specify test size and random state
    )

    clf = DecisionTree(max_depth=10)  # Instantiate the DecisionTree classifier with specified max depth
    clf.fit(X_train, y_train)  # Train the model on training data

    y_pred = clf.predict(X_test)  # Make predictions on test data
    acc = accuracy(y_test, y_pred)  # Calculate accuracy of predictions

    print("Accuracy:", acc)  # Print accuracy
