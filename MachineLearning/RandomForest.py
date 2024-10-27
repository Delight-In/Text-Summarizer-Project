from collections import Counter
import numpy as np
from DecisionTree import DecisionTree  # Importing DecisionTree class

def bootstrap_sample(X, y):
    # Create a bootstrap sample from the dataset
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)  # Randomly sample with replacement
    return X[idxs], y[idxs]  # Return the sampled features and labels

def most_common_label(y):
    # Find the most common label in the array y
    counter = Counter(y)  # Count occurrences of each label
    most_common = counter.most_common(1)[0][0]  # Get the most common label
    return most_common

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees  # Number of trees in the forest
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.max_depth = max_depth  # Maximum depth of each tree
        self.n_feats = n_feats  # Number of features to consider for best split
        self.trees = []  # List to store the individual trees

    def fit(self, X, y):
        # Train the Random Forest model
        self.trees = []  # Reset the trees list for new training
        for _ in range(self.n_trees):
            tree = DecisionTree(  # Create a new Decision Tree
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X, y)  # Get a bootstrap sample
            tree.fit(X_samp, y_samp)  # Fit the tree to the sampled data
            self.trees.append(tree)  # Add the trained tree to the forest

    def predict(self, X):
        # Predict labels for the input data using the trained trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # Get predictions from each tree
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # Swap axes to organize predictions
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]  # Get the most common prediction for each sample
        return np.array(y_pred)  # Return predictions as a numpy array


# Testing the Random Forest implementation
if __name__ == "__main__":
    # Imports for dataset and accuracy function
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        # Calculate accuracy of predictions
        return np.sum(y_true == y_pred) / len(y_true)

    # Load the breast cancer dataset
    data = datasets.load_breast_cancer()
    X = data.data  # Features
    y = data.target  # Labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Initialize and train the Random Forest classifier
    clf = RandomForest(n_trees=3, max_depth=10)
    clf.fit(X_train, y_train)  # Fit the model to training data

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)  # Calculate accuracy

    # Print the accuracy of the model
    print("Accuracy:", acc)
