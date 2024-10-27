from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors to consider

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X  # Training features
        self.y_train = y  # Training labels

    def predict(self, X):
        """Predict the labels for the given input data."""
        # Predict label for each sample in X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """Predict the label for a single sample."""
        # Compute distances between the sample x and all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort the distances and get the indices of the k nearest neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # Return the most common class label among the neighbors
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]  # Return the most common label


if __name__ == "__main__":
    # Imports for visualization and dataset
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Define a color map for visualization
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        """Calculate the accuracy of predictions."""
        return np.sum(y_true == y_pred) / len(y_true)

    # Load the iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3  # Set the number of neighbors
    clf = KNN(k=k)  # Initialize the KNN classifier
    clf.fit(X_train, y_train)  # Fit the model to the training data
    predictions = clf.predict(X_test)  # Make predictions on the test data
    print("KNN classification accuracy:", accuracy(y_test, predictions))  # Print accuracy
