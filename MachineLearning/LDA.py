import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components  # Number of linear discriminants to retain
        self.linear_discriminants = None  # Placeholder for the linear discriminants

    def fit(self, X, y):
        n_features = X.shape[1]  # Number of features in the dataset
        class_labels = np.unique(y)  # Unique class labels in the target variable

        # Overall mean of the dataset
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))  # Within-class scatter matrix
        SB = np.zeros((n_features, n_features))  # Between-class scatter matrix

        for c in class_labels:
            X_c = X[y == c]  # Data points for class c
            mean_c = np.mean(X_c, axis=0)  # Mean of class c

            # Calculate within-class scatter
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            n_c = X_c.shape[0]  # Number of samples in class c
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)  # Difference between class mean and overall mean

            # Calculate between-class scatter
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Compute the inverse of SW and then multiply by SB
        A = np.linalg.inv(SW).dot(SB)

        # Obtain eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort eigenvalues and eigenvectors in descending order
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors.T[idxs]  # Transpose for easier indexing

        # Store the first n eigenvectors as linear discriminants
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        """Project the data onto the linear discriminants."""
        return np.dot(X, self.linear_discriminants.T)


# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # Load the Iris dataset
    data = datasets.load_iris()
    X, y = data.data, data.target

    # Create an LDA instance with 2 components and fit the model
    lda = LDA(2)
    lda.fit(X, y)

    # Transform the data to the new space
    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    # Plot the results
    x1, x2 = X_projected[:, 0], X_projected[:, 1]
    plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.title("LDA of Iris Dataset")
    plt.show()
