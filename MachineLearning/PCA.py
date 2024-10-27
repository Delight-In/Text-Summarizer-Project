import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # Number of principal components to keep
        self.components = None  # Placeholder for principal components
        self.mean = None  # Placeholder for mean of the dataset

    def fit(self, X):
        # Mean centering the data
        self.mean = np.mean(X, axis=0)  # Compute mean for each feature
        X = X - self.mean  # Center the data by subtracting the mean

        # Compute the covariance matrix
        cov = np.cov(X.T)  # Transpose to get features as columns

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        eigenvectors = eigenvectors.T  # Transpose eigenvectors for easier indexing
        idxs = np.argsort(eigenvalues)[::-1]  # Sort indices by eigenvalue size
        eigenvalues = eigenvalues[idxs]  # Sort eigenvalues
        eigenvectors = eigenvectors[idxs]  # Sort eigenvectors

        # Store the first n eigenvectors as components
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project the data onto the principal components
        X = X - self.mean  # Center the data using the mean computed during fit
        return np.dot(X, self.components.T)  # Project data onto principal components


# Testing the PCA implementation
if __name__ == "__main__":
    # Imports for data visualization and dataset
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # Load the iris dataset
    data = datasets.load_iris()
    X = data.data  # Feature data
    y = data.target  # Target labels

    # Project the data onto the 2 primary principal components
    pca = PCA(2)  # Initialize PCA with 2 components
    pca.fit(X)  # Fit PCA model to the data
    X_projected = pca.transform(X)  # Transform the data

    # Print shapes of original and transformed data
    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    # Extract the projected components for plotting
    x1 = X_projected[:, 0]  # First principal component
    x2 = X_projected[:, 1]  # Second principal component

    # Plot the projected data
    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")  # Label for x-axis
    plt.ylabel("Principal Component 2")  # Label for y-axis
    plt.colorbar()  # Show colorbar for class labels
    plt.show()  # Display the plot
