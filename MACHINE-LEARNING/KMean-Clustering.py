import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # Randomly initialize centroids
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = self._calculate_centroids(X, labels)

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def predict(self, X):
        return self._assign_clusters(X)

    def get_centroids(self):
        return self.centroids

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.vstack((
        np.random.normal(loc=0.0, scale=1.0, size=(100, 2)),
        np.random.normal(loc=5.0, scale=1.0, size=(100, 2)),
        np.random.normal(loc=10.0, scale=1.0, size=(100, 2))
    ))

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    
    # Predict clusters
    labels = kmeans.predict(X)
    centroids = kmeans.get_centroids()

    # Plotting the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
