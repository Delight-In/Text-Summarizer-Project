import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = -1 * np.ones(n_samples)  # -1 indicates noise
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already processed
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:  # Mark as noise
                self.labels_[i] = -1
            else:  # Start a new cluster
                self.labels_[i] = cluster_id
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        for neighbor_idx in neighbors:
            if self.labels_[neighbor_idx] == -1:  # Change noise to border point
                self.labels_[neighbor_idx] = cluster_id
            if self.labels_[neighbor_idx] != -1:  # Already in a cluster
                continue

            self.labels_[neighbor_idx] = cluster_id
            new_neighbors = self._get_neighbors(X, neighbor_idx)

            if len(new_neighbors) >= self.min_samples:
                neighbors = np.union1d(neighbors, new_neighbors)

# Example:
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X1 = np.random.normal(loc=0.0, scale=0.5, size=(100, 2))
    X2 = np.random.normal(loc=5.0, scale=0.5, size=(100, 2))
    X3 = np.random.normal(loc=10.0, scale=0.5, size=(100, 2))
    X = np.vstack((X1, X2, X3))

    # Fit the DBSCAN model
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X)

    # Get labels
    labels = dbscan.labels_

    # Plotting the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=30)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.show()
