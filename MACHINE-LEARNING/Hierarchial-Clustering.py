import numpy as np
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        # Create a distance matrix
        self.distance_matrix = self._compute_distance_matrix(X)
        # Create a list of clusters
        self.clusters = [[i] for i in range(len(X))]

        while len(self.clusters) > self.n_clusters:
            # Find the two closest clusters
            i, j = self._find_closest_clusters()
            # Merge these clusters
            self._merge_clusters(i, j)

        # Assign labels to the data points
        self.labels_ = np.zeros(len(X), dtype=int)
        for cluster_id, cluster in enumerate(self.clusters):
            for point in cluster:
                self.labels_[point] = cluster_id

    def _compute_distance_matrix(self, X):
        # Compute Euclidean distance matrix
        distance_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        return distance_matrix

    def _find_closest_clusters(self):
        # Find the indices of the two closest clusters
        min_distance = np.inf
        closest_clusters = (None, None)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = self._linkage(self.clusters[i], self.clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_clusters = (i, j)

        return closest_clusters

    def _linkage(self, cluster1, cluster2):
        # Compute the distance between two clusters (average linkage)
        dist = np.mean([self.distance_matrix[i, j] for i in cluster1 for j in cluster2])
        return dist

    def _merge_clusters(self, i, j):
        # Merge clusters i and j
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]

    def predict(self):
        return self.labels_

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.vstack((
        np.random.normal(loc=0.0, scale=1.0, size=(100, 2)),
        np.random.normal(loc=5.0, scale=1.0, size=(100, 2)),
        np.random.normal(loc=10.0, scale=1.0, size=(100, 2))
    ))

    # Fit the Hierarchical Clustering model
    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)

    # Predict clusters
    labels = hc.predict()

    # Plotting the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=30)
    plt.title('Hierarchical Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
