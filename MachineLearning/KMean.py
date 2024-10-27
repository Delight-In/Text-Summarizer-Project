import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing NumPy for numerical operations

np.random.seed(42)  # Set random seed for reproducibility


def euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K  # Number of clusters
        self.max_iters = max_iters  # Maximum iterations for convergence
        self.plot_steps = plot_steps  # Flag to plot steps during training

        # List to hold indices of samples for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # Centroids of the clusters
        self.centroids = []

    def predict(self, X):
        """Assign clusters to each sample in X."""
        self.X = X  # Store the data
        self.n_samples, self.n_features = X.shape  # Get dimensions of data

        # Initialize centroids with random samples from X
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Create clusters based on the closest centroids
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()  # Plotting the current step

            # Store old centroids for convergence check
            centroids_old = self.centroids
            # Calculate new centroids from the clusters
            self.centroids = self._get_centroids(self.clusters)

            # Check if the centroids have changed
            if self._is_converged(centroids_old, self.centroids):
                break  # Stop if converged

            if self.plot_steps:
                self.plot()  # Plotting again if needed

        # Assign each sample to the cluster index
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        """Get the labels for each sample based on cluster assignments."""
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx  # Assign cluster index to each sample
        return labels

    def _create_clusters(self, centroids):
        """Assign samples to the nearest centroids to create clusters."""
        clusters = [[] for _ in range(self.K)]  # Initialize empty clusters
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)  # Find closest centroid
            clusters[centroid_idx].append(idx)  # Add sample index to the corresponding cluster
        return clusters

    def _closest_centroid(self, sample, centroids):
        """Find the index of the closest centroid to a sample."""
        # Compute distances from the sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)  # Get index of the closest centroid
        return closest_index

    def _get_centroids(self, clusters):
        """Calculate the new centroids as the mean of each cluster."""
        centroids = np.zeros((self.K, self.n_features))  # Initialize centroids
        for cluster_idx, cluster in enumerate(clusters):
            # Calculate mean for each cluster
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean  # Update the centroid
        return centroids

    def _is_converged(self, centroids_old, centroids):
        """Check if the centroids have changed (convergence check)."""
        # Calculate distances between old and new centroids
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0  # Return True if all distances are zero

    def plot(self):
        """Plot the clusters and centroids."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each cluster
        for i, index in enumerate(self.clusters):
            point = self.X[index].T  # Get points for the cluster
            ax.scatter(*point)  # Scatter plot for points

        # Plot centroids
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)  # Mark centroids

        plt.show()  # Display the plot


# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_blobs  # Import function to create synthetic datasets

    # Create synthetic data with 3 clusters
    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)  # Print the shape of the dataset

    clusters = len(np.unique(y))  # Number of unique clusters in the dataset
    print(clusters)  # Print number of clusters

    # Initialize and fit the KMeans model
    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)  # Predict cluster labels for the dataset

    k.plot()  # Plot the final clusters and centroids
