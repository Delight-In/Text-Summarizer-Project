import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Initialize the SVM parameters
        self.lr = learning_rate  # Learning rate for weight updates
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get number of samples and features

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the condition for the SVM is satisfied
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Update weights for correctly classified samples
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Update weights and bias for misclassified samples
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Make predictions using the learned weights and bias
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)  # Return -1 or 1 based on the sign


# Testing the SVM implementation
if __name__ == "__main__":
    # Imports for dataset generation and visualization
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Create a synthetic dataset with two classes
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM

    # Initialize and train the SVM model
    clf = SVM()
    clf.fit(X, y)
    
    # Print learned weights and bias
    print(clf.w, clf.b)

    def visualize_svm():
        # Function to visualize the SVM decision boundary
        def get_hyperplane_value(x, w, b, offset):
            # Calculate the y-value of the hyperplane for a given x
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)  # Scatter plot of data points

        # Get x-coordinates for plotting
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # Calculate y-coordinates for the decision boundary and margins
        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)  # Lower margin
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)  # Upper margin
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        # Plot the decision boundary and margins
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # Decision boundary
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # Lower margin
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")  # Upper margin

        # Set y-axis limits for better visualization
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()  # Show the plot

    visualize_svm()  # Call the visualization function
