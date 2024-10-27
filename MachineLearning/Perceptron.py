import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate  # Learning rate for weight updates
        self.n_iters = n_iters  # Number of iterations for training
        self.activation_func = self._unit_step_func  # Activation function
        self.weights = None  # Placeholder for weights
        self.bias = None  # Placeholder for bias

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Number of samples and features

        # Initialize parameters
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        # Convert labels to binary (1 for positive class, 0 for negative)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function to get predicted output
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)  # Calculate update
                self.weights += update * x_i  # Update weights
                self.bias += update  # Update bias

    def predict(self, X):
        # Calculate linear output for input features
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply activation function to get predictions
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        # Activation function: step function
        return np.where(x >= 0, 1, 0)


# Testing the Perceptron implementation
if __name__ == "__main__":
    # Imports for data visualization and dataset
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        # Calculate accuracy of predictions
        return np.sum(y_true == y_pred) / len(y_true)

    # Create a dataset with two classes
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Initialize and train the Perceptron model
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)  # Fit the model to training data
    predictions = p.predict(X_test)  # Make predictions on test data

    # Print the classification accuracy
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    # Plot the training data and the decision boundary
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # Calculate decision boundary
    x0_1 = np.amin(X_train[:, 0])  # Minimum x value
    x0_2 = np.amax(X_train[:, 0])  # Maximum x value

    # Calculate corresponding y values for the decision boundary
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # Plot the decision boundary
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # Set limits for y-axis
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()  # Display the plot
