import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, reg_strength=0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.num_iters = num_iters
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.num_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Update rule for the correct classification
                    self.w -= self.learning_rate * (2 * self.reg_strength * self.w)
                else:
                    # Update rule for the misclassified point
                    self.w -= self.learning_rate * (2 * self.reg_strength * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Example usage
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset (e.g., Iris dataset)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # We will classify only two classes for simplicity
    X = X[y != 2]
    y = y[y != 2]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the SVM
    svm = LinearSVM(learning_rate=0.001, reg_strength=0.01, num_iters=1000)
    svm.fit(X_train, y_train)

    # Predict
    predictions = svm.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    print(f'Accuracy: {acc * 100:.2f}%')
