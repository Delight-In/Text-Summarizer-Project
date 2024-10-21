import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.0001, n_iter = 1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_features = X.shape

        # Init parameters

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iter):

            # approximate value of y
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply sigmoid function on y
            y_predicted = self._sigmoid(linear_model)

            # Compute Gradient
            dw = (1/n_sample) * np.dot(X.T, (y_predicted - y))
            db = (1/n_sample) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self. learning_rate * db

    def predict(self,X):
       linear_model = np.dot(X, self.weights) + self.bias
       y_predicted = self._sigmoid(linear_model)
       y_predicted_class = [1 if i>0.5 else 0 for i in y_predicted]
       return np.array(y_predicted_class)
    
    def _sigmoid(self, x):
       return 1 / (1 + np.exp(-x))
    

# Example

if __name__ == "__main__":

    # Imports Libraries
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iter=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))


