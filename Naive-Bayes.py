import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._gaussian_pdf(x, c)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def _gaussian_pdf(self, x, c):
        mean = self.means[c]
        var = self.variances[c]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Example usage
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset (e.g., Iris dataset)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Naive Bayes model
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # Predict
    predictions = gnb.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, predictions)
    print(f'Accuracy: {acc * 100:.2f}%')
