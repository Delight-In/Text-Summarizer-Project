import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get the number of samples and features
        self._classes = np.unique(y)  # Identify unique class labels
        n_classes = len(self._classes)  # Get the number of classes

        # Initialize arrays to store mean, variance, and prior probabilities for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]  # Filter the data for the current class
            self._mean[idx, :] = X_c.mean(axis=0)  # Compute the mean of the features for this class
            self._var[idx, :] = X_c.var(axis=0)  # Compute the variance of the features for this class
            self._priors[idx] = X_c.shape[0] / float(n_samples)  # Compute prior probability of the class

    def predict(self, X):
        # Predict the class labels for the provided data
        y_pred = [self._predict(x) for x in X]  # Predict for each sample
        return np.array(y_pred)  # Return predictions as a numpy array

    def _predict(self, x):
        posteriors = []  # List to hold posterior probabilities for each class

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # Log prior probability for the class
            posterior = np.sum(np.log(self._pdf(idx, x)))  # Log likelihood of the data given the class
            posterior = prior + posterior  # Combine prior and likelihood
            posteriors.append(posterior)  # Append to the list of posteriors

        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Calculate the probability density function for a given class
        mean = self._mean[class_idx]  # Get the mean for the class
        var = self._var[class_idx]  # Get the variance for the class
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Compute the numerator of the Gaussian
        denominator = np.sqrt(2 * np.pi * var)  # Compute the denominator of the Gaussian
        return numerator / denominator  # Return the probability density


# Testing the Naive Bayes classifier
if __name__ == "__main__":
    # Imports for dataset generation and model evaluation
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        # Calculate accuracy as the proportion of correct predictions
        return np.sum(y_true == y_pred) / len(y_true)

    # Generate a synthetic classification dataset
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Initialize and train the Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)  # Make predictions on the test set

    # Output the classification accuracy
    print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))
