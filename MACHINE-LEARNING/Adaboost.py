import numpy as np

# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1  # Polarity indicates whether the threshold is a greater or lesser condition
        self.feature_idx = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for the split
        self.alpha = None  # Weight of the classifier in the final prediction

    def predict(self, X):
        n_samples = X.shape[0]  # Get the number of samples
        X_column = X[:, self.feature_idx]  # Extract the relevant feature column
        predictions = np.ones(n_samples)  # Initialize predictions to 1 (positive class)
        
        # Set predictions based on the polarity and threshold
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1  # Classify as -1 if below the threshold
        else:
            predictions[X_column > self.threshold] = -1  # Classify as -1 if above the threshold

        return predictions  # Return the final predictions


class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf  # Number of classifiers to use
        self.clfs = []  # List to store weak classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get number of samples and features

        # Initialize weights to equal for all samples
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []  # Reset classifiers list

        # Iterate through the number of classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()  # Create a new decision stump
            min_error = float("inf")  # Initialize minimum error to infinity

            # Greedily search for the best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]  # Extract the current feature column
                thresholds = np.unique(X_column)  # Get unique values to consider as thresholds

                # Iterate through each threshold
                for threshold in thresholds:
                    # Predict with polarity 1
                    p = 1  # Initialize polarity
                    predictions = np.ones(n_samples)  # Default predictions to 1
                    predictions[X_column < threshold] = -1  # Update predictions based on threshold

                    # Calculate error as the weighted sum of misclassified samples
                    misclassified = w[y != predictions]  # Weights of misclassified samples
                    error = sum(misclassified)  # Total error

                    # Adjust error and polarity if error is greater than 0.5
                    if error > 0.5:
                        error = 1 - error  # Adjust error
                        p = -1  # Flip polarity

                    # Update the best configuration if the current error is lower
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error  # Update minimum error

            # Calculate alpha, the weight of the classifier
            EPS = 1e-10  # Small constant to avoid division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))  # Calculate classifier weight

            # Get predictions and update weights
            predictions = clf.predict(X)  # Get predictions from the stump

            # Update weights for misclassified samples
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize weights so they sum to one
            w /= np.sum(w)

            # Save the classifier
            self.clfs.append(clf)

    def predict(self, X):
        # Get predictions from all classifiers and sum them
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)  # Sum predictions
        y_pred = np.sign(y_pred)  # Convert to binary predictions (-1 or 1)

        return y_pred  # Return final predictions


# Testing
if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Function to calculate accuracy
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Calculate accuracy as the ratio of correct predictions
        return accuracy

    data = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = data.data, data.target  # Extract features and target

    y[y == 0] = -1  # Convert target labels from 0/1 to -1/1

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)  # Initialize Adaboost with 5 classifiers
    clf.fit(X_train, y_train)  # Fit the model to the training data
    y_pred = clf.predict(X_test)  # Make predictions on the test data

    acc = accuracy(y_test, y_pred)  # Calculate accuracy of the predictions
    print("Accuracy:", acc)  # Print the accuracy
