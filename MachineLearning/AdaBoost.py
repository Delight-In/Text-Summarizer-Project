import numpy as np  # Import the NumPy library for numerical operations


# Decision stump used as weak classifier
class DecisionStump:  # Define the DecisionStump class
    def __init__(self):  # Initialize the class
        self.polarity = 1  # Set polarity (direction of prediction)
        self.feature_idx = None  # Initialize feature index
        self.threshold = None  # Initialize threshold
        self.alpha = None  # Initialize classifier weight

    def predict(self, X):  # Method to make predictions
        n_samples = X.shape[0]  # Get number of samples
        X_column = X[:, self.feature_idx]  # Select the relevant feature
        predictions = np.ones(n_samples)  # Initialize predictions to 1

        # Set predictions based on threshold and polarity
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1  # Assign -1 where condition is met
        else:
            predictions[X_column > self.threshold] = -1  # Assign -1 for the opposite condition

        return predictions  # Return final predictions


class Adaboost:  # Define the Adaboost class
    def __init__(self, n_clf=5):  # Initialize with number of classifiers
        self.n_clf = n_clf  # Set number of weak classifiers
        self.clfs = []  # Initialize list to hold classifiers

    def fit(self, X, y):  # Method to train the model
        n_samples, n_features = X.shape  # Get number of samples and features

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))  # Set all weights to equal value

        self.clfs = []  # Clear classifiers list for new training

        # Iterate through classifiers
        for _ in range(self.n_clf):  # Loop for the number of classifiers
            clf = DecisionStump()  # Create a new DecisionStump classifier
            min_error = float("inf")  # Initialize minimum error

            # Greedy search to find best threshold and feature
            for feature_i in range(n_features):  # Iterate through each feature
                X_column = X[:, feature_i]  # Get the current feature column
                thresholds = np.unique(X_column)  # Get unique values for thresholds

                for threshold in thresholds:  # Iterate through each threshold
                    # Predict with polarity 1
                    p = 1  # Assume initial polarity is 1
                    predictions = np.ones(n_samples)  # Initialize predictions to 1
                    predictions[X_column < threshold] = -1  # Set predictions based on threshold

                    # Calculate error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]  # Get weights of misclassified samples
                    error = sum(misclassified)  # Calculate error

                    # Adjust for polarity if error is greater than 0.5
                    if error > 0.5:
                        error = 1 - error  # Adjust error
                        p = -1  # Change polarity

                    # Store the best configuration if error is minimized
                    if error < min_error:
                        clf.polarity = p  # Set best polarity
                        clf.threshold = threshold  # Set best threshold
                        clf.feature_idx = feature_i  # Set best feature index
                        min_error = error  # Update minimum error

            # Calculate alpha (classifier weight)
            EPS = 1e-10  # Small value to prevent division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))  # Calculate classifier weight

            # Calculate predictions and update weights
            predictions = clf.predict(X)  # Get predictions from the current classifier

            # Update weights based on predictions
            w *= np.exp(-clf.alpha * y * predictions)  # Adjust weights
            # Normalize weights to sum to one
            w /= np.sum(w)  # Normalize weights

            # Save classifier
            self.clfs.append(clf)  # Add classifier to the list

    def predict(self, X):  # Method to make final predictions
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]  # Get weighted predictions from each classifier
        y_pred = np.sum(clf_preds, axis=0)  # Sum all classifier predictions
        y_pred = np.sign(y_pred)  # Convert to -1 or 1 based on sign

        return y_pred  # Return final predicted labels


# Testing the implementation
if __name__ == "__main__":  # Ensure this runs only if the script is executed directly
    # Imports for dataset loading and model evaluation
    from sklearn import datasets  # Import datasets from sklearn
    from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting

    def accuracy(y_true, y_pred):  # Function to compute accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # Calculate accuracy
        return accuracy  # Return computed accuracy

    data = datasets.load_breast_cancer()  # Load breast cancer dataset
    X, y = data.data, data.target  # Get features and target labels

    y[y == 0] = -1  # Convert target values from 0 to -1 for Adaboost

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into training and testing sets
        X, y, test_size=0.2, random_state=5  # Specify test size and random state
    )

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)  # Instantiate the Adaboost classifier
    clf.fit(X_train, y_train)  # Train the model on training data
    y_pred = clf.predict(X_test)  # Make predictions on test data

    acc = accuracy(y_test, y_pred)  # Calculate accuracy of predictions
    print("Accuracy:", acc)  # Print accuracy
