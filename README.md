# Machine Learning Algorithms Overview

## 1. Linear Regression
Linear regression is a fundamental algorithm used for predicting a continuous target variable based on one or more input features. It establishes a linear relationship by fitting a line through the data points. The goal is to minimize the difference between predicted and actual values, typically using the least squares method.

**Key Concepts:**
- **Equation:** `y = b0 + b1 x_1 + ...... + b_n x_n`
- **Assumptions:** Linearity, independence, homoscedasticity, normal distribution of errors.

---

## 2. Logistic Regression
Logistic regression is used for binary classification tasks. Unlike linear regression, it predicts the probability of a binary outcome using the logistic function (sigmoid function), which maps predicted values to a range between 0 and 1.

**Key Concepts:**
- **Output:** Probabilities (0 to 1) of class membership.
- **Loss Function:** Binary cross-entropy loss.
- **Decision Boundary:** Determined by a threshold (e.g., 0.5).

---

## 3. Perceptron
The Perceptron is a simple type of artificial neuron used in binary classification. It learns a linear decision boundary by adjusting weights based on misclassified instances. This is achieved through a series of updates during the training phase.

**Key Concepts:**
- **Activation Function:** Typically a step function.
- **Learning Rule:** Weights are updated based on prediction errors.
- **Limitations:** Can only classify linearly separable data.

---

## 4. Decision Tree
Decision trees are a non-parametric supervised learning method used for both classification and regression tasks. They split data into branches based on feature values, leading to a tree-like model of decisions.

**Key Concepts:**
- **Splitting Criteria:** Gini impurity, entropy (for classification), mean squared error (for regression).
- **Tree Depth:** Controls model complexity and overfitting.
- **Interpretability:** Easy to visualize and understand.

---

## 5. Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees during training and merges their predictions to improve accuracy and control overfitting. It combines the strengths of individual trees to achieve better performance.

**Key Concepts:**
- **Bootstrapping:** Random sampling with replacement to create diverse datasets.
- **Feature Randomness:** Randomly selecting a subset of features for each tree to enhance diversity.
- **Voting Mechanism:** Aggregates predictions from all trees.

---

## 6. Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms data into a lower-dimensional space while preserving as much variance as possible. It identifies the directions (principal components) along which the data varies the most.

**Key Concepts:**
- **Mean Centering:** Subtracting the mean to center the data.
- **Eigenvalues and Eigenvectors:** Used to determine principal components.
- **Variance Preservation:** Focuses on directions of maximum variance.

---

## 7. Support Vector Machine (SVM)
SVM is a powerful classification algorithm that aims to find the optimal hyperplane that maximizes the margin between different classes. It can handle both linear and non-linear classification through the use of kernel functions.

**Key Concepts:**
- **Margin:** The distance between the hyperplane and the nearest data points from either class (support vectors).
- **Kernel Trick:** Allows SVM to perform in high-dimensional spaces.
- **Regularization:** Controls the trade-off between maximizing margin and minimizing classification error.

---

## 8. K-Nearest Neighbors (KNN)
KNN is a simple, instance-based learning algorithm used for classification and regression. It classifies a data point based on the majority class of its `k` nearest neighbors in the feature space.

**Key Concepts:**
- **Distance Metric:** Typically Euclidean distance.
- **Voting Mechanism:** Assigns class based on the majority vote of neighbors.
- **Non-parametric:** No explicit training phase.

---

## 9. K-Means Clustering
K-Means is an unsupervised learning algorithm used for clustering. It partitions the dataset into `k` distinct clusters based on feature similarity. Each cluster is defined by its centroid, which is the mean of the data points assigned to that cluster.

**Key Concepts:**
- **Centroid Initialization:** Randomly selecting `k` points as initial centroids.
- **Iteration:** Assigning points to the nearest centroid and updating centroids until convergence.
- **Distance Metric:** Typically Euclidean distance.

---

## 10. Linear Discriminant Analysis (LDA)
LDA is a supervised learning technique used for dimensionality reduction and classification. It aims to find a linear combination of features that best separates two or more classes by maximizing the ratio of between-class variance to within-class variance.

**Key Concepts:**
- **Scatter Matrices:** Within-class and between-class scatter matrices are computed to measure class separability.
- **Eigenvalues and Eigenvectors:** Used to determine the optimal projection direction.
- **Class Separation:** Focuses on maximizing distance between classes.

---

## 11. Naive Bayes
Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem. It assumes that the presence of a feature in a class is independent of the presence of any other feature, which simplifies the computation.

**Key Concepts:**
- **Conditional Probability:** Calculates the probability of a class given the features.
- **Independence Assumption:** Features are assumed to be independent.
- **Types:** Gaussian Naive Bayes, Multinomial Naive Bayes, etc.

---

## 12. AdaBoost
AdaBoost (Adaptive Boosting) is an ensemble learning technique that combines multiple weak classifiers to create a strong classifier. It focuses on training the classifiers sequentially, giving more weight to misclassified instances.

**Key Concepts:**
- **Weight Adjustment:** Misclassified samples receive higher weights in subsequent classifiers.
- **Final Model:** Combines predictions from all weak classifiers, often using weighted voting.
- **Boosting Framework:** Effective for improving performance of weak learners.

---

