# ML-DL-Algorithms

### Machine learning and Deep learning algorithms collection from scratch.

* Regression
* Decision Tree
* Random Forest
* SVM
* Naive bayes
* Bagging
* Gradient Boosting
* Kmean
* Hierarchial
* DBSCAN
* XGBoost


Here’s a brief description and a mathematical formula for each of the mentioned algorithms and techniques:

### 1. Regression
**Description:** A statistical method to model the relationship between a dependent variable and one or more independent variables.  
**Formula:** \( Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n + \epsilon \)

### 2. Decision Tree
**Description:** A tree-like model used for classification and regression that splits data into branches based on feature values.  
**Formula:** No specific formula; uses Gini impurity or entropy for splits.

### 3. Random Forest
**Description:** An ensemble method that builds multiple decision trees and merges their outputs for improved accuracy and robustness.  
**Formula:** Average of predictions from multiple trees: \( \hat{Y} = \frac{1}{M} \sum_{m=1}^{M} T_m(X) \)

### 4. SVM (Support Vector Machine)
**Description:** A classification technique that finds the hyperplane that best separates classes in the feature space.  
**Formula:** Maximize \( \frac{2}{\|w\|} \) subject to \( y_i (w^T x_i + b) \geq 1 \)

### 5. Naive Bayes
**Description:** A probabilistic classifier based on Bayes' theorem, assuming independence between features.  
**Formula:** \( P(C|X) = \frac{P(X|C)P(C)}{P(X)} \)

### 6. Bagging (Bootstrap Aggregating)
**Description:** An ensemble method that generates multiple subsets of data by sampling with replacement and trains a model on each.  
**Formula:** Combined prediction: \( \hat{Y} = \frac{1}{M} \sum_{m=1}^{M} f_m(X) \)

### 7. Gradient Boosting
**Description:** An ensemble technique that builds models sequentially, each trying to correct the errors of the previous one.  
**Formula:** \( F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x) \) where \( h_m(x) \) is the new model.

### 8. K-means
**Description:** A clustering algorithm that partitions data into K distinct clusters based on distance from cluster centroids.  
**Formula:** \( J = \sum_{i=1}^{K} \sum_{j=1}^{n} ||x_j - \mu_i||^2 \)

### 9. Hierarchical Clustering
**Description:** A clustering method that builds a hierarchy of clusters either through agglomerative or divisive approaches.  
**Formula:** No specific formula; uses distance metrics to create dendrograms.

### 10. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
**Description:** A clustering method that groups points closely packed together while marking points in low-density regions as outliers.  
**Formula:** No explicit formula; defines clusters based on density (epsilon and min points).

### 11. XGBoost (Extreme Gradient Boosting)
**Description:** An optimized implementation of gradient boosting that includes regularization to improve model performance and prevent overfitting.  
**Formula:** Similar to gradient boosting: \( F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x) \), with additional regularization terms.
