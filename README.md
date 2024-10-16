# ML-DL-Algorithms

### Machine Learning and Deep Learning Algorithms Collection from Scratch

This repository contains a collection of fundamental machine learning and deep learning algorithms implemented from scratch. Each algorithm includes a brief description and a relevant mathematical formula.

## Algorithms Included

- Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Bagging (Bootstrap Aggregating)
- Gradient Boosting
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- XGBoost (Extreme Gradient Boosting)

---

## Algorithm Descriptions

### 1. Regression
**Description:** A statistical method to model the relationship between a dependent variable and one or more independent variables.  
**Formula:**  
`Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε`

### 2. Decision Tree
**Description:** A tree-like model used for classification and regression that splits data into branches based on feature values.  
**Formula:** No specific formula; uses Gini impurity or entropy for splits.

### 3. Random Forest
**Description:** An ensemble method that builds multiple decision trees and merges their outputs for improved accuracy and robustness.  
**Formula:**  
`Ŷ = (1/M) * Σ (Tₘ(X))`, where `M` is the number of trees.

### 4. Support Vector Machine (SVM)
**Description:** A classification technique that finds the hyperplane that best separates classes in the feature space.  
**Formula:** Maximize  
`2 / ||w||`  
subject to  
`yᵢ (w^T xᵢ + b) ≥ 1`.

### 5. Naive Bayes
**Description:** A probabilistic classifier based on Bayes' theorem, assuming independence between features.  
**Formula:**  
`P(C|X) = (P(X|C) * P(C)) / P(X)`.

### 6. Bagging (Bootstrap Aggregating)
**Description:** An ensemble method that generates multiple subsets of data by sampling with replacement and trains a model on each.  
**Formula:**  
`Ŷ = (1/M) * Σ (fₘ(X))`, where `M` is the number of models.

### 7. Gradient Boosting
**Description:** An ensemble technique that builds models sequentially, each trying to correct the errors of the previous one.  
**Formula:**  
`Fₘ(x) = Fₘ₋₁(x) + η * hₘ(x)`,  
where `hₘ(x)` is the new model.

### 8. K-Means Clustering
**Description:** A clustering algorithm that partitions data into K distinct clusters based on distance from cluster centroids.  
**Formula:**  
`J = Σ (i=1 to K) Σ (j=1 to n) ||xᵢ - μₖ||²`.

### 9. Hierarchical Clustering
**Description:** A clustering method that builds a hierarchy of clusters either through agglomerative or divisive approaches.  
**Formula:** No specific formula; uses distance metrics to create dendrograms.

### 10. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
**Description:** A clustering method that groups points closely packed together while marking points in low-density regions as outliers.  
**Formula:** No explicit formula; defines clusters based on density (epsilon and min points).

### 11. XGBoost (Extreme Gradient Boosting)
**Description:** An optimized implementation of gradient boosting that includes regularization to improve model performance and prevent overfitting.  
**Formula:** Similar to gradient boosting:  
`Fₘ(x) = Fₘ₋₁(x) + η * hₘ(x)`,  
with additional regularization terms.

---

## Getting Started

To run these algorithms, clone this repository and follow the instructions in each algorithm's directory for setup and execution.

### Prerequisites

- Python 3.x
- Required libraries (e.g., NumPy, Pandas, scikit-learn)

### Installation

```bash
git clone https://github.com/yourusername/ML-DL-Algorithms.git
cd ML-DL-Algorithms
# Install required libraries
pip install -r requirements.txt
