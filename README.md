# 🌸 Iris Flower Classification with Machine Learning

An end-to-end Machine Learning classification project to predict and categorize Iris flower species (*Setosa, Versicolor, Virginica*) based on sepal and petal morphological measurements.

---

## 📌 Project Overview
- **Objective:** Classify Iris flower species accurately using multi-feature botanical measurements and compare diverse classification algorithms alongside ensemble voting methods.
- **Dataset:** Scikit-Learn Built-in Iris Dataset (150 samples, 4 continuous features).
- **Target Classes:** 
  - `0`: Setosa
  - `1`: Versicolor
  - `2`: Virginica
- **Key Features:**
  - `sepal length (cm)`[cite: 2]
  - `sepal width (cm)`[cite: 2]
  - `petal length (cm)`[cite: 2]
  - `petal width (cm)`[cite: 2]

---

## ⚙️ Machine Learning Pipeline Workflow

### 1. Data Inspection & Preprocessing
* Loaded and mapped target integer labels to botanical species names (*Setosa, Versicolor, Virginica*)[cite: 2].
* Verified data quality (0 missing/null values across all columns)[cite: 2].
* Computed statistical metrics (mean, standard deviation, quartiles)[cite: 2].

### 2. Feature Relationship Visualization
* Generated a multi-variable **Pairplot** using Seaborn to observe species clustering and linear separability across petal/sepal dimensions[cite: 2].

### 3. Data Splitting & Feature Standardization
* Stratified 80/20 train-test split (`test_size=0.2`, `random_state=42`, `stratify=y`) ensuring balanced class distributions across folds[cite: 2].
* Standardized feature distributions using **`StandardScaler`** ($z = \frac{x - \mu}{\sigma}$) to prevent feature scale bias[cite: 2].

### 4. Model Training & Evaluation
Benchmarked four machine learning classification algorithms:
* **Support Vector Machine (SVM):** Linear kernel with probability calibration[cite: 2].
* **Logistic Regression:** Regularized linear classifier (`max_iter=200`)[cite: 2].
* **Decision Tree Classifier:** Tree-based partitioning algorithm[cite: 2].
* **Random Forest Classifier:** Bagging ensemble of 100 decision trees[cite: 2].

### 5. Ensemble Learning
* **Soft Voting Classifier:** Weighted probability averaging across all 4 base models[cite: 2].
* **Hard Voting Classifier:** Majority rule consensus classification[cite: 2].

---

## 📊 Model Performance Benchmarking

| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **100.00%** | **1.00** | **1.00** | **1.00** |
| **Logistic Regression** | 93.33% | 0.93 | 0.93 | 0.93 |
| **Decision Tree** | 93.33% | 0.93 | 0.93 | 0.93 |
| **Random Forest** | 90.00% | 0.90 | 0.90 | 0.90 |
| **Soft Voting Classifier (Ensemble)** | **93.33%** | 0.93 | 0.93 | 0.93 |

---

## 🎯 Confusion Matrix Analysis (Best Model - SVM)

The Support Vector Machine classifier achieved perfect zero-error classification across the entire test set[cite: 2]:

```text
               Predicted Setosa   Predicted Versicolor   Predicted Virginica
Actual Setosa          10                   0                      0
Actual Versicolor       0                  10                      0
Actual Virginica        0                   0                     10
