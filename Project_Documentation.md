# Project Documentation: Breast Cancer Diagnostic System

## 1. Abstract
Early and accurate detection of breast cancer is crucial for successful medical treatment. The **Breast Cancer Diagnostic System** is an end-to-end data science and machine learning application that utilizes the Wisconsin Breast Cancer Dataset to classify tumors as Benign or Malignant based on cell nuclear measurements.

## 2. Methodology
1. **Data Acquisition:** Loaded the benchmark breast cancer dataset via `scikit-learn` with simulated noise and missing values.
2. **Data Cleaning:** Handled missing values using median imputation and removed duplicate entries.
3. **Exploratory Data Analysis (EDA):** Generated correlation matrices and distribution visualizations stored in `outputs/`.
4. **Predictive Modeling:** Trained a Random Forest Classifier to evaluate tumor characteristics with high classification accuracy.