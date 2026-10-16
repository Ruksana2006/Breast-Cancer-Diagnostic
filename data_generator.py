"""
File: src/data_generator.py
Description: Loads the Breast Cancer Wisconsin dataset using scikit-learn,
             injects minor missing values/noise to demonstrate Data Cleaning,
             and saves the raw dataset to the 'data/' folder.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

def generate_cancer_dataset():
    # Ensure 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Load standard Breast Cancer dataset from sklearn
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    
    # Add target column (0 = malignant, 1 = benign)
    df['diagnosis'] = cancer.target

    # Select a few key features to keep it clean and structured
    key_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 
        'mean smoothness', 'mean compactness', 'mean concavity', 'diagnosis'
    ]
    df = df[key_features]

    # Rename diagnosis values for readability: 0 -> malignant, 1 -> benign
    df['diagnosis'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})

    # Inject a few random missing values and duplicate rows to demonstrate Data Cleaning
    np.random.seed(42)
    for col in ['mean radius', 'mean texture', 'mean area']:
        missing_indices = np.random.choice(df.index, size=10, replace=False)
        df.loc[missing_indices, col] = np.nan

    duplicates = df.sample(12, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Save raw dataset
    output_path = "data/raw_cancer_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Raw Breast Cancer dataset generated at '{output_path}' with {len(df)} records.")

if __name__ == "__main__":
    generate_cancer_dataset()