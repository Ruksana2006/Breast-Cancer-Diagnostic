"""
File: src/main_pipeline.py
Description: Complete Data Science & ML pipeline for Breast Cancer Diagnostics.
             Performs Data Cleaning, EDA, Visualization, and Model Training.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def step_1_clean_data(filepath="data/raw_cancer_data.csv"):
    """Loads raw dataset, removes duplicates, handles missing values, and saves clean CSV."""
    print("\n--- STEP 1: DATA CLEANING ---")
    df = pd.read_csv(filepath)
    print(f"Initial Raw Dataset Shape: {df.shape}")
    print(f"Missing Values Before Cleaning:\n{df.isnull().sum()}\n")

    # 1. Remove duplicate records
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate rows.")

    # 2. Impute missing numerical values using Median
    for col in df.columns:
        if col != 'diagnosis' and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val:.2f}")

    # Save cleaned dataset
    clean_path = "data/cleaned_cancer_data.csv"
    df.to_csv(clean_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to '{clean_path}' with shape {df.shape}.")
    return df

def step_2_generate_visualizations(df):
    """Generates and saves analytical charts to the 'outputs/' folder."""
    print("\n--- STEP 2: GENERATING VISUALIZATIONS ---")
    sns.set_theme(style="whitegrid")

    # Chart 1: Diagnosis Distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='diagnosis', palette='Set2')
    plt.title("Breast Cancer Diagnosis Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Diagnosis (benign vs malignant)")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 5))
    plt.tight_layout()
    plt.savefig("outputs/01_diagnosis_distribution.png", dpi=300)
    plt.close()

    # Chart 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.drop(columns=['diagnosis'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/02_correlation_heatmap.png", dpi=300)
    plt.close()

    print("[SUCCESS] Saved visualization charts in 'outputs/'.")

def step_3_train_model(df):
    """Trains a Random Forest Classifier and saves outputs & model."""
    print("\n--- STEP 3: MACHINE LEARNING MODEL ---")
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'malignant': 0, 'benign': 1})

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['malignant', 'benign'])

    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:\n", report)

    # Save model evaluation report
    with open("outputs/model_evaluation_report.txt", "w") as f:
        f.write("BREAST CANCER DIAGNOSTIC ML MODEL REPORT\n")
        f.write("========================================\n")
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # Chart 3: Feature Importance
    plt.figure(figsize=(8, 4))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', color='purple')
    plt.title("Feature Importance in Tumor Diagnosis", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/03_feature_importance.png", dpi=300)
    plt.close()

    # Save trained model
    joblib.dump(model, "outputs/cancer_model.pkl")
    print("[SUCCESS] Saved trained model and report to 'outputs/'.")

if __name__ == "__main__":
    df_clean = step_1_clean_data()
    step_2_generate_visualizations(df_clean)
    step_3_train_model(df_clean)