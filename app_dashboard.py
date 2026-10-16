"""
File: src/app_dashboard.py
Description: Interactive CLI Dashboard for Breast Cancer Diagnostic Predictions.
"""

import joblib
import pandas as pd

def run_diagnostic_dashboard():
    print("==========================================================")
    print("      🩺 BREAST CANCER DIAGNOSTIC PREDICTION DASHBOARD      ")
    print("==========================================================")

    # Load trained ML model
    try:
        model = joblib.load("outputs/cancer_model.pkl")
    except FileNotFoundError:
        print("[ERROR] Model file not found! Please run 'python src/main_pipeline.py' first.")
        return

    print("\nEnter Tumor Cell Measurements for AI Diagnostics:")
    try:
        mean_radius = float(input("1. Mean Radius [default: 14.12]: ") or 14.12)
        mean_texture = float(input("2. Mean Texture [default: 19.28]: ") or 19.28)
        mean_perimeter = float(input("3. Mean Perimeter [default: 91.93]: ") or 91.93)
        mean_area = float(input("4. Mean Area [default: 654.8]: ") or 654.8)
        mean_smoothness = float(input("5. Mean Smoothness [default: 0.096]: ") or 0.096)
        mean_compactness = float(input("6. Mean Compactness [default: 0.104]: ") or 0.104)
        mean_concavity = float(input("7. Mean Concavity [default: 0.088]: ") or 0.088)
    except ValueError:
        print("[ERROR] Please enter valid numerical values.")
        return

    # Prepare input dataframe
    input_data = pd.DataFrame([[
        mean_radius, mean_texture, mean_perimeter, mean_area,
        mean_smoothness, mean_compactness, mean_concavity
    ]], columns=[
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity'
    ])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    # 0 = malignant, 1 = benign
    result_label = "BENIGN (Non-Cancerous / Safe) ✅" if prediction == 1 else "MALIGNANT (Cancerous / High Risk) ⚠️"

    print("\n==========================================================")
    print("                 📊 DIAGNOSTIC REPORT                     ")
    print("==========================================================")
    print(f"• AI Prediction Result : {result_label}")
    print(f"• Model Confidence     : {confidence:.1f}%")
    print("==========================================================\n")

if __name__ == "__main__":
    run_diagnostic_dashboard()