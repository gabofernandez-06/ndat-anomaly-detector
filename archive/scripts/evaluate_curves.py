import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def main():
    # Load dataset
    df = pd.read_csv("training_data.csv")
    
    # Convert labels to binary (0 = NORMAL, 1 = ATTACK_CANDIDATE)
    y = (df["label"] == "ATTACK_CANDIDATE").astype(int)
    X = df[["avg_size", "size_variance", "timing_variance", "port_count"]]
    
    # Use the same train/test split configuration as train_model.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the saved scaler and model
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")
    
    # Transform test features
    X_test_scaled = scaler.transform(X_test)
    
    # Get predicted probabilities for the attack class
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    
    print(f"=== Model Evaluation Metrics ===")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    
    # Plotting the curves
    plt.figure(figsize=(12, 5))
    
    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR-AUC)')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig("evaluation_curves.png", dpi=300)
    print("Saved high-resolution plots to evaluation_curves.png")
    plt.show()

if __name__ == "__main__":
    main()
