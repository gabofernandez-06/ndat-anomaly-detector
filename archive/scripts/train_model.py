"""
train_model.py — trains a binary (NORMAL vs ATTACK) logistic regression
classifier on training_data.csv, using the 4 candidate features:
avg_size, size_variance, timing_variance, port_count.

Uses class weighting to handle the NORMAL/ATTACK imbalance, and a held-out
test set to check real generalization, not just training-set memorization.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_FILE = "training_data.csv"
MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"

FEATURE_COLS = ["avg_size", "size_variance", "timing_variance", "port_count"]

def main():
    df = pd.read_csv(DATA_FILE)

    # Binary label: 0 = NORMAL, 1 = any attack type
    df["binary_label"] = (df["label"] != "NORMAL").astype(int)

    X = df[FEATURE_COLS].values
    y = df["binary_label"].values

    print(f"Total rows: {len(df)}")
    print(f"NORMAL: {(y == 0).sum()}  |  ATTACK: {(y == 1).sum()}")
    print()

    # Stratified split keeps the same NORMAL/ATTACK ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features — fit scaler on TRAIN only, apply to both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # class_weight='balanced' automatically up-weights the minority class
    # (ATTACK) so the model can't just learn to always predict NORMAL
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("=== Test set results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "ATTACK"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("=== Learned weights (on standardized features) ===")
    for name, coef in zip(FEATURE_COLS, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {model.intercept_[0]:.4f}")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print()
    print(f"Saved model to {MODEL_FILE}, scaler to {SCALER_FILE}")

if __name__ == "__main__":
    main()
