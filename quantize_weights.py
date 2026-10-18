import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def float_to_fp8_e4m3(val):
    """
    Simulates conversion of a floating-point weight into an FP8 E4M3 format 
    (1 sign bit, 4 exponent bits, 3 mantissa bits) as defined in the CIM architecture[cite: 4, 5].
    """
    if val == 0.0:
        return 0, 0, 0
    
    # 1. Extract Sign
    sign = 1 if val < 0 else 0
    val = abs(val)
    
    # 2. Extract Exponent and Mantissa (E4M3 format: bias = 7)
    bias = 7
    exponent = int(np.floor(np.log2(val))) if val > 0 else -bias
    exponent = max(-7, min(8, exponent)) # E4M3 range clamp
    
    scaled_val = val / (2 ** exponent)
    # 3-bit mantissa scaling (values between 1.0 and 1.875)
    mantissa = int(round((scaled_val - 1.0) * 8))
    mantissa = max(0, min(7, mantissa))
    
    return sign, exponent + bias, mantissa

def fp8_e4m3_to_float(sign, encoded_exp, mantissa):
    """Reconstructs float value from FP8 E4M3 representation."""
    bias = 7
    exponent = encoded_exp - bias
    val = (1.0 + (mantissa / 8.0)) * (2 ** exponent)
    return -val if sign == 1 else val

def main():
    # Load dataset and model
    df = pd.read_csv("training_data.csv")
    y = (df["label"] == "ATTACK_CANDIDATE").astype(int)
    X = df[["avg_size", "size_variance", "timing_variance", "port_count"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")
    
    print("=== Original Scikit-Learn Weights ===")
    feature_names = ["avg_size", "size_variance", "timing_variance", "port_count"]
    orig_weights = model.coef_[0]
    intercept = model.intercept_[0]
    
    for name, w in zip(feature_names, orig_weights):
        print(f"  {name}: {w:.4f}")
    print(f"  Intercept: {intercept:.4f}\n")
    
    print("=== Quantized FP8 (E4M3) Hardware Mapping ===")
    quantized_weights = []
    for name, w in zip(feature_names, orig_weights):
        s, e, m = float_to_fp8_e4m3(w)
        reconstructed = fp8_e4m3_to_float(s, e, m)
        quantized_weights.append(reconstructed)
        print(f"  {name:15} -> Sign: {s}, Exp(4b): {e:04b}, Mantissa(3b): {m:03b} | Quantized Val: {reconstructed:8.4f}")
        
    # Simulate inference using quantized weights and compare against the float baseline
    X_test_scaled = scaler.transform(X_test)
    
    # Standard floating-point prediction
    orig_preds = model.predict(X_test_scaled)
    
    # Quantized hardware-emulated prediction
    quantized_preds = []
    for row in X_test_scaled:
        score = np.dot(row, quantized_weights) + intercept
        pred = 1 if score >= 0 else 0
        quantized_preds.append(pred)
        
    orig_acc = accuracy_score(y_test, orig_preds)
    quant_acc = accuracy_score(y_test, quantized_preds)
    
    print(f"\n=== Validation Results ===")
    print(f"Original Model Accuracy:     {orig_acc:.4f}")
    print(f"Quantized FP8 MAC Accuracy:  {quant_acc:.4f}")
    print("Quantized-weight inference complete. Compare recall/PR-AUC against the float baseline.")

if __name__ == "__main__":
    main()
