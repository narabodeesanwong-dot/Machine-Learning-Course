import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_features(X):
    # Standardize the input features before training (Mean = 0, Std = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape เป็น (จำนวนตัวอย่าง, 11 features, 1 channel) สำหรับ 1D-DCNN
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    print(f"[Preprocessing] Standardized & reshaped features to: {X_reshaped.shape}")
    return X_reshaped, scaler