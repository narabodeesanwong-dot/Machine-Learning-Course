import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_features(X_train, X_test, output_dir='outputs'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    return X_train_scaled, X_test_scaled, scaler