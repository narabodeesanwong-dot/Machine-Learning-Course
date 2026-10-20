import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_features(X_train, X_test, output_dir='outputs'):
    # เรียกใช้ StandardScaler เพื่อปรับให้ข้อมูลทุกคอลัมน์มีสเกลเดียวกัน (Mean=0, SD=1)
    scaler = StandardScaler()
    
    # ชุด Train ใช้ fit_transform เพื่อให้โมเดลเรียนรู้สเกลและแปลงค่า
    X_train_scaled = scaler.fit_transform(X_train)
    
    # **จุดสำคัญ:** ชุด Test ใช้แค่ transform เท่านั้น ห้ามใช้ fit เด็ดขาด เพื่อป้องกัน Data Leakage
    X_test_scaled = scaler.transform(X_test)
    
    # บันทึกตัวปรับสเกล (Scaler) เก็บไว้ เพื่อนำไปใช้กับข้อมูลใหม่ตอนนำระบบไปใช้งานจริง
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, scaler
