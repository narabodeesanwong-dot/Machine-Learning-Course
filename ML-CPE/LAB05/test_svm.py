import joblib
import numpy as np

def test_inference():
    scaler = joblib.load('outputs/scaler.pkl')
    model = joblib.load('outputs/svm_model.pkl')
    
    # ตัวอย่างข้อมูลทดสอบ 1 sample (11 features)
    sample = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
    sample_scaled = scaler.transform(sample)
    
    prediction = model.predict(sample_scaled)
    print(f"Sample Prediction (Quality): {prediction[0]}")

if __name__ == '__main__':
    test_inference()