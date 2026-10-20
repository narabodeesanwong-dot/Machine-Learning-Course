import joblib
import numpy as np

def test_inference():
    # โหลดไฟล์ Scaler และ Model ที่ถูกเซฟไว้ขึ้นมาเตรียมใช้งานจริง
    scaler = joblib.load('outputs/scaler.pkl')
    model = joblib.load('outputs/svm_model.pkl')
    
    # จำลองข้อมูลไวน์ใหม่ 1 ขวด (ประกอบด้วยค่าสารเคมี 11 ชนิด)
    sample = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
    
    # **จุดสำคัญ:** ต้องเอาข้อมูลใหม่ไปผ่าน Scaler ตัวเดิมก่อน เพื่อให้สเกลตรงกับตอนที่เทรนโมเดล
    sample_scaled = scaler.transform(sample)
    
    # สั่งให้โมเดลทำนายระดับคุณภาพไวน์
    prediction = model.predict(sample_scaled)
    print(f"Sample Prediction (Quality): {prediction[0]}")

if __name__ == '__main__':
    test_inference()
