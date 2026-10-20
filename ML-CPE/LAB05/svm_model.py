from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train, kernel='rbf', C=1.0, degree=3, gamma='scale'):
    # สร้างโมเดล Support Vector Machine (SVC) โดยเปิดให้ตั้งค่า พารามิเตอร์ต่างๆ ได้จากภายนอก
    model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, random_state=42)
    
    # เริ่มการสอน (Train) โมเดลด้วยข้อมูล X_train และ y_train
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='outputs/svm_model.pkl'):
    # ฟังก์ชันสำหรับบันทึกโมเดลที่ผ่านการเทรนแล้วให้อยู่ในรูปแบบไฟล์ .pkl
    joblib.dump(model, filepath)
