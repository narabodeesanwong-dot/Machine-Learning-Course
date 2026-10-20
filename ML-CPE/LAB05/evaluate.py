import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, kernel_name="", save_plot_path=None):
    # นำโมเดลที่เทรนแล้ว มาทำนาย (Predict) ข้อมูลชุด Test
    y_pred = model.predict(X_test)
    
    # คำนวณความแม่นยำ (Accuracy) 
    acc = accuracy_score(y_test, y_pred)
    
    # ปริ้นท์สรุปผลการทดสอบออกทางหน้าจอ
    print(f"=== Kernel: {kernel_name} ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # ถ้าระบุ path สำหรับเซฟรูปภาพ ระบบจะสร้างกราฟ Confusion Matrix อัตโนมัติ
    if save_plot_path:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        # ใช้ sns.heatmap วาดกราฟลงตาราง ใส่สี ฟ้า (Blues) และแสดงตัวเลข (annot=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({kernel_name} Kernel)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_plot_path) # เซฟรูปลงโฟลเดอร์
        plt.close()
        
    return acc, y_pred
