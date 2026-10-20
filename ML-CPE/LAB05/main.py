import os
from data_loader import load_data
from split_data import prepare_and_split
from preprocessing import preprocess_features
from svm_model import train_svm, save_model
from evaluate import evaluate_model

def run_pipeline():
    # สร้างโฟลเดอร์ outputs อัตโนมัติ (ถ้ายังไม่มี) เพื่อเก็บผลลัพธ์
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Load Data: เรียกใช้งานฟังก์ชันโหลดไฟล์ข้อมูล
    df = load_data('winequality-red.csv')
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Split Data: นำข้อมูลมาแบ่งเป็นชุด Train และ Test
    X_train, X_test, y_train, y_test = prepare_and_split(df)
    
    # 3. Standardize Features: ปรับสเกลข้อมูลให้เป็นมาตรฐานเดียวกัน
    X_train_scaled, X_test_scaled, _ = preprocess_features(X_train, X_test)
    
    # 4. Train & Compare Kernels: กระบวนการคัดเลือกโมเดลที่ดีที่สุดแบบอัตโนมัติ
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    best_model = None
    best_acc = 0.0
    best_kernel = ""
    
    # นำ Kernel ทั้ง 3 ชนิด มาวนลูปเทรนและทดสอบแข่งกัน
    for k in kernels:
        model = train_svm(X_train_scaled, y_train, kernel=k)
        acc, _ = evaluate_model(model, X_test_scaled, y_test, kernel_name=k)
        results[k] = acc # เก็บค่าความแม่นยำของแต่ละตัวไว้
        
        # ถ้ารอบนี้คะแนนดีกว่ารอบก่อนหน้า ให้บันทึกตัวนี้เป็น "โมเดลที่ดีที่สุด (Best Model)"
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_kernel = k
            
    # เมื่อจบลูป จะทำการบันทึกเฉพาะโมเดลที่ดีที่สุด (ผู้ชนะ) และสร้างกราฟ Confusion Matrix ของตัวที่ชนะ
    save_model(best_model, 'outputs/svm_model.pkl')
    evaluate_model(best_model, X_test_scaled, y_test, kernel_name=best_kernel, save_plot_path='outputs/confusion_matrix.png')
    
    # สรุปผลลัพธ์การแข่งขันท้ายสุด
    print("--- Summary Results ---")
    for k, acc in results.items():
        print(f"SVM ({k.capitalize()} Kernel) Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    run_pipeline()
