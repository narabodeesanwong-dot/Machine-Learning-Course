import os
import pandas as pd
from data_loader import load_and_preprocess_data
from knn_tf import train_knn, predict_knn
from evaluate import evaluate_accuracy, plot_k_curve, plot_confusion_matrix

def main():
    # สร้างโฟลเดอร์ outputs หากยังไม่มี
    os.makedirs('outputs', exist_ok=True)
    
    # โหลดและ Preprocess ข้อมูล (ระบุ path ของไฟล์ให้ถูกต้อง)
    filepath = 'winequality-red.csv' 
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    k_values = [3, 5, 7]
    accuracies = []
    
    best_k = 0
    best_acc = 0
    best_pred = None
    
    print("--- KNN Classification Results ---")
    
    # เทรนและประเมินผลในแต่ละค่า k
    for k in k_values:
        model = train_knn(X_train, y_train, k)
        y_pred = predict_knn(model, X_test)
        
        acc = evaluate_accuracy(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy for k={k}: {acc:.4f}")
        
        # หาค่า K ที่ดีที่สุด
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_pred = y_pred
            
    print(f"\nBest k value: {best_k} with an accuracy of {best_acc:.4f}")
    
    # บันทึกผลลัพธ์เป็นภาพและ CSV ตามโครงสร้าง
    plot_k_curve(k_values, accuracies, 'outputs/01_k_curve.png')
    plot_confusion_matrix(y_test, best_pred, 'outputs/02_confusion_matrix.png', best_k)
    
    df_preds = pd.DataFrame({'Actual': y_test.values, 'Predicted': best_pred})
    df_preds.to_csv('outputs/predictions.csv', index=False)
    print("All output files saved to 'outputs/' directory.")

if __name__ == "__main__":
    main()