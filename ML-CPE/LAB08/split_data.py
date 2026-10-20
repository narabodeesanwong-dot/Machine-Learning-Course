import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save_data(X, y, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # แบ่ง Train 70% และ Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # แบ่ง Temp ออกเป็น Validation 15% และ Test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # บันทึกไฟล์ลงในโฟลเดอร์ outputs/ ตามโครงสร้างตัวอย่าง
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"[Split Data] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test