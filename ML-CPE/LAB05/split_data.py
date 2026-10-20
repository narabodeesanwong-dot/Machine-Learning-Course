import numpy as np
import json
from sklearn.model_selection import train_test_split

def prepare_and_split(df, target_col='quality', test_size=0.2, random_state=42, output_dir='outputs'):
    # แยก Features (X) โดยดึงทุกคอลัมน์ยกเว้น quality และแยก Label (y) โดยดึงแค่ quality
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # บันทึก Features และ Labels ทั้งหมดเป็นไฟล์ .npy เพื่อเก็บไว้ตรวจสอบย้อนหลัง
    np.save(f'{output_dir}/features.npy', X)
    np.save(f'{output_dir}/labels.npy', y)
    
    # ดึงคลาสคุณภาพไวน์ทั้งหมดที่ไม่ซ้ำกัน (เช่น 3, 4, 5, 6, 7, 8) บันทึกเป็นไฟล์ JSON
    classes = [int(c) for c in np.unique(y)]
    with open(f'{output_dir}/classes.json', 'w') as f:
        json.dump(classes, f)
        
    # แบ่งข้อมูลเป็นชุด Train 80% และ Test 20%
    # **จุดสำคัญ:** stratify=y คือการบังคับให้สัดส่วนคุณภาพไวน์กระจายตัวเท่ากัน ป้องกันโมเดลลำเอียง
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # บันทึกข้อมูลที่แบ่งเสร็จแล้วลงโฟลเดอร์ outputs เพื่อให้ไฟล์อื่นดึงไปใช้ได้ง่าย
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test
