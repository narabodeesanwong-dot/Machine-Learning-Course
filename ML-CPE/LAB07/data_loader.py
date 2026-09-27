import pandas as pd
import numpy as np

def load_data(filepath):
    # อ่านไฟล์ข้อมูล
    df = pd.read_csv(filepath)
    
    # แยก X และ y
    X = df.drop('quality', axis=1).values
    y_raw = df['quality'].values
    
    # ปรับค่า y ให้เริ่มจาก 0 (เช่น คุณภาพ 3-8 จะกลายเป็น 0-5)
    y = y_raw - np.min(y_raw)
    num_classes = len(np.unique(y))
    
    return X, y, num_classes