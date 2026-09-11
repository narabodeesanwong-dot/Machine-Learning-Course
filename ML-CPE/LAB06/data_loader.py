import pandas as pd

def load_data(filepath):
    # Select and load a dataset
    df = pd.read_csv(filepath)
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    
    # ปรับค่า y ให้เริ่มจาก 0 (เนื่องจากคุณภาพไวน์ในชุดข้อมูลนี้มีค่าเริ่มที่ 3 ถึง 8)
    # เพื่อให้เข้ากับ Loss Function ของ Neural Network
    y_min = y.min()
    y_shifted = y - y_min
    num_classes = len(set(y_shifted))
    
    return X, y_shifted, y_min, num_classes