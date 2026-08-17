import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # 1. โหลดข้อมูล
    df = pd.read_csv(filepath)
    
    # 2. แยก Features (X) และ Target (y)
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # 3. แบ่งข้อมูล Train (80%) และ Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Standardize input features ก่อนเทรนโมเดล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test