from sklearn.preprocessing import StandardScaler

def preprocess_for_cnn(X_train, X_test):
    scaler = StandardScaler()
    # Standardize input features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape สำหรับ Conv1D -> (จำนวนข้อมูล, จำนวนฟีเจอร์, 1)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return X_train_reshaped, X_test_reshaped, scaler