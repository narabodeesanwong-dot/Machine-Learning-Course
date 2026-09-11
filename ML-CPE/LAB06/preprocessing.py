from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test):
    # Standardize the input features before training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler