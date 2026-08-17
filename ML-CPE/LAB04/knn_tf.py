from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, k):
    # สร้างและเทรนโมเดล KNN ตามค่า k ที่ระบุ
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def predict_knn(model, X_test):
    # พยากรณ์ผลลัพธ์
    return model.predict(X_test)