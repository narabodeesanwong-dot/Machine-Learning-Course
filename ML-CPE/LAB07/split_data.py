from sklearn.model_selection import train_test_split

def split_dataset(X, y):
    # แบ่งข้อมูล Train 80% และ Test 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test