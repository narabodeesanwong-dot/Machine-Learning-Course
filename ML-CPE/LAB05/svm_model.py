from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train, kernel='rbf', C=1.0, degree=3, gamma='scale'):
    model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='outputs/svm_model.pkl'):
    joblib.dump(model, filepath)