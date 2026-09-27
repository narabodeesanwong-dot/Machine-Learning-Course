import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_data
from split_data import split_dataset
from preprocessing import preprocess_for_cnn

def test_predictions():
    # Load model and data
    model = load_model('outputs/cnn_model.keras')
    # แก้ไขพาธไฟล์ตรงนี้เช่นเดียวกัน
    X, y, _ = load_data('winequality-red.csv.csv')
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    _, X_test_cnn, _ = preprocess_for_cnn(X_train, X_test)
    
    # 7. Predictions on selected dataset
    print("\n--- Making Predictions on 5 Random Samples ---")
    random_indices = np.random.choice(len(X_test_cnn), 5, replace=False)
    
    for idx in random_indices:
        sample = np.expand_dims(X_test_cnn[idx], axis=0)
        true_label = y_test[idx]
        
        prediction_probs = model.predict(sample, verbose=0)
        predicted_label = np.argmax(prediction_probs)
        
        print(f"Sample {idx}: True Label (Mapped) = {true_label}, Predicted = {predicted_label}")

if __name__ == "__main__":
    test_predictions()