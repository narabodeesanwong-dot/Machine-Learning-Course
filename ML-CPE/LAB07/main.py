from data_loader import load_data
from split_data import split_dataset
from preprocessing import preprocess_for_cnn
from cnn_model import build_cnn
from evaluate import plot_history

def main():
    # 1. Select and load dataset
    print("Loading data...")
    # แก้ไขพาธไฟล์ตรงนี้ จาก '../winequality-red.csv.csv' เป็น 'winequality-red.csv.csv'
    X, y, num_classes = load_data('winequality-red.csv.csv')
    
    # 2. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 3. Standardize features
    X_train_cnn, X_test_cnn, scaler = preprocess_for_cnn(X_train, X_test)
    input_shape = (X_train_cnn.shape[1], 1)
    
    # กำหนดค่าที่ต้องการเปรียบเทียบ
    configs = ['basic', 'deep']
    epochs_list = [10, 30] 
    
    results = {}

    for config in configs:
        for epochs in epochs_list:
            print(f"\n--- Training CNN: Config='{config}', Epochs={epochs} ---")
            
            # 4. Build CNN
            model = build_cnn(input_shape, num_classes, config_type=config)
            
            # 5. Train CNN
            history = model.fit(X_train_cnn, y_train, 
                                epochs=epochs, 
                                validation_data=(X_test_cnn, y_test),
                                verbose=0)
            
            # 6. Evaluate
            test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
            print(f"Accuracy for {config} ({epochs} epochs): {test_acc:.4f}")
            
            # Save metrics & plots
            plot_name = f"history_{config}_{epochs}_epochs"
            plot_history(history, f"Config: {config} | Epochs: {epochs}", plot_name)
            
            results[f"{config}_{epochs}"] = test_acc
            
            # Save model for testing
            if config == 'deep' and epochs == 30:
                model.save('outputs/cnn_model.keras')

    # Output สรุปผล
    print("\n=== Final Accuracy Comparison ===")
    for key, acc in results.items():
        print(f"Model {key}: {acc:.4f}")

if __name__ == "__main__":
    main()