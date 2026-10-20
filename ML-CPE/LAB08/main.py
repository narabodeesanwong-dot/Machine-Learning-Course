import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from data_loader import load_dataset
from preprocessing import preprocess_features
from split_data import split_and_save_data
from vgg_model import build_vgg_dcnn, train_and_save_model
from evaluate import evaluate_model
from test_vgg import test_random_samples

def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    output_dir = "outputs"
    csv_path = "winequality-red.csv.csv"

    # 1. Load dataset & sort classes
    X_raw, y_encoded, class_names, feature_names = load_dataset(csv_path, output_dir)

    # 2. Standardize input features before training
    X_processed, _ = preprocess_features(X_raw)

    # 3. Split dataset into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_data(
        X_processed, y_encoded, output_dir
    )

    input_shape = (X_train.shape[1], 1)
    num_classes = len(class_names)

    # 4. Compare different DCNN configurations (1 Conv, 2 Conv, 3 Conv VGG-style)
    print("\n" + "="*60)
    print("COMPARING DCNN CONFIGURATIONS (Fixed at 50 Epochs)")
    print("="*60)
    configs = ["config1_1conv", "config2_2conv", "vgg_3block"]
    config_scores = {}

    for cfg in configs:
        tf.random.set_seed(42)
        np.random.seed(42)
        is_main_vgg = (cfg == "vgg_3block")
        model = build_vgg_dcnn(input_shape, num_classes, config_type=cfg)
        history = train_and_save_model(
            model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=32, output_dir=output_dir, save_files=is_main_vgg
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        test_acc = accuracy_score(y_test, y_pred)
        config_scores[cfg] = {
            "train_acc": history["accuracy"][-1],
            "val_acc": history["val_accuracy"][-1],
            "test_acc": test_acc
        }
        if is_main_vgg:
            evaluate_model(model, X_test, y_test, history, class_names, output_dir)

    # 5. Compare different numbers of epochs (10, 30, 50 Epochs)
    print("\n" + "="*60)
    print("COMPARING DIFFERENT NUMBERS OF EPOCHS (VGG-3Block Model)")
    print("="*60)
    epoch_list = [10, 30, 50]
    epoch_scores = {}

    for ep in epoch_list:
        tf.random.set_seed(42)
        np.random.seed(42)
        model = build_vgg_dcnn(input_shape, num_classes, config_type="vgg_3block")
        history = train_and_save_model(
            model, X_train, y_train, X_val, y_val,
            epochs=ep, batch_size=32, output_dir=output_dir, save_files=False
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        epoch_scores[ep] = {
            "train_acc": history["accuracy"][-1],
            "val_acc": history["val_accuracy"][-1],
            "test_acc": accuracy_score(y_test, y_pred)
        }

    # 6. Test the trained VGG model using 4 random samples
    test_random_samples(output_dir=output_dir, feature_names=feature_names)

    # 7. Print Final Summary Output
    print("\n--- Summary 1: Accuracy Scores by DCNN Configuration ---")
    for cfg, sc in config_scores.items():
        print(f"{cfg:15s} | Train Acc: {sc['train_acc']:.4f} | Val Acc: {sc['val_acc']:.4f} | Test Acc: {sc['test_acc']:.4f}")

    print("\n--- Summary 2: Accuracy Comparison by Number of Epochs ---")
    for ep, sc in epoch_scores.items():
        print(f"Epochs = {ep:<4d}   | Train Acc: {sc['train_acc']:.4f} | Val Acc: {sc['val_acc']:.4f} | Test Acc: {sc['test_acc']:.4f}")

if __name__ == "__main__":
    main()