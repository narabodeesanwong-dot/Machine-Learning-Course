import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, history, class_names, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ทำนายผลบนชุดข้อมูล Test
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Evaluate] Test Accuracy: {acc:.4f}")
    print("\n[Evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # 1. สร้างและบันทึก Confusion Matrix (confusion_matrix.png)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Wine Quality Classification (VGG-1D)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 2. สร้างและบันทึกกราฟประวัติการเทรน (training_history.png)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close()

    return acc