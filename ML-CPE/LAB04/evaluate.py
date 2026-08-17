from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def plot_k_curve(k_values, accuracies, output_path):
    plt.figure()
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('K-Value vs Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path, k):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (k={k})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.savefig(output_path)
    plt.close()