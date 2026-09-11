import matplotlib.pyplot as plt
import pandas as pd

def plot_history(history, title_suffix, filename="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training and validation accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'Accuracy {title_suffix}')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    
    # Training and validation loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'Loss {title_suffix}')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    
    plt.savefig(filename)
    plt.close()

def show_results(results_list):
    df = pd.DataFrame(results_list)
    print("\n--- Accuracy scores for each NN configuration ---")
    print(df.to_string(index=False)) # <--- ใช้ to_string หรือ print(df) ธรรมดา