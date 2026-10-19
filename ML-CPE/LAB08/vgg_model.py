import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg_dcnn(input_shape, num_classes, config_type="vgg_3block"):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    if config_type == "config1_1conv":
        # Config 1: 1 Convolutional Layer (32 filters) + 64 Dense Neurons
        model.add(layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.3))

    elif config_type == "config2_2conv":
        # Config 2: 2 Convolutional Layers (64, 128 filters) + 128 Dense Neurons
        model.add(layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))

    elif config_type == "vgg_3block":
        # Config 3 (VGG-Style): 3 Convolutional Blocks (64, 128, 256 filters) + 256 Dense Neurons
        model.add(layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(128, activation="relu"))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_save_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, output_dir="outputs", save_files=True):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    if save_files:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "vgg_model.keras")
        history_path = os.path.join(output_dir, "history.json")

        model.save(model_path)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=4)
        print(f"[VGG Model] Saved model to {model_path} and history to {history_path}")

    return history.history