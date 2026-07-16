"""
CNN-based face classifier.

Each registered user is one softmax class (label_id from the users table).
Training data lives on disk at dataset/<label_id>/*.png, written during
registration capture. Training rebuilds the model from scratch over all
classes whenever a new user is added — simplest correct approach for a
small (classroom-sized) roster.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "face_cnn.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

IMG_SIZE = 100


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _load_dataset():
    """Reads every dataset/<label_id>/*.png into (X, y)."""
    X, y = [], []
    if not os.path.isdir(DATASET_DIR):
        return np.array(X), np.array(y)
    for label_folder in sorted(os.listdir(DATASET_DIR), key=lambda s: int(s) if s.isdigit() else -1):
        folder_path = os.path.join(DATASET_DIR, label_folder)
        if not os.path.isdir(folder_path) or not label_folder.isdigit():
            continue
        label_id = int(label_folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(folder_path, fname)
            img = tf.keras.utils.load_img(img_path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
            arr = tf.keras.utils.img_to_array(img)
            X.append(arr)
            y.append(label_id)
    return np.array(X), np.array(y)


def train_model(min_samples_per_class=15, epochs=25):
    """Retrains the CNN over the full dataset. Returns a dict with training stats."""
    X, y = _load_dataset()
    if len(X) == 0:
        return {"success": False, "error": "No training images found. Register at least one user first."}

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"success": False, "error": "Need at least 2 registered users before training (softmax needs 2+ classes)."}
    if counts.min() < min_samples_per_class:
        return {"success": False, "error": f"Every user needs at least {min_samples_per_class} captured samples."}

    # remap arbitrary label_ids -> contiguous 0..N-1 indices for the softmax head
    label_ids_sorted = sorted(classes.tolist())
    label_to_index = {lbl: i for i, lbl in enumerate(label_ids_sorted)}
    y_mapped = np.array([label_to_index[v] for v in y])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, brightness_range=(0.8, 1.2), horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_model(num_classes=len(label_ids_sorted))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    )
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop],
        verbose=0
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_to_index, f)  # label_id (db) -> softmax index

    val_acc = float(history.history["val_accuracy"][-1])
    return {
        "success": True,
        "num_classes": len(label_ids_sorted),
        "num_samples": len(X),
        "val_accuracy": round(val_acc, 4)
    }


class FaceRecognizer:
    """Lazily loads the trained model once and reuses it across requests."""
    _model = None
    _index_to_label = None

    @classmethod
    def _ensure_loaded(cls):
        if cls._model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model not trained yet. Train the model from the Register page first.")
            cls._model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, "rb") as f:
                label_to_index = pickle.load(f)
            cls._index_to_label = {v: k for k, v in label_to_index.items()}

    @classmethod
    def reload(cls):
        cls._model = None
        cls._index_to_label = None

    @classmethod
    def predict(cls, face_array):
        """face_array: (IMG_SIZE, IMG_SIZE) uint8 grayscale. Returns (label_id, confidence)."""
        cls._ensure_loaded()
        x = face_array.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
        probs = cls._model.predict(x, verbose=0)[0]
        best_index = int(np.argmax(probs))
        confidence = float(probs[best_index])
        label_id = cls._index_to_label[best_index]
        return label_id, confidence
