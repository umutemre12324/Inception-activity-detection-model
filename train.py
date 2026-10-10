
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from model import build_inception_model

# Verileri yükle
x_train_scaled = np.load("x_train_scaled.npy")
y_train_to = np.load("y_train_to.npy")
x_test_scaled = np.load("x_test_scaled.npy")
y_test_to = np.load("y_test_to.npy")

input_shape = x_train_scaled.shape[1:]
num_classes = y_train_to.shape[1]

# Eğitim verisinden doğrulama verisi ayır
x_train, x_val, y_train, y_val = train_test_split(
    x_train_scaled,
    y_train_to,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train_to, axis=1)
)

# Modeli oluştur
model = build_inception_model(input_shape, num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callback'ler
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Tek seferde eğitim
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# En iyi modeli yükle
model = tf.keras.models.load_model("best_model.keras")

# Test kümesinde son değerlendirme
test_loss, test_accuracy = model.evaluate(
    x_test_scaled, y_test_to, verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Eğitim grafikleri
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
