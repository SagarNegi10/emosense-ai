import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json

# ==============================
# CPU Thread Optimization (Ryzen 5600H - 12 threads)
# ==============================
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

# ==============================
# Config
# ==============================
IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 40
DATA_DIR = "dataset"
if not Path(DATA_DIR).exists():
    DATA_DIR = "dataset"

tf.keras.utils.set_random_seed(42)

# ==============================
# Load Dataset with Auto Split
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)
num_classes = len(class_names)

# Save class order used for training so inference labels stay aligned.
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)
print("Saved class label mapping to class_names.json")

# Compute class weights from the training split to reduce impact of imbalance.
train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
class_counts = np.bincount(train_labels, minlength=num_classes)
class_weight = {
    i: float(train_labels.shape[0] / (num_classes * class_counts[i]))
    for i in range(num_classes)
}
print("Class counts (train):", class_counts.tolist())
print("Class weights:", class_weight)

# ==============================
# Normalization
# ==============================
normalization_layer = layers.Rescaling(1./255)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.12),
    ],
    name="augmentation",
)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.cache().shuffle(20000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==============================
# Model Architecture
# ==============================
model = keras.Sequential([
    layers.Input(shape=(48,48,1)),
    data_augmentation,

    layers.Conv2D(32, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(32, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(64, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(128, 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ==============================
# Callbacks
# ==============================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    mode="max",
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "emotion_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# Train
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

print("Training Complete. Best model saved as emotion_model.keras")

# ==============================
# Validation Diagnostics (TTA + Confusion Matrix + Per-class Metrics)
# ==============================
all_y_true = []
all_y_pred = []

for batch_x, batch_y in val_ds:
    # Test-time augmentation: average predictions from original + horizontal flip.
    pred_original = model(batch_x, training=False)
    pred_flipped = model(tf.image.flip_left_right(batch_x), training=False)
    pred_avg = (pred_original + pred_flipped) / 2.0

    all_y_true.extend(batch_y.numpy().tolist())
    all_y_pred.extend(tf.argmax(pred_avg, axis=1).numpy().tolist())

all_y_true = np.array(all_y_true, dtype=np.int32)
all_y_pred = np.array(all_y_pred, dtype=np.int32)

cm = tf.math.confusion_matrix(
    all_y_true, all_y_pred, num_classes=num_classes
).numpy()

tp = np.diag(cm)
pred_pos = np.sum(cm, axis=0)
actual_pos = np.sum(cm, axis=1)

precision = tp / np.maximum(pred_pos, 1)
recall = tp / np.maximum(actual_pos, 1)
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)

print("\nValidation per-class metrics:")
for i, name in enumerate(class_names):
    print(
        f"{name:>10s} | "
        f"precision: {precision[i]:.4f} | "
        f"recall: {recall[i]:.4f} | "
        f"f1: {f1[i]:.4f} | "
        f"support: {actual_pos[i]}"
    )

print(f"\nMacro F1: {np.mean(f1):.4f}")

# ==============================
# Plot Accuracy + Confusion Matrix
# ==============================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Validation Confusion Matrix (with TTA predictions)")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

threshold = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j, i, str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()
plt.show()