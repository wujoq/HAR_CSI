import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Data loading and preprocessing ---

def parse_amp_phase_cell(cell_str):
    """Parse string like '(1.2345,0.9876)' into tuple of floats."""
    try:
        amp, phase = cell_str.strip("()").split(",")
        return float(amp), float(phase)
    except:
        return 0.0, 0.0  # fallback in case of missing or malformed values

def load_x_data(x_folder):
    x_data = []
    file_names = sorted(os.listdir(x_folder))
    for fname in file_names:
        fpath = os.path.join(x_folder, fname)
        df = pd.read_csv(fpath, header=0)
        # df shape: (subcarriers, packets) with string tuples
        arr = np.empty((df.shape[0], df.shape[1], 2), dtype=np.float32)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                amp, phase = parse_amp_phase_cell(df.iloc[i, j])
                arr[i, j, 0] = amp
                arr[i, j, 1] = phase
        x_data.append(arr)
    return np.array(x_data)  # shape: (samples, subcarriers, packets, 2)

def load_y_data(y_folder):
    y_data = []
    file_names = sorted(os.listdir(y_folder))
    for fname in file_names:
        fpath = os.path.join(y_folder, fname)
        with open(fpath, 'r') as f:
            label = f.readline().strip()
            y_data.append(label)
    return np.array(y_data)

# Paths
x_folder = 'dataset_phase/x'
y_folder = 'dataset_phase/y'

# Load data
X = load_x_data(x_folder)  # shape: (samples, subcarriers, packets, 2)
y = load_y_data(y_folder)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Expand dims if needed (Conv2D expects 4D input: (samples, height, width, channels))
# Our shape: (samples, subcarriers, packets, 2) -> already OK for Conv2D
print(f"X shape: {X.shape}")

# --- Split data ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_categorical, test_size=0.10, random_state=42, stratify=y_categorical
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
)

# --- Model definition ---
number_of_classes = y_categorical.shape[1]

inputs = Input(shape=(X.shape[1], X.shape[2], 2))  # 2 channels: amplitude + phase
x = Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(number_of_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Training ---
checkpoint = ModelCheckpoint(
    'best_model_amp_phase.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# --- Plotting ---
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

# --- Test evaluation ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
