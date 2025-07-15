import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Data loading and preprocessing ---

def load_x_data(x_folder):
    x_data = []
    file_names = sorted(os.listdir(x_folder))
    for fname in file_names:
        fpath = os.path.join(x_folder, fname)
        df = pd.read_csv(fpath, header=None)
        # Skip first line (packet_id), use the rest as subcarrier x amplitude matrix
        arr = df.iloc[1:].values.astype(np.float32)
        x_data.append(arr)
    return np.array(x_data)

def load_y_data(y_folder):
    y_data = []
    file_names = sorted(os.listdir(y_folder))
    for fname in file_names:
        fpath = os.path.join(y_folder, fname)
        # Assuming label is in the first line of each file
        with open(fpath, 'r') as f:
            label = f.readline().strip()
            y_data.append(label)
    return np.array(y_data)

# Paths
x_folder = 'dataset/x'
y_folder = 'dataset/y'

# Load data
X = load_x_data(x_folder)  # shape: (samples, subcarriers, amplitudes)
y = load_y_data(y_folder)  # shape: (samples,)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X for Conv2D: (samples, timesteps, subcarriers, amplitudes, 1)
# Here, treat each sample as a timestep=1 sequence
X = X[..., np.newaxis]  # (samples, subcarriers, amplitudes, 1)
X = np.expand_dims(X, axis=1)  # (samples, 1, subcarriers, amplitudes, 1)

# --- Split data ---

# Split: 70% train, 20% val, 10% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_categorical, test_size=0.10, random_state=42, stratify=y_categorical
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp
)
# 2/9 ≈ 0.222, so 20% of total data

# --- Model definition ---

number_of_classes = y_categorical.shape[1]

# inputs = Input(shape=(X.shape[1], X.shape[2], X.shape[3], 1))
# print(f"Input shape: {inputs}")
# x = TimeDistributed(Conv2D(32, (3, 1), activation='relu', padding="same"))(inputs)
# x = TimeDistributed(MaxPooling2D((2,1)))(x)
# x = TimeDistributed(Conv2D(64, (3, 1), activation='relu', padding="same"))(x)
# x = TimeDistributed(MaxPooling2D((2,1)))(x)
# x = TimeDistributed(Flatten())(x)
# x = TimeDistributed(Dense(128, activation='relu'))(x)
# x = Flatten()(x)
# x = Dropout(0.5)(x)
# outputs = Dense(number_of_classes, activation='softmax')(x)

# model = Model(inputs, outputs)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
X = X.squeeze(axis=1)
X_train = X_train.squeeze(axis=1)
X_val = X_val.squeeze(axis=1)
X_test = X_test.squeeze(axis=1)

inputs = Input(shape=(X.shape[1], X.shape[2], 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) 
x = MaxPooling2D((2, 2))(x)                                   
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(number_of_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# --- Training ---
model.fit(X_train, y_train, epochs=35, batch_size=16, validation_data=(X_val, y_val))

# --- Test evaluation ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
