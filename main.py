import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout


# CSI data, packet, subcarrier, amplitude
data = ...


# Input layer !!Make sure the shape is OK!!

inputs = Input(shape=(data.shape[1], data.shape[2], 1))
number_of_classes = 4 # standing, lying, walking, sitting on a chair 
# First layer 2D convolution
x = TimeDistributed(Conv2D(32, (3, 1), activation='relu', padding="same"))(input)
x = TimeDistributed(MaxPooling2D(2,1))(x)
x = TimeDistributed(Conv2D(64, (3, 1), activation='relu', padding="same"))(x)
x = TimeDistributed(MaxPooling2D(2,1))(x)

x = TimeDistributed(Flatten())(x)  

# LSTM do przetwarzania sekwencji
x = LSTM(128)(x)
x = Dropout(0.5)(x)

outputs = Dense(number_of_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
