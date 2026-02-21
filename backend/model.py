# ============================================================
# model.py – Train Emotion Model
# Output: emotion_model.hdf5
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Dataset Path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
train_dir = os.path.join(BASE_DIR, "dataset", "train")
test_dir = os.path.join(BASE_DIR, "dataset", "test")

# -----------------------------
# Image Generators
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    train_data,
    validation_data=test_data,
    epochs=25
)

# -----------------------------
# Save Model
# -----------------------------
save_path = os.path.join(os.path.dirname(__file__), "emotion_model.hdf5")
model.save(save_path)

print("Model saved at:", save_path)