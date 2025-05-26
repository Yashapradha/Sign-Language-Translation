import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
import numpy as np

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and clean data
data = pd.read_csv("keypoint.csv", header=None)
data = data.iloc[:, 0:43].dropna()

# Encode labels to ensure 0-based indexing
le = LabelEncoder()
y = le.fit_transform(data[0])  # Transformed to 0-based labels
X = data.iloc[:, 1:]

# Number of classes
class_label_count = len(le.classes_)
print("Class count:", class_label_count)
print("Encoded labels:", sorted(set(y)))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=50, activation='relu', input_shape=(42,)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=class_label_count, activation='softmax')  # Final layer matches class count
])

# Compile with sparse categorical crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)


# Evaluate the model on the test set
_, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)


model.save("keypoint_classifier_new.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

with open("keypoint_classifier.tflite","wb") as f:
    f.write(tflite_model)