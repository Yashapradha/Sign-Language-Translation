import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data=pd.read_csv("keypoint.csv",header=None)
data=data.iloc[:,0:43]
data=data.dropna()

class_label_count = data[0].nunique()
print("Number of unique class labels:", class_label_count)

X=data.iloc[:,1:]
y=data.iloc[:,0]



# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling


# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=50, activation='relu', input_shape=(42,)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=class_label_count, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Evaluate the model on the test set
_, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)


model.save("keypoint1.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

with open("keypoint1.tflite","wb") as f:
    f.write(tflite_model)