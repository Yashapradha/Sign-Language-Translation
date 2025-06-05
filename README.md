---
# ✋ Hand Gesture Recognition Using MediaPipe

Estimate hand poses and recognize finger gestures using **MediaPipe** in Python. This project utilizes a simple Multi-Layer Perceptron (MLP) to classify hand signs based on detected key points.

> 🎯 Ideal for researchers, developers, and hobbyists working on real-time hand gesture control and recognition systems.
---

## 📁 Repository Contents

This repository includes:

- ✅ Sample program for gesture recognition
- ✅ Pre-trained TFLite model for hand sign recognition
- ✅ Labeled dataset for training hand sign recognition model
- ✅ Jupyter notebook for training the model

---

## 📦 Requirements

To run this project, make sure the following dependencies are installed:

| Package         | Version      | Notes                                           |
| --------------- | ------------ | ----------------------------------------------- |
| `mediapipe`     | >= 0.8.1     | Hand landmark detection                         |
| `opencv-python` | >= 3.4.2     | Webcam and image processing                     |
| `tensorflow`    | >= 2.3.0     | For running the MLP model                       |
| `tf-nightly`    | >= 2.5.0.dev | _Only if creating a TFLite for an LSTM model_   |
| `scikit-learn`  | >= 0.23.2    | _Optional: for displaying the confusion matrix_ |
| `matplotlib`    | >= 3.3.2     | _Optional: for displaying the confusion matrix_ |

You can install them using:

```bash
pip install -r requirements.txt
```

_(Ensure to create a `requirements.txt` with the above versions if it doesn't exist.)_

---

## 🚀 Running the Demo

You can test the hand gesture recognition demo using your webcam:

```bash
python main.py
```

---

---

## 📌 Notes

- The project is designed to run in real time with decent performance on standard CPUs.
- This system currently focuses on static gesture recognition using 21 MediaPipe hand landmarks.
- You can extend this to dynamic gestures using LSTM or other sequence models.

---

## 📬 Feedback & Contributions

Feel free to fork the repo, improve it, or submit issues and pull requests. Contributions are always welcome!

---
