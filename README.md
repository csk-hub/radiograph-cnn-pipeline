# radiograph-cnn-pipeline
A TensorFlow/Keras-based Convolutional Neural Network (CNN) model for binary classification of bone fractures from multi-region X-ray images, achieving 95.65% test accuracy.
# Bone Fracture Binary Classification using CNN

This project implements a Convolutional Neural Network (CNN) in Python using TensorFlow/Keras to perform binary classification on X-ray images to detect bone fractures (`fractured` vs. `not fractured`).

## 📌 Features
- **Data Augmentation:** Applies rotation, zoom, and horizontal flipping to prevent overfitting.
- **Model Architecture:** Custom 3-layer CNN with Max-Pooling, Dense layers, and Dropout regularization.
- **Hardware Acceleration:** Configured with `tf.distribute.MirroredStrategy` for multi-GPU training.
- **Evaluation & Visualization:** Generates accuracy/loss curves, confusion matrices, classification reports, and random sample inference visualizations.

## 📊 Dataset
The dataset used is the **Bone Fracture Multi-Region X-Ray Data** from Kaggle:
- **Source:** `bmadushanirodrigo/fracture-multi-region-x-ray-data`
- **Classes:** `fractured`, `not fractured`
- **Splits:** Train (9,246 images), Validation (829 images), Test (506 images)

## 🛠️ Model Architecture & Training Settings
- **Input Size:** 150x150x3
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Epochs:** 15
- **Test Accuracy:** ~95.65%

## 🚀 Getting Started

### Prerequisites
Ensure you have the required libraries installed:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow kagglehub
