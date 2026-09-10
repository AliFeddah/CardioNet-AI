# 🫀 CardioNet-AI

An end-to-end Deep Learning system for automated ECG arrhythmia classification, deployed as an interactive Streamlit web application.

> Accuracy: 98.34% on the MIT-BIH Arrhythmia test dataset.

---

## 📌 Overview

CardioNet-AI uses a custom 1D Convolutional Neural Network (1D-CNN) to classify ECG heartbeats into 5 categories based on the AAMI standard. The model is trained on the well-known MIT-BIH Arrhythmia Database, and deployed as a user-friendly web application where users can upload a CSV file containing 187 time points and receive an instant classification.

---

## 🎯 Key Features

- Deep Learning Model: Custom 1D-CNN architecture built with TensorFlow/Keras
- 5-Class Classification: Normal, SVEB, VEB, Fusion, Unknown (AAMI standard)
- Interactive Web App: Built with Streamlit for real-time predictions
- Confidence Scores: Displays prediction confidence and full class probability distribution
- Signal Visualization: Renders the uploaded ECG signal for visual inspection
- Lightweight Deployment: Model and scaler bundled directly with the app

---

## 🧠 Model Architecture

The 1D-CNN consists of:

- 3 Convolutional Blocks:
  - Block 1: Conv1D (32 filters, kernel=5) + BatchNorm + MaxPool
  - Block 2: Conv1D (64 filters, kernel=5) + BatchNorm + MaxPool
  - Block 3: Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool
- Global Average Pooling layer
- Dense head: 128 → Dropout(0.5) → 64 → Dropout(0.3) → 5 (Softmax)

Training details:
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch size: 64
- Epochs: 10

---

## 📊 Dataset

- Source: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- Samples: ~100,000 heartbeats (87,554 training / 21,892 testing)
- Features: 187 time points per heartbeat
- Classes: 5 (AAMI standard)

---

## 📈 Results

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Normal   | 0.97      | 1.00   | 0.99     | 18,118  |
| SVEB     | 0.99      | 0.61   | 0.75     | 556     |
| VEB      | 0.98      | 0.88   | 0.93     | 1,448   |
| Fusion   | 0.88      | 0.64   | 0.74     | 162     |
| Unknown  | 0.99      | 0.94   | 0.97     | 1,608   |
| Overall Accuracy | | | 98.34% | 21,892 |

> Note: Lower scores for SVEB and Fusion are due to severe class imbalance and morphological similarity to normal beats — a well-known challenge in the MIT-BIH database.

---

## 🚀 How to Run Locally

### 1. Clone the repository

`bash
git clone https://github.com/AliFeddah/CardioNet-AI.git
cd CardioNet-AI