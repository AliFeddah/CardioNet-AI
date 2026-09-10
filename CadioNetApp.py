import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# ============================================================
# 1. Load model and scaler (from local files)
# ============================================================
@st.cache_resource
def load_assets():
    model = load_model('ecg_cnn_model.h5')
    scaler = joblib.load('scaler_cnn.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Make sure 'ecg_cnn_model.h5' and 'scaler_cnn.pkl' are in the same folder as app.py")
    st.stop()

# ============================================================
# 2. Streamlit UI
# ============================================================
st.set_page_config(
    page_title="CardioNet-AI",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 CardioNet-AI")
st.markdown("### Deep Learning ECG Arrhythmia Classifier")
st.markdown(
    """
    Upload a CSV file containing 187 time points to classify the heartbeat 
    into one of 5 categories (AAMI standard).
    
    Model: 1D Convolutional Neural Network (98.34% accuracy)
    """
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        
        if data.shape[1] != 187:
            st.error(f"File has {data.shape[1]} columns. Expected 187.")
        else:
            scaled = scaler.transform(data.values)
            scaled_cnn = scaled.reshape((scaled.shape[0], 187, 1))
            
            with st.spinner("Analyzing ECG signal..."):
                probs = model.predict(scaled_cnn, verbose=0)
            
            pred = int(np.argmax(probs[0]))
            labels = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']
            result = labels[pred]
            confidence = probs[0][pred] * 100
            
            # Result
            if result == 'Normal':
                st.success(f"✅ Prediction: {result} ({confidence:.2f}%)")
            else:
                st.error(f"⚠️ Prediction: {result} ({confidence:.2f}%)")
            
            # Probabilities
            with st.expander("📊 Show class probabilities"):
                for i, label in enumerate(labels):
                    st.write(f"{label}: {probs[0][i]*100:.2f}%")
            
            # Signal
            with st.expander("📈 Show ECG signal"):
                st.line_chart(data.values[0])
                
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("⚠️ Educational project only. Not for clinical use.")