
import streamlit as st
import numpy as np
import wfdb
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks

import tempfile              
import shutil                
import warnings            
import zipfile

# Optional (biar tampilan lebih clean)
warnings.filterwarnings("ignore")
plt.style.use("default")

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="🫀 Brugada Syndrome Detection",
    layout="centered",
)

# =============================
# LOAD MODEL
# =============================

@st.cache_resource
def load_model_from_zip(zip_path="model_brugada_1dcnn_saved.zip"):
    # Buat temporary folder
    temp_dir = tempfile.mkdtemp()
    
    # Unzip model ke temp_dir
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # folder SavedModel biasanya ada di dalam temp_dir/model_brugada_1dcnn_saved/
    saved_model_folder = temp_dir + "/model_brugada_1dcnn_saved"
    
    # Load model
    model = tf.keras.models.load_model(saved_model_folder)
    return model

model = load_model_from_zip()

# =============================
# BANDPASS FILTER
# =============================
def butter_bandpass_filter(data, fs, lowcut=0.5, highcut=40, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# =============================
# PREPROCESS FUNCTION
# =============================
def preprocess_ecg(record_path):
    target_leads = ['V1', 'V2', 'V3', 'II']

    try:
        rec = wfdb.rdrecord(record_path)
    except Exception as e:
        st.error(f"Failed to read WFDB file: {e}")
        return None, None

    sig = rec.p_signal

    lead_indices = [
        rec.sig_name.index(lead)
        for lead in target_leads
        if lead in rec.sig_name
    ]

    if len(lead_indices) != 4:
        return None, None

    raw_matrix = sig[:, lead_indices]

    mean = np.mean(raw_matrix, axis=0)
    std = np.std(raw_matrix, axis=0)
    normalized = (raw_matrix - mean) / (std + 1e-8)

    return np.expand_dims(normalized, axis=0), rec

# =============================
# ECG PLOT 3 DETIK, 4 LEAD
# =============================
def plot_ecg_3sec(rec):
    target_leads = ['V1', 'V2', 'V3', 'II']
    sig = rec.p_signal
    fs = rec.fs
    n_samples = min(int(fs*3), sig.shape[0])  # 3 detik
    time = np.arange(0, n_samples)/fs  # dalam detik

    lead_indices = [rec.sig_name.index(l) for l in target_leads if l in rec.sig_name]

    fig, axes = plt.subplots(len(lead_indices), 1, figsize=(12, 2.5*len(lead_indices)), sharex=True)

    if len(lead_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(lead_indices):
        filtered = butter_bandpass_filter(sig[:n_samples, idx], fs=fs)
        axes[i].plot(time, filtered, color='navy')
        axes[i].set_ylabel(f"{rec.sig_name[idx]}")
        axes[i].grid(True)
        axes[i].set_ylim(-1.2, 2.5)
        axes[i].axhline(0, color='black', lw=0.8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ECG Signal - First 3 Seconds (Leads: {', '.join(target_leads)})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

# =============================
# HIGHLIGHT FUNCTION
# =============================
def highlight_signal(signal):
    threshold = np.mean(signal) + 2 * np.std(signal)
    return signal > threshold

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

# =============================
# UI
# =============================
st.title("🫀 Brugada Syndrome Detection")
st.write("Predict Brugada Syndrome from ECG signals using 1D-CNN")

st.markdown("---")

patient_id = st.text_input("Enter Patient ID: ")

uploaded_files = st.file_uploader(
    "Upload ECG files (.dat & .hea)",
    type=["dat", "hea"],
    accept_multiple_files=True
)
manual_path = st.text_input("Or enter ECG path (without extension)")

ecg_path = None

if uploaded_files:
    temp_dir = tempfile.mkdtemp()

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    # ambil nama record tanpa extension
    for file in uploaded_files:
        if file.name.endswith(".dat"):
            ecg_path = os.path.join(temp_dir, file.name.replace(".dat", ""))

elif manual_path:
    ecg_path = manual_path

# =============================
# PREDICTION
# =============================
if st.button("🔍 Predict"):

    if not ecg_path:
        st.warning("Please upload file or input path!")
        st.stop()

    X, rec = preprocess_ecg(ecg_path)

    if X is None:
        st.error("Missing required leads (V1, V2, V3, II)")
        st.stop()

    # =============================
    # MODEL PREDICTION
    # =============================
    infer = model.signatures["serving_default"]
    output = infer(tf.constant(X))

 
    y_prob = float(list(output.values())[0].numpy()[0][0])
    y_pred = int(y_prob >= threshold)

    # =============================
    # RESULT DISPLAY
    # =============================
    st.subheader("📊 Prediction Results")
    st.metric("Probability", f"{y_prob:.4f}")

    progress_val = int(y_prob * 100)
    st.progress(progress_val)

    if progress_val >= 70:
        st.warning("🔥 High Risk")
    elif progress_val >= 40:
        st.info("⚠️ Moderate Risk")
    else:
        st.success("✅ Low Risk")

    if y_pred == 1:
        st.error("⚠️ Brugada Detected")
    else:
        st.success("✅ Normal")

    # =============================
    # ECG VISUALIZATION 3 DETIK
    # =============================
    st.markdown("---")
    st.subheader("📈 ECG Visualization (4 Leads, 3 Seconds)")
    fig = plot_ecg_3sec(rec)
    st.pyplot(fig)
    plt.close(fig)



    # =============================
    # DOWNLOAD RESULT
    # =============================
    result_df = pd.DataFrame({
        "Probability": [y_prob],
        "Prediction": ["Brugada" if y_pred else "Normal"]
    })

    st.download_button(
        "💾 Download Result",
        result_df.to_csv(index=False),
        "result.csv",
        "text/csv"
    )

    st.caption(f"Threshold: {threshold}")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("💡 Created by Vector Victory Team | 2026")
