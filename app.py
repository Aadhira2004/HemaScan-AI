import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="HemaScan AI | Clinical Diagnostics",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL MEDICAL CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3.5em; 
        background-color: #c62828; color: white; font-weight: bold; border: none;
    }
    .prediction-card { 
        padding: 30px; border-radius: 15px; background-color: white; 
        border-left: 10px solid #c62828; box-shadow: 0 10px 20px rgba(0,0,0,0.08); 
        text-align: center; margin-top: 20px;
    }
    .status-text { font-size: 14px; color: #7f8c8d; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ENGINE ---
@st.cache_resource
def load_engine():
    if not os.path.exists('blood_group_model.h5'): return None, None
    model = tf.keras.models.load_model('blood_group_model.h5', compile=False)
    if os.path.exists('labels.txt'):
        with open('labels.txt', 'r') as f:
            labels = [l.strip() for l in f.read().split(',')]
    else:
        labels = ['A', 'B', 'AB', 'O']
    return model, labels

# --- SIDEBAR CREDENTIALS ---
with st.sidebar:
    st.title("🩸 HemaScan AI")
    st.markdown("**Clinical Version 1.0.4**")
    st.divider()
    st.write("**Team Members:**")
    st.caption("Z. Najla, D. Renuga Devi, T. Nivetha, G. Aswinekha")
    st.divider()
    st.write("**Guided By:**")
    st.info("Mr. K. Ilango Xavier\nHOD, Dept of AI & DS")
    st.caption("M.A.M. College of Engineering")

# --- MAIN APP INTERFACE ---
st.title("Automated Fingerprint-Based Blood Group Detection")
st.write("A non-invasive diagnostic tool using Swin Transformer Architecture.")

# TABBED NAVIGATION
tab_scan, tab_logs, tab_info = st.tabs(["🔍 Live Diagnostic Scanner", "📋 Session History", "🔬 Technology Detail"])

with tab_scan:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Step 1: Acquisition")
        # CAMERA vs UPLOAD TOGGLE
        input_mode = st.radio("Select Input Source:", ["Live Camera", "File Upload"], horizontal=True)
        
        raw_file = None
        if input_mode == "Live Camera":
            raw_file = st.camera_input("Scan Patient Fingerprint")
        else:
            raw_file = st.file_uploader("Upload Scanned Image", type=["jpg", "png", "jpeg"])

        if raw_file:
            img = Image.open(raw_file).convert('RGB')
            st.image(img, caption="Target Pattern for Minutiae Extraction", width=300)
            
            if st.button("RUN SWIN TRANSFORMER ANALYSIS"):
                model, labels = load_engine()
                if model:
                    with st.status("Analyzing Ridge Hierarchies...", expanded=True) as status:
                        # Preprocessing
                        test_img = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                        test_img = np.expand_dims(test_img, axis=0)
                        
                        # INFERENCE LATENCY TRACKING
                        start = time.time()
                        preds = model.predict(test_img, verbose=0)
                        end = time.time()
                        latency = round(end - start, 3)
                        
                        # UPDATE STATE
                        res = labels[np.argmax(preds)]
                        conf = np.max(preds) * 100
                        st.session_state['active_res'] = {"label": res, "conf": conf, "latency": latency}
                        
                        # LOG TO HISTORY
                        if 'logs' not in st.session_state: st.session_state['logs'] = []
                        st.session_state['logs'].append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Result": res,
                            "Confidence": f"{conf:.1f}%",
                            "Latency": f"{latency}s"
                        })
                        
                        status.update(label="Analysis Successful", state="complete")
                else:
                    st.error("Engine failure: AI weights missing.")

    with col_result:
        st.subheader("Step 2: Analysis Output")
        if st.session_state.get('active_res'):
            data = st.session_state['active_res']
            st.markdown(f'''
                <div class="prediction-card">
                    <p style="color: #7f8c8d; font-size: 16px; letter-spacing: 1px;">PREDICTED BLOOD TYPE</p>
                    <h1 style="color: #c62828; font-size: 110px; margin: 0;">{data['label']}</h1>
                    <p style="font-weight: bold; font-size: 20px; color: #2c3e50;">Confidence: {data['conf']:.2f}%</p>
                    <hr>
                    <p class="status-text">Inference Latency: {data['latency']}s | Model: Swin-T</p>
                </div>
            ''', unsafe_allow_html=True)
            
            if st.button("Clear for Next Patient"):
                st.session_state['active_res'] = None
                st.rerun()
        else:
            st.info("Please provide an input scan to generate the diagnostic report.")

with tab_logs:
    st.subheader("Clinical Session Logs")
    if 'logs' in st.session_state and st.session_state['logs']:
        df = pd.DataFrame(st.session_state['logs'])
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Session Report (CSV)", df.to_csv(index=False), "hemascan_report.csv")
    else:
        st.write("No patients scanned in current session.")

with tab_info:
    st.subheader("Technical Methodology")
    st.write("The HemaScan AI utilizes a **Swin Transformer** for hierarchical feature extraction.")
    st.markdown("""
    - **Step 1:** Image Acquisition via 224x224 RGB sensor input.
    - **Step 2:** Feature extraction focusing on minutiae points (ridge endings and bifurcations).
    - **Step 3:** Mapping hierarchical features to blood group classifications using Softmax.
    """)

st.divider()
st.caption("Department of AI & DS | M.A.M. College of Engineering")
