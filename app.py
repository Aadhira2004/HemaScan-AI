import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="HemaScan AI | M.A.M. Engineering",
    page_icon="🩸",
    layout="wide"
)

# --- ADVANCED PASTEL AI/DS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f2f7f7; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #e0f2f1 0%, #ffffff 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border-bottom: 6px solid #4db6ac;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    /* Result Card */
    .result-card {
        background-color: #ffffff;
        padding: 45px;
        border-radius: 25px;
        border: 1px solid #e0f2f1;
        box-shadow: 0 15px 35px rgba(0,121,107,0.1);
        text-align: center;
    }
    
    /* Team & Info Cards */
    .info-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #80cbc4;
        margin-bottom: 12px;
        font-size: 0.9rem;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.03);
    }

    /* Buttons */
    .stButton>button {
        background-color: #00796b !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        height: 3.5em !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,121,107,0.2) !important;
    }
    .stButton>button:hover {
        background-color: #004d40 !important;
        transform: translateY(-2px);
    }
    
    h1, h2, h3 { color: #263238; }
    </style>
    """, unsafe_allow_html=True)

# --- UI: TOP INSTITUTIONAL HEADER ---
st.markdown(f'''
    <div class="main-header">
        <h3 style="margin:0; color: #004d40; letter-spacing: 2px; font-weight: 800;">M.A.M. COLLEGE OF ENGINEERING</h3>
        <p style="margin:5px 0; color: #546e7a; font-size: 1.1rem; font-weight: 500;">Department of Artificial Intelligence and Data Science</p>
        <hr style="width: 40%; margin: 15px auto; border: 0.5px solid #b2dfdb;">
        <h1 style="margin:10px 0; font-size: 2.8rem; color: #00796b;">🩸 HemaScan AI</h1>
        <p style="color: #004d40; font-weight: 600; opacity: 0.8;">Precision Non-Invasive Blood Group Diagnostic Suite</p>
    </div>
''', unsafe_allow_html=True)

# --- SIDEBAR: TEAM & GUIDE DETAILS ---
with st.sidebar:
    st.markdown("### 👨‍🏫 Project Guidance")
    st.info("**Mr. K. Ilango Xavier**\nHOD, Dept of AI & DS")
    
    st.markdown("### 👥 Research Team")
    st.markdown('<div class="info-card"><b>Aadhira Suleim A R</b><br>Final Year, B.E. BME</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>Z. Najla</b><br>Core AI Contributor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>D. Renuga Devi</b><br>Data Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>T. Nivetha</b><br>Model Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>G. Aswinekha</b><br>Documentation</div>', unsafe_allow_html=True)
    
    st.divider()
    st.caption("M.A.M. Engineering Project 2026")

# --- APP LOGIC: ENGINE LOADING ---
@st.cache_resource
def load_clinical_model():
    model_path = 'blood_group_model.h5'
    if not os.path.exists(model_path): return None, None
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = open('labels.txt', 'r').read().split(',') if os.path.exists('labels.txt') else ['A', 'B', 'AB', 'O']
    return model, labels

# --- MAIN TABS ---
tab_scan, tab_tech, tab_logs = st.tabs(["🔍 Patient Analysis", "🧬 AI Architecture", "📊 Diagnostic History"])

with tab_scan:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Fingerprint Acquisition")
        mode = st.radio("Input Source:", ["Live Camera Capture", "Upload Static Scan"], horizontal=True)
        
        file = st.camera_input("Scanner") if mode == "Live Camera Capture" else st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, width=320, caption="Normalized Biometric Input")
            
            if st.button("EXECUTE DIAGNOSTIC ENGINE"):
                model, labels = load_clinical_model()
                if model:
                    with st.spinner("Extracting Hierarchical Ridge Features..."):
                        img_arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                        start = time.time()
                        preds = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
                        latency = round(time.time() - start, 3)
                        
                        res = labels[np.argmax(preds)]
                        conf = np.max(preds) * 100
                        st.session_state['res_data'] = {"label": res, "conf": conf, "time": latency}
                        
                        if 'history' not in st.session_state: st.session_state['history'] = []
                        st.session_state['history'].append({"Time": time.strftime("%H:%M"), "Group": res, "Confidence": f"{conf:.1f}%"})
                else:
                    st.error("System Error: AI Model weights not found on server.")

    with col_output:
        st.subheader("Clinical Result")
        if 'res_data' in st.session_state:
            data = st.session_state['res_data']
            st.markdown(f'''
                <div class="result-card">
                    <p style="color: #90a4ae; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Identified Blood Group</p>
                    <h1 style="color: #c62828; font-size: 6.5rem; margin: 10px 0; font-weight: 800;">{data['label']}</h1>
                    <hr style="width: 30%; margin: 10px auto; border-top: 2px solid #ef5350;">
                    <p style="font-size: 1.4rem; color: #37474f;">Matching Confidence: <b>{data['conf']:.2f}%</b></p>
                    <p style="font-size: 0.8rem; color: #90a4ae;">Processing Latency: {data['time']}s | Engine: Swin-Transformer</p>
                </div>
            ''', unsafe_allow_html=True)
            if st.button("Reset for Next Patient"):
                del st.session_state['res_data']
                st.rerun()
        else:
            st.info("System Standby. Please provide biometric input to initiate diagnosis.")

with tab_tech:
    st.markdown("### Swin Transformer (Shifted Window) Architecture")
    st.write("This project leverages the **Swin Transformer**, a state-of-the-art Vision Transformer that uses hierarchical feature maps and shifted window partitioning. This allows the AI to capture high-resolution minutiae details from fingerprint ridges that traditional CNNs might miss.")

with tab_logs:
    st.subheader("Session Log")
    if 'history' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
        st.download_button("Export Session Report (CSV)", pd.DataFrame(st.session_state['history']).to_csv(index=False), "hemascan_report.csv")
    else:
        st.write("No patient records for this active session.")

st.divider()
st.caption("© 2026 M.A.M. College of Engineering | Dept of Artificial Intelligence and Data Science")
