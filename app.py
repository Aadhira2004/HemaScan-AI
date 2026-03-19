import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="HemaScan AI | Clinical Suite",
    page_icon="🩸",
    layout="wide"
)

# --- ADVANCED PASTEL MEDICAL STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f0f7f7; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #e0f2f1 0%, #ffffff 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border-bottom: 6px solid #80cbc4;
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
        border-left: 5px solid #4db6ac;
        margin-bottom: 12px;
        font-size: 0.9rem;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.03);
    }

    /* Buttons */
    .stButton>button {
        background-color: #4db6ac !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        height: 3.5em !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(77,182,172,0.3) !important;
    }
    .stButton>button:hover {
        background-color: #00897b !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,137,123,0.4) !important;
    }
    
    h1, h2, h3 { color: #263238; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent; 
        border-radius: 10px; 
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI: TOP INSTITUTIONAL HEADER ---
st.markdown(f'''
    <div class="main-header">
        <h3 style="margin:0; color: #00796b; letter-spacing: 2px; font-weight: 800;">ROHINI COLLEGE OF ENGINEERING AND TECHNOLOGY</h3>
        <p style="margin:5px 0; color: #546e7a; font-size: 1.1rem;">Department of Biomedical Engineering</p>
        <hr style="width: 50%; margin: 15px auto; border: 0.5px solid #b2dfdb;">
        <h1 style="margin:10px 0; font-size: 2.8rem; color: #004d40;">🩸 HemaScan AI</h1>
        <p style="color: #00796b; font-weight: 600;">Automated Non-Invasive Blood Group Diagnostic Suite</p>
    </div>
''', unsafe_allow_html=True)

# --- SIDEBAR: TEAM & GUIDE DETAILS ---
with st.sidebar:
    st.markdown("### 👨‍🏫 Project Guidance")
    st.info("**Mr. K. Ilango Xavier**\nHOD, Dept of AI & DS")
    
    st.markdown("### 👥 Research Team")
    st.markdown('<div class="info-card"><b>Aadhira Suleim A R</b><br>Final Year, B.E. BME</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>Z. Najla</b><br>Research Contributor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>D. Renuga Devi</b><br>Research Contributor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>T. Nivetha</b><br>Research Contributor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>G. Aswinekha</b><br>Research Contributor</div>', unsafe_allow_html=True)
    
    st.divider()
    st.caption("Developed for Academic Excellence 2026")

# --- APP LOGIC: ENGINE LOADING ---
@st.cache_resource
def load_clinical_model():
    if not os.path.exists('blood_group_model.h5'): return None, None
    model = tf.keras.models.load_model('blood_group_model.h5', compile=False)
    labels = open('labels.txt', 'r').read().split(',') if os.path.exists('labels.txt') else ['A', 'B', 'AB', 'O']
    return model, labels

# --- MAIN TABS ---
tab_scan, tab_tech, tab_logs = st.tabs(["🔍 Patient Analysis", "🧬 Swin-T Architecture", "📊 Clinical Logs"])

with tab_scan:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Data Acquisition")
        mode = st.radio("Input Selection:", ["Capture Live", "Upload Scan"], horizontal=True)
        
        file = st.camera_input("Scanner") if mode == "Capture Live" else st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, width=320, caption="Normalized Ridge Pattern")
            
            if st.button("EXECUTE ANALYSIS"):
                model, labels = load_clinical_model()
                if model:
                    with st.spinner("Processing Hierarchical Windows..."):
                        # Preprocessing
                        img_arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                        start = time.time()
                        preds = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
                        latency = round(time.time() - start, 3)
                        
                        # Result
                        res = labels[np.argmax(preds)]
                        conf = np.max(preds) * 100
                        st.session_state['res_data'] = {"label": res, "conf": conf, "time": latency}
                        
                        # History
                        if 'history' not in st.session_state: st.session_state['history'] = []
                        st.session_state['history'].append({"Time": time.strftime("%H:%M"), "Group": res, "Accuracy": f"{conf:.1f}%"})
                else:
                    st.error("Engine Error: .h5 file not found on server.")

    with col_output:
        st.subheader("Diagnostic Result")
        if 'res_data' in st.session_state:
            data = st.session_state['res_data']
            st.markdown(f'''
                <div class="result-card">
                    <p style="color: #90a4ae; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Predicted Blood Type</p>
                    <h1 style="color: #ef5350; font-size: 6rem; margin: 10px 0;">{data['label']}</h1>
                    <p style="font-size: 1.4rem; color: #37474f;">Confidence: <b>{data['conf']:.2f}%</b></p>
                    <p style="font-size: 0.8rem; color: #90a4ae;">Latency: {data['time']}s | Model: Swin-Transformer</p>
                </div>
            ''', unsafe_allow_html=True)
            if st.button("Reset for New Patient"):
                del st.session_state['res_data']
                st.rerun()
        else:
            st.info("System ready. Please provide a fingerprint scan to begin.")

with tab_tech:
    st.markdown("### Swin Transformer Methodology")
    st.write("This clinical tool utilizes a **Shifted Window (Swin) Transformer** which performs hierarchical feature extraction. Unlike standard CNNs, Swin-T maintains a global context by shifting windows between layers, making it exceptionally accurate at identifying fine minutiae in fingerprint ridges.")

with tab_logs:
    st.subheader("Session Diagnostic History")
    if 'history' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
    else:
        st.write("No patient records in current session.")

st.divider()
st.caption("© 2026 Rohini College of Engineering and Technology | BME Final Year Project")
