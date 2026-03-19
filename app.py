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

# --- ADVANCED PASTEL CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8fbfc; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #e0f2f1 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-bottom: 5px solid #80cbc4;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    /* Professional Cards */
    .result-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .team-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #ffab91;
        margin-bottom: 10px;
        font-size: 0.9rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.02);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #4db6ac !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3em !important;
        font-weight: 600 !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #00897b !important;
        transform: translateY(-2px);
    }
    
    h1, h2, h3 { color: #37474f; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC: ASSET LOADING ---
@st.cache_resource
def load_assets():
    if not os.path.exists('blood_group_model.h5'): return None, None
    model = tf.keras.models.load_model('blood_group_model.h5', compile=False)
    labels = open('labels.txt', 'r').read().split(',') if os.path.exists('labels.txt') else ['A', 'B', 'AB', 'O']
    return model, labels

# --- UI: TOP HEADER SECTION ---
st.markdown(f'''
    <div class="main-header">
        <h3 style="margin:0; color: #00796b; letter-spacing: 2px;">ROHINI COLLEGE OF ENGINEERING AND TECHNOLOGY</h3>
        <p style="margin:5px 0; color: #546e7a; font-weight: 500;">Department of Biomedical Engineering</p>
        <h1 style="margin:10px 0; font-size: 2.5rem;">🩸 HemaScan AI: Non-Invasive Diagnostics</h1>
    </div>
''', unsafe_allow_html=True)

# --- SIDEBAR: TEAM & GUIDE ---
with st.sidebar:
    st.markdown("### 👨‍🏫 Project Guidance")
    st.info("**Mr. K. Ilango Xavier**\n\nHOD, Dept of AI & DS")
    
    st.markdown("### 👥 Research Team")
    st.markdown('<div class="team-card"><b>Aadhira Suleim A R</b><br>Final Year B.E. (BME)</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-card"><b>Z. Najla</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="team-card"><b>D. Renuga Devi</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="team-card"><b>T. Nivetha</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="team-card"><b>G. Aswinekha</b></div>', unsafe_allow_html=True)
    
    st.divider()
    st.caption("Developed for Final Year Project 2026")

# --- MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["🔍 Analysis Chamber", "🧬 Swin Transformer Specs", "📊 Diagnostic Logs"])

with tab1:
    c1, c2 = st.columns([1, 1], gap="large")
    
    with c1:
        st.subheader("Patient Data Input")
        input_type = st.radio("Source:", ["Scan/Upload", "Live Camera"], horizontal=True)
        
        file = st.camera_input("Scanner") if input_type == "Live Camera" else st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, width=280, caption="Captured Minutiae Map")
            
            if st.button("START AI DIAGNOSIS"):
                model, labels = load_assets()
                if model:
                    with st.spinner("Processing Hierarchical Patches..."):
                        # Prep & Predict
                        arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                        start = time.time()
                        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)
                        latency = round(time.time() - start, 3)
                        
                        # Store Result
                        res = labels[np.argmax(preds)]
                        conf = np.max(preds) * 100
                        st.session_state['last_res'] = {"label": res, "conf": conf, "time": latency}
                        
                        # Log it
                        if 'history' not in st.session_state: st.session_state['history'] = []
                        st.session_state['history'].append({"Time": time.strftime("%H:%M"), "Group": res, "Confidence": f"{conf:.1f}%"})
                else:
                    st.error("Model file (.h5) missing from server.")

    with c2:
        st.subheader("Clinical Result")
        if 'last_res' in st.session_state:
            data = st.session_state['last_res']
            st.markdown(f'''
                <div class="result-card">
                    <p style="color: #90a4ae; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 2px;">Result Identified</p>
                    <h1 style="color: #e53935; font-size: 5rem; margin: 0;">{data['label']}</h1>
                    <p style="font-size: 1.2rem; color: #455a64;">Confidence Score: <b>{data['conf']:.2f}%</b></p>
                    <p style="font-size: 0.8rem; color: #90a4ae;">Inference Time: {data['time']}s</p>
                </div>
            ''', unsafe_allow_html=True)
            if st.button("Reset for New Patient"):
                del st.session_state['last_res']
                st.rerun()
        else:
            st.info("Awaiting input for diagnostic processing.")

with tab2:
    st.markdown("### Swin Transformer (Shifted Window) Architecture")
    st.write("Our system partitions the fingerprint into local windows, applying self-attention across shifted layers to capture both fine ridge details and global patterns.")
    

with tab3:
    st.subheader("Session History")
    if 'history' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
    else:
        st.write("No logs for current session.")

