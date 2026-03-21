import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Blood Group AI | MAMCE", page_icon="🩸", layout="wide")

# --- 2. PROFESSIONAL UI THEME ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f1f8e9 0%, #ffffff 100%); }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .header-container {
        background: white; padding: 30px; border-radius: 20px;
        border-bottom: 8px solid #66bb6a; box-shadow: 0 10px 30px rgba(0,0,0,0.04);
        text-align: center; margin-bottom: 30px;
    }
    .result-glass {
        background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px);
        padding: 40px; border-radius: 30px; border: 1px solid #c8e6c9;
        text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.03);
    }
    .stButton>button {
        width: 100%; border-radius: 12px; background: #66bb6a; color: white;
        font-weight: 700; height: 3.5em; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INITIALIZE SESSION STATE (Fixes the KeyError) ---
if 'res_label' not in st.session_state:
    st.session_state['res_label'] = None
if 'res_conf' not in st.session_state:
    st.session_state['res_conf'] = 0
if 'res_index' not in st.session_state:
    st.session_state['res_index'] = 0

# --- 4. SIDEBAR (GUIDE & TEAM) ---
with st.sidebar:
    st.markdown("### 👨‍🏫 Project Guidance")
    st.success("**Mr. K. Ilango Xavier**")
    st.caption("Assistant Professor | AI & DS")
    st.markdown("---")
    st.markdown("### 👥 Research Team")
    for member in ["Z. Najla", "D. Renuga Devi", "T. Nivetha", "G. Aswinekha"]:
        st.markdown(f"**• {member}**")
    st.markdown("---")
    st.info("M.A.M. College of Engineering")

# --- 5. MAIN HEADER ---
st.markdown('''
    <div class="header-container">
        <h2 style="color: #1b5e20; margin:0;">M.A.M. COLLEGE OF ENGINEERING</h2>
        <p style="color: #43a047; font-weight: 500;">Department of Artificial Intelligence & Data Science</p>
        <h4 style="color: #2e7d32; font-weight: 400; margin-top:10px;">Non-Invasive Blood Group Detection System</h4>
    </div>
''', unsafe_allow_html=True)

# --- 6. CORE LAYOUT ---
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.subheader("📁 Biometric Input")
    file = st.file_uploader("Upload Fingerprint Image", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        if st.button("EXECUTE AI ANALYSIS"):
            if os.path.exists('blood_group_model.h5'):
                model = tf.keras.models.load_model('blood_group_model.h5', compile=False)
                with st.spinner("Processing Swin Transformer Layers..."):
                    img_res = img.resize((224, 224))
                    img_arr = np.array(img_res).astype('float32') / 255.0
                    img_batch = np.expand_dims(img_arr, axis=0)
                    preds = model.predict(img_batch, verbose=0)
                    
                    labels = ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']
                    st.session_state['res_index'] = np.argmax(preds)
                    st.session_state['res_label'] = labels[st.session_state['res_index']]
                    st.session_state['res_conf'] = np.max(preds) * 100
            else:
                st.error("Model file (.h5) not detected in repository.")

with col_right:
    st.subheader("📊 Result Inference")
    if st.session_state['res_label']:
        st.markdown(f'''
            <div class="result-glass">
                <p style="color: #66bb6a; font-weight: 700;">CLASSIFICATION</p>
                <h1 style="color: #1b5e20; font-size: 100px; margin: 0;">{st.session_state['res_label']}</h1>
                <p>AI Confidence: <b>{st.session_state['res_conf']:.2f}%</b></p>
                <hr style="border: 0.5px solid #e8f5e9;">
                <p style="color: #999; font-size: 12px;">Internal Model Index: {st.session_state['res_index']}</p>
            </div>
        ''', unsafe_allow_html=True)
        if st.button("Clear Results"):
            st.session_state['res_label'] = None
            st.rerun()
    else:
        st.info("Awaiting fingerprint scan for deep learning analysis...")

