import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Blood Group AI", page_icon="🩸", layout="wide")

# --- 2. PROFESSIONAL PASTEL CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0f7f4 0%, #ffffff 100%); }
    .header-card {
        background: #ffffff; padding: 30px; border-radius: 20px;
        border-left: 8px solid #81c784; box-shadow: 0 10px 25px rgba(0,0,0,0.03);
        text-align: center; margin-bottom: 40px;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.9); padding: 50px; border-radius: 30px;
        border: 1px solid #c8e6c9; text-align: center;
        box-shadow: 0 15px 35px rgba(129, 199, 132, 0.1);
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3em;
        background-color: #81c784; color: white; border: none;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
st.markdown('''
    <div class="header-card">
        <h1 style="color: #2e7d32; margin:0;">M.A.M. COLLEGE OF ENGINEERING</h1>
        <p style="color: #66bb6a; font-weight: 500;">Department of AI & Data Science</p>
        <h4 style="color: #1b5e20; font-weight: 400;">Non-Invasive Blood Group Detection System</h4>
    </div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Project Guide")
    st.info("Mr. K. Ilango Xavier")
    st.subheader("Project Team")
    st.write("• Z. Najla"); st.write("• D. Renuga Devi")
    st.write("• T. Nivetha"); st.write("• G. Aswinekha")

# --- 4. CORE LOGIC ---
@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

# Initialize session state keys to prevent KeyErrors
if 'res_label' not in st.session_state:
    st.session_state['res_label'] = None
if 'res_conf' not in st.session_state:
    st.session_state['res_conf'] = 0

# --- 5. MAIN INTERFACE ---
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("#### 📤 Step 1: Upload Scan")
    file = st.file_uploader("Upload Fingerprint", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        if st.button("RUN ANALYSIS"):
            model = load_engine()
            if model:
                with st.spinner("Analyzing..."):
                    img_resized = img.resize((224, 224))
                    img_arr = np.array(img_resized).astype('float32') / 255.0
                    img_batch = np.expand_dims(img_arr, axis=0)
                    preds = model.predict(img_batch, verbose=0)
                    
                    labels = ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']
                    st.session_state['res_label'] = labels[np.argmax(preds)]
                    st.session_state['res_conf'] = np.max(preds) * 100
            else:
                st.error("Model not found.")

with c2:
    st.markdown("#### 🔍 Step 2: Prediction")
    if st.session_state['res_label']:
        st.markdown(f'''
            <div class="result-card">
                <p style="color: #81c784; font-weight:600; letter-spacing:1px;">RESULT</p>
                <h1 style="color: #2e7d32; font-size: 90px; margin:0;">{st.session_state['res_label']}</h1>
                <p>Confidence: <b>{st.session_state['res_conf']:.2f}%</b></p>
            </div>
        ''', unsafe_allow_html=True)
        if st.button("Reset"):
            st.session_state['res_label'] = None
            st.rerun()
    else:
        st.info("Awaiting biometric input...")
