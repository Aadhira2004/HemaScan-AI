import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Blood Group AI", page_icon="🩸", layout="wide")

# --- 2. THE PASTEL GREEN CSS ---
st.markdown("""
    <style>
    /* Full Page Background */
    .stApp {
        background-color: #f1f8e9; /* Pastel Green / Mint */
    }
    
    /* Header Card */
    .header-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border-bottom: 6px solid #81c784;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 40px;
    }

    /* Result Card */
    .result-card {
        background: #ffffff;
        padding: 40px;
        border-radius: 25px;
        border: 1px solid #c8e6c9;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #e8f5e9;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #81c784;
        color: white;
        border: none;
        font-weight: 600;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. UI CONTENT ---
st.markdown('''
    <div class="header-card">
        <h2 style="color: #2e7d32; margin:0;">M.A.M. COLLEGE OF ENGINEERING</h2>
        <p style="color: #66bb6a; font-weight: 500; margin-top:5px;">Department of AI & Data Science</p>
        <h4 style="color: #1b5e20; font-weight: 400; margin-top:10px;">Non-Invasive Blood Group Detection</h4>
    </div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Project Guide")
    st.info("Mr. K. Ilango Xavier")
    st.markdown("---")
    st.subheader("Project Team")
    st.write("• Z. Najla")
    st.write("• D. Renuga Devi")
    st.write("• T. Nivetha")
    st.write("• G. Aswinekha")

# --- 4. ENGINE ---
@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

if 'res_label' not in st.session_state:
    st.session_state['res_label'] = None

# --- 5. MAIN LAYOUT ---
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("#### 📤 Upload Fingerprint")
    file = st.file_uploader("Select Image File", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        if st.button("RUN AI ANALYSIS"):
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
                st.error("Model Engine Offline.")

with c2:
    st.markdown("#### 🔍 AI Result")
    if st.session_state['res_label']:
        st.markdown(f'''
            <div class="result-card">
                <p style="color: #81c784; font-weight: 600; letter-spacing: 1px;">PREDICTED GROUP</p>
                <h1 style="color: #2e7d32; font-size: 80px; margin: 10px 0;">{st.session_state['res_label']}</h1>
                <p style="color: #66bb6a;">Confidence: <b>{st.session_state['res_conf']:.2f}%</b></p>
            </div>
        ''', unsafe_allow_html=True)
        if st.button("Clear and Scan New"):
            st.session_state['res_label'] = None
            st.rerun()
    else:
        st.info("Awaiting fingerprint scan...")
