import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Blood Group Predictor", page_icon="🩸", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f5fafa; }
    .header-box { background: white; padding: 20px; border-radius: 12px; border-bottom: 5px solid #4db6ac; text-align: center; margin-bottom: 25px; }
    .result-box { background: white; padding: 35px; border-radius: 15px; border: 2px solid #e0f2f1; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-box"><h2 style="color: #00695c;">M.A.M. COLLEGE OF ENGINEERING</h2><p>AI & Data Science | Blood Group Detection</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Project Guide")
    st.info("Mr. K. Ilango Xavier")
    st.subheader("Project Team")
    st.write("Z. Najla"); st.write("D. Renuga Devi"); st.write("T. Nivetha"); st.write("G. Aswinekha")

@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

def get_labels():
    # UPDATED: Matching your labels.txt exactly
    return ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']

c1, c2 = st.columns(2)
with c1:
    st.subheader("Scan Input")
    file = st.file_uploader("Upload Fingerprint", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, width=250)
        if st.button("START AI ANALYSIS"):
            model = load_engine()
            if model:
                with st.spinner("Swin Transformer Analyzing..."):
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype('float32') / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)
                    
                    preds = model.predict(img_batch, verbose=0)
                    labels = get_labels()
                    
                    st.session_state['res_label'] = labels[np.argmax(preds)]
                    st.session_state['res_conf'] = np.max(preds) * 100
            else:
                st.error("Model file not found.")

with c2:
    st.subheader("Detection Result")
    if 'res_label' in st.session_state:
        st.markdown(f'<div class="result-box"><h1 style="color: #b71c1c; font-size: 70px;">{st.session_state["res_label"]}</h1><p>Confidence: {st.session_state["res_conf"]:.2f}%</p></div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting fingerprint scan...")
