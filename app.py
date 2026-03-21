import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Blood Group Predictor", page_icon="🩸", layout="wide")

if 'res_label' not in st.session_state:
    st.session_state['res_label'] = None
    st.session_state['res_conf'] = 0

st.markdown('<div style="text-align:center; padding:20px; border-bottom:5px solid #4db6ac;"><h2>M.A.M. COLLEGE OF ENGINEERING</h2><p>AI & Data Science | Official Blood Group Detection</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

def get_labels():
    return ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Upload Fingerprint")
    file = st.file_uploader("Choose a clear JPG/PNG", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, width=250)
        if st.button("RUN AI ANALYSIS"):
            model = load_engine()
            if model:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized).astype('float32') / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_batch, verbose=0)
                labels = get_labels()
                st.session_state['res_label'] = labels[np.argmax(preds)]
                st.session_state['res_conf'] = np.max(preds) * 100
            else:
                st.error("Engine Offline.")

with c2:
    st.subheader("2. AI Prediction")
    if st.session_state['res_label']:
        st.success(f"Result: {st.session_state['res_label']}")
        st.info(f"Confidence: {st.session_state['res_conf']:.2f}%")
        if st.button("Scan Another Fingerprint"):
            st.session_state['res_label'] = None
            st.rerun()
    else:
        st.warning("Awaiting fingerprint upload...")
