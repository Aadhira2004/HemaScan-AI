import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Blood Group AI", page_icon="🩸", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #f1f8e9; }
    .header-card { background: white; padding: 25px; border-radius: 15px; border-bottom: 5px solid #81c784; text-align: center; margin-bottom: 20px; }
    .result-card { background: white; padding: 30px; border-radius: 20px; border: 1px solid #c8e6c9; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-card"><h2 style="color: #2e7d32;">M.A.M. COLLEGE OF ENGINEERING</h2><p>AI & Data Science | System Diagnostic Mode</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

# --- CRITICAL: THE LABEL ORDER ---
# If the output is wrong, we might need to change the order of this list!
def get_labels():
    return ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']

if 'res_label' not in st.session_state:
    st.session_state['res_label'] = None

c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Analysis")
    file = st.file_uploader("Upload Fingerprint", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, width=250)
        if st.button("RUN AI"):
            model = load_engine()
            if model:
                # Preprocessing
                img_res = img.resize((224, 224))
                img_arr = np.array(img_res).astype('float32') / 255.0
                img_batch = np.expand_dims(img_arr, axis=0)
                
                preds = model.predict(img_batch, verbose=0)
                predicted_index = np.argmax(preds)
                labels = get_labels()
                
                st.session_state['res_label'] = labels[predicted_index]
                st.session_state['res_index'] = predicted_index # Save the math index
                st.session_state['res_conf'] = np.max(preds) * 100

with c2:
    st.subheader("2. Result")
    if st.session_state.get('res_label'):
        st.markdown(f'''
            <div class="result-card">
                <h1 style="color: #2e7d32;">{st.session_state['res_label']}</h1>
                <p>Confidence: {st.session_state['res_conf']:.2f}%</p>
                <p style="color: gray; font-size: 12px;">Model Index Picked: {st.session_state['res_index']}</p>
            </div>
        ''', unsafe_allow_html=True)
