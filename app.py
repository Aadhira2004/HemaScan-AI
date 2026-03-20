import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Blood Group Predictor", page_icon="🩸", layout="wide")

st.markdown('<div style="text-align:center; padding:20px; border-bottom:5px solid #4db6ac;"><h2>M.A.M. COLLEGE OF ENGINEERING</h2><p>AI & Data Science | Swin Transformer Analysis</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    if os.path.exists('blood_group_model.h5'):
        return tf.keras.models.load_model('blood_group_model.h5', compile=False)
    return None

def get_labels():
    return ['A', 'A+', 'A-', 'AB', 'AB+', 'AB-', 'B', 'B+', 'B-', 'O', 'O+', 'O-']

c1, c2 = st.columns(2)
with c1:
    st.subheader("Scan Input")
    file = st.file_uploader("Upload Fingerprint", type=['jpg','png','jpeg'])
    
    # Checkbox to try Grayscale if the model is stuck on 'O'
    use_gray = st.checkbox("Use Grayscale Preprocessing", value=False)

    if file:
        img = Image.open(file).convert('RGB')
        if use_gray:
            img = ImageOps.grayscale(img).convert('RGB')
        st.image(img, width=250)
        
        if st.button("RUN AI ANALYSIS"):
            model = load_engine()
            if model:
                with st.spinner("Analyzing..."):
                    # Preprocessing
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype('float32') / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)
                    
                    # Prediction
                    preds = model.predict(img_batch, verbose=0)
                    labels = get_labels()
                    
                    st.session_state['res_label'] = labels[np.argmax(preds)]
                    st.session_state['res_conf'] = np.max(preds) * 100
                    st.session_state['raw_scores'] = preds[0].tolist() # Convert to list for session state
            else:
                st.error("Model file not found.")

with c2:
    st.subheader("Result")
    if 'res_label' in st.session_state:
        st.success(f"Predicted: {st.session_state['res_label']} ({st.session_state['res_conf']:.2f}%)")
        
        # Safe check for raw_scores to avoid KeyError
        if 'raw_scores' in st.session_state:
            with st.expander("See Raw Probability Scores"):
                st.bar_chart(st.session_state['raw_scores'])
                st.write("Labels: A, A+, A-, AB, AB+, AB-, B, B+, B-, O, O+, O-")
        
        if st.button("Clear and Reset"):
            for key in ['res_label', 'res_conf', 'raw_scores']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    else:
        st.info("Awaiting scan...")
