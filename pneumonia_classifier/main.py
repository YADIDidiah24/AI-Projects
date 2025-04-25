import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import io
from util import classify, generate_heatmap, show_intermediate_activations, plot_prediction_confidence

# Page configuration
st.set_page_config(
    page_title="PneumoScan AI - Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

# Set custom styles
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #3366ff;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #505050;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .normal-result {
        color: #00cc66;
        font-size: 28px;
        font-weight: bold;
    }
    .pneumonia-result {
        color: #ff5050;
        font-size: 28px;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Optional background
# set_background('./bgs/bg5.png')

# Header
st.markdown('<div class="main-header">🫁 PneumoScan AI - Advanced Pneumonia Detection System</div>', unsafe_allow_html=True)
# Sidebar with information
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 15px 0; color: white; margin-top: -50">
            <h2 style="color: #3498db;">🫁 PneumoScan AI</h2>
            <p style="font-size: 14px; color: #bdc3c7;">Version 1.0</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("## About")
    st.info("This application uses deep learning to detect pneumonia from chest X-ray images.")
    
    st.markdown("## How to use")
    st.write("1. Upload a chest X-ray image using the file uploader")
    st.write("2. The system will analyze the image and provide a diagnosis")
    st.write("3. View the heatmap to understand which areas influenced the prediction")
    
    # Model metrics (from your training results)
    st.markdown("## Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "98.42%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.metric("AUC", "0.9913")
        st.markdown('</div>', unsafe_allow_html=True)
    
    metrics_expander = st.expander("View detailed metrics")
    with metrics_expander:
        st.write("Precision (Pneumonia): 70.8%")
        st.write("Recall (Pneumonia): 99.5%")
        st.write("Specificity (Normal): 31.6%")
        st.write("F1 Score: 0.83")
    
    # Explanation about pneumonia X-rays
    pneumonia_expander = st.expander("What does pneumonia look like in X-rays?")
    with pneumonia_expander:
        st.markdown("""
        **Key indicators of pneumonia in chest X-rays:**
        
        - **White opacities**: Pneumonia shows as white patches or infiltrates in the lungs
        - **Location**: Often appears in lower lobes of the lungs
        - **Consolidation**: Areas where air spaces are filled with fluid (appearing white)
        - **Air bronchograms**: Dark bronchi surrounded by white infiltrate
        
        Normal chest X-rays show clear, black lung fields with visible blood vessels appearing as thin white lines.
        """)

# Main content
st.markdown('<div class="sub-header">Upload a chest X-ray image</div>', unsafe_allow_html=True)

# Upload file
file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

# Load classifier with appropriate error handling
try:
    # Try loading the model directly
    model = load_model('pneumonia_model.h5')
except:
    # If that fails, try with custom objects
    from keras.utils import get_custom_objects
    from keras.layers import Dropout
    
    # Define a custom Dropout layer that can be serialized/deserialized
    class FixedDropout(Dropout):
        def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
            super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        
        def get_config(self):
            config = super().get_config()
            return config
    
    # Register the custom layer
    get_custom_objects().update({'FixedDropout': FixedDropout})
    model = load_model('pneumonia_model.h5')

# Load class names
class_names = ['PNEUMONIA', 'NORMAL']

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_container_width=True, caption="Uploaded X-ray")

    # Classify image
    class_name, conf_score = classify(image, model, class_names)
    
    # Generate heatmap
    with col2:

        show_intermediate_activations(image, model)
    
    st.markdown('<div class="sub-header">Diagnosis Result</div>', unsafe_allow_html=True)
    
    result_class = "pneumonia-result" if class_name == "PNEUMONIA" else "normal-result"
    st.markdown(f'<div class="{result_class}">{class_name}</div>', unsafe_allow_html=True)
    
    # Display confidence with progress bar
    st.write(f"Confidence Score: {conf_score:.1%}")
    st.progress(float(conf_score))
    
    # Additional information based on diagnosis
    if class_name == "PNEUMONIA":
        st.markdown('<div >', unsafe_allow_html=True)
        st.markdown("""
        ### Pneumonia Detected
        The model has detected patterns consistent with pneumonia in this X-ray. Key observations:
        - Presence of opacities in the lung fields
        - Possible consolidation in affected areas
        - Reduced lung clarity compared to healthy X-rays
        
        **Note:** This is a preliminary AI-based assessment. Please consult with a healthcare professional for proper diagnosis and treatment.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div >', unsafe_allow_html=True)
        st.markdown("""
        ### No Pneumonia Detected
        The model suggests this X-ray appears normal. Key observations:
        - Clear lung fields
        - Normal vascular markings
        - No significant opacities or consolidation
        
        **Note:** This is a preliminary AI-based assessment. Please consult with a healthcare professional for proper diagnosis and treatment.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Display sample images if no file is uploaded
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Instructions
    Please upload a clear, frontal chest X-ray image.
    
    For best results:
    - Use high-resolution images
    - Ensure the entire lung field is visible
    - Upload PA (posteroanterior) view X-rays when possible
    """)
    st.markdown('</div>', unsafe_allow_html=True)
