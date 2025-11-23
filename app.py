"""
Streamlit Web Application for MNIST Digit Classifier
Deploys a trained Deep Learning Classifier as an interactive web app
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from tensorflow import keras

# Configure page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-success {
        padding: 1rem;
        background-color: #90EE90;
        border-radius: 0.5rem;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        color: #2d5016;
    }
    .confidence-high {
        color: #2d5016;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## MNIST Digit Classifier")
    st.markdown("---")
    
    st.markdown("""
    ### About This Application
    This web application classifies handwritten digits (0-9) using a trained 
    Deep Learning CNN model.
    
    ### How to Use
    1. **Upload an Image**: Click the upload area to select a handwritten digit image
       - Supported formats: `.jpg`, `.jpeg`, `.png`
       - Image should be roughly square (similar to 28×28 pixels)
    
    2. **View Results**: After clicking "Predict", the app will:
       - Display the predicted digit (highlighted in green)
       - Show confidence scores for top-5 predictions
       - Visualize the probabilities in a bar chart
    
    ### Model Details
    - **Architecture**: CNN (Convolutional Neural Network)
    - **Input Size**: 28×28 grayscale images
    - **Output Classes**: 10 (digits 0-9)
    - **Framework**: TensorFlow/Keras
    
    ### Tips for Best Results
    - Ensure the digit is centered and clear
    - Use high contrast (dark digit on light background)
    - Avoid image that are too small or blurry
    """)
    
    st.markdown("---")
    st.markdown("**Model Status**: Ready for prediction")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Title
st.markdown('<h1 class="main-header">Handwritten Digit Classifier</h1>', 
            unsafe_allow_html=True)

st.markdown("""
---
Upload an image of a handwritten digit and the model will predict what digit it is!
""")

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    """Load the trained model and class labels"""
    model_path = "dnn_model.keras"
    # model_path = "baseline_cnn.keras"
    # model_path = "dnn_dropout.keras"
    # model_path = "mobilenetv2_transfer.keras"
    # model_path = "resnet50_transfer.keras"
    labels_path = "class_labels.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please ensure `mnist_classifier.keras` is in the same directory as this app")
        return None, None
    
    if not os.path.exists(labels_path):
        st.warning(f"⚠ Labels file not found: {labels_path}. Using default labels (0-9)")
        labels = [str(i) for i in range(10)]
    else:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    
    try:
        model = keras.models.load_model(model_path)
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(img, target_size=(28, 28)):
    """
    Preprocess the uploaded image for model prediction
    - Resize to 28×28
    - Convert to grayscale if needed
    - Normalize pixel values (0-1)
    - Invert if needed (MNIST was trained on dark backgrounds with light digits)
    """
    # Convert to grayscale if RGB
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # IMPORTANT: Detect and fix image orientation
    # MNIST was trained on images with DARK backgrounds and LIGHT digits
    # User uploads typically have LIGHT backgrounds and DARK digits
    # Calculate mean pixel value: if > 0.5, the background is light, so invert
    mean_pixel = np.mean(img_array)
    if mean_pixel > 0.5:
        # Image has light background - invert it to match MNIST training format
        img_array = 1.0 - img_array
    
    # Reshape for model input (add batch and channel dimensions)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, img

def get_top_5_predictions(probabilities, labels):
    """Get top 5 predictions with confidence scores"""
    top_indices = np.argsort(probabilities[0])[::-1][:5]
    top_predictions = [(labels[idx], probabilities[0][idx]) for idx in top_indices]
    return top_predictions

# ============================================================================
# IMAGE UPLOAD AND PREDICTION
# ============================================================================

st.subheader("Upload Image")
st.markdown("Upload a handwritten digit image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a handwritten digit"
)

if uploaded_file is not None:
    # Display uploaded image centered and constrained
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Store image for later use
    st.session_state.uploaded_image = image

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================

st.markdown("---")

if st.button("Predict Digit", type="primary", use_container_width=True):
    if "uploaded_image" not in st.session_state:
        st.error("Please upload an image first")
    else:
        # Load model and labels
        model, labels = load_model_and_labels()
        
        if model is None or labels is None:
            st.error("Could not load model or labels")
        else:
            # Preprocess image
            img_array, processed_img = preprocess_image(st.session_state.uploaded_image)
            
            # Make prediction
            with st.spinner("Analyzing digit..."):
                probabilities = model.predict(img_array, verbose=0)
            
            # Get predictions
            predicted_class = labels[np.argmax(probabilities[0])]
            confidence = np.max(probabilities[0])
            top_5 = get_top_5_predictions(probabilities, labels)
            
            # Display results
            st.markdown("---")
            
            # Top-1 Prediction (in green box)
            st.markdown(
                f'<div class="prediction-success">Predicted Digit: <span class="confidence-high">{predicted_class}</span></div>',
                unsafe_allow_html=True
            )
            
            # Confidence metric
            st.metric(
                label="Confidence Score",
                value=f"{confidence*100:.2f}%",
                delta=None
            )
            
            # Top-5 Predictions Table
            st.subheader("Top-5 Predictions")
            
            top_5_data = {
                "Rank": list(range(1, 6)),
                "Digit": [pred[0] for pred in top_5],
                "Confidence": [f"{pred[1]*100:.2f}%" for pred in top_5]
            }
            top_5_df = pd.DataFrame(top_5_data)
            st.dataframe(top_5_df, use_container_width=True, hide_index=True)
            
            # Top-5 Predictions Bar Chart
            st.subheader("Confidence Distribution")
            chart_data = pd.DataFrame({
                "Digit": [pred[0] for pred in top_5],
                "Confidence": [pred[1]*100 for pred in top_5]
            })
            st.bar_chart(chart_data.set_index("Digit"), use_container_width=True)
            
            # Additional info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Prediction", predicted_class)
            with col_info2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            with col_info3:
                next_best = top_5[1][0] if len(top_5) > 1 else "N/A"
                st.metric("Second Best", next_best)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>MNIST Digit Classifier | Deep Learning Web Application</p>
    <p>Built with Streamlit and TensorFlow/Keras</p>
</div>
""", unsafe_allow_html=True)
